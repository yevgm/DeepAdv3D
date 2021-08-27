import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# variable definitions
from config import *


from deep_adv_3d.loss import *
from deep_adv_3d.utils import *
from deep_adv_3d.tensor_board import *
from vista.subprocess_plotter import AdversarialPlotter
from utils.gradflow_check import *
from vista.geom_vis import plot_mesh

# debug imports
# from graphviz import Digraph, Source
import re
import torch
import wandb
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models
# from torchviz import make_dot

class Trainer:

    def __init__(self, train_data: torch.utils.data.DataLoader,
                       validation_data: torch.utils.data.DataLoader,
                       test_data: torch.utils.data.DataLoader,
                       model: nn.Module,
                       classifier:nn.Module=None):

        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.batch_size = TRAIN_BATCH_SIZE
        self.num_batch = len(self.train_data)
        self.test_num_batch = len(self.test_data)
        self.n_epoch = N_EPOCH
        self.weight_decay = WEIGHT_DECAY
        self.lr = LR
        self.classifier = classifier

        if classifier is not None:
            self.classifier.eval()
            for param in self.classifier.parameters():
                param.requires_grad = False
            self.classifier.to(DEVICE)
        self.model = model
        self.model.to(DEVICE)
        # W and B
        if LOG_GRADIENTS_WANDB:
            wandb.watch((self.model), log="all", log_freq=5)
        # early stop
        self.early_stopping = EarlyStopping(patience=EARLY_STOP_WAIT)  # hardcoded for validation loss early stop
        # checkpoints regulator
        self.init_tensor_board()
        self.checkpoint_callback = ModelCheckpoint(filepath=self.tensor_log_dir, model=self.model)

        # attributes initializations
        self.optimizer, self.scheduler = None, None


        # plotter init
        if classifier is not None:
            self.plt = AdversarialPlotter()

    def init_tensor_board(self):
        '''
        Create output directory wrapper and initialize tensorboard object instance
        '''
        create_data_output_dir()
        now = datetime.now()
        d = now.strftime("%b-%d-%Y_%H-%M-%S")
        self.tensor_log_dir = generate_new_tensorboard_results_dir(d)
        self.writer = SummaryWriter(self.tensor_log_dir, flush_secs=FLUSH_RESULTS)

    def train(self):
        # pre-train preparations

        self.optimizer, self.scheduler = self.define_optimizer()
        val_loss = 0
        for epoch in range(self.n_epoch):
            # train step
            self.model.train()
            self.one_epoch_step(epoch=epoch, split='train')

            # validation step
            if epoch % VAL_STEP_EVERY == 0:
                # pass validation through model
                val_loss = self.validation_step(epoch)
                # check if model parameters should be saved
                self.checkpoint_callback.on_validation_end(epoch=epoch, monitored_value=val_loss)
                # check if training is finished
                stop_training = self.early_stopping.on_epoch_end(epoch=epoch, monitored_value=val_loss)
                if stop_training:
                    self.early_stopping.on_train_end()
                    self.plt.finalize()
                    self.evaluate()
                    exit()

            self.scheduler.step(val_loss)


    def define_optimizer(self):
        '''
        choose the optizimer and it's hyper-parameters
        '''
        if OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(list(self.model.parameters()), lr=self.lr, betas=(0.9, 0.999),
                                          weight_decay=self.weight_decay)
        elif OPTIMIZER == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                        nesterov=True, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LR_SCHEDULER_WAIT)

        return optimizer, scheduler

    def one_epoch_step(self, epoch=0, split='train'):
        if split == 'train':
            data = self.train_data

        elif split == 'validation':
            data = self.validation_data
        else:
            raise('Split not specified')

        loss, orig_vertices, adex, faces = None, None, None, None
        epoch_loss, epoch_misclassified, epoch_classified = 0, 0, 0
        num_clas, num_misclassified = 0,0

        for i, data in enumerate(data, 0):
            orig_vertices, label, _, eigvecs, _, targets, faces, edges = data
            orig_vertices = orig_vertices.transpose(2, 1)

            if not TRAINING_CLASSIFIER:
                eigen_space_v = self.model(orig_vertices).transpose(2, 1)
                # perturbation = self.model(orig_vertices)
                # create the adversarial example
                # adex = perturbation
                # adex = orig_vertices + perturbation
                adex = orig_vertices + torch.bmm(eigvecs, eigen_space_v).transpose(2, 1)

                perturbed_logits = self.classifier(adex)  # no grad is already implemented in the constructor
                # pred_choice_adex = perturbed_logits.data.max(1)[1]
                # print('adex:   ', pred_choice_adex)
                #
                # logits = self.classifier(orig_vertices)
                # pred_choice_orig = logits.data.max(1)[1]
                # num_clas = pred_choice_orig.eq(label.squeeze()).sum().cpu()
                # print('logits: ',pred_choice_orig)
                # print('labels: ', label.squeeze())
                # print('Classified: ',num_clas)
                loss = self.calculate_loss(perturbed_logits=perturbed_logits, labels=label, targets=targets)
                pred_choice = perturbed_logits.data.max(1)[1]
                # num_misclassified = (~pred_choice.eq(label)).sum().cpu()  # for untargeted attack such as crossentropy
                num_misclassified = (pred_choice.eq(targets)).sum().cpu()  # for targeted attack
            else:
                pred = self.model(orig_vertices)
                pred = F.log_softmax(pred, dim=1)
                loss = F.nll_loss(pred, label, reduction='sum')

                pred_choice = pred.data.max(1)[1]
                num_clas = pred_choice.eq(label).cpu().sum()

            # Back-propagation step
            if split == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss = epoch_loss + loss.item()
            epoch_classified = epoch_classified + num_clas
            epoch_misclassified = epoch_misclassified + num_misclassified

        # END OF TRAIN

        if TRAINING_CLASSIFIER:
            report_to_wandb_classifier(epoch=epoch, split=split, epoch_loss=epoch_loss / 70,
                                       epoch_classified=epoch_classified)
        else:
            report_to_wandb_regressor(epoch=epoch, split=split, epoch_loss=epoch_loss / 70,
                                      epoch_misclassified=epoch_misclassified)

        # push to visualizer every epoch - last batch
        self.push_data_to_plotter(orig_vertices, adex, faces, epoch, split)

        return loss.item()

    def evaluate(self):
        data = self.test_data
        self.model.eval()

        with torch.no_grad():
            loss, orig_vertices, adex, faces = None, None, None, None
            epoch_loss, epoch_misclassified, epoch_classified = 0, 0, 0

            for i, data in enumerate(data, 0):
                orig_vertices, label, _, _, _, targets, faces, edges = data
                orig_vertices = orig_vertices.transpose(2, 1)

                if not TRAINING_CLASSIFIER:
                    perturbation = self.model(orig_vertices)
                    # create the adversarial example
                    adex = orig_vertices + perturbation

                    perturbed_logits = self.classifier(adex)  # no grad is already implemented in the constructor

                    loss = self.calculate_loss(perturbed_logits=perturbed_logits, labels=label, targets=targets)
                    pred_choice = perturbed_logits.data.max(1)[1]
                    # num_misclass = (~pred_choice.eq(label)).sum().cpu()  # for untargeted attack such as crossentropy
                    num_misclass = (pred_choice.eq(targets)).sum().cpu()  # for targeted attack

                else:
                    pred = self.model(orig_vertices)
                    pred = F.log_softmax(pred, dim=1)
                    loss = F.nll_loss(pred, label, reduction='sum')

                    pred_choice = pred.data.max(1)[1]
                    num_clas = pred_choice.eq(label).cpu().sum()


        if USE_WANDB and TRAINING_CLASSIFIER:
            report_to_wandb_classifier(epoch=0, split="test", epoch_loss=loss.item() / 15, epoch_classified=num_clas.item())
        elif USE_WANDB:
            report_to_wandb_regressor(epoch=0, split="test", epoch_loss=loss.item() / 15, epoch_misclassified=num_misclass.item())



    def calculate_loss(self, perturbed_logits, labels, targets=None):


        # if CHOOSE_LOSS == 1:
        misclassification_loss = AdversarialLoss()
        # loss = misclassification_loss(perturbed_logits, labels) # for untargeted attack
        loss = misclassification_loss(perturbed_logits, targets) # for targeted attack
            # loss = missloss
            # missloss_out, reconstruction_loss_out = 0, 0
        # elif CHOOSE_LOSS == 2:
        #     if LOSS == 'l2':
        #         reconstruction_loss = L2Similarity(orig_vertices, adex, vertex_area)
        #     else:
        #         reconstruction_loss = LocalEuclideanSimilarity(orig_vertices.transpose(2, 1), adex.transpose(2, 1),
        #                                                        edges)
        #
        #     reconstruction_loss = reconstruction_loss()
        #     loss = reconstruction_loss
        #     missloss_out, reconstruction_loss_out = 0, 0
        # else:
        #     misclassification_loss = AdversarialLoss()
        #     missloss = misclassification_loss(perturbed_logits, targets)
        #
        #     if LOSS == 'l2':
        #         reconstruction_loss = L2Similarity(orig_vertices, adex, vertex_area)
        #     else:
        #         reconstruction_loss = LocalEuclideanSimilarity(orig_vertices.transpose(2, 1), adex.transpose(2, 1), edges)
        #
        #     reconstruction_loss = reconstruction_loss()
        #     loss = missloss + RECON_LOSS_CONST * reconstruction_loss
        #     missloss_out = missloss.item()
        #     reconstruction_loss_out = reconstruction_loss.item()

        return loss

    def validation_step(self, epoch):
        self.model.eval()
        with torch.no_grad():
            total_loss = self.one_epoch_step(epoch=epoch, split='validation')
        return total_loss

    def push_data_to_plotter(self, orig_vertices, adex, faces, epoch, split):
        if split == 'train':
            data_dict = self.plt.prepare_plotter_dict(orig_vertices[:VIS_N_MESH_SETS, :, :],
                                                      adex[:VIS_N_MESH_SETS, :, :],
                                                      faces[:VIS_N_MESH_SETS, :, :])
            # cache data to use later at validation step
            self.plt.cache(data_dict)
        elif split == 'validation':
            val_data_dict = self.plt.prepare_plotter_dict(orig_vertices[:VIS_N_MESH_SETS, :, :],
                                                      adex[:VIS_N_MESH_SETS, :, :],
                                                      faces[:VIS_N_MESH_SETS, :, :])
            new_data = (self.plt.uncache(), val_data_dict)
            self.plt.push(new_epoch=epoch, new_data=new_data)


    # def evaluate(self, test_param_dir=TEST_PARAMS_DIR):  # TODO - fix this when train loop is finished
    #     # pre-test preparations
    #     s_writer = SummaryWriter(test_param_dir, flush_secs=FLUSH_RESULTS)
    #     test_param_file = get_param_file(test_param_dir)
    #     self.model.load_state_dict(torch.load(test_param_file, map_location=DEVICE), strict=MODEL_STRICT_PARAM_LOADING)
    #     self.model = self.model.eval()  # set to test mode
    #
    #     running_total_loss = 0.0
    #     running_reconstruction_loss = 0.0
    #     running_misclassify_loss = 0.0
    #     num_misclassified = 0
    #     test_len = len(self.test_data.dataset)
    #     with torch.no_grad():
    #         # the evaluation is based purely on the misclassifications amount on the test set
    #         for i, data in enumerate(self.test_data):
    #             orig_vertices, label, _, eigvecs, vertex_area, targets, faces, edges = data
    #             label = label[:, 0]
    #             orig_vertices = orig_vertices.transpose(2, 1)
    #
    #
    #             # get Eigenspace vector field
    #             eigen_space_v = self.model(orig_vertices).transpose(2, 1)
    #             # adversarial example (smoothly perturbed)
    #             adex = orig_vertices + torch.bmm(eigvecs, eigen_space_v).transpose(2, 1)
    #             # pass through classifier
    #             logits, _, _ = self.classifier(orig_vertices)
    #             perturbed_logits, _, _ = self.classifier(adex)
    #
    #             # debug
    #             # pred_orig = logits.data.max(1)[1]
    #             # pred_choice = perturbed_logits.data.max(1)[1]
    #             # num_classified = pred_orig.eq(label).cpu().sum()
    #             # num_misclassified = pred_choice.eq(targets).cpu().sum()
    #
    #             # visualize the output TODO: think about how exactly
    #             if PLOT_TEST_SAMPLE & (i == 0):  # save first batch (20 examples) as png
    #                 dump_adversarial_example_image_batch(orig_vertices, adex, faces, label, targets, logits, perturbed_logits, test_param_dir)
    #
    #             pred_choice = perturbed_logits.data.max(1)[1]
    #             num_misclassified += pred_choice.eq(targets).cpu().sum()
    #
    #             loss, missloss, similarity_loss = self.calculate_loss(orig_vertices, adex, vertex_area,
    #                                                                   perturbed_logits, targets, edges)
    #
    #             running_total_loss += loss
    #             running_reconstruction_loss += RECON_LOSS_CONST * similarity_loss
    #             running_misclassify_loss += missloss
    #
    #         report_test_to_tensorboard(s_writer, running_total_loss / self.test_num_batch,
    #                                    running_reconstruction_loss / self.test_num_batch,
    #                                    running_misclassify_loss / self.test_num_batch,
    #                                    num_misclassified, test_len)



