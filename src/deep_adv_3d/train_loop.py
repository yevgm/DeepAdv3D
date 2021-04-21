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


class Trainer:

    def __init__(self, train_data: torch.utils.data.DataLoader,
                       validation_data: torch.utils.data.DataLoader,
                       test_data: torch.utils.data.DataLoader,
                       model: nn.Module,
                       classifier: nn.Module):

        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.batch_size = TRAIN_BATCH_SIZE
        self.num_batch = len(self.train_data)
        self.test_num_batch = len(self.test_data)
        self.scheduler_step = SCHEDULER_STEP_SIZE
        self.n_epoch = N_EPOCH
        self.weight_decay = WEIGHT_DECAY
        self.lr = LR

        self.classifier = classifier
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.to(DEVICE)
        self.model = model
        self.model.to(DEVICE)

        # attributes initializations
        self.loss_values = []
        self.optimizer, self.scheduler = None, None
        self.tensor_log_dir = ''
        self.writer = None
        self.train_step_cntr, self.val_step_cntr = 0, 0

        # plotter init
        self.plt = AdversarialPlotter()

    def train(self):
        # pre-train preparations
        save_weights_file = self.init_tensor_board()
        self.optimizer, self.scheduler = self.define_optimizer()
        self.model = self.model.train()  # set to train mode

        for epoch in range(self.n_epoch):
            # train step
            self.one_epoch_step(epoch=epoch, split='train')
            # validation step
            if epoch % VAL_STEP_EVERY == 0:
                self.validation_step(epoch)

            self.scheduler.step()
            # TODO: validation early stop
            if (self.train_step_cntr > 0) & (self.train_step_cntr % SAVE_PARAMS_EVERY == 0):
                torch.save(self.model.state_dict(), save_weights_file)

        # save weights also at the end
        torch.save(self.model.state_dict(), save_weights_file)

        # evaluate the model at the end of training
        self.evaluate(self.tensor_log_dir)


    def define_optimizer(self):
        if OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999),
                                          weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999),
                                         weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=0.5)

        return optimizer, scheduler

    def init_tensor_board(self):
        '''
        Create output directory wrapper and initialize tensorboard object instance
        '''
        create_data_output_dir()
        now = datetime.now()
        d = now.strftime("%b-%d-%Y_%H-%M-%S")
        self.tensor_log_dir = generate_new_tensorboard_results_dir(d)
        self.writer = SummaryWriter(self.tensor_log_dir, flush_secs=FLUSH_RESULTS)
        save_weights_file = os.path.join(self.tensor_log_dir, PARAM_FILE_NAME)

        return save_weights_file

    def one_epoch_step(self, epoch=0, split='train'):
        if split == 'train':
            data = self.train_data
            step_cntr = self.train_step_cntr
        elif split == 'validation':
            data = self.validation_data
            step_cntr = self.train_step_cntr  # this is on purpose
        else:
            data = self.test_data
            step_cntr = None

        loss = None
        orig_vertices = None
        adex = None
        faces = None
        for i, data in enumerate(data, 0):
            orig_vertices, label, _, eigvecs, vertex_area, targets, faces, edges = data
            # plot_mesh(orig_vertices[0], faces[0], grid_on=True, strategy='cloud')  # plot some mesh for debugging
            cur_batch_len = orig_vertices.shape[0]
            orig_vertices = orig_vertices.transpose(2, 1)

            # get Eigenspace vector field
            eigen_space_v = self.model(orig_vertices).transpose(2, 1)
            # create the adversarial example (smoothly perturbed)
            adex = orig_vertices + torch.bmm(eigvecs, eigen_space_v).transpose(2, 1)
            perturbed_logits, _, _ = self.classifier(adex)  # no grad is already implemented in the constructor

            # DEBUG - visualize the adex
            # if epoch == 10:  # plot the first adex in the batch
            # plot_mesh(adex[0].transpose(0, 1), faces[0], grid_on=True)
            dump_adversarial_example_image(orig_vertices, adex, faces, step_cntr, self.tensor_log_dir)

            # debug
            # logits, _, _ = self.classifier(orig_vertices)
            # label = label[:, 0]
            # pred_orig = logits.data.max(1)[1]
            # pred_choice = perturbed_logits.data.max(1)[1]
            # num_classified = pred_orig.eq(label).cpu().sum()
            # num_misclassified = pred_choice.eq(targets).cpu().sum()

            loss, missloss, similarity_loss = self.calculate_loss(orig_vertices, adex, vertex_area,
                                                                  perturbed_logits, targets, edges)

            # compare batch loss vs single loss
            # single_batch_adv_loss = AdversarialLoss_single_batch()
            # loss_sum = 0
            # for k in range(adex.shape[0]):
            #     one_batch_loss = single_batch_adv_loss(perturbed_logits[k], targets[k])
            #     loss_sum += one_batch_loss
            # debug_loss = loss_sum / adex.shape[0]
            # print('diff {}'.format(missloss - loss_sum / adex.shape[0]))
            # a=1
            # Back-propagation step
            self.optimizer.zero_grad()
            loss.backward()
            # if epoch > 1:  # trying to draw the gradients flow, not working since our grads are None for some reason
            #     plot_grad_flow_v2(self.model.named_parameters())
            self.optimizer.step()

            # Metrics
            self.loss_values.append(loss.item())
            pred_choice = perturbed_logits.data.max(1)[1]
            num_misclassified = pred_choice.eq(targets).sum().cpu()

            # report to tensorboard
            report_to_tensorboard(split, self.writer, i, step_cntr, cur_batch_len, epoch, self.num_batch, loss.item(),
                                  similarity_loss.item(), missloss.item(), num_misclassified)

            if step_cntr is not None:
                step_cntr += 1

        # update relevant step counter
        if split == 'train':
            self.train_step_cntr = step_cntr
        elif split == 'validation':
            self.val_step_cntr = step_cntr

        # push to visualizer every epoch - last batch
        self.push_data_to_plotter(orig_vertices, adex, faces, epoch, split)

        return loss.item()

    def validation_step(self, epoch):
        total_loss = self.one_epoch_step(epoch=epoch, split='validation')
        # TODO: if total loss is satisfying some condition, stop the train


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


    def calculate_loss(self, orig_vertices, adex, vertex_area,
                       perturbed_logits, targets, edges):

        MisclassifyLoss = AdversarialLoss()
        if LOSS == 'l2':
            Similarity_loss = L2Similarity(orig_vertices, adex, vertex_area)
        else:
            Similarity_loss = LocalEuclideanSimilarity(orig_vertices.transpose(2, 1), adex.transpose(2, 1), edges)

        missloss = MisclassifyLoss(perturbed_logits, targets)
        similarity_loss = Similarity_loss()
        loss = missloss + RECON_LOSS_CONST * similarity_loss

        return loss, missloss, similarity_loss


    def evaluate(self, test_param_dir=TEST_PARAMS_DIR): # TODO - fix this when train loop is finished
        # pre-test preparations
        s_writer = SummaryWriter(test_param_dir, flush_secs=FLUSH_RESULTS)
        test_param_file = get_param_file(test_param_dir)
        self.model.load_state_dict(torch.load(test_param_file, map_location=DEVICE), strict=MODEL_STRICT_PARAM_LOADING)
        self.model = self.model.eval()  # set to test mode

        running_total_loss = 0.0
        running_reconstruction_loss = 0.0
        running_misclassify_loss = 0.0
        num_misclassified = 0
        test_len = len(self.test_data.dataset)
        with torch.no_grad():
            # the evaluation is based purely on the misclassifications amount on the test set
            for i, data in enumerate(self.test_data):
                orig_vertices, label, _, eigvecs, vertex_area, targets, faces, edges = data
                label = label[:, 0]
                orig_vertices = orig_vertices.transpose(2, 1)


                # get Eigenspace vector field
                eigen_space_v = self.model(orig_vertices).transpose(2, 1)
                # adversarial example (smoothly perturbed)
                adex = orig_vertices + torch.bmm(eigvecs, eigen_space_v).transpose(2, 1)
                # pass through classifier
                logits, _, _ = self.classifier(orig_vertices)
                perturbed_logits, _, _ = self.classifier(adex)

                # debug
                # pred_orig = logits.data.max(1)[1]
                # pred_choice = perturbed_logits.data.max(1)[1]
                # num_classified = pred_orig.eq(label).cpu().sum()
                # num_misclassified = pred_choice.eq(targets).cpu().sum()

                # visualize the output TODO: think about how exactly
                if PLOT_TEST_SAMPLE & (i == 0):  # save first batch (20 examples) as png
                    dump_adversarial_example_image_batch(orig_vertices, adex, faces, label, targets, logits, perturbed_logits, test_param_dir)

                pred_choice = perturbed_logits.data.max(1)[1]
                num_misclassified += pred_choice.eq(targets).cpu().sum()

                loss, missloss, similarity_loss = self.calculate_loss(orig_vertices, adex, vertex_area,
                                                                      perturbed_logits, targets, edges)

                running_total_loss += loss
                running_reconstruction_loss += RECON_LOSS_CONST * similarity_loss
                running_misclassify_loss += missloss

            report_test_to_tensorboard(s_writer, running_total_loss / self.test_num_batch,
                                       running_reconstruction_loss / self.test_num_batch,
                                       running_misclassify_loss / self.test_num_batch,
                                       num_misclassified, test_len)




