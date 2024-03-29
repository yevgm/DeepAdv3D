import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


from deep_adv_3d.loss import *
from deep_adv_3d.utils import *
from deep_adv_3d.tensor_board import *
import utils.mesh as mesh
# from vista.subprocess_plotter import AdversarialPlotter

import torch
import wandb
import torch.nn.functional as F

class Trainer:

    def __init__(self, train_data: torch.utils.data.DataLoader,
                       validation_data: torch.utils.data.DataLoader,
                       test_data: torch.utils.data.DataLoader,
                       model: nn.Module,
                       run_config,
                       classifier:nn.Module=None
                       ):
        self.run_config = run_config
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.batch_size = run_config['TRAIN_BATCH_SIZE']
        self.num_batch = len(self.train_data)
        self.test_num_batch = len(self.test_data)
        self.n_epoch = run_config['N_EPOCH']
        self.weight_decay = run_config['WEIGHT_DECAY']
        self.lr = run_config['LR']
        self.dropout_prob = run_config['DROPOUT_PROB']
        self.classifier = classifier

        if classifier is not None:
            self.classifier.eval()
            for param in self.classifier.parameters():
                param.requires_grad = False
            self.classifier.to(run_config['DEVICE'])
        self.model = model
        self.model.to(run_config['DEVICE'])
        # W and B
        if run_config['LOG_GRADIENTS_WANDB']:
            wandb.watch((self.model), log="all", log_freq=10)
        # early stop
        self.early_stopping = EarlyStopping(patience=run_config['EARLY_STOP_WAIT'])  # hardcoded for validation loss early stop
        # checkpoints regulator
        self.init_tensor_board()
        self.checkpoint_callback = ModelCheckpoint(filepath=self.tensor_log_dir, model=self.model, save_model=run_config['SAVE_WEIGHTS'])

        # attributes initializations
        self.optimizer, self.scheduler = None, None

        # cache adversarial examples
        self.train_cache, self.val_cache = None, None

        # plotter init
        # if run_config['USE_PLOTTER']:
        #     self.plt = AdversarialPlotter(run_config)


    def train(self):
        # pre-train preparations

        self.optimizer, self.scheduler = self.define_optimizer()
        val_loss = 0
        for epoch in range(self.n_epoch):
            # train step
            self.model.train()
            self.one_epoch_step(epoch=epoch, split='train')

            # validation step
            if epoch % self.run_config['VAL_STEP_EVERY'] == 0:
                # pass validation through model
                val_loss = self.validation_step(epoch)
                # check if model parameters should be saved
                self.checkpoint_callback.on_validation_end(epoch=epoch, monitored_value=val_loss)
                # check if training is finished
                stop_training = self.early_stopping.on_epoch_end(epoch=epoch, monitored_value=val_loss)
                if stop_training:
                    self.save_adex_to_drive()
                    self.early_stopping.on_train_end()
                    torch.save(self.model.state_dict(), os.path.join(self.tensor_log_dir, 'last_epoch_weights.pt'))
                    # if self.run_config['USE_PLOTTER']:
                    #     self.plt.finalize()
                    # self.evaluate()  # TODO: uncomment for test
                    exit()

            self.scheduler.step(val_loss)

        # if self.run_config['USE_PLOTTER']:
        #     self.plt.finalize()
        self.save_adex_to_drive()
        torch.save(self.model.state_dict(), os.path.join(self.tensor_log_dir, 'last_epoch_weights.pt'))
        self.evaluate()


    def define_optimizer(self):
        '''
        choose the optizimer and it's hyper-parameters
        '''
        optimizer_type = self.run_config['OPTIMIZER']
        if optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                        nesterov=True, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.run_config['LR_SCHEDULER_WAIT'])

        return optimizer, scheduler

    def one_epoch_step(self, epoch=0, split='train'):
        if split == 'train':
            data = self.train_data

        elif split == 'validation':
            data = self.validation_data
        else:
            raise Exception('Split not specified')

        loss, orig_vertices, adex, faces = None, None, None, None
        epoch_loss, epoch_misclassified, epoch_classified, epoch_misclass_loss, epoch_recon_loss = 0, 0, 0, 0, 0
        epoch_chamfer_loss = 0
        num_clas, num_misclassified, misloss, recon_loss, chamfer = 0, 0, 0, 0, 0

        for i, data in enumerate(data, 0):
            orig_vertices, label, _, eigvecs, vertex_area, targets, faces, edges = data
            orig_vertices = orig_vertices.transpose(2, 1)


            if not self.run_config['TRAINING_CLASSIFIER']:
                # perturbation = self.model(orig_vertices)
                eigen_space_v = self.model(orig_vertices)

                # create the adversarial example
                # adex = orig_vertices + perturbation
                adex = orig_vertices + torch.bmm(eigvecs, eigen_space_v.transpose(2, 1)).transpose(2, 1)  # with addition
                # adex = adex - adex.mean(dim=2).reshape(-1, 3, 1)

                perturbed_logits = self.classifier(adex)  # no grad is already implemented in the constructor
                loss, misloss, recon_loss = self.calculate_loss(perturbed_logits=perturbed_logits, labels=label,
                                                                targets=targets, orig_vertices=orig_vertices,
                                                                adex=adex, vertex_area=vertex_area, edges=edges,
                                                                faces=faces, epoch=epoch)
                pred_choice = perturbed_logits.data.max(1)[1]
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
            epoch_misclass_loss = epoch_misclass_loss + misloss
            epoch_recon_loss = epoch_recon_loss + recon_loss
            epoch_chamfer_loss = epoch_chamfer_loss + chamfer

        # END OF TRAIN

        if self.run_config['TRAINING_CLASSIFIER']:
            report_to_wandb_classifier(run_config=self.run_config, epoch=epoch, split=split, epoch_loss=epoch_loss / self.run_config['DATASET_TRAIN_SIZE'],
                                       epoch_classified=epoch_classified)
        else:
            report_to_wandb_regressor(run_config=self.run_config, epoch=epoch, split=split, epoch_loss=epoch_loss / self.run_config['DATASET_TRAIN_SIZE'],
                                      epoch_misclassified=epoch_misclassified, misloss=epoch_misclass_loss / self.run_config['DATASET_TRAIN_SIZE'],
                                      recon_loss=epoch_recon_loss / self.run_config['DATASET_TRAIN_SIZE'])

        # push to visualizer every epoch - last batch
        if self.run_config['USE_PLOTTER'] or self.run_config['SAVE_EXAMPLES_TO_DRIVE']:
            self.push_data_to_plotter(orig_vertices, adex, faces, epoch, split)

        return loss.item()

    def evaluate(self):
        data = self.test_data
        self.model.eval()

        with torch.no_grad():
            loss, orig_vertices, adex, faces = None, None, None, None
            epoch_loss, epoch_misclassified, epoch_classified = 0, 0, 0

            for i, data in enumerate(data, 0):
                orig_vertices, label, _, eigvecs, vertex_area, targets, faces, edges = data
                orig_vertices = orig_vertices.transpose(2, 1)

                if not self.run_config['TRAINING_CLASSIFIER']:
                    # perturbation = self.model(orig_vertices)
                    eigen_space_v = self.model(orig_vertices)
                    eigvecs = eigvecs.to(torch.float32)


                    # create the adversarial example
                    adex = orig_vertices + torch.bmm(eigvecs, eigen_space_v.transpose(2, 1)).transpose(2, 1)
                    # adex = orig_vertices + perturbation

                    perturbed_logits = self.classifier(adex)  # no grad is already implemented in the constructor

                    loss, missloss, recon_loss = self.calculate_loss(orig_vertices=orig_vertices, perturbed_logits=perturbed_logits,
                                               labels=label, targets=targets, adex=adex, vertex_area=vertex_area)
                    pred_choice = perturbed_logits.data.max(1)[1]
                    num_misclass = (pred_choice.eq(targets)).sum().cpu()  # for targeted attack

                else:
                    pred = self.model(orig_vertices)
                    pred = F.log_softmax(pred, dim=1)
                    loss = F.nll_loss(pred, label, reduction='sum')

                    pred_choice = pred.data.max(1)[1]
                    num_clas = pred_choice.eq(label).cpu().sum()


        if self.run_config['TRAINING_CLASSIFIER']:
            report_to_wandb_classifier(run_config=self.run_config, epoch=0, split="test",
                                       epoch_loss=loss.item() / self.run_config['DATASET_VAL_SIZE'], epoch_classified=num_clas.item())
        else:
            report_to_wandb_regressor(run_config=self.run_config, epoch=0, split="test",
                                      epoch_loss=loss.item() / self.run_config['DATASET_VAL_SIZE'], epoch_misclassified=num_misclass.item())



    def calculate_loss(self, perturbed_logits, labels, orig_vertices=None, adex=None, vertex_area=None, targets=None,
                       epoch=None, edges=None, faces=None):
        recon_const = self.run_config['RECON_LOSS_CONST']
        l2_const = self.run_config['L2_LOSS_CONST']
        edge_loss_const = self.run_config['EDGE_LOSS_CONST']
        laplacian_loss_const = self.run_config['LAPLACIAN_LOSS_CONST']

        # only misclassification loss
        if self.run_config['CHOOSE_LOSS'] == 1:
            misclassification_loss = AdversarialLoss()
            loss = misclassification_loss(perturbed_logits, targets) # for targeted attack

        # only reconstruction loss
        elif self.run_config['CHOOSE_LOSS'] == 2:
            if self.run_config['LOSS'] == 'l2':
                reconstruction_loss = L2Similarity(orig_vertices, adex, vertex_area)
            elif self.run_config['LOSS'] == 'EUCLIDEAN':
                reconstruction_loss = LocalEuclideanBatch(original_pos=orig_vertices, perturbed_pos=adex,
                                                          run_config=self.run_config)
                return reconstruction_loss, 0, reconstruction_loss
            else:
                raise('Not implemented reonstruction loss')

            recon_loss = reconstruction_loss()
            loss = recon_loss, 0, recon_loss

        else:
            misclassification_loss = AdversarialLoss()
            missloss = misclassification_loss(perturbed_logits, targets)

            if self.run_config['LOSS'] == 'l2':
                reconstruction_loss = L2Similarity(orig_vertices, adex, vertex_area)
                recon_loss = reconstruction_loss()
            elif self.run_config['LOSS'] == 'EUCLIDEAN':
                recon_loss = LocalEuclideanBatch(original_pos=orig_vertices, perturbed_pos=adex,
                                                          run_config=self.run_config)
                # edgeloss = MeshEdgeLoss()
                # recon_loss = edgeloss(adex, faces)
                # l2_loss = L2Similarity(orig_vertices, adex, vertex_area)
                # l2 = l2_loss()
            else:
                raise('Not implemented reonstruction loss')

            loss = missloss + recon_const * recon_loss
            return loss, missloss, recon_loss

        return loss

    def validation_step(self, epoch):
        self.model.eval()
        with torch.no_grad():
            total_loss = self.one_epoch_step(epoch=epoch, split='validation')
        return total_loss

    def init_tensor_board(self):
        '''
        Create output directory wrapper and initialize tensorboard object instance
        '''
        create_data_output_dir(self.run_config)
        now = datetime.now()
        d = now.strftime("%b-%d-%Y_%H-%M-%S")
        self.tensor_log_dir = generate_new_results_dir(date=d, run_config=self.run_config)
        self.writer = SummaryWriter(self.tensor_log_dir, flush_secs=self.run_config['FLUSH_RESULTS'])


    def push_data_to_plotter(self, orig_vertices, adex, faces, epoch, split):
        VIS_N_MESH_SETS = self.run_config['VIS_N_MESH_SETS']
        save_to_drive = self.run_config['SAVE_EXAMPLES_TO_DRIVE']
        if split == 'train':
            data_dict = self.prepare_plotter_dict(orig_vertices[:VIS_N_MESH_SETS, :, :].detach().clone(),
                                                      adex[:VIS_N_MESH_SETS, :, :].detach().clone(),
                                                      faces[:VIS_N_MESH_SETS, :, :].detach().clone())
            if save_to_drive == False:
                # cache data to use later at validation step
                self.plt.cache(data_dict)
            else:
                self.train_cache = data_dict

        elif split == 'validation':
            val_data_dict = self.prepare_plotter_dict(orig_vertices[:VIS_N_MESH_SETS, :, :].detach().clone(),
                                                      adex[:VIS_N_MESH_SETS, :, :].detach().clone(),
                                                      faces[:VIS_N_MESH_SETS, :, :].detach().clone())

            if save_to_drive == False:
                new_data = (self.plt.uncache(), val_data_dict)
                self.plt.push(new_epoch=epoch, new_data=new_data)
            else:
                self.val_cache = val_data_dict

    def prepare_plotter_dict(self, orig_vertices, adexs, faces):

        max_b_idx = self.run_config['VIS_N_MESH_SETS']
        dict = {'orig_vertices': orig_vertices.detach().cpu().numpy()[:max_b_idx, :, :],
                'adexs': adexs.detach().cpu().numpy()[:max_b_idx, :, :],
                'faces': faces.detach().cpu().numpy()[:max_b_idx]}
        return dict


    def save_adex_to_drive(self):
        if self.run_config['SAVE_EXAMPLES_TO_DRIVE']:
            if self.run_config['USE_WANDB']:
                run_name = self.run_config['RUN_NAME']
            else:
                run_name = 'test_run'
            save_examples_dir = os.path.abspath(
                os.path.join(self.run_config['REPO_ROOT'], '..', 'adex_sweep', run_name))
            output_dir = os.path.abspath(os.path.join(self.run_config['REPO_ROOT'], '..', 'adex_sweep'))

            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            if not os.path.isdir(save_examples_dir):
                os.mkdir(save_examples_dir)

            train_orig_num = self.train_cache['orig_vertices'].shape[0]
            val_orig_num = self.val_cache['orig_vertices'].shape[0]

            for i in range(train_orig_num):
                fp = os.path.join(save_examples_dir, 'train_orig_{}.ply'.format(i + 1))
                with open(fp, 'w'):
                    mesh.write_mesh(fp=fp,
                                    v=self.train_cache['orig_vertices'][i].T,
                                    f=self.train_cache['faces'][i])

                fp = os.path.join(save_examples_dir, 'train_adex_{}.ply'.format(i + 1))
                with open(fp, 'w'):
                    mesh.write_mesh(fp=fp,
                                    v=self.train_cache['adexs'][i].T,
                                    f=self.train_cache['faces'][i])

            for i in range(val_orig_num):
                fp = os.path.join(save_examples_dir, 'val_orig_{}.ply'.format(i + 1))
                with open(fp, 'w'):
                    mesh.write_mesh(fp=fp,
                                    v=self.val_cache['orig_vertices'][i].T,
                                    f=self.val_cache['faces'][i])

                fp = os.path.join(save_examples_dir, 'val_adex_{}.ply'.format(i + 1))
                with open(fp, 'w'):
                    mesh.write_mesh(fp=fp,
                                    v=self.val_cache['adexs'][i].T,
                                    f=self.val_cache['faces'][i])


