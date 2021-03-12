import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# variable definitions
from config import *

# repository modules
from models.Origin_pointnet import PointNetCls, Regressor
from model1.loss import *
from model1.utils import *
from model1.tensor_board import *

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Trainer Class
# ----------------------------------------------------------------------------------------------------------------------#

# Model 1 with classifier inside. Commenented out since I'm trying to shove the classifier inside the loss function,
# instead of combining them
# class Model1(nn.Module):
#
#     def __init__(self, outDim, classifier_model: nn.Module):
#         pass
#         # Definition of:
#         # 1. regressor
#         # 2. original pointNet
#         super(Model1, self).__init__()
#         self.regressor = Regressor(outDim, feature_transform=False)  # do we need feature transform false?
#         self.classifier = classifier_model
#
#     def forward(self, x):
#         pass
#         # Model forward pass, return output from trained classifier AND from regressor
#         # run the pass, and with no grad run classifier
#
#         v = self.regressor.forward(x)  # this is the perturbation
#
#         with torch.no_grad():
#             perturbed_pos = x + v
#             classification = self.classifier.forward(perturbed_pos)
#
#         return classification, v


class trainer:

    def __init__(self, train_data: torch.utils.data.DataLoader,
                       test_data: torch.utils.data.DataLoader,
                       model: nn.Module,
                       classifier: nn.Module):


        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = TRAIN_BATCH_SIZE
        self.num_batch = len(self.train_data)
        self.scheduler_step = SCHEDULER_STEP_SIZE
        self.n_epoch = N_EPOCH
        self.weight_decay = WEIGHT_DECAY
        self.lr = LR

        self.classifier = classifier
        self.classifier.eval()
        self.classifier.to(DEVICE)
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.model = model
        self.model.to(DEVICE)

        self.loss_values = []

    def train(self):

        # pre-train preparations
        generate_data_output_dir()
        now = datetime.now()
        d = now.strftime("%b-%d-%Y_%H-%M-%S")
        tensor_log_dir = generate_new_tensorboard_results_dir(d)
        writer = SummaryWriter(tensor_log_dir, flush_secs=FLUSH_RESULTS)
        save_weights_dir = tensor_log_dir

        if OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=0.5)

        self.model = self.model.train()  # set to train mode
        step_cntr = 0
        for epoch in range(self.n_epoch):
            if epoch != 0:
                scheduler.step()
            for i, data in enumerate(self.train_data, 0):
                orig_vertices, label, _, eigvecs, vertex_area, targets, faces, edges= data
                # label = label[:, 0].to(DEVICE) TODO: remove this later on
                cur_batch_len = orig_vertices.shape[0]
                orig_vertices = orig_vertices.transpose(2, 1)

                optimizer.zero_grad()
                # get Eigenspace vector field
                eigen_space_v = self.model(orig_vertices).transpose(2, 1)

                # adversarial example (smoothly perturbed)
                adex = orig_vertices + torch.bmm(eigvecs, eigen_space_v).transpose(2, 1)
                # DEBUG - visualize the adex
                if DEBUG & (step_cntr > 0) & (step_cntr % SHOW_TRAIN_SAMPLE_EVERY == 0):
                    dump_adversarial_example_image(orig_vertices, adex, faces, step_cntr, self.tensor_log_dir)


                # no grad is already implemented in the constructor
                perturbed_logits, _, _ = self.classifier(adex)

                MisclassifyLoss = AdversarialLoss(perturbed_logits, targets)
                if LOSS == 'l2':
                    Similarity_loss = L2Similarity(orig_vertices, adex, vertex_area)
                else:
                    Similarity_loss = LocalEuclideanSimilarity(orig_vertices.transpose(2, 1), adex.transpose(2, 1), edges)


                missloss = MisclassifyLoss()
                similarity_loss = Similarity_loss()
                loss = missloss + RECON_LOSS_CONST * similarity_loss

                # Back-propagation step
                loss.backward()
                optimizer.step()

                # Metrics
                self.loss_values.append(loss.item())
                pred_choice = perturbed_logits.data.max(1)[1]
                num_misclassified = pred_choice.eq(targets).cpu().sum()

                # report to tensorboard
                report_to_tensorboard(writer, i, step_cntr, cur_batch_len, epoch, self.num_batch, loss.item(),
                                      similarity_loss.item(), missloss.item(), num_misclassified)

                step_cntr += 1

            if (step_cntr > 0) & (step_cntr % SAVE_PARAMS_EVERY == 0):
                torch.save(self.model.state_dict(), save_weights_dir)

        # save at the end also
        torch.save(self.model.state_dict(), save_weights_dir)

        # evaluate the model
        self.evaluate(tensor_log_dir)


    def evaluate(self, test_param_dir=TEST_PARAMS_DIR):
        # pre-test preparations
        writer = SummaryWriter(test_param_dir, flush_secs=FLUSH_RESULTS)
        test_param_file = get_param_file(test_param_dir)
        self.model.load_state_dict(torch.load(test_param_file, map_location=DEVICE))
        self.model = self.model.eval()  # set to test mode

        running_total_loss = 0.0
        running_reconstruction_loss = 0.0
        running_misclassify_loss = 0.0
        dataset_size = len(self.test_data.dataset)
        with torch.no_grad():
            # the evaluation is based purely on the misclassifications amount on the test set
            for i, data in enumerate(self.test_data):
                orig_vertices, label, _, eigvecs, vertex_area, targets, faces, edges = data
                orig_vertices = orig_vertices.transpose(2, 1)

                # get Eigenspace vector field
                eigen_space_v = self.model(orig_vertices).transpose(2, 1)
                # adversarial example (smoothly perturbed)
                adex = orig_vertices + torch.bmm(eigvecs, eigen_space_v).transpose(2, 1)

                # visualize the output TODO: think about how exactly, maybe augment the test also?
                # if PLOT_TEST_SAMPLE & (i == 0): # save first batch (16 examples) as png
                #     dump_adversarial_example_image(orig_vertices, adex, faces, step_cntr, self.tensor_log_dir)

                perturbed_logits, _, _ = self.classifier(adex)
                pred_choice = perturbed_logits.data.max(1)[1]
                num_misclassified = pred_choice.eq(targets).cpu().sum()

                MisclassifyLoss = AdversarialLoss(perturbed_logits, targets)
                if LOSS == 'l2':
                    Similarity_loss = L2Similarity(orig_vertices, adex, vertex_area)
                else:
                    Similarity_loss = LocalEuclideanSimilarity(orig_vertices.transpose(2, 1), adex.transpose(2, 1), edges)

                missloss = MisclassifyLoss()
                similarity_loss = Similarity_loss()
                loss = missloss + RECON_LOSS_CONST * similarity_loss

                running_total_loss += loss
                running_reconstruction_loss += RECON_LOSS_CONST * similarity_loss
                running_misclassify_loss += missloss

            report_test_to_tensorboard(writer, running_total_loss / dataset_size,
                                       running_reconstruction_loss / dataset_size,
                                       running_misclassify_loss / dataset_size,
                                       num_misclassified, orig_vertices.shape[0])




