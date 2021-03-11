import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# variable definitions
from config import *

# repository modules
from models.Origin_pointnet import PointNetCls, Regressor
import torch.nn.functional as F
from tqdm import tqdm
from model1.loss import *
from model1.utils import *
from model1.tensor_board import *
from utils import laplacebeltrami_FEM_v2
from utils import eigenpairs
from utils.laplacian import tri_areas_batch
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

        generate_data_output_dir()

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
        now = datetime.now()
        d = now.strftime("%b-%d-%Y_%H-%M-%S")
        self.tensor_log_dir = generate_new_tensorboard_results_dir(d)
        self.writer = SummaryWriter(self.tensor_log_dir, flush_secs=FLUSH_RESULTS)

        self.save_weights_dir = generate_unique_params_name(d)


    def train(self):
        if OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=0.5)

        step_cntr = 0
        for epoch in range(self.n_epoch):
            if epoch != 0:
                scheduler.step()
            for i, data in enumerate(self.train_data, 0):
                orig_vertices, label, _, eigvecs, vertex_area, targets, faces, edges = data
                # label = label[:, 0].to(DEVICE) TODO: remove this later on
                cur_batch_len = orig_vertices.shape[0]
                orig_vertices = orig_vertices.transpose(2, 1)

                optimizer.zero_grad()
                self.model = self.model.train()  # set to train mode
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
                report_to_tensorboard(self.writer, i, step_cntr, cur_batch_len, epoch, self.num_batch, loss.item(),
                                      similarity_loss.item(), missloss.item(), num_misclassified)

                step_cntr += 1

            if (step_cntr > 0) & (step_cntr % SAVE_PARAMS_EVERY == 0):
                torch.save(self.model.state_dict(), self.save_weights_dir)

        # save at the end also
        torch.save(self.model.state_dict(), self.save_weights_dir)
        return self.loss_values

    # def evaluate(self): ## TODO: remove this later on
    #     # the evaluation is based purely on the misclassifications amount on the test set
    #     total_misclassified = 0
    #     total_testset = 0
    #     total_loss = 0
    #     test_loss_values = []
    #     for i, data in tqdm(enumerate(self.test_data, 0)):
    #         points, target = data
    #         target = target[:, 0]
    #         points = points.transpose(2, 1)
    #         if torch.cuda.is_available():
    #             points, target = points.cuda(), target.cuda()
    #         self.model = self.model.eval()
    #         perturbed_ex = self.model(points)
    #
    #         logits = self.classifier(perturbed_ex)
    #         pred = F.log_softmax(logits, dim=1)  # CW page 5: we don't use this (this if F), we need Z
    #         classifier_loss = F.nll_loss(pred, target)
    #
    #         pred_choice = pred.data.max(1)[1]
    #         correct = pred_choice.eq(target.data).cpu().sum()
    #         test_loss_values.append(classifier_loss.item())
    #         total_misclassified += 1 if correct.item() != 0 else total_misclassified  # not sure it works like that
    #         total_testset += points.size()[0]
    #     test_accuracy = total_misclassified / len(self.test_data.dataset)
    #     test_mean_loss = sum(test_loss_values) / len(test_loss_values)
    #
    #     return test_mean_loss, test_accuracy

