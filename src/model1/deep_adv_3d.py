import torch
import torch.nn as nn
import adversarial.carlini_wagner as cw

# variable definitions
from config import *

# repository modules
from models.Origin_pointnet import PointNetCls, Regressor
import torch.nn.functional as F
from tqdm import tqdm
from utils.misc import kNN
from utils import laplacebeltrami_FEM_v2
from utils import eigenpairs
from utils.laplacian import tri_areas_batch

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Classes Definition
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

    def __init__(self, train_data, test_data, model: nn.Module, classifier): #TODO: add datatype assert
        self.classifier = classifier
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.batch_size = TRAIN_BATCH_SIZE
        self.num_batch = int(len(train_data.dataset) / self.batch_size)
        self.scheduler_step = SCH_STEP
        self.n_epoch = N_EPOCH
        self.lr = LR

        self.loss_values = []  # TODO: add tensorboard support instead
        self.save_weights_dir = PARAMS_FILE

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=0.5)

        self.model.to(DEVICE)

        target_class = TARGET_CLASS  # temporary and will need access from outside later on

        for epoch in range(self.n_epoch):
            if epoch is not 0:
                scheduler.step()
            for i, data in enumerate(self.train_data, 0):
                points, target, faces = data
                target = target[:, 0]
                cur_batch_len = len(points)
                points = points.transpose(2, 1)

                points = points.to(DEVICE)
                target = target.to(DEVICE)

                # adex = cw.generate_adversarial_example(mesh=data, classifier=self.classifier,
                #                                        target=target, lowband_perturbation=False)

                optimizer.zero_grad()
                self.model = self.model.train()
                self.classifier = self.classifier.eval()
                perturbed_ex = self.model(points)

                with torch.no_grad():
                    logits, _, _ = self.classifier(perturbed_ex)
                    pred = torch.nn.functional.log_softmax(logits, dim=1)  # CW page 5: we don't use this (this if F), we need Z


                # loss = F.nll_loss(pred, target)  # for training classifier
                MisclassifyLoss = AdversarialLoss(perturbed_ex, logits, target_class)  # target is constant 5 for now
                L2Loss = L2Similarity(points, perturbed_ex, faces)
                # loss = MisclassifyLoss() + L2Loss()  # commented out cuz its hard to debug
                missloss = MisclassifyLoss()
                l2 = L2Loss()
                loss = missloss + l2

                self.loss_values.append(loss.item())

                loss.backward()
                optimizer.step()
                pred_choice = pred.data.max(1)[1]
                num_misclassified = pred_choice.eq(target_class).cpu().sum()
                print('[Epoch #%d: Batch %d/%d] train loss: %f, num misclassified: [%d/%d]' % (
                    epoch, i, self.num_batch, loss.item(), num_misclassified.item(), float(cur_batch_len)))

        torch.save(self.model.state_dict(), self.save_weights_dir)
        return self.loss_values

    def evaluate(self):
        # the evaluation is based purely on the misclassifications amount on the test set
        total_misclassified = 0
        total_testset = 0
        total_loss = 0
        test_loss_values = []
        for i, data in tqdm(enumerate(self.test_data, 0)):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            if torch.cuda.is_available():
                points, target = points.cuda(), target.cuda()
            self.model = self.model.eval()
            perturbed_ex = self.model(points)

            logits = self.classifier(perturbed_ex)
            pred = F.log_softmax(logits, dim=1)  # CW page 5: we don't use this (this if F), we need Z
            classifier_loss = F.nll_loss(pred, target)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            test_loss_values.append(classifier_loss.item())
            total_misclassified += 1 if correct.item() is 0 else total_misclassified  # not sure it works like that
            total_testset += points.size()[0]
        test_accuracy = total_misclassified / len(self.test_data.dataset)
        test_mean_loss = sum(test_loss_values) / len(test_loss_values)

        return test_mean_loss, test_accuracy

# losses ----------------------------------------------------------


class LossFunction(object):
    def __init__(self, adv_example):
        self.adv_example = adv_example

    def __call__(self) -> torch.Tensor:
        raise NotImplementedError


class AdversarialLoss(LossFunction):
    def __init__(self, adv_example: torch.Tensor, perturbed_logits, target, k: float = 0):
        super().__init__(adv_example)
        self.k = torch.tensor([k], device=adv_example.device, dtype=torch.float32)
        self.perturbed_logits = perturbed_logits
        self.target = target

    def __call__(self) -> torch.Tensor:
        Z = self.perturbed_logits
        values, index = torch.sort(Z, dim=1)
        index = index[-1]
        argmax = index[-1] if index[-1] != self.target else index[-2]  # max{Z(i): i != target}
        Z = Z[-1]
        Ztarget, Zmax = Z[self.target], Z[argmax]
        return torch.max(Zmax - Ztarget, -self.k)


class L2Similarity(LossFunction):
    def __init__(self, original_pos: torch.Tensor, perturbed_pos: torch.Tensor, faces):
        super().__init__(original_pos)
        self.original_pos = original_pos
        self.perturbed_pos = perturbed_pos
        self.faces = faces

    def __call__(self) -> torch.Tensor:
        area_values = tri_areas_batch(self.original_pos, self.faces)
        diff = self.perturbed_pos - self.original_pos
        weight_diff = diff
        # weight_diff = diff * torch.sqrt( # TODO: uncomment
        #     area_values.view(-1, 1))  # (sqrt(ai)*(xi-perturbed(xi)) )^2  = ai*(x-perturbed(xi))^2
        L2 = weight_diff.norm(
            p="fro")  # this reformulation uses the sub-gradient (hence ensuring a valid behaviour at zero)
        return L2

# -----------------------------------------------------------------

## TODO: I am not sure about this one at all
class ModelHandler:

    def __init__(self, model: nn.Module, param_file: str = "", auto_grad=True):
        """
        This class handles the correct switch between evaluation and training for a given model
        NOTE loads pretrained parameters
        """
        self.model = model.to(DEVICE)

        # load parameters
        if auto_grad == False:
            self.model.load_state_dict(torch.load(param_file, map_location=DEVICE))
            for param in model.features.parameters():
                param.requires_grad = False
            self.model.eval()
        # else: //TODO: Do we need this?
        #     for param in model.features.parameters():
        #         param.requires_grad = True

    def __call__(self):
        return self.model
