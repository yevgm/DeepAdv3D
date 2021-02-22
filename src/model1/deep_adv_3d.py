import torch
import torch.nn as nn

# variable definitions
from config import *

# repository modules
from models.Origin_pointnet import PointNetCls, Regressor
import torch.nn.functional as F
from tqdm import tqdm
from utils.misc import kNN
import adversarial.carlini_wagner as cw

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Classes Definition
# ----------------------------------------------------------------------------------------------------------------------#

# Model 1 with classifier inside. Commenented out since I'm trying to shove the classifier inside the loss function now
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

    def __init__(self, train_data, test_data, model: nn.Module, lossFunction, classifier): #TODO: add datatype assert
        self.classifier = classifier
        self.train_data = train_data
        self.test_data = test_data
        self.lossFunction = lossFunction
        self.model = model
        self.batch_size = BATCH_SIZE
        self.num_batch = int(len(train_data.dataset) / self.batch_size)
        self.scheduler_step = SCH_STEP
        self.n_epoch = N_EPOCH
        self.lr = LR

        self.loss_values = [] #TODO: add tensorboard support instead
        self.save_weights_dir = PARAMS_FILE

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))  # TODO: change from model.parameters() to per-parameter options?
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=0.5)

        self.model.to(DEVICE)

        for epoch in range(self.n_epoch):
            scheduler.step()
            for i, data in enumerate(self.train_data, 0):
                points, target = data
                # points, _ = data
                target = target[:, 0]
                cur_batch_len = len(points)
                points = points.transpose(2, 1)

                points.to(DEVICE)
                target.to(DEVICE)

                adex = cw.generate_adversarial_example(mesh=data, classifier=self.classifier,
                                                       target=target, lowband_perturbation=False)



                optimizer.zero_grad()
                self.model = self.model.train()  # TODO: overload this to only apply to certain parts of the model? (exclude the classifier)
                classfication, v = self.model(points)

                pred = F.log_softmax(classfication, dim=1)
                # loss = F.nll_loss(pred, target)
                loss = self.lossFunction(points, v, pred, target)  # TODO: create lossFunction in regressor_trainer.py

                self.loss_values.append(loss.item())

                loss.backward()
                optimizer.step()
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                    epoch, i, self.num_batch, loss.item(), correct.item() / float(cur_batch_len)))

        torch.save(self.model.state_dict(), self.save_weights_dir)
        return self.loss_values

    def evaluate(self):
        # TODO: rewrite this. Do we need to evaluate here only the misclassification or include the similarity too?
        total_correct = 0
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
            classfication, v = self.model(points)
            pred = F.log_softmax(classfication, dim=1)
            loss = F.nll_loss(pred, target)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            test_loss_values.append(loss.item())
            total_correct += correct.item()
            total_testset += points.size()[0]
        test_accuracy = total_correct / len(self.test_data.dataset)
        test_mean_loss = sum(test_loss_values) / len(test_loss_values)

        return test_mean_loss, test_accuracy

# losses ----------------------------------------------------------


class LossFunction(object):
    def __init__(self, adv_example):
        self.adv_example = adv_example

    def __call__(self) -> torch.Tensor:
        raise NotImplementedError


class AdversarialLoss(LossFunction):
    def __init__(self, adv_example, perturbed_logits, k: float = 0):
        super().__init__(adv_example)
        self.k = torch.tensor([k], device=adv_example.device, dtype=adv_example.dtype_float)
        self.perturbed_logits = perturbed_logits

    def __call__(self) -> torch.Tensor:
        Z = self.perturbed_logits
        values, index = torch.sort(Z, dim=1)
        index = index[-1]
        argmax = index[-1] if index[-1] != self.adv_example.target else index[-2]  # max{Z(i): i != target}
        Z = Z[-1]
        Ztarget, Zmax = Z[self.adv_example.target], Z[argmax]
        return torch.max(Zmax - Ztarget, -self.k)

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
