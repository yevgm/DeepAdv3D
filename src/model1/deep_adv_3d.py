import torch
import torch.nn as nn

# variable definitions
from config import *

# repository modules
from models.Origin_pointnet import PointNetCls, Regressor


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Classes Definition
# ----------------------------------------------------------------------------------------------------------------------#

class Model1(nn.Module):

    def __init__(self, outDim, classifier_model: nn.Module):
        pass
        # Definition of:
        # 1. regressor
        # 2. original pointNet
        super(Model1, self).__init__()
        self.regressor = Regressor(outDim, feature_transform=False)  # do we need feature transform false?
        self.classifier = PointNetCls()

    def forward(self, x):
        pass
        # Model forward pass, return output from trained classifier AND from regressor
        # run the pass, and with no grad run classifier
        # TODO: ASSERT that gradient is correctly backpropagated !

        v = self.regressor.forward(x)  # this is the perturbation

        with torch.no_grad():
            perturbed_pos = x + v
            classification = self.classifier.forward(perturbed_pos)

        return classification, v



class trainer:

    def __init__(self, train_data, test_data, model: nn.Module): #TODO: add datatype assert
        self.train_data = train_data
        self.test_data = test_data

        self.model = model
        self.batch_size = BATCH_SIZE
        self.num_batch = int(len(train_data.dataset) / self.batch_size)
        self.scheduler_step = SCH_STEP
        self.n_epoch = N_EPOCH
        self.lr = LR

        self.loss_values = [] #TODO: add tensorboard support instead
        self.save_weights_dir = PARAMS_FILE

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=0.5)

        self.model.to(DEVICE)

        for epoch in range(self.n_epoch):
            scheduler.step()
            for i, data in enumerate(self.train_data, 0):
                # TODO: completely rewrite this function to fit our model
                points, target = data
                target = target[:, 0]
                cur_batch_len = len(points)
                points = points.transpose(2, 1)

                points.to(DEVICE)
                target.to(DEVICE)

                optimizer.zero_grad()
                classifier = classifier.train()
                pred, trans, trans_feat = self.model(points) # output should be adex example
                pred = F.log_softmax(pred, dim=1)
                loss = F.nll_loss(pred, target)

                self.loss_values.append(loss.item())

                loss.backward()
                optimizer.step()
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                epoch, i, self.num_batch, loss.item(), correct.item() / float(cur_batch_len)))

        torch.save(self.model.state_dict(), self.save_weights_dir)
        return  loss_values, test_mean_loss, test_accuracy



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
