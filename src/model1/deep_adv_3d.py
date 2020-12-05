import torch
import torch.nn as nn

# variable definitions
from config import *

# repository modules
from models.Origin_pointnet import PointNetCls, Regressor


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Classes Definition
# ----------------------------------------------------------------------------------------------------------------------#

class Model1:

    def __init__(self, outDim, classifier_model: nn.Module):
        pass
        # Definition of:
        # 1. regressor
        # 2. original pointNet
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

    def __init__(self):
        pass

    def train(self):
        pass


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