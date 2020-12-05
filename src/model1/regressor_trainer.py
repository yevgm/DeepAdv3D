# This is the main script that runs out model1:

# Using the central article's method in the neural setting
# '(revolves around regressing for the optimal smooth deformation field parameters needed to optimally deform the target'
# ' shape to achieve target/untargeted adverserial attack success)
# Architecture: Simple PointNet (Without T-Nets, see implementation in the Shape Completion Repo)
# + switch last layer to regression layer

# variable definitions
from config import *

# repository modules
from models.Origin_pointnet import PointNetCls, Regressor
from model1.deep_adv_3d import Model1, trainer
import dataset

if __name__ == '__main__':
    pass
    # data loading
    traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=True)
    testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True, transform_data=True)
    # model definition
    num_nodes = testdata.get(0).num_nodes
    classifier = PointNetCls(k=10, feature_transform=False, global_transform=False)
    classifier.to(DEVICE)
    model1 = Model1(outDim=num_nodes, classifier_model=classifier)
    model1 = model1.to(DEVICE)

    # train network
    train_ins = trainer(train_data=traindata, test_data=testdata, model=model1)
    train_ins.train()


