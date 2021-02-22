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
from test_trainer_pointnet import load_datasets

# # geometric loader
def load_datasets_for_regressor(train_batch=8, test_batch=20):
    # it uses carlini's FaustDataset class that inherits from torch_geometric.data.InMemoryDataset
    traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=True)
    testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True, transform_data=True)

    trainLoader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_batch,
                                                   shuffle=True,
                                                   num_workers=4)
    testLoader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=test_batch,
                                                   shuffle=False,
                                                   num_workers=4)

    return trainLoader, testLoader, traindata, testdata


def RegressorLoss(points, v, pred, target):  # TODO: write this
    pass


if __name__ == '__main__':
    pass
    # data loading
    # traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=True)
    # testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True, transform_data=True)
    trainLoader, testLoader, traindata, testdata = load_datasets(train_batch=32, test_batch=20)

    # model definition
    num_nodes = testdata.get(0).num_nodes
    classifier = PointNetCls(k=10, feature_transform=False, global_transform=False)
    classifier.to(DEVICE)
    # model1 = Model1(outDim=num_nodes, classifier_model=classifier)
    # model1 = model1.to(DEVICE)
    regressor = Regressor()

    # train network
    train_ins = trainer(train_data=trainLoader, test_data=testLoader,
                        model=model1, lossFunction=RegressorLoss,
                        classifier=classifier)
    train_ins.train()


