# This is the main script that runs out model1:

# Using the central article's method in the neural setting
# revolves around regressing for the optimal smooth deformation field parameters needed to optimally deform the target
# shape to achieve target/untargeted adversarial attack success
# Architecture: Simple PointNet (Without T-Nets, see implementation in the Shape Completion Repo)
# + switch last layer to regression layer

# variable definitions
from config import *

# repository modules
from models.Origin_pointnet import PointNetCls, Model1
from model1.deep_adv_3d import *

import dataset
from test_trainer_pointnet import load_datasets

# # geometric loader
# def load_datasets_for_regressor(train_batch=8, test_batch=20):
#     # it uses carlini's FaustDataset class that inherits from torch_geometric.data.InMemoryDataset
#     traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=True)
#     testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True, transform_data=True)
#
#     trainLoader = torch.utils.data.DataLoader(train_dataset,
#                                                    batch_size=train_batch,
#                                                    shuffle=True,
#                                                    num_workers=4)
#     testLoader = torch.utils.data.DataLoader(test_dataset,
#                                                    batch_size=test_batch,
#                                                    shuffle=False,
#                                                    num_workers=4)
#
#     return trainLoader, testLoader, traindata, testdata


if __name__ == '__main__':
    # data loading
    # traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=True)
    # testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True, transform_data=True)
    # torch data, not geometric!
    trainLoader, testLoader, traindata, testdata = load_datasets(train_batch=TRAIN_BATCH_SIZE, test_batch=TEST_BATCH_SIZE)

    # model definition
    num_nodes = testdata.get(0).num_nodes
    # TODO: load classifier parameters
    classifier = PointNetCls(k=10, feature_transform=False, global_transform=False)
    classifier.to(DEVICE)
    model = Model1(numVertices=6890)  # FAUST has 6890

    # train network
    train_ins = trainer(train_data=trainLoader, test_data=testLoader,
                        model=model, classifier=classifier)
    train_ins.train()


