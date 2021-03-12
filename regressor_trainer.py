# This is the main script that runs out model1:

# Using the central article's method in the neural setting
# revolves around regressing for the optimal smooth deformation field parameters needed to optimally deform the target
# shape to achieve target/untargeted adversarial attack success
# Architecture: Simple PointNet (Without T-Nets, see implementation in the Shape Completion Repo)
# + switch last layer to regression layer

# variable definitions
from config import *

# repository modules
from models.Origin_pointnet import PointNetCls, Regressor
from model1.deep_adv_3d import *
import dataset
from dataset.data_loaders import FaustDataset, FaustDatasetInMemory

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

def load_datasets(train_batch=8, test_batch=20):
    # here we use FaustDataset class that inherits from torch.utils.data.Dataloader. it's a map-style dataset.
    train_dataset = FaustDatasetInMemory(
        root=os.path.join(FAUST, r'raw'),
        split='train',
        data_augmentation=TRAIN_DATA_AUG)

    test_dataset = FaustDatasetInMemory(
        root=os.path.join(FAUST, r'raw'),
        split='test',
        data_augmentation=TEST_DATA_AUG)

    trainLoader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=NUM_WORKERS)
    testLoader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_batch,
                                               shuffle=False,
                                               num_workers=NUM_WORKERS)

    return trainLoader, testLoader

if __name__ == '__main__':
    # data loading
    # traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=True)
    # testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True, transform_data=True)
    # torch data, not geometric!
    trainLoader, testLoader = load_datasets(train_batch=TRAIN_BATCH_SIZE, test_batch=TEST_BATCH_SIZE)

    # classifier and model definition
    classifier = PointNetCls(k=10, feature_transform=False, global_transform=False)
    classifier.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE))
    model = Regressor(numVertices=K)  # K - additive vector field (V) dimension in eigen-space

    # train network
    train_ins = trainer(train_data=trainLoader, test_data=testLoader,
                        model=model, classifier=classifier)
    # train_ins.train()
    train_ins.evaluate(TEST_PARAMS_DIR)

