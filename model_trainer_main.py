# This is the main script that runs out model1:

# Using the central article's method in the neural setting
# revolves around regressing for the optimal smooth deformation field parameters needed to optimally deform the target
# shape to achieve target/untargeted adversarial attack success
# Architecture: Simple PointNet (Without T-Nets, see implementation in the Shape Completion Repo)
# + switch last layer to regression layer

# variable definitions
from config import *

# repository modules
from models.pointnet import PointNet
from models.deep_adv_3d_model1 import Regressor
from deep_adv_3d.train_loop import *
from dataset.data_loaders import FaustDataset, FaustDatasetInMemory


def load_datasets(train_batch=8, test_batch=20):
    # here we use FaustDataset class that inherits from torch.utils.data.Dataloader. it's a map-style dataset.
    if LOAD_WHOLE_DATA_TO_MEMORY:
        train_dataset = FaustDatasetInMemory(
            root=os.path.join(FAUST, r'raw'),
            split='train',
            data_augmentation=TRAIN_DATA_AUG)

        test_dataset = FaustDatasetInMemory(
            root=os.path.join(FAUST, r'raw'),
            split='test',
            data_augmentation=TEST_DATA_AUG)
    else:
        train_dataset = FaustDataset(
            root=os.path.join(FAUST, r'raw'),
            split='train',
            data_augmentation=TRAIN_DATA_AUG)

        test_dataset = FaustDataset(
            root=os.path.join(FAUST, r'raw'),
            split='test',
            data_augmentation=TEST_DATA_AUG)

    trainLoader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=NUM_WORKERS)
    testLoader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_batch,
                                               shuffle=SHUFFLE_TEST_DATA,
                                               num_workers=NUM_WORKERS)

    return trainLoader, testLoader


if __name__ == '__main__':
    # Data Loading and pre-processing
    trainLoader, testLoader = load_datasets(train_batch=TRAIN_BATCH_SIZE, test_batch=TEST_BATCH_SIZE)

    # classifier and model definition
    classifier = PointNet(k=10, feature_transform=False, global_transform=False)
    classifier.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE), strict=CLS_STRICT_PARAM_LOADING)  # strict = False for dropping running mean and var of train batchnorm
    model = Regressor(numVertices=K)  # K - additive vector field (V) dimension in eigen-space
    train_ins = Trainer(train_data=trainLoader, test_data=testLoader,
                        model=model, classifier=classifier)

    # train network
    train_ins.train()

    # evaluate network
    # train_ins.evaluate(TEST_PARAMS_DIR)

