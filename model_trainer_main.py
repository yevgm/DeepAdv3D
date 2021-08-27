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
from models.deep_adv_3d_model1 import RegressorOriginalPointnet, OshriRegressor, Regressor
from deep_adv_3d.train_loop import *
from dataset.data_loaders import *
from utils.torch.nn import *

import torch.nn as nn
import wandb

def load_datasets(dataset, train_batch=8, test_batch=20, val_batch=20):
    if dataset == 'Faust':
        dataset_path = FAUST
        if LOAD_WHOLE_DATA_TO_MEMORY:
            class_inst = FaustDatasetInMemory
        else:
            class_inst = FaustDataset
    elif dataset == 'Shrec14':
        dataset_path = SHREC14
        if LOAD_WHOLE_DATA_TO_MEMORY:
            class_inst = Shrec14DatasetInMemory
            pass
        else:
            class_inst = Shrec14Dataset

    train_dataset = class_inst(
        root=os.path.join(dataset_path, r'raw'),
        split='train',
        data_augmentation=TRAIN_DATA_AUG)

    validation_dataset = class_inst(
        root=os.path.join(dataset_path, r'raw'),
        split='validation',
        data_augmentation=VAL_DATA_AUG)

    test_dataset = class_inst(
        root=os.path.join(dataset_path, r'raw'),
        split='test',
        data_augmentation=TEST_DATA_AUG)

    trainLoader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=NUM_WORKERS)
    validationLoader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size=val_batch,
                                               shuffle=SHUFFLE_VAL_DATA,
                                               num_workers=NUM_WORKERS)
    testLoader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_batch,
                                               shuffle=SHUFFLE_TEST_DATA,
                                               num_workers=NUM_WORKERS)

    return trainLoader, validationLoader, testLoader


if __name__ == '__main__':
    # set seed for all platforms
    set_determinsitic_run()

    if USE_WANDB:
        wandb.init(entity="deepadv3d", project="DeepAdv3D")

    # Data Loading and pre-processing
    trainLoader, validationLoader, testLoader = load_datasets(dataset=DATASET_NAME, train_batch=TRAIN_BATCH_SIZE,
                                                              test_batch=TEST_BATCH_SIZE, val_batch=VAL_BATCH_SIZE)

    # classifier and model definition
    classifier = PointNet(k=10)
    classifier.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE))
    model = RegressorOriginalPointnet()
    # model = OshriRegressor()
    # model = Regressor(numVertices=6890)
    train_ins = Trainer(train_data=trainLoader, validation_data=validationLoader, test_data=testLoader,
                        model=model, classifier=classifier)

    # train network
    train_ins.train()

