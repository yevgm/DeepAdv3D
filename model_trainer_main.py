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
from dataset.data_loaders import *
from utils.torch.nn import *

import wandb
import torch.nn.init as init

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

def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.01)



if __name__ == '__main__':
    # close tensorboard process and exit
    if TERMINATE_TB:
        finalize_tensorboard()
        exit()

    wandb.init(entity="deepadv3d", project="DeepAdv3D")

    # set seed for all platforms
    set_determinsitic_run()

    # Data Loading and pre-processing
    trainLoader, validationLoader, testLoader = load_datasets(dataset=DATASET_NAME, train_batch=TRAIN_BATCH_SIZE,
                                                              test_batch=TEST_BATCH_SIZE, val_batch=VAL_BATCH_SIZE)

    # classifier and model definition
    classifier = PointNet(k=10)
    classifier.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE), strict=CLS_STRICT_PARAM_LOADING)  # strict = False for dropping running mean and var of train batchnorm
    model = Regressor(numVertices=K)  # K - additive vector field (V) dimension in eigen-space
    model.apply(initialize_weights)  # TODO: remove
    train_ins = Trainer(train_data=trainLoader, validation_data=validationLoader, test_data=testLoader,
                        model=model, classifier=classifier)
    # open tensorboard process if it's not already open
    if RUN_TB:
        tensor_board_sub_proccess_handler = TensorboardSupervisor(mode= RUN_TB + 2 * RUN_BROWSER)  # opens tensor at port 6006 if available

    # train network
    train_ins.train()

    # evaluate network
    # train_ins.evaluate(TEST_PARAMS_DIR)





