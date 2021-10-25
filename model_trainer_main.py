# This is the main script that runs out model1:

# Using the central article's method in the neural setting
# revolves around regressing for the optimal smooth deformation field parameters needed to optimally deform the target
# shape to achieve target/untargeted adversarial attack success
# Architecture: Simple PointNet (Without T-Nets, see implementation in the Shape Completion Repo)
# + switch last layer to regression layer

# variable definitions
from run_config import *

# repository modules
from models.pointnet import PointNet
from models.deep_adv_3d_model1 import RegressorOriginalPointnet, OshriRegressor, RegressorOriginalPointnetEigen, RegressorEigenSeptember, RegressorEigenSeptemberDeep
from deep_adv_3d.train_loop import *
from dataset.data_loaders import *
from utils.torch.nn import *
import wandb
# fix for 2Right
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_datasets(run_config):
    dataset = run_config['DATASET_NAME']
    train_batch = run_config['TRAIN_BATCH_SIZE']
    test_batch = run_config['TEST_BATCH_SIZE']
    val_batch = run_config['VAL_BATCH_SIZE']

    if dataset == 'Faust':
        dataset_path = run_config['FAUST']
        if run_config['LOAD_WHOLE_DATA_TO_MEMORY']:
            class_inst = FaustDatasetInMemory
        else:
            pass
            # class_inst = FaustDataset
    elif dataset == 'Shrec14':
        dataset_path = run_config['SHREC14']
        if run_config['LOAD_WHOLE_DATA_TO_MEMORY']:
            class_inst = Shrec14DatasetInMemory
            pass
        else:
            pass
            # class_inst = Shrec14Dataset

    train_dataset = class_inst(
        run_config=run_config,
        root=os.path.join(dataset_path, r'raw'),
        split='train',
        data_augmentation=run_config['TRAIN_DATA_AUG'])

    validation_dataset = class_inst(
        run_config=run_config,
        root=os.path.join(dataset_path, r'raw'),
        split='validation',
        data_augmentation=run_config['VAL_DATA_AUG'])

    test_dataset = class_inst(
        run_config=run_config,
        root=os.path.join(dataset_path, r'raw'),
        split='test',
        data_augmentation=run_config['TEST_DATA_AUG'])

    trainLoader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=run_config['NUM_WORKERS'])
    validationLoader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size=val_batch,
                                               shuffle=run_config['SHUFFLE_VAL_DATA'],
                                               num_workers=run_config['NUM_WORKERS'])
    testLoader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_batch,
                                               shuffle=run_config['SHUFFLE_TEST_DATA'],
                                               num_workers=run_config['NUM_WORKERS'])

    return trainLoader, validationLoader, testLoader


if __name__ == '__main__':
    # set seed for all platforms
    set_determinsitic_run(run_config=run_config)

    config = run_config  # default for debug
    if run_config['USE_WANDB']:
        wandb.init(entity="deepadv3d", project="DeepAdv3d_sweeps", config=run_config)
        config = wandb.config._items
        config['RUN_NAME'] = wandb.run.name

    # Data Loading and pre-processing
    trainLoader, validationLoader, testLoader = load_datasets(run_config=config)

    # classifier and model definition
    classifier = PointNet(config, k=10)
    classifier.load_state_dict(torch.load(config['PARAMS_FILE'], map_location=config['DEVICE']))
    model = RegressorOriginalPointnet(config)
    # model = RegressorEigenSeptember(config)
    # model = nn.DataParallel(model)
    # classifier = nn.DataParallel(classifier)
    # model = RegressorOriginalPointnetEigen(config)
    # model.load_state_dict(torch.load(MODEL_PARAMS_FILE, map_location=run_config['DEVICE']))
    # model = OshriRegressor()
    # model = Regressor(numVertices=6890)
    # model = RegressorOriginalPointnetEigen(K=K)
    train_ins = Trainer(train_data=trainLoader, validation_data=validationLoader, test_data=testLoader,
                            model=model, classifier=classifier, run_config=config)

    # train network
    train_ins.train()

    if run_config['USE_WANDB']:
        wandb.finish()
