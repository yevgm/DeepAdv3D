import os
import sys
import torch
import inspect

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   DEVICE
# ----------------------------------------------------------------------------------------------------------------------#
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   PATH
# ----------------------------------------------------------------------------------------------------------------------#
REPO_ROOT = os.path.dirname(os.path.realpath(__file__))
MODEL_DATA_DIR = os.path.abspath(os.path.join(REPO_ROOT, "..", "model_data"))
SRC_DIR = os.path.join(REPO_ROOT, "src")
FAUST = os.path.join(REPO_ROOT, "datasets", "faust")
sys.path.insert(0, SRC_DIR)
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   MODEL
# ----------------------------------------------------------------------------------------------------------------------#
# classifier:
PARAMS_FILE = os.path.join(MODEL_DATA_DIR, "FAUST10_pointnet_no_bn.pt") # FAUST10_pointnet_rot_b128.pt
# model1:
# MODEL1_PARAMS_DIR = os.path.join(MODEL_DATA_DIR, "model1_params") # .pt will be added in the code
PARAM_FILE_NAME = "model_params.pt"
SAVE_PARAMS_EVERY = 300 # steps
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TENSORBOARD
# ----------------------------------------------------------------------------------------------------------------------#
TENSOR_LOG_DIR = os.path.abspath(os.path.join(REPO_ROOT, "..", "tensor_board_logs"))
SHOW_LOSS_EVERY = 1 # log the loss value every SHOW_LOSS_EVERY mini-batches
FLUSH_RESULTS = 5 # in seconds
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TRAIN HYPERPARAMETERS
# ----------------------------------------------------------------------------------------------------------------------#
TRAINING_CLASSIFIER = True # turn on to switch between classifier train and model train
LR = 4e-3  # learning rate
OPTIMIZER = 'Adam' # 'Adam', 'AdamW'
WEIGHT_DECAY = 0.1 # regularization
SCHEDULER_STEP_SIZE = 200
TRAIN_BATCH_SIZE = 4  # number of data examples in one batch
TEST_BATCH_SIZE = 20
N_EPOCH = 3  # number of train epochs
RECON_LOSS_CONST = 400 # ratio between reconstruction loss and missclasificaition loss 
TRAIN_DATA_AUG = True

# adversarial example params:
K = 40 # number of laplacian eigenvectors to take. NOTE: up to 70. more then that the model decoder is "broken" - see model
LOSS = 'l2' # 'l2', 'local_euclidean'
# local euclidean loss params:
CUTOFF = 5 # 40
NEIGHBORS = 20 # 140
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TEST
# ----------------------------------------------------------------------------------------------------------------------#
# Don't forget to update the test parameters to the original train!
TEST_PARAMS_DIR = os.path.join(TENSOR_LOG_DIR, "3_Faust_Lr_0.004_Batch_8_l2_epoch_100_K_40")
TARGET_CLASS = 5 # the attack target - still not used\
TEST_DATA_AUG = True
PLOT_TEST_SAMPLE = True
TEST_EPOCHS = 1 # valid use only with "TEST_DATA_AUG = True"
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   DATA
# ----------------------------------------------------------------------------------------------------------------------#
NUM_WORKERS = 0
DATASET_CLASSES = 10
DATASET_NAME = "Faust"
EPS = 1e-9 # for numerical stability - used in calculating eigenvectors
LOAD_WHOLE_DATA_TO_MEMORY = False # use InMemory of Not in dataset loader stage
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   DEBUG
# ----------------------------------------------------------------------------------------------------------------------#
DEBUG = True
SHOW_TRAIN_SAMPLE_EVERY = 100 # plot vista / save image to folder every SHOW_TRAIN_SAMPLE_EVERY gradient steps
BATCH_NORM_USE_STATISTICS = True
BATCH_NORM_MOMENTUM = 0.1 # default is 0.1
USE_BN = False