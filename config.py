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
PARAMS_FILE = os.path.join(MODEL_DATA_DIR, "momentum_03.pt") # "FAUST10_pointnet_rot_b128.pt" / FAUST10_pointnet_rot_b128.pt
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
TRAINING_CLASSIFIER = False # turn on to switch between classifier train and model train
LR = 5e-3  # learning rate
OPTIMIZER = 'AdamW' # 'Adam', 'AdamW'
WEIGHT_DECAY = 0.5 # regularization
SCHEDULER_STEP_SIZE = 150
TRAIN_BATCH_SIZE = 32  # number of data examples in one batch
TEST_BATCH_SIZE = 20
N_EPOCH = 230  # number of train epochs
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
TEST_PARAMS_DIR = os.path.join(TENSOR_LOG_DIR, "Mar-29-2021_19-28-21_Faust_Lr_0.001_Batch_80_l2_epoch_1500_classifier_train")
TARGET_CLASS = 5 # the attack target - still not used\
TEST_DATA_AUG = True
PLOT_TEST_SAMPLE = True
TEST_EPOCHS = 50 # valid use only with "TEST_DATA_AUG = True"
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   DATA
# ----------------------------------------------------------------------------------------------------------------------#
NUM_WORKERS = 0
DATASET_CLASSES = 10
DATASET_NAME = "Faust"
EPS = 1e-9 # for numerical stability - used in calculating eigenvectors
LOAD_WHOLE_DATA_TO_MEMORY = True # use InMemory of Not in dataset loader stage
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   DEBUG
# ----------------------------------------------------------------------------------------------------------------------#
DEBUG = True
SHOW_TRAIN_SAMPLE_EVERY = 100 # plot vista / save image to folder every SHOW_TRAIN_SAMPLE_EVERY gradient steps
BATCH_NORM_USE_STATISTICS = False
BATCH_NORM_MOMENTUM = 0.3 # default is 0.1
USE_BN = True