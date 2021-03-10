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
SRC_DIR = os.path.join(REPO_ROOT, "src")
FAUST = os.path.join(REPO_ROOT, "datasets/faust")
sys.path.insert(0, SRC_DIR)
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   MODEL
# ----------------------------------------------------------------------------------------------------------------------#
# classifier:
PARAMS_FILE = os.path.join(REPO_ROOT, "model_data/FAUST10_pointnet_rot_b128.pt")
# model1:
MODEL1_PARAMS_FILE = os.path.join(REPO_ROOT, "model_data/model1_params.pt")
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TRAIN HYPERPARAMETERS
# ----------------------------------------------------------------------------------------------------------------------#
LR = 4e-3  # learning rate
OPTIMIZER = 'Adam' # 'Adam', 'AdamW'
WEIGHT_DECAY = 0 # regularization
SCHEDULER_STEP_SIZE = 500
TRAIN_BATCH_SIZE = 4  # number of data examples in one batch
TEST_BATCH_SIZE = 20
N_EPOCH = 100  # number of train epochs
RECON_LOSS_CONST = 200 # ratio between reconstruction loss and missclasificaition loss 

# adversarial example params:
K = 40 # number of laplacian eigenvectors to take. NOTE: up to 70. more then that the model decoder is "broken" - see model
LOSS = 'local_euclidean' # 'l2', 'local_euclidean'
# local euclidean loss params:
CUTOFF = 5 # 40
NEIGHBORS = 20 # 140
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TEST
# ----------------------------------------------------------------------------------------------------------------------#
TARGET_CLASS = 5 # the attack target - still not used
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   DATA
# ----------------------------------------------------------------------------------------------------------------------#
NUM_WORKERS = 4
DATASET_CLASSES = 10
DATASET_NAME = "Faust"
EPS = 1e-9 # for numerical stability - used in calculating eigenvectors
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TENSORBOARD
# ----------------------------------------------------------------------------------------------------------------------#
TENSOR_LOG_DIR = os.path.abspath(os.path.join(REPO_ROOT, "..", "tensor_board_logs"))
SHOW_LOSS_EVERY = 1 # log the loss value every SHOW_LOSS_EVERY mini-batches
FLUSH_RESULTS = 5 # in seconds
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   DEBUG
# ----------------------------------------------------------------------------------------------------------------------#
DEBUG = True
SHOW_TRAIN_SAMPLE_EVERY = 30 # plot vista every SHOW_TRAIN_SAMPLE_EVERY batches
