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
SHREC14 = os.path.join(REPO_ROOT, "datasets", "shrec14")
sys.path.insert(0, SRC_DIR)
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   MODEL
# ----------------------------------------------------------------------------------------------------------------------#
# classifier:
PARAMS_FILE = os.path.join(MODEL_DATA_DIR, "FAUST_classifier_august.ckpt")  # FAUST10_pointnet_rot_b128_v2.pt, FAUST10_pointnet_rot_b128.pt, momentum_05.pt, shrec14_71percent_acc_momentum05.pt
# model1:
# MODEL1_PARAMS_DIR = os.path.join(MODEL_DATA_DIR, "model1_params") # .pt will be added in the code
# PARAM_FILE_NAME = "model_params.pt"
# SAVE_PARAMS_EVERY = 300  # steps
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TENSORBOARD
# ----------------------------------------------------------------------------------------------------------------------#
RUN_TB = False  # run tensorboard server 
RUN_BROWSER = False
TERMINATE_TB = False
TENSOR_LOG_DIR = os.path.abspath(os.path.join(REPO_ROOT, "..", "tensor_board_logs"))
SHOW_LOSS_EVERY = 1  # log the loss value every SHOW_LOSS_EVERY mini-batches
FLUSH_RESULTS = 5  # in seconds
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Weights and biases
# ----------------------------------------------------------------------------------------------------------------------#
USE_WANDB = True
LOG_GRADIENTS_WANDB = False  # slows down the training significantly. 
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TRAIN HYPERPARAMETERS
# ----------------------------------------------------------------------------------------------------------------------#
UNIVERSAL_RAND_SEED = 143 #143
EARLY_STOP_WAIT = 120  # epochs
LR_SCHEDULER_WAIT =  60 # epochs
SCHEDULER_STEP_SIZE = 250
OPTIMIZER = 'Adam' # 'Adam', 'AdamW', 'sgd'

TRAINING_CLASSIFIER = False  # turn on to switch between classifier train and model train
CALCULATE_EIGENVECTORS = True
LR = 1e-3 # learning rate
WEIGHT_DECAY = 1e-4 # regularization 1e-4
TRAIN_BATCH_SIZE = 32  # number of data examples in one batch
TEST_BATCH_SIZE = 20
N_EPOCH = 500  # number of train epochs
RECON_LOSS_CONST = 200  # ratio between reconstruction loss and missclasificaition loss
TRAIN_DATA_AUG = False

# adversarial example params:
K = 50  #40 number of laplacian eigenvectors to take. NOTE: up to 70. more then that the model decoder is "broken" - see model
LOSS = 'l2'  # 'l2', 'local_euclidean'
# local euclidean loss params:
CUTOFF = 5  # 40
NEIGHBORS = 20  # 140
CHOOSE_LOSS = 1  ## 1 for only misclassification, 2 for only reconstruction, 3 - both
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TEST
# ----------------------------------------------------------------------------------------------------------------------#
# Don't forget to update the test parameters to the original train!
SHUFFLE_TEST_DATA = False
TEST_PARAMS_DIR = os.path.join(TENSOR_LOG_DIR, "Mar-29-2021_23-51-05_Faust_Lr_0.005_Batch_32_l2_epoch_800_K_40")  # here you put the tensor_board_logs foldername to test the model
TARGET_CLASS = 5  # the attack target - still not used\
TEST_DATA_AUG = False
PLOT_TEST_SAMPLE = False
TEST_EPOCHS = 1  # valid use only with "TEST_DATA_AUG = True"
# validation set: 
VAL_BATCH_SIZE = 20
SHUFFLE_VAL_DATA = False
VAL_STEP_EVERY = 1  # epochs
VAL_DATA_AUG = False
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   DATA
# ----------------------------------------------------------------------------------------------------------------------#
NUM_WORKERS = 0
DATASET_CLASSES = 10
DATASET_NAME = "Faust" # 'Faust', 'Shrec14'
EPS = 1e-9  # for numerical stability - used in calculating eigenvectors
LOAD_WHOLE_DATA_TO_MEMORY = True  # use InMemory of Not in dataset loader stage
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   VISTA
# ----------------------------------------------------------------------------------------------------------------------#
SHOW_TRAINING = True  # interactive training
CLIM = [0, 0.01]  # None or [0, 0.2] - it's the color limit of the shapes

#testing: 
VIS_N_MESH_SETS = 2  # Parallel plot will plot 8 meshes for each mesh set - 4 from train, 4 from vald
VIS_STRATEGY = 'mesh'  # spheres,cloud,mesh  - Choose how to display the meshes
VIS_CMAP = 'OrRd'  # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
# We use two colors: one for the mask verts [Right end of the spectrum] and one for the rest [Left end of the spectrum].
VIS_SMOOTH_SHADING = False  # Smooth out the mesh before visualization?  Applicable only for 'mesh' method
VIS_SHOW_EDGES = False  # Visualize with edges? Applicable only for 'mesh' method
VIS_SHOW_GRID = True
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   DEBUG
# ----------------------------------------------------------------------------------------------------------------------#
PLOT_TRAIN_IMAGES = False
SHOW_TRAIN_SAMPLE_EVERY = 100  # plot vista / save image to folder every SHOW_TRAIN_SAMPLE_EVERY gradient steps
# classifier bn
CLS_USE_BN = False
CLS_BATCH_NORM_USE_STATISTICS = False
CLS_BATCH_NORM_MOMENTUM = 0.1  # default is 0.1
CLS_STRICT_PARAM_LOADING = False  # strict = False for dropping running mean and var of train batchnorm
# model bn
MODEL_USE_BN = False
MODEL_BATCH_NORM_USE_STATISTICS = False
MODEL_BATCH_NORM_MOMENTUM = 0.5  # default is 0.1
MODEL_STRICT_PARAM_LOADING = False  # strict = False for dropping running mean and var of train batchnorm
# model dropout
MODEL_USE_DROPOUT = False
