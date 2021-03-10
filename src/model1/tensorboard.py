# variable definitions
from config import *

from shutil import copyfile
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Auxiliary
# ----------------------------------------------------------------------------------------------------------------------#


def generate_new_tensorboard_results_dir(mode="train"):

    # find folder name
    if mode == "train":
        batch_size = TRAIN_BATCH_SIZE
    else:
        batch_size = TEST_BATCH_SIZE

    dir_list = os.listdir(TENSOR_LOG_DIR)
    cur_idx = len(dir_list) + 1
    new_tensorboard_name = str(cur_idx)+"_"+str(DATASET_NAME)+"_Lr_"+str(LR)+"_Batch_"+str(batch_size)+"_"\
                          +LOSS+"_epoch_"+str(N_EPOCH)+"_K_"+str(K)
    new_dir_name = os.path.join(TENSOR_LOG_DIR, new_tensorboard_name)

    # create the folder if not exists
    if not os.path.isdir(new_dir_name):
        try:
            os.mkdir(new_dir_name)
        except:
            sys.exit("New tensorboard folder could not be created")
    else:
        sys.exit("New tensorboard folder already exists")

    # copy current config file to log hyper parameters
    try:
        copyfile(os.path.join(REPO_ROOT, "config.py"), os.path.join(new_dir_name, "config.py"))
    except IOError:
        sys.exit("Can't copy config file to new tensorboard log dir")

    return new_dir_name