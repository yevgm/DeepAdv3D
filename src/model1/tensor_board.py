# variable definitions
from config import *

from shutil import copyfile
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Auxiliary
# ----------------------------------------------------------------------------------------------------------------------#


def generate_new_tensorboard_results_dir(date, mode="train"):

    # find folder name
    if mode == "train":
        batch_size = TRAIN_BATCH_SIZE
    else:
        batch_size = TEST_BATCH_SIZE

    # create the main folder if not exists
    if not os.path.isdir(TENSOR_LOG_DIR):
        try:
            os.mkdir(TENSOR_LOG_DIR)
        except:
            sys.exit("New tensorboard folder could not be created")

    new_tensorboard_name = date + "_" + str(DATASET_NAME)+"_Lr_"+str(LR)+"_Batch_"+str(batch_size)+"_"\
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


def report_to_tensorboard(tensor_obj, idx, batch_size, cur_batch_len, running_loss,
                          running_recon_loss, running_missclassify_loss, num_misclassified):
    if idx % SHOW_LOSS_EVERY == SHOW_LOSS_EVERY - 1:  # every SHOW_LOSS_EVERY mini-batches

        # ...log the running loss
        tensor_obj.add_scalar('Loss/Train_total',
                               running_loss / SHOW_LOSS_EVERY, idx)
        tensor_obj.add_scalar('Loss/Train_reconstruction_loss',
                               RECON_LOSS_CONST * running_recon_loss / SHOW_LOSS_EVERY, idx)
        tensor_obj.add_scalar('Loss/Train_misclassification_loss',
                               running_missclassify_loss / SHOW_LOSS_EVERY, idx)
        tensor_obj.add_scalar('Accuracy/Train_Misclassified_targets',
                               num_misclassified / float(cur_batch_len), idx)

        running_loss = 0.0
        running_recon_loss = 0.0
        running_missclassify_loss = 0.0
        num_misclassified = 0
    return running_loss, running_recon_loss, running_missclassify_loss, num_misclassified