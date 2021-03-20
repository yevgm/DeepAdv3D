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


def classifier_report_to_tensorboard(tensor_obj, batch_idx, step_cntr, cur_batch_len, epoch, n_batches, total_loss,
                                    num_classified):
    if step_cntr % SHOW_LOSS_EVERY == SHOW_LOSS_EVERY - 1:  # every SHOW_LOSS_EVERY mini-batches
        # old stdout prints
        print('[Epoch #%d: Batch %d/%d] train loss: %f, Classified: [%d/%d]' % (
            epoch, n_batches, batch_idx, total_loss, float(cur_batch_len), num_classified.item()))

        # ...log the running loss
        tensor_obj.add_scalar('Loss/Train_total',
                               total_loss, step_cntr)
        tensor_obj.add_scalar('Accuracy/Train_classified',
                               num_classified / float(cur_batch_len), step_cntr)

def report_to_tensorboard(tensor_obj, batch_idx, step_cntr, cur_batch_len, epoch, n_batches, total_loss,
                          recon_loss, missclassify_loss, num_misclassified):
    if step_cntr % SHOW_LOSS_EVERY == SHOW_LOSS_EVERY - 1:  # every SHOW_LOSS_EVERY mini-batches
        # old stdout prints
        print('[Epoch #%d: Batch %d/%d] train loss: %f, Misclassified: [%d/%d]' % (
            epoch, n_batches, batch_idx, total_loss, float(cur_batch_len), num_misclassified.item()))

        # ...log the running loss
        tensor_obj.add_scalar('Loss/Train_total',
                               total_loss, step_cntr)
        tensor_obj.add_scalar('Loss/Train_reconstruction_loss',
                               RECON_LOSS_CONST * recon_loss, step_cntr)
        tensor_obj.add_scalar('Loss/Train_misclassification_loss',
                               missclassify_loss, step_cntr)
        tensor_obj.add_scalar('Accuracy/Train_Misclassified_targets',
                               num_misclassified / float(cur_batch_len), step_cntr)

def report_test_to_tensorboard(tensor_obj, total_loss, recon_loss, missclassify_loss, num_misclassified, cur_batch_len):

    print('test loss: %f, Misclassified: [%d/%d]' % (total_loss, float(cur_batch_len), num_misclassified.item()))

    # ...log the running loss
    tensor_obj.add_scalar('Loss/Test_total',
                          total_loss, 1)
    tensor_obj.add_scalar('Loss/Test_reconstruction_loss',
                          recon_loss, 1)
    tensor_obj.add_scalar('Loss/Test_misclassification_loss',
                          missclassify_loss, 1)
    tensor_obj.add_scalar('Accuracy/Test_Misclassified_targets',
                          num_misclassified / float(cur_batch_len), 1)

def classifier_report_test_to_tensorboard(tensor_obj, total_loss, num_classified, number_of_test_samples):
    print('test loss: %f, classified: [%d/%d]' % (total_loss, float(number_of_test_samples), num_classified.item()))

    # ...log the running loss
    tensor_obj.add_scalar('Loss/Test_total',
                          total_loss, 1)
    tensor_obj.add_scalar('Accuracy/Test_classified_targets',
                          num_classified / float(number_of_test_samples), 1)