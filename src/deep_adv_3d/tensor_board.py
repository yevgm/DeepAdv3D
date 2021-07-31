# variable definitions
from config import *

from shutil import copyfile
from multiprocessing import Process
import os
import logging
import psutil
import signal
import wandb

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Auxiliary
# ----------------------------------------------------------------------------------------------------------------------#
class TensorboardSupervisor:

    def __init__(self, log_dp=None, mode=3):
        # 'Mode: 0 - Does nothing. 1 - Opens up only server. 2 - Opens up only chrome. 3- Opens up both '
        super().__init__()
        self.mode = mode
        if mode not in [1, 2, 3]:
            raise ValueError(f'Invalid mode: {mode}')
        if log_dp is None:
            from config import TENSOR_LOG_DIR
            log_dp = TENSOR_LOG_DIR
        if mode != 2:
            self.server = TensorboardServer(log_dp)
            self.server.start()
            logging.info("Started Tensorboard Server")
        if mode != 1:
            self.chrome = ChromeProcess()
            self.chrome.start()
            logging.info("Opened Chrome Tab")

def finalize_tensorboard():
    logging.info('Killing Tensorboard Server')
    list_of_proccesses = findProcessIdByName('tensorboard.main')
    for process in list_of_proccesses:
        os.kill(process['pid'], signal.SIGTERM)


class TensorboardServer(Process):
    def __init__(self, log_dp):
        super().__init__()
        self.os_name = os.name
        self.log_dp = str(log_dp)
        self.daemon = True

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.log_dp}" --port=6006 2> NUL')
        elif self.os_name == 'posix':  # Linux
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.log_dp}" --port=6006 >/dev/null 2>&1')
        else:
            raise NotImplementedError(f'No support for OS : {self.os_name}')


class ChromeProcess(Process):
    def __init__(self):
        super().__init__()
        self.os_name = os.name
        self.daemon = True

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'start chrome  http://localhost:6006/')
        elif self.os_name == 'posix':  # Linux
            # os.system(f'google-chrome http://localhost:6006/')
            # start firefox instead
            os.system(f'firefox http://localhost:6006/')
        else:
            raise NotImplementedError(f'No support for OS : {self.os_name}')

def findProcessIdByName(processName):
    '''
    Get a list of all the PIDs of a all the running process whose name contains
    the given string processName
    '''
    listOfProcessObjects = []
    #Iterate over the all the running process
    for proc in psutil.process_iter():
       try:
           pinfo_pid = proc.as_dict(attrs=['pid'])
           pinfo_cmd = " ".join(proc.cmdline())
           # Check if process name contains the given name string.
           if pinfo_cmd is not None:
               if processName.lower() in pinfo_cmd.lower() :
                   listOfProcessObjects.append(pinfo_pid)
       except (psutil.NoSuchProcess, psutil.AccessDenied , psutil.ZombieProcess):
           pass
    return listOfProcessObjects

def generate_new_tensorboard_results_dir(date, mode="train", model='adv3d'):

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
    if model == "adv3d":
        new_tensorboard_name = date + "_" + str(DATASET_NAME)+"_Lr_"+str(LR)+"_Batch_"+str(batch_size)+"_"\
                              +LOSS+"_epoch_"+str(N_EPOCH)+"_K_"+str(K)
    else:
        new_tensorboard_name = date + "_" + str(DATASET_NAME) + "_Lr_" + str(LR) + "_Batch_" + str(batch_size) + "_" \
                               + LOSS + "_epoch_" + str(N_EPOCH) + "_classifier_train"
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

def report_to_tensorboard(split, tensor_obj, batch_idx, step_cntr, cur_batch_len, epoch, n_batches, total_loss,
                          recon_loss, missclassify_loss, perturbed_logits, targets):
    # report to wandb
    wandb_log_dict = {"Total Loss": total_loss,
    "Misclassification Loss": missclassify_loss}

    # if USE_RECONSTRUCTION_LOSS:
    if True:
        wandb_log_dict.update({"Reconstruction Loss": recon_loss})

    wandb.log(wandb_log_dict)

    # Metrics
    pred_choice = perturbed_logits.data.max(1)[1]
    num_misclassified = pred_choice.eq(targets).sum().cpu()
    if split == 'train':
        if step_cntr % SHOW_LOSS_EVERY == SHOW_LOSS_EVERY - 1:  # every SHOW_LOSS_EVERY mini-batches
            # old stdout prints
            print('[Epoch #%d: Batch %d/%d] Train loss: %f, Misclassified: [%d/%d]' % (
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
    elif split == 'validation':
        # old stdout prints
        print('[Epoch #%d: Batch %d/%d] Validation loss: %f, Misclassified: [%d/%d]' % (
            epoch, n_batches, batch_idx, total_loss, float(cur_batch_len), num_misclassified.item()))

        # ...log the running loss
        tensor_obj.add_scalar('Loss/Val_Total',
                              total_loss, step_cntr)
        tensor_obj.add_scalar('Loss/Val_Reconstruction_Loss',
                              RECON_LOSS_CONST * recon_loss, step_cntr)
        tensor_obj.add_scalar('Loss/Val_Misclassification_Loss',
                              missclassify_loss, step_cntr)
        tensor_obj.add_scalar('Accuracy/Val_Misclassified_Targets',
                              num_misclassified / float(cur_batch_len), step_cntr)
    else:
        print('Test loss: %f, Misclassified: [%d/%d]' % (total_loss, float(cur_batch_len), num_misclassified.item()))

        # ...log the running loss
        tensor_obj.add_scalar('Loss/Test_total',
                              total_loss, 1)
        tensor_obj.add_scalar('Loss/Test_reconstruction_loss',
                              recon_loss, 1)
        tensor_obj.add_scalar('Loss/Test_misclassification_loss',
                              missclassify_loss, 1)
        tensor_obj.add_scalar('Accuracy/Test_Misclassified_targets',
                              num_misclassified / float(cur_batch_len), 1)

# def report_test_to_tensorboard(tensor_obj, total_loss, recon_loss, missclassify_loss, num_misclassified, cur_batch_len):
#
#     print('test loss: %f, Misclassified: [%d/%d]' % (total_loss, float(cur_batch_len), num_misclassified.item()))
#
#     # ...log the running loss
#     tensor_obj.add_scalar('Loss/Test_total',
#                           total_loss, 1)
#     tensor_obj.add_scalar('Loss/Test_reconstruction_loss',
#                           recon_loss, 1)
#     tensor_obj.add_scalar('Loss/Test_misclassification_loss',
#                           missclassify_loss, 1)
#     tensor_obj.add_scalar('Accuracy/Test_Misclassified_targets',
#                           num_misclassified / float(cur_batch_len), 1)

def classifier_report_test_to_tensorboard(tensor_obj, total_loss, num_classified, number_of_test_samples):
    print('test loss: %f, classified: [%d/%d]' % (total_loss, float(number_of_test_samples), num_classified.item()))

    # ...log the running loss
    tensor_obj.add_scalar('Loss/Test_total',
                          total_loss, 1)
    tensor_obj.add_scalar('Accuracy/Test_classified_targets',
                          num_classified / float(number_of_test_samples), 1)


if __name__ == '__main__':
    list = findProcessIdByName('firefox')
    a=1