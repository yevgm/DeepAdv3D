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

def report_to_wandb_classifier(epoch, split, epoch_loss, epoch_classified=0):

    if split == 'train':
        if USE_WANDB:
            wandb.log({
                "Train\Epoch Loss": epoch_loss,
                "Train\Epoch Classified": epoch_classified})


        print('[Epoch #%d] Train loss: %f, Classified: [%d/70]' % (
            epoch, epoch_loss, epoch_classified))
    elif split == 'validation':
        if USE_WANDB:
            wandb.log({
                "Validation\Epoch Loss": epoch_loss,
                "Validation\Epoch Classified": epoch_classified})

        print('[Epoch #%d] Validation loss: %f, Classified: [%d/15]' % (
            epoch, epoch_loss, epoch_classified))
    elif split== 'test':
        if USE_WANDB:
            my_data = [
                ["TestLoss", epoch_loss],
                ["TestAccuracy", epoch_classified]
            ]
            columns = ["Name", "Values"]
            data_table = wandb.Table(data=my_data, columns=columns)
            wandb.log({"Test_Results":data_table})


def report_to_wandb_regressor(epoch, split, epoch_loss, epoch_misclassified):

    if split == 'train':
        if USE_WANDB:
            wandb.log({
                "Train\Epoch Loss": epoch_loss,
                "Train\Epoch Misclassified": epoch_misclassified})

        print('[Epoch #%d] Train loss: %f, Misclassified: [%d/70]' % (
            epoch, epoch_loss, epoch_misclassified))
    elif split == 'validation':
        if USE_WANDB:
            wandb.log({
                "Validation\Epoch Loss": epoch_loss,
                "Validation\Epoch Misclassified": epoch_misclassified})

        print('[Epoch #%d] Validation loss: %f, Misclassified: [%d/15]' % (
            epoch, epoch_loss, epoch_misclassified))
    elif split == 'test':
        if USE_WANDB:
            my_data = [
                ["Test Loss", epoch_loss],
                ["Test Misclassified/15", epoch_misclassified]
            ]
            columns = ["Name", "Values"]
            data_table = wandb.Table(data=my_data, columns=columns)
            wandb.log({"Test_Results": data_table})



if __name__ == '__main__':
    list = findProcessIdByName('firefox')
    a=1