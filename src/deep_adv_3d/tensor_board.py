from shutil import copyfile
from multiprocessing import Process
import os
import sys
import logging
import psutil
import signal
import wandb

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Auxiliary
# ----------------------------------------------------------------------------------------------------------------------#
class TensorboardSupervisor:

    def __init__(self, run_config, log_dp=None, mode=3):
        # 'Mode: 0 - Does nothing. 1 - Opens up only server. 2 - Opens up only chrome. 3- Opens up both '
        super().__init__()
        self.mode = mode
        if mode not in [1, 2, 3]:
            raise ValueError(f'Invalid mode: {mode}')
        if log_dp is None:
            log_dp = run_config['TENSOR_LOG_DIR']
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

def generate_new_results_dir(date, run_config, model='adv3d'):

    batch_size = run_config['TRAIN_BATCH_SIZE']
    TENSOR_LOG_DIR = run_config['TENSOR_LOG_DIR']
    DATASET_NAME = run_config['DATASET_NAME']
    LOSS = run_config['LOSS']
    LR = run_config['LR']
    K = run_config['K']
    N_EPOCH = run_config['N_EPOCH']
    REPO_ROOT = run_config['REPO_ROOT']

    # create the main folder if not exists
    if not os.path.isdir(TENSOR_LOG_DIR):
        try:
            os.mkdir(TENSOR_LOG_DIR)
        except:
            sys.exit("New tensorboard folder could not be created")

    if run_config['RUN_NAME'] is not None:
        new_tensorboard_name = date + "_" + run_config['RUN_NAME']
    else:
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
        copyfile(os.path.join(REPO_ROOT, "run_config.py"), os.path.join(new_dir_name, "run_config.py"))
    except IOError:
        sys.exit("Can't copy config file to new tensorboard log dir")

    return new_dir_name


# def classifier_report_to_tensorboard(tensor_obj, batch_idx, step_cntr, cur_batch_len, epoch, n_batches, total_loss,
#                                     num_classified):
#     if step_cntr % SHOW_LOSS_EVERY == SHOW_LOSS_EVERY - 1:  # every SHOW_LOSS_EVERY mini-batches
#         # old stdout prints
#         print('[Epoch #%d: Batch %d/%d] train loss: %f, Classified: [%d/%d]' % (
#             epoch, n_batches, batch_idx, total_loss, float(cur_batch_len), num_classified.item()))
#
#         # ...log the running loss
#         tensor_obj.add_scalar('Loss/Train_total',
#                                total_loss, step_cntr)
#         tensor_obj.add_scalar('Accuracy/Train_classified',
#                                num_classified / float(cur_batch_len), step_cntr)

def report_to_wandb_classifier(run_config, epoch, split, epoch_loss, epoch_classified=0):

    if split == 'train':
        if run_config['USE_WANDB']:
            wandb.log({
                "Train/Epoch Loss": epoch_loss,
                "Train/Epoch Classified": epoch_classified})


        print('[Epoch #%d] Train loss: %f, Classified: [%d/%d]' % (
            epoch, epoch_loss, epoch_classified, run_config['DATASET_TRAIN_SIZE']))
    elif split == 'validation':
        if run_config['USE_WANDB']:
            wandb.log({
                "Validation/Epoch Loss": epoch_loss,
                "Validation/Epoch Classified": epoch_classified})

        print('[Epoch #%d] Validation loss: %f, Classified: [%d/%d]' % (
            epoch, epoch_loss, epoch_classified, run_config['DATASET_VAL_SIZE']))
    elif split== 'test':

        if run_config['USE_WANDB']:
            wandb.log({
                "Test/loss": epoch_loss,
                "Test/classified": epoch_classified})

            print('Test loss: %f, Classified: [%d/15]' % (epoch_loss, epoch_classified))
            # my_data = [
            #     ["TestLoss", epoch_loss],
            #     ["TestAccuracy", epoch_classified]
            # ]
            # columns = ["Name", "Values"]
            # data_table = wandb.Table(data=my_data, columns=columns)
            # wandb.log({"Test_Results":data_table})


def report_to_wandb_regressor(run_config, epoch, split, epoch_loss, epoch_misclassified, misloss=None, recon_loss=None):

    if split == 'train':
        if run_config['USE_WANDB']:
            logdict = {"Train/Epoch Loss": epoch_loss, "Train/Epoch Misclassified": epoch_misclassified}
            if misloss is not None:
                logdict.update({"Train/Misclass Loss": misloss})
            if recon_loss is not None:
                logdict.update({"Train/Reconstruction Loss": recon_loss})
            wandb.log(logdict)

        print('[Epoch #%d] Train loss: %f, Misclassified: [%d/%d]' % (
            epoch, epoch_loss, epoch_misclassified, run_config['DATASET_TRAIN_SIZE']))
    elif split == 'validation':
        if run_config['USE_WANDB']:
            logdict = {"Validation\Epoch Loss": epoch_loss, "Validation/Epoch Misclassified": epoch_misclassified}
            if misloss is not None:
                logdict.update({"Validation/Misclass Loss": misloss})
            if recon_loss is not None:

                logdict.update({"Validation/Reconstruction Loss": recon_loss})
            wandb.log(logdict)

        print('[Epoch #%d] Validation loss: %f, Misclassified: [%d/%d]' % (
            epoch, epoch_loss, epoch_misclassified, run_config['DATASET_VAL_SIZE']))
    elif split == 'test':

        if run_config['USE_WANDB']:
            wandb.log({
                "Test/loss": epoch_loss,
                "Test/misclassified": epoch_misclassified})

        print('Test loss: %f, misclassified: [%d/15]' % (epoch_loss, epoch_misclassified))

            # my_data = [
            #     ["Test Loss", epoch_loss],
            #     ["Test Misclassified/15", epoch_misclassified]
            # ]
            # columns = ["Name", "Values"]
            # data_table = wandb.Table(data=my_data, columns=columns)
            # wandb.log({"Test_Results": data_table})



if __name__ == '__main__':
    list = findProcessIdByName('firefox')
    a=1