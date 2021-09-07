import os
import shutil
import logging as log
import warnings
import numpy as np
import torch
import sys
from vista.adv_plotter import labels
from vista.geom_vis import plot_mesh, plot_mesh_montage

"""
Callbacks
=========

Callbacks supported by Lightning
"""

class Callback(object):
    """Abstract base class used to build new callbacks."""

    def __init__(self):
        pass

    def on_epoch_begin(self):
        """Called when the epoch begins."""
        pass

    def on_epoch_end(self, epoch, monitored_val):
        """Called when the epoch ends."""
        pass

    def on_batch_begin(self):
        """Called when the training batch begins."""
        pass

    def on_batch_end(self):
        """Called when the training batch ends."""
        pass

    def on_train_begin(self):
        """Called when the train begins."""
        pass

    def on_train_end(self):
        """Called when the train ends."""
        pass

    def on_validation_begin(self):
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, epoch, monitored_val):
        """Called when the validation loop ends."""
        pass

    def on_test_begin(self):
        """Called when the test begins."""
        pass

    def on_test_end(self):
        """Called when the test ends."""
        pass


class EarlyStopping(Callback):
    r"""
    Stop training when a monitored quantity has stopped improving.

    Args:
        monitor (str): quantity to be monitored. Default: ``'val_loss'``.
        min_delta (float): minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no
            improvement. Default: ``0``.
        patience (int): number of epochs with no improvement
            after which training will be stopped. Default: ``0``.
        verbose (bool): verbosity mode. Default: ``0``.
        mode (str): one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity. Default: ``'auto'``.
        strict (bool): whether to crash the training if `monitor` is
            not found in the metrics. Default: ``True``.

    Example::

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import EarlyStopping

        early_stopping = EarlyStopping('val_loss')
        Trainer(early_stop_callback=early_stopping)
    """

    def __init__(self, min_delta=0.0, patience=0, verbose=False, mode='min', strict=True):
        super(EarlyStopping, self).__init__()

        self.patience = patience
        self.verbose = verbose
        self.strict = strict
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            if self.verbose:
                log.info(f'EarlyStopping mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.on_train_begin()

    def check_metrics(self, monitor_val=None):
        error_msg = 'monitored value is not available'
        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose:
                warnings.warn(error_msg, RuntimeWarning)

            return False

        return True

    def on_train_begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, monitored_value):
        stop_training = False
        if not self.check_metrics(monitored_value):
            return stop_training

        current = monitored_value
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                stop_training = True
                self.on_train_end()

        return stop_training

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose:
            log.info(f'NOTE: Did not improve criterion for over {self.patience} epochs - STOPPING TRAIN')  # MANO


class ModelCheckpoint(Callback):
    r"""

    Save the model after every epoch.

    Args:
        filepath (str): path to save the model file.
            Can contain named formatting options to be auto-filled.

            Example::

                # save epoch and val_loss in name
                ModelCheckpoint(filepath='{epoch:02d}-{val_loss:.2f}.hdf5')
                # saves file like: /path/epoch_2-val_loss_0.2.hdf5
        monitor (str): quantity to monitor.
        verbose (bool): verbosity mode, 0 or 1.
        save_top_k (int): if `save_top_k == k`,
            the best k models according to
            the quantity monitored will be saved.
            if `save_top_k == 0`, no models are saved.
            if `save_top_k == -1`, all models are saved.
            Please note that the monitors are checked every `period` epochs.
            if `save_top_k >= 2` and the callback is called multiple
            times inside an epoch, the name of the saved file will be
            appended with a version count starting with `v0`.
        mode (str): one of {min, max}.
            If `save_top_k != 0`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc.
        save_weights_only (bool): if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period (int): Interval (number of epochs) between checkpoints.

    Example::

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

        checkpoint_callback = ModelCheckpoint(filepath='my_path')
        Trainer(checkpoint_callback=checkpoint_callback)

        # saves checkpoints to my_path whenever 'val_loss' has a new min
    """

    def __init__(self, filepath, model, verbose=0,
                 save_top_k=1, save_weights_only=False,
                 mode='min', period=1, prefix=''):
        super(ModelCheckpoint, self).__init__()

        self.model = model
        self.verbose = verbose
        self.filepath = filepath
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_check = 0
        self.prefix = prefix
        self.best_k_models = {}
        self.kth_best_model = ''
        self.best = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.kth_value = np.Inf
            self.mode = 'min'
        elif mode == 'max':
            self.monitor_op = np.greater
            self.kth_value = -np.Inf
            self.mode = 'max'


    def _del_model(self, filepath):
        dirpath = os.path.dirname(filepath)

        # make paths
        os.makedirs(dirpath, exist_ok=True)

        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

    def _save_model(self, filepath):
        # dirpath = os.path.dirname(filepath)
        #
        # # make paths
        # os.makedirs(dirpath, exist_ok=True)

        # delegate the saving to the model
        torch.save(self.model.state_dict(), filepath)

    def check_monitor_top_k(self, current):
        less_than_k_models = len(self.best_k_models.keys()) < self.save_top_k
        if less_than_k_models:
            return True
        return self.monitor_op(current, self.best_k_models[self.kth_best_model])

    def on_validation_end(self, epoch, monitored_value):

        self.epochs_since_last_check += 1

        if self.save_top_k == 0:
            # no models are saved
            return
        if self.epochs_since_last_check >= self.period:
            self.epochs_since_last_check = 0
            filepath = f'{self.filepath}/{self.prefix}_ckpt_epoch_{epoch}.ckpt'
            version_cnt = 0
            while os.path.isfile(filepath):
                # this epoch called before
                filepath = f'{self.filepath}/{self.prefix}_ckpt_epoch_{epoch}_v{version_cnt}.ckpt'
                version_cnt += 1

            if self.save_top_k != -1:
                current = monitored_value

                if current is None:
                    warnings.warn(
                        f'Can save best model only with {self.monitor} available,'
                        ' skipping.', RuntimeWarning)
                else:
                    if self.check_monitor_top_k(current):

                        # remove kth
                        if len(self.best_k_models.keys()) == self.save_top_k:
                            delpath = self.kth_best_model
                            self.best_k_models.pop(self.kth_best_model)
                            self._del_model(delpath)

                        self.best_k_models[filepath] = current
                        if len(self.best_k_models.keys()) == self.save_top_k:
                            # monitor dict has reached k elements
                            if self.mode == 'min':
                                self.kth_best_model = max(self.best_k_models, key=self.best_k_models.get)
                            else:
                                self.kth_best_model = min(self.best_k_models, key=self.best_k_models.get)
                            self.kth_value = self.best_k_models[self.kth_best_model]

                        if self.mode == 'min':
                            self.best = min(self.best_k_models.values())
                        else:
                            self.best = max(self.best_k_models.values())
                        if self.verbose > 0:
                            print('',end='\r')
                            log.info(
                                f'Epoch {epoch:05d}: {self.monitor} reached' # MANO - Remove \n
                                f' {current:0.5f} (best {self.best:0.5f}), saving model to'
                                f' {filepath} as top {self.save_top_k}')
                        self._save_model(filepath)

                    else:
                        if self.verbose > 0:
                            print('',end='\r')
                            log.info(f'Epoch {epoch:05d}: {self.monitor} was not in top {self.save_top_k}')
                            # MANO - Remove \n

            else:
                if self.verbose > 0:
                    print('',end='\r') # MANO - Removed \n, added print
                    log.info(f'Epoch {epoch:05d}: saving model to {filepath}')
                self._save_model(filepath)

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Misc Functions
# ----------------------------------------------------------------------------------------------------------------------#

def create_data_output_dir(run_config):

    if not os.path.isdir(run_config['MODEL_DATA_DIR']):
        try:
            os.mkdir(run_config['MODEL_DATA_DIR'])
        except:
            sys.exit("New model data folder could not be created")


def get_test_pic_filename(dir_name):
    dir_list = os.listdir(dir_name)
    cntr = 0
    for file in dir_list:
        if file.endswith(".png") & file.startswith("test_examples"):
            cntr += 1
    if cntr == 0:
        return os.path.join(dir_name, "test_examples")
    else:
        return os.path.join(dir_name, "test_examples_" + str(cntr))


def get_param_file(dir_name):
    dir_list = os.listdir(dir_name)
    for file in dir_list:
        if file.endswith(".pt"):
            return os.path.join(dir_name, file)


# def dump_adversarial_example_image(orig_vertices, adex, faces, step_num, file_path):
#     if PLOT_TRAIN_IMAGES & (step_num > 0) & (step_num % SHOW_TRAIN_SAMPLE_EVERY == 0):
#         p, _ = plot_mesh_montage([orig_vertices[0].T, adex[0].T], [faces[0], faces[0]], screenshot=True)
#         path = os.path.join(file_path, "step_" + str(step_num) + ".png")
#         p.show(screenshot=path, full_screen=True)

def dump_adversarial_example_image_batch(orig_vertices, adex, faces, orig_class, targets, logits, perturbed_logits, file_path):
    # orig_v_list = []
    # adex_v_list = []
    # color_list = []
    # faces_list = []
    # target_list = []
    # success_list = []
    # for i in range(orig_vertices.shape[0]):
    #     color = (orig_vertices[i] - adex[i]).norm(p=2, dim=-1)
    #     orig_v_list.append(orig_vertices[i])
    #     adex_v_list.append(adex[i])
    #     faces_list.append(faces[i])

    # p, _ = plot_mesh_montage([orig_vertices[0].T, adex[0].T], [faces[0], faces[0]], screenshot=True)
    # path = os.path.join(file_path, "test_examples.png")
    # p.show(screenshot=path, full_screen=True)

    perturbed_l = []
    faces_l = []
    color_l = []
    target_l = []
    success_l = []
    class_success_l = []
    classified_as_ = logits.data.max(1)[1]
    perturbed_class_ = perturbed_logits.data.max(1)[1]
    # fill the lists with needed information from main list
    # hardcoded to show 16 shapes for now (range(orig_vertices.shape[0]))
    for i in range(orig_vertices.shape[0]):
        perturbed = adex[i]
        pos = orig_vertices[i]
        color = (pos - perturbed).norm(p=2, dim=0)
        original_class = orig_class[i].item()
        classified_as = classified_as_[i].item()
        perturbed_class = perturbed_class_[i].item()
        target = targets[i].item()

        class_success_l.append((classified_as == original_class))
        success_l.append((perturbed_class == target) | (original_class == target))

        target_l.append([labels[original_class], labels[target]])
        perturbed_l.append(perturbed.T)
        faces_l.append(faces[i])
        color_l.append(color)

    # Plot all:
    p, _ = plot_mesh_montage(perturbed_l, fb=faces_l, clrb=color_l, labelb=target_l,
                             success=success_l, classifier_success=class_success_l,
                             cmap='OrRd', screenshot=True)
    path = get_test_pic_filename(file_path)
    p.show(screenshot=path, full_screen=True)