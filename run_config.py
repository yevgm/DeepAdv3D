import os
import sys
import torch
import inspect

repo_root = os.path.dirname(os.path.realpath(__file__))
model_data_dir =  os.path.abspath(os.path.join(repo_root, "..", "model_data"))

run_config = {
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   run_config['DEVICE']
# ----------------------------------------------------------------------------------------------------------------------#
	'DEVICE': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   PATH
# ----------------------------------------------------------------------------------------------------------------------#
	'REPO_ROOT': repo_root,
	'MODEL_DATA_DIR': os.path.abspath(os.path.join(repo_root, "..", "model_data")),
	'SRC_DIR': os.path.join(repo_root, "src"),
	'FAUST': os.path.join(repo_root, "datasets", "faust"),
	'SHREC14': os.path.join(repo_root, "datasets", "shrec14_downsampled"),  # shrec14_downsampled, shrec14
	'RUN_NAME': None,
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   MODEL
# ----------------------------------------------------------------------------------------------------------------------#
# classifier:
	'PARAMS_FILE':  os.path.join(model_data_dir, "shrec14_no_aug_sep_100percent.ckpt"), #os.path.join(repo_root,'saved_params', "FAUST_classifier_august.ckpt"),  # FAUST10_pointnet_rot_b128_v2.pt, FAUST10_pointnet_rot_b128.pt, momentum_05.pt, shrec14_71percent_acc_momentum05.pt
	'MODEL_PARAMS_FILE':  os.path.join(repo_root,'saved_params', "eigens_and_augmentation_recon_loss.ckpt"),
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TENSORBOARD
# ----------------------------------------------------------------------------------------------------------------------#
	'RUN_TB': False,  # run tensorboard server 
	'RUN_BROWSER': False,
	'TERMINATE_TB': False,
	'TENSOR_LOG_DIR': os.path.abspath(os.path.join(repo_root, "..", "tensor_board_logs")),
	'SHOW_LOSS_EVERY': 1,  # log the loss value every SHOW_LOSS_EVERY mini-batches
	'FLUSH_RESULTS': 5, # in seconds
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Weights and biases
# ----------------------------------------------------------------------------------------------------------------------#
	'USE_WANDB': False,
	'USE_PLOTTER': True,
	'SAVE_WEIGHTS': False,
	'LOG_GRADIENTS_WANDB': False,  # slows down the training significantly. 
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TRAIN HYPERPARAMETERS
# ----------------------------------------------------------------------------------------------------------------------#
	'UNIVERSAL_RAND_SEED': 143, #143
	'EARLY_STOP_WAIT': 250, # epochs
	'LR_SCHEDULER_WAIT':  80, # epochs
	'SCHEDULER_STEP_SIZE': 250,
	'OPTIMIZER': 'AdamW', # 'Adam', 'AdamW', 'sgd'

	'TRAINING_CLASSIFIER': False,  # turn on to switch between classifier train and model train
	'CALCULATE_EIGENVECTORS': False,
	'LR': 0.050572566231955045, # learning rate
	'WEIGHT_DECAY': 0.0844692091146692, # regularization 1e-4

	'TRAIN_BATCH_SIZE': 40,  # number of data examples in one batch
	'N_EPOCH': 2000,  # number of train epochs
	'RECON_LOSS_CONST': 1000,  # ratio between reconstruction loss and missclasificaition loss
	'TRAIN_DATA_AUG': False,
	'DROPOUT_PROB': 0.06799470785277999,

# Architecture parameters - Do not change after classifier has been trained! number 247 in the sweep
# parameters: --DROPOUT_PROB=0.06799470785277999 --LATENT_SPACE_FEAT=1024 --LR=0.050572566231955045 --OPTIMIZER=AdamW --POINTNET_LAST_LAYER_SIZE=128 --TRAIN_BATCH_SIZE=70 --WEIGHT_DECAY=0.0844692091146692
	'DROPOUT_PROB_CLS': 0.06799470785277999,
	'POINTNET_LAST_LAYER_SIZE': 256, # 128
	'LATENT_SPACE_FEAT': 1024,
	'MODEL_LAST_LAYER_SIZE': 256,  # Important: the model transfers 512 to this, and this to K*3, so it cant be too large nor too small
	# adversarial example params:
	'K': 40,  #40 number of laplacian eigenvectors to take. NOTE: up to 70. more then that the model decoder is "broken" - see model
	'LOSS': 'l2',  # 'l2', 'local_euclidean'
	'CHOOSE_LOSS': 3,  ## 1 for only misclassification, 2 for only reconstruction, 3 - both
	# local euclidean loss params:
	'CUTOFF': 5,  # 40
	'NEIGHBORS': 20,  # 140
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TEST
# ----------------------------------------------------------------------------------------------------------------------#
# Don't forget to update the test parameters to the original train!
	'SHUFFLE_TEST_DATA': False,
	'TARGET_CLASS': 5,  # the attack target - still not used\
	'TEST_DATA_AUG': False,
	'PLOT_TEST_SAMPLE': False,
	'TEST_EPOCHS': 1,  # valid use only with "TEST_DATA_AUG = True"
	# validation set: 
	'VAL_BATCH_SIZE': 20,
	'SHUFFLE_VAL_DATA': False,
	'VAL_STEP_EVERY': 1,  # epochs
	'VAL_DATA_AUG': False,
	'TEST_BATCH_SIZE': 20,
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   DATA
# ----------------------------------------------------------------------------------------------------------------------#
	'NUM_WORKERS': 0,
	'DATASET_CLASSES': 10,
	'DATASET_TRAIN_SIZE': 320,
	'DATASET_NAME': "Shrec14", # 'Faust', 'Shrec14'
	'LOAD_WHOLE_DATA_TO_MEMORY': True,  # use InMemory of Not in dataset loader stage
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   VISTA
# ----------------------------------------------------------------------------------------------------------------------#
	'SHOW_TRAINING': True,  # interactive training
	'CLIM': [0, 0.01],  # None or [0, 0.2] - it's the color limit of the shapes

	#testing: 
	'VIS_N_MESH_SETS': 2,  # Parallel plot will plot 8 meshes for each mesh set - 4 from train, 4 from vald
	'VIS_STRATEGY': 'mesh',  # spheres,cloud,mesh  - Choose how to display the meshes
	'VIS_CMAP': 'OrRd',  # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
	# We use two colors: one for the mask verts [Right end of the spectrum] and one for the rest [Left end of the spectrum].
	'VIS_SMOOTH_SHADING': False,  # Smooth out the mesh before visualization?  Applicable only for 'mesh' method
	'VIS_SHOW_EDGES': False,  # Visualize with edges? Applicable only for 'mesh' method
	'VIS_SHOW_GRID': True,
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   DEBUG
# ----------------------------------------------------------------------------------------------------------------------#
	# classifier bn
	'CLS_USE_BN': False,
	'CLS_BATCH_NORM_USE_STATISTICS': False,
	'CLS_BATCH_NORM_MOMENTUM': 0.1,  # default is 0.1
	'CLS_STRICT_PARAM_LOADING': False, # strict = False for dropping running mean and var of train batchnorm
	# model bn
	'MODEL_USE_BN': True


}

sys.path.insert(0, run_config['SRC_DIR'])





# ----------------------------------------------------------------------------------------------------------------------#
#                                                   NO LONGER IN USE
# ----------------------------------------------------------------------------------------------------------------------#
# MODEL_BATCH_NORM_USE_STATISTICS = False
# MODEL_BATCH_NORM_MOMENTUM = 0.5  # default is 0.1
# MODEL_STRICT_PARAM_LOADING = False  # strict = False for dropping running mean and var of train batchnorm
# model dropout
# MODEL_USE_DROPOUT = False