# built-in libraries
import sys
import os
from datetime import datetime

# third party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.nn.functional as func
import random
import wandb

# repository modules

from models.pointnet import PointNet
from model_trainer_main import load_datasets
from utils.torch.nn import *
from src.deep_adv_3d.train_loop import Trainer

# variable definitions
from run_config import *


if __name__ == "__main__":
    # set seed for all platforms
    set_determinsitic_run()

    if run_config['USE_WANDB']:
        wandb.init(entity="deepadv3d", project="DeepAdv3D")

    model = PointNet(k=10)
    model = model.to(run_config['DEVICE'])


    # Data Loading and pre-processing
    trainLoader, validationLoader, testLoader = load_datasets(run_config=run_config)

    train_ins = Trainer(run_config=run_config, train_data=trainLoader, validation_data=validationLoader, test_data=testLoader,
                            model=model)

    train_ins.train()