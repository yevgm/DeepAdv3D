import dataset
import models
import train
'''
FAUST = "../../Downloads/Mesh-Datasets/MyFaustDataset"
COMA = "../../Downloads/Mesh-Datasets/MyComaDataset"
SHREC14 =  "../../Downloads/Mesh-Datasets/MyShrec14"

PARAMS_FILE = "../model_data/FAUST10.pt"


traindata = dataset.FaustDataset(FAUST, train=True, test=False)
traindata = dataset.FaustAugmented(FAUST, train=True, test=False)
testdata = dataset.FaustDataset(FAUST, train=False, test=True)

model = models.ChebnetClassifier(
    param_conv_layers=[128,128,64,64],
    D_t=traindata.downscale_matrices,
    E_t=traindata.downscaled_edges,
    num_classes = traindata.num_classes,
    parameters_file=PARAMS_FILE)

#train network
train.train(
    train_data=traindata,
    classifier=model,
    parameters_file=PARAMS_FILE,
    epoch_number=0)


#compute accuracy
accuracy, confusion_matrix = train.evaluate(eval_data=testdata,classifier=model)
print(accuracy)

i = 20
x = traindata[i].pos
e = traindata[i].edge_index.t()
f = traindata[i].face.t()
y = traindata[i].y
t = 2
n = x.shape[0]
eigs_num = 100

import adversarial.carlini_wagner as cw
# targeted attack using C&W method
logger = cw.ValueLogger({"adversarial": lambda x:x.adversarial_loss()})
builder = cw.CWBuilder(search_iterations=1)
builder.set_classifier(model).set_mesh(x,e,f).set_target(t)

builder.set_distortion_function(cw.L2_distortion).set_perturbation_type("lowband", eigs_num=eigs_num)
builder.set_minimization_iterations(0).set_adversarial_coeff(0.1)
adex_cw = builder.build(usetqdm="standard")

# untargeted attack using FGSM
adex_it = pgd.FGSMBuilder().set_classifier(model).set_mesh(x,e,f).build()
'''

# built-in libraries
import os 

# third party libraries
import matplotlib.pyplot as plt 
import numpy as np
import tqdm
import torch 
import torch.nn.functional as func

# repository modules
import models
import train
import adversarial.carlini_wagner as cw
import adversarial.pgd as pgd
import dataset
import utils

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),".."))
FAUST = os.path.join(REPO_ROOT,"datasets/faust")
SHREC14 = os.path.join(REPO_ROOT,"datasets/shrec14")
SMAL = os.path.join(REPO_ROOT,"datasets/smal")
PARAMS_FILE = os.path.join(REPO_ROOT, "model_data/data.pt")




DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SRC_DIR = os.path.join(REPO_ROOT,"src")
SHREC14 =  os.path.join(REPO_ROOT,"datasets/shrec14")
traindata = dataset.Shrec14Dataset(SHREC14,device=DEVICE, train=True, test=False)
testdata = dataset.Shrec14Dataset(SHREC14, device=DEVICE, train=False, test=True, transform_data=False)
from torch import nn
from models.pointnet import SimplePointNet 

#autoencoder
LATENT_SPACE = 128
NUM_POINTS = 7000
ENC = SimplePointNet(
    latent_dimensionality=LATENT_SPACE*2,
    convolutional_output_dim=512,
    conv_layer_sizes=[32, 128, 256],
    fc_layer_sizes=[512, 256, 128],
    transformer_positions=[0]).to(DEVICE)

# classifier
CLA = nn.Sequential(nn.Linear(LATENT_SPACE, 64), nn.ReLU(), nn.Linear(64,10)).to(DEVICE)
params = sum([np.prod(p.size()) for p in ENC.parameters()])
ENC(traindata[0].pos)

traindata = dataset.FaustDataset(FAUST, train=True, test=False, transform_data=True)
testdata = dataset.FaustDataset(FAUST, train=False, test=True,  transform_data=True)

model = models.ChebnetClassifier(
    param_conv_layers=[128,128,64,64],
    D_t = traindata.downscale_matrices,
    E_t = traindata.downscaled_edges,
    num_classes = traindata.num_classes,
    parameters_file=PARAMS_FILE)

model = model.to(torch.device("cpu"))


#train network
train.train(
    train_data=traindata,
    classifier=model,
    parameters_file=PARAMS_FILE,
    epoch_number=0)