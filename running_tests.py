# built-in libraries
import sys
import os

# third party libraries
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.nn.functional as func
import random
import pyvista as pv


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),".."))
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SRC_DIR = os.path.join(REPO_ROOT,"src")
FAUST = os.path.join(REPO_ROOT,"datasets/faust")
PARAMS_FILE = os.path.join(REPO_ROOT, "model_data/FAUST10_pointnet.pt")

# repository modules
sys.path.insert(0, SRC_DIR)
import models
# import train
import nTrain
import dataset
import utils

# traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=True)
# testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True,  transform_data=False)

from torch import nn
from models.pointnet import SimplePointNet

LATENT_SPACE = 256

# model = SimplePointNet(
#     latent_dimensionality=LATENT_SPACE,
#     convolutional_output_dim=1024,
#     conv_layer_sizes=[64, 64, 128],
#     fc_layer_sizes=[512],
#     transformer_positions=[0]).to(DEVICE)

from models.model import PointNetCls
model = PointNetCls(k=10, feature_transform=False)
# print(model)


# from torch_geometric.data import DataLoader

# batchsize = 8
# trainLoader = DataLoader(traindata,
#                 batch_size=batchsize,
#                 shuffle=True,
#                 num_workers=0)
# testLoader = DataLoader(testdata,
#                 batch_size=batchsize,
#                 shuffle=True,
#                 num_workers=0)

from dataset.dataset import FaustDataset
train_dataset = FaustDataset(
    root=os.path.join(FAUST, r'raw'),
    classification=True,
    split='train')

test_dataset = FaustDataset(
    root=os.path.join(FAUST, r'raw'),
    classification=True,
    split='test',
    data_augmentation=False)

batchsize = 8
trainLoader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batchsize,
                                           shuffle=True,
                                           num_workers=0)
testLoader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=20,
                                           shuffle=False,
                                           num_workers=0)

#train network
# loss_values, test_mean_loss, test_accuracy = nTrain.train(
#                                                         train_data=trainLoader,
#                                                         test_data=testLoader,
#                                                         classifier=model,
#                                                         batchSize=batchsize,
#                                                         parameters_file=PARAMS_FILE,
#                                                         epoch_number=50, # <- change here the number of epochs used for training
#                                                         learning_rate=4e-3,
#                                                         train=True)
#
# print('test mean loss:',test_mean_loss,' test_accuracy:',test_accuracy)
# loss_values = np.array(loss_values)
# sliced_loss = loss_values[0::5]#sliced
#
# fig, axs = plt.subplots(2)
# fig.suptitle('losses')
# axs[0].plot(np.arange(1,len(sliced_loss)+1,1), sliced_loss)
# axs[1].plot(np.arange(1,len(loss_values)+1,1), loss_values)
#
# axs[0].set(xlabel='5*batches index', ylabel='loss')
# axs[0].grid()
# axs[1].set(xlabel='batches index', ylabel='loss')
# axs[1].grid()
# plt.show()
# a=1



# load parameters
model.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE))
model.eval()

# loss_values, test_mean_loss, test_accuracy = nTrain.train(
#                                                         train_data=trainLoader,
#                                                         test_data=testLoader,
#                                                         classifier=model,
#                                                         batchSize=batchsize,
#                                                         parameters_file=PARAMS_FILE,
#                                                         learning_rate=1e-3,
#                                                         train=False)

# print('test mean loss:',test_mean_loss,' test_accuracy:',test_accuracy)

import plotly
import plotly.graph_objects as go


def visualize(pos, faces, intensity=None):
    cpu = torch.device("cpu")
    if type(pos) != np.ndarray:
        pos = pos.to(cpu).clone().detach().numpy()
    if pos.shape[-1] != 3:
        raise ValueError("Vertices positions must have shape [n,3]")
    if type(faces) != np.ndarray:
        faces = faces.to(cpu).clone().detach().numpy()
    if faces.shape[-1] != 3:
        raise ValueError("Face indices must have shape [m,3]")
    if intensity is None:
        intensity = np.ones([pos.shape[0]])
    elif type(intensity) != np.ndarray:
        intensity = intensity.to(cpu).clone().detach().numpy()

    x, z, y = pos.T
    i, j, k = faces.T

    mesh = go.Mesh3d(x=x, y=y, z=z,
                     color='lightpink',
                     intensity=intensity,
                     opacity=1,
                     colorscale=[[0, 'gold'], [0.5, 'mediumturquoise'], [1, 'magenta']],
                     i=i, j=j, k=k,
                     showscale=True)
    layout = go.Layout(scene=go.layout.Scene(aspectmode="data"))

    # pio.renderers.default="plotly_mimetype"
    fig = go.Figure(data=[mesh],
                    layout=layout)
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue")
    fig.show()


def compare(pos1, faces1, pos2, faces2):
    n, m = pos1.shape[0], pos2.shape[0]
    tmpx = torch.cat([pos1, pos2], dim=0)
    tmpf = torch.cat([faces1, faces2 + n], dim=0)
    color = torch.zeros([n + m], dtype=pos1.dtype, device=pos1.device)
    color[n:] = (pos1 - pos2).norm(p=2, dim=-1)
    visualize(tmpx, tmpf, color)


def show_perturbation(adex):
    perturbed = adex.perturbed_pos.cpu()
    pos = adex.pos.cpu()
    p1 = adex.logits.cpu().detach().numpy()
    p2 = adex.perturbed_logits.cpu().detach().numpy()
    m = min([p1.min(), p2.min()])
    num_classes = p1.shape[1]

    x_ticks = np.array(range(num_classes), dtype=float)
    ax = plt.subplot(111)
    ax.bar(x_ticks - 0.2, (p1 - m)[0], width=0.4, color='b', align='center')
    ax.bar(x_ticks + 0.2, (p2 - m)[0], width=0.4, color='y', align='center')
    ax.legend(["standard", "perturbed towards " + str(adex.target.item())])
    ax.set_title("Class Probabilities with/without Perturbation")
    plt.show()

    color = (pos - perturbed).norm(p=2, dim=-1)
    visualize(perturbed, adex.faces, intensity=color)
    # p_obj = pv.PolyData(perturbed.detach().numpy(), adex.faces.detach().numpy())
    # p_obj.plot(show_edges=True, style='wireframe')
    # cpu = torch.device("cpu")
    # p_obj = pv.PolyData(perturbed.to(cpu).clone().detach().numpy(), adex.faces.to(cpu).clone().detach().numpy())
    # p_obj["elevation"] = color.to(cpu).clone().detach().numpy()
    # p_obj.plot(show_edges=True)

# load data in different format for this code
traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=False)
testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True,  transform_data=False)

import adversarial.carlini_wagner as cw
from adversarial.carlini_wagner import CWBuilder, LowbandPerturbation


params = {
    CWBuilder.USETQDM:True,
    CWBuilder.MIN_IT:100,
    CWBuilder.LEARN_RATE:1e-4,
    CWBuilder.ADV_COEFF:1,
    CWBuilder.REG_COEFF:1,
    LowbandPerturbation.EIGS_NUMBER:40}

#choose random target
while True:
    i = random.randint(0, len(testdata)-1)
    target = random.randint(0, testdata.num_classes-1)
    y = testdata[i].y.item()
    if y != target: break
mesh = testdata[i]
print(target)

# search for adversarial example
adex = cw.generate_adversarial_example(
    mesh=mesh, classifier=model, target=target,
    search_iterations=1,  # was 1
    lowband_perturbation=False,  # was true
    adversarial_loss="carlini_wagner",
    similarity_loss="local_euclidean",
    **params)

show_perturbation(adex)
