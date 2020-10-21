# built-in libraries
import sys
import os

# third party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.nn.functional as func
import random
import pyvista as pv
from torch import nn

# variable definitions
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),".."))
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SRC_DIR = os.path.join(REPO_ROOT,"src")
FAUST = os.path.join(REPO_ROOT,"datasets/faust")
PARAMS_FILE = os.path.join(REPO_ROOT, "model_data/FAUST10_pointnet.pt")

# repository modules
sys.path.insert(0, SRC_DIR)
import vista
from vista import plot_mesh, plot_mesh_montage
import models
# import train
import ntrain
import dataset
import utils
from models.pointnet import SimplePointNet
from models.Origin_pointnet import PointNetCls
from dataset.data_loaders import FaustDataset
import adversarial.carlini_wagner as cw
from adversarial.carlini_wagner import CWBuilder, LowbandPerturbation


def load_datasets(train_batch=8,test_batch=20):
    train_dataset = FaustDataset(
        root=os.path.join(FAUST, r'raw'),
        classification=True,
        split='train')

    test_dataset = FaustDataset(
        root=os.path.join(FAUST, r'raw'),
        classification=True,
        split='test',
        data_augmentation=False)

    trainLoader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=10)
    testLoader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_batch,
                                               shuffle=False,
                                               num_workers=10)
    # load data in different format for Adversarial code
    traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=False)
    testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True, transform_data=False)

    return trainLoader,testLoader,traindata,testdata

def show_model_accuracy(PARAMS_FILE,model):
    loss_values, test_mean_loss, test_accuracy = ntrain.train(
        train_data=trainLoader,
        test_data=testLoader,
        classifier=model,
        batchSize=20,
        parameters_file=PARAMS_FILE,
        learning_rate=1e-3,
        train=False)

    print('test mean loss:', test_mean_loss, ' test_accuracy:', test_accuracy)

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

    # Plot one mesh:
    # plot_mesh(perturbed, f=adex.faces, clr=color, label='Perturbed',
    #           slabel='L2 perturbation difference', lighting=1, clr_map='rainbow')

    # Plot multiple meshes using lists:
    gt_const_mid = color.max().item() / 2
    gt_const_min = color.min().item()
    vlist = [pos, perturbed, [pos, perturbed]]
    flist = [adex.faces, adex.faces, [adex.faces, adex.faces]]
    clist = [torch.zeros_like(color)+gt_const_min, color, [torch.zeros_like(color)+gt_const_mid, color]]
    Opacitylist = [1, 1, [0.35, 1]]
    plot_mesh_montage(vlist, fb=flist, clrb=clist, labelb=['Ground Truth', 'Perturbed', 'Both'],
                      slabelb=[ 'GT', 'L2 difference', 'L2 difference'], lighting=1, opacity=Opacitylist)



if __name__ == "__main__":
    model = PointNetCls(k=10, feature_transform=False)
    # print(model)
    trainLoader,testLoader, traindata, testdata = load_datasets(train_batch=8, test_batch=20)

    # train network
    # loss_values, test_mean_loss, test_accuracy = nTrain.train(
    #                                                         train_data=trainLoader,
    #                                                         test_data=testLoader,
    #                                                         classifier=model,
    #                                                         batchSize=batchsize,
    #                                                         parameters_file=PARAMS_FILE,
    #                                                         epoch_number=50,
    #                                                         learning_rate=4e-3,
    #                                                         train=True)
    # temp train visualizer - in the future : add tensorboard?
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

    # load parameters
    model.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE))
    model.eval()
    # show_model_accuracy(PARAMS_FILE, model)

    params = {
        CWBuilder.USETQDM: True,
        CWBuilder.MIN_IT: 100,
        CWBuilder.LEARN_RATE: 1e-4,
        CWBuilder.ADV_COEFF: 1,
        CWBuilder.REG_COEFF: 15,
        LowbandPerturbation.EIGS_NUMBER: 40}

    # choose random target
    while True:
        i = random.randint(0, len(testdata) - 1)
        target = random.randint(0, testdata.num_classes - 1)
        y = testdata[i].y.item()
        if y != target: break
    mesh = testdata[i]
    print(target)

    # search for adversarial example
    adex = cw.generate_adversarial_example(
        mesh=mesh, classifier=model, target=target,
        search_iterations=1,
        lowband_perturbation=True,
        adversarial_loss="carlini_wagner",
        similarity_loss="local_euclidean",
        **params)

    # show the found perturbed figure

    show_perturbation(adex)

    # # DEBUG vista
    pos = mesh.pos
    perturbed = mesh.pos
    faces = mesh.face
    color = (pos - perturbed+1).norm(p=2, dim=-1)
    # vlist = [pos,pos]
    # flist = [faces.T, faces.T]
    # clist = [color, color]
    # plot_mesh_montage(vlist, fb=flist, clrb=clist, labelb=['test1','test2'], slabelb=['L2 perturbation difference', 'original'],
    #           lighting=1)
    # vlist = [pos, pos, [pos, pos]]
    # flist = [faces.T, faces.T, [faces.T, faces.T]]
    # clist = [torch.zeros_like(color), color, [torch.zeros_like(color), color]]
    # plot_mesh_montage(vlist, fb=flist, clrb=clist, labelb=['Ground Truth', 'Perturbed', 'Both'],
    #                   slabelb=[ 'GT', 'L2 difference', 'L2 difference'], lighting=1)
    # plot_mesh(perturbed, f=faces.T, clr=color, label='test', slabel='L2 perturbation difference',
    #           lighting=1)
