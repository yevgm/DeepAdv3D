from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from utils.torch.nn import *

# class STN3d(nn.Module):
#     def __init__(self):
#         super(STN3d, self).__init__()
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 9)
#         self.relu = nn.ReLU()
#
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)
#
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#
#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)
#
#         iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
#         iden = iden.to(x.device)
#         x = x + iden
#         x = x.view(-1, 3, 3)
#         return x


# class STNkd(nn.Module):
#     def __init__(self, k=64):
#         super(STNkd, self).__init__()
#         self.conv1 = torch.nn.Conv1d(k, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k*k)
#         self.relu = nn.ReLU()
#
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)
#
#         self.k = k
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#
#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)
#
#         iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k*self.k).repeat(batchsize,1)
#         x = x + iden
#         x = x.view(-1, self.k, self.k)
#         return x

class PointNetfeat(nn.Module):
    def __init__(self, run_config):
        super(PointNetfeat, self).__init__()
        self.latent_space_feat = run_config['LATENT_SPACE_FEAT']
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, run_config['LATENT_SPACE_FEAT'], 1)
        if run_config['CLS_USE_BN']:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(run_config['LATENT_SPACE_FEAT'])
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()



    def forward(self, x):
        x = F.relu(self.conv1(x))  # the first MLP layer (mlp64,64 shared)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


class PointNet(nn.Module):

    def __init__(self, run_config, k=16):
        super(PointNet, self).__init__()

        self.classes = k
        self.feat = PointNetfeat(run_config)
        self.fc1 = nn.Linear(run_config['LATENT_SPACE_FEAT'], 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, run_config['POINTNET_LAST_LAYER_SIZE'])
        self.dropout = nn.Dropout(p=run_config['DROPOUT_PROB_CLS'])
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(run_config['POINTNET_LAST_LAYER_SIZE'], self.classes)

        if run_config['CLS_USE_BN']:
            self.bn1 = nn.BatchNorm1d(run_config['LATENT_SPACE_FEAT'])
            self.bn2 = nn.BatchNorm1d(512)
            self.bn3 = nn.BatchNorm1d(run_config['POINTNET_LAST_LAYER_SIZE'])
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()


    def forward(self, x):
        # x, trans, trans_feat = self.feat(x)  # x is 1024, trans is exit from TNET1, trans_Feat is exit from tnet2
        x = self.feat(x)  # x is 1024, trans is exit from TNET1, trans_Feat is exit from tnet2
        x = F.relu(self.fc1(x))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x


# def feature_transform_regularizer(trans):
#     d = trans.size()[1]
#     batchsize = trans.size()[0]
#     I = torch.eye(d)[None, :, :]
#     loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
#     return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 3, 6890))

    cls = PointNet(k=10)
    out, _, _ = cls(sim_data)
    print('class', out.size())
    #
    # seg = PointNetDenseCls(k = 3)
    # out, _, _ = seg(sim_data)
    # print('seg', out.size())

    # model = Regressor(numVertices=6890)
    # out = model(sim_data)
    print('Regressor', out.size())
