import torch.nn.parallel
import torch.utils.data
from torch import nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from src.models.pointnet import PointNetfeat
# variable definitions
from config import *


class Encoder(nn.Module):
    def __init__(self, firstDim = 64):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, firstDim, 1)  # default 3, 64
        if MODEL_USE_BN:
            self.bn1 = nn.BatchNorm1d(firstDim, momentum=MODEL_BATCH_NORM_MOMENTUM, track_running_stats=MODEL_BATCH_NORM_USE_STATISTICS)  # default 64
        else:
            self.bn1 = nn.Identity()
        self.conv2 = torch.nn.Conv1d(firstDim, 2*firstDim, 1)  # default 64, 128
        if MODEL_USE_BN:
            self.bn2 = nn.BatchNorm1d(2*firstDim, momentum=MODEL_BATCH_NORM_MOMENTUM, track_running_stats=MODEL_BATCH_NORM_USE_STATISTICS)  # default 128
        else:
            self.bn2 = nn.Identity()
        self.conv3 = torch.nn.Conv1d(2*firstDim, 16*firstDim, 1)  # default 128,1024
        if MODEL_USE_BN:
            self.bn3 = nn.BatchNorm1d(16*firstDim, momentum=MODEL_BATCH_NORM_MOMENTUM, track_running_stats=MODEL_BATCH_NORM_USE_STATISTICS)  # default 1024
        else:
            self.bn3 = nn.Identity()

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))  # the first MLP layer (mlp64,64 shared)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


class Decoder(nn.Module):

    def __init__(self, outDim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(1024, 4 * outDim)  # 1024, 2048
        if MODEL_USE_BN:
            self.bn1 = nn.BatchNorm1d(4 * outDim, momentum=MODEL_BATCH_NORM_MOMENTUM, track_running_stats=MODEL_BATCH_NORM_USE_STATISTICS)  # 2048
        else:
            self.bn1 = nn.Identity()

        self.fc2 = nn.Linear(4 * outDim, 2 * outDim)  # 2048, 8192
        if MODEL_USE_DROPOUT:
            self.dropout = nn.Dropout(p=0.3)
        else:
            self.dropout = nn.Identity()
        if MODEL_USE_BN:
            self.bn2 = nn.BatchNorm1d(2 * outDim, momentum=MODEL_BATCH_NORM_MOMENTUM, track_running_stats=MODEL_BATCH_NORM_USE_STATISTICS)  # 8192
        else:
            self.bn2 = nn.Identity()
        self.fc3 = nn.Linear(2 * outDim, outDim)  # 8192, outDim , 6890 is for faust

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x


class Regressor(nn.Module):

    def __init__(self, numVertices, firstDim=64):
        super(Regressor, self).__init__()
        self.numVertices = numVertices
        self.outDim = 3*numVertices
        self.enc = Encoder(firstDim)
        self.dec = Decoder(self.outDim)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        x = x.view(-1, 3, self.numVertices)
        return x


class RegressorOriginalPointnet(nn.Module):

    def __init__(self):
        super(RegressorOriginalPointnet, self).__init__()
        self.feat = PointNetfeat()
        self.fc1 = nn.Linear(1024, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 8192)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(8192, 6890*3)

    def forward(self, x):
        # x, trans, trans_feat = self.feat(x)  # x is 1024, trans is exit from TNET1, trans_Feat is exit from tnet2
        x = self.feat(x)  # x is 1024, trans is exit from TNET1, trans_Feat is exit from tnet2
        x = F.relu(self.fc1(x))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, 3, 6890)  # that's the only difference from pointnet, along with layer sizes
        return x