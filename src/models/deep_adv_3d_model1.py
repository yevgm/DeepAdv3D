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

    def __init__(self, in_feat, outDim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(in_feat, 4 * in_feat)  # 1024, 2048
        if MODEL_USE_BN:
            self.bn1 = nn.BatchNorm1d(4 * in_feat, momentum=MODEL_BATCH_NORM_MOMENTUM, track_running_stats=MODEL_BATCH_NORM_USE_STATISTICS)  # 2048
        else:
            self.bn1 = nn.Identity()

        self.fc2 = nn.Linear(4 * in_feat, 8 * in_feat)  # 2048, 8192
        if MODEL_USE_DROPOUT:
            self.dropout = nn.Dropout(p=0.3)
        else:
            self.dropout = nn.Identity()
        if MODEL_USE_BN:
            self.bn2 = nn.BatchNorm1d(8 * in_feat, momentum=MODEL_BATCH_NORM_MOMENTUM, track_running_stats=MODEL_BATCH_NORM_USE_STATISTICS)  # 8192
        else:
            self.bn2 = nn.Identity()
        self.fc3 = nn.Linear(8 * in_feat, outDim)  # 8192, outDim , 6890 is for faust

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
        self.dec = Decoder(in_feat=1024,outDim=self.outDim)

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
        self.fc3 = nn.Linear(8192, 6892*3)

    def forward(self, x):
        # x, trans, trans_feat = self.feat(x)  # x is 1024, trans is exit from TNET1, trans_Feat is exit from tnet2
        x = self.feat(x)  # x is 1024, trans is exit from TNET1, trans_Feat is exit from tnet2
        x = F.relu(self.fc1(x))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, 3, 6892)  # that's the only difference from pointnet, along with layer sizes
        return x

    ################### OSHRI MODEL ###############################

class DensePointNetFeatures(nn.Module):
    def __init__(self, code_size, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(in_channels + 64, 128, 1)
        self.conv3 = nn.Conv1d(in_channels + 64 + 128, code_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(code_size)

    def init_weights(self):
        conv_mu = 0.0
        conv_sigma = 0.02
        bn_gamma_mu = 1.0
        bn_gamma_sigma = 0.02
        bn_betta_bias = 0.0

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=conv_mu, std=conv_sigma)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, mean=bn_gamma_mu, std=bn_gamma_sigma)  # weight=gamma, bias=betta
                nn.init.constant_(m.bias, bn_betta_bias)

    # noinspection PyTypeChecker
    def forward(self, x):
        # Input: Batch of Point Clouds : [b x num_vertices X in_channels]
        # Output: The global feature vector : [b x code_size]
        x = x.transpose(2, 1).contiguous()  # [b x in_channels x num_vertices]
        y = F.relu(self.bn1(self.conv1(x)))  # [B x 64 x n]
        z = F.relu(self.bn2(self.conv2(torch.cat((x, y), 1))))  # [B x 128 x n]
        z = self.bn3(self.conv3(torch.cat((x, y, z), 1)))  # [B x code_size x n]
        z, _ = torch.max(z, 2)  # [B x code_size]
        return z

class ShapeEncoder(nn.Module):
    def __init__(self, code_size=1024, in_channels=3):
        super().__init__()
        self.code_size = code_size
        self.in_channels = in_channels

        features = DensePointNetFeatures(self.code_size, self.in_channels)


        self.encoder = nn.Sequential(
            features,
            nn.Linear(self.code_size, self.code_size),
            nn.BatchNorm1d(self.code_size),
            nn.ReLU()
        )

    def init_weights(self):
        bn_gamma_mu = 1.0
        bn_gamma_sigma = 0.02
        bn_betta_bias = 0.0

        nn.init.normal_(self.encoder[2].weight, mean=bn_gamma_mu, std=bn_gamma_sigma)  # weight=gamma
        nn.init.constant_(self.encoder[2].bias, bn_betta_bias)  # bias=betta

# Input: Batch of Point Clouds : [b x num_vertices X in_channels]
# Output: The global feature vector : [b x code_size]
    def forward(self, shape):
        return self.encoder(shape)

class ShapeDecoder(nn.Module):
    CCFG = [1, 2, 4, 8]  # Enlarge this if you need more

    def __init__(self, pnt_code_size, out_channels, num_convl):
        super().__init__()

        self.pnt_code_size = pnt_code_size
        self.out_channels = out_channels
        if num_convl > len(self.CCFG):
            raise NotImplementedError("Please enlarge the Conv Config vector")

        self.thl = nn.Tanh()
        self.convls = []
        self.bnls = []
        for i in range(num_convl - 1):
            self.convls.append(nn.Conv1d(self.pnt_code_size * self.CCFG[i], self.pnt_code_size * self.CCFG[i + 1], 1))
            self.bnls.append(nn.BatchNorm1d(self.pnt_code_size * self.CCFG[i + 1]))
        self.convls.append(nn.Conv1d(self.pnt_code_size * self.CCFG[num_convl - 1], self.out_channels, 1))
        self.convls = nn.ModuleList(self.convls)
        self.bnls = nn.ModuleList(self.bnls)

    def init_weights(self):
        conv_mu = 0.0
        conv_sigma = 0.02
        bn_gamma_mu = 1.0
        bn_gamma_sigma = 0.02
        bn_betta_bias = 0.0

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=conv_mu, std=conv_sigma)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, mean=bn_gamma_mu, std=bn_gamma_sigma)  # weight=gamma
                nn.init.constant_(m.bias, bn_betta_bias)  # bias=betta

    # noinspection PyTypeChecker
    # Input: Point code for each point: [b x nv x pnt_code_size]
    # Where pnt_code_size == in_channels + 2*shape_code
    # Output: predicted coordinates for each point, after the deformation [B x nv x 3]
    def forward(self, x):
        # x = x.transpose(2, 1).contiguous()  # [b x nv x in_channels]
        for convl, bnl in zip(self.convls[:-1], self.bnls):
            x = F.relu(bnl(convl(x)))
        out = 2 * self.thl(self.convls[-1](x))  # TODO - Fix this constant - we need a global scale
        # out = out.transpose(2, 1)
        out = out.squeeze()
        return out

class OshriRegressor(nn.Module):

    def __init__(self, numVertices=6890):
        super(OshriRegressor, self).__init__()
        self.numVertices = numVertices
        self.outDim = 3 * numVertices
        self.enc = ShapeEncoder(code_size=1024, in_channels=3)
        self.dec = ShapeDecoder(pnt_code_size=1024, out_channels=3 * numVertices, num_convl=4)

        self.enc.init_weights()
        self.dec.init_weights()

        # self.fc1 = nn.Linear(1024, 4096)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(4096, 8192)
        # self.dropout = nn.Dropout(p=0.3)
        # self.relu = nn.ReLU()
        # self.fc3 = nn.Linear(8192, 6890*3)

    def forward(self, x):
        x = x.transpose(2,1)
        x = self.enc(x)
        x = x.unsqueeze(dim=2)
        x = self.dec(x)
        x = x.view(-1, 3, self.numVertices)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.dropout(self.fc2(x)))
        # x = self.fc3(x)
        # x = x.view(-1, 3, 6890)  # that's the only difference from pointnet, along with layer sizes
        return x

################ MODELS THAT SUPPORT EIGEN VECTORS #############


class RegressorOriginalPointnetEigen(nn.Module):

    def __init__(self, K):
        super(RegressorOriginalPointnetEigen, self).__init__()
        self.feat = PointNetfeat()
        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, K*3)

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, 3, K)
        return x