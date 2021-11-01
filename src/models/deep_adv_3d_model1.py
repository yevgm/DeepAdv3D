import torch.nn.parallel
import torch.utils.data
from torch import nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from src.models.pointnet import PointNetfeat

class PointNetfeatModel(nn.Module):
    def __init__(self, run_config):
        super(PointNetfeatModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        if run_config['MODEL_USE_BN']:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()


    def forward(self, x):
        # if len(x.shape)<3:
        #     x = torch.unsqueeze(x.T, 0) ## fix for geometric data loader

        x = F.relu(self.bn1(self.conv1(x)))  # the first MLP layer (mlp64,64 shared)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

class RegressorOriginalPointnet(nn.Module):

    def __init__(self, run_config):
        super(RegressorOriginalPointnet, self).__init__()
        self.run_config = run_config
        self.v_size = run_config['NUM_VERTICES']
        self.dropout_p = run_config['DROPOUT_PROB']
        self.feat = PointNetfeatModel(run_config)
        self.fc1 = nn.Linear(1024, 2048)

        self.drop1 = nn.Dropout(p=self.dropout_p)
        self.fc2 = nn.Linear(2048, 4096)

        self.drop2 = nn.Dropout(p=self.dropout_p)
        self.fc3 = nn.Linear(4096, 8192)

        self.drop3 = nn.Dropout(p=self.dropout_p)
        self.fc4 = nn.Linear(8192, 6890*3)

        if self.run_config['MODEL_USE_BN']:
            self.bn1 = nn.BatchNorm1d(2048)
            self.bn2 = nn.BatchNorm1d(4096)
            self.bn3 = nn.BatchNorm1d(8192)

        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

    def forward(self, x):
        # x, trans, trans_feat = self.feat(x)  # x is 1024, trans is exit from TNET1, trans_Feat is exit from tnet2
        x = self.feat(x)  # x is 1024, trans is exit from TNET1, trans_Feat is exit from tnet2
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2((self.fc2(x)))))
        x = self.drop3(F.relu(self.bn3((self.fc3(x)))))
        x = self.fc4(x)
        x = x.view(-1, 3, self.v_size)  # that's the only difference from pointnet, along with layer sizes
        return x


################ MODELS THAT SUPPORT EIGEN VECTORS #############

class RegressorEigenSeptember(nn.Module):

    def __init__(self, run_config):
        super(RegressorEigenSeptember, self).__init__()
        self.run_config = run_config
        self.v_size = run_config['NUM_VERTICES']
        self.dropout_p = run_config['DROPOUT_PROB']
        self.k = run_config['K']
        self.feat = PointNetfeatModel(run_config)
        self.fc1 = nn.Linear(1024, 512)

        self.drop1 = nn.Dropout(p=self.dropout_p)
        self.fc2 = nn.Linear(512, 256) # 256

        self.drop2 = nn.Dropout(p=self.dropout_p)
        self.fc3 = nn.Linear(256, self.k * 3) # 256

        if self.run_config['MODEL_USE_BN']:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256) # 256

        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

    def forward(self, x):
        x = self.feat(x)  # x is 1024
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2((self.fc2(x)))))
        x = self.fc3(x)
        x = x.view(-1, 3, self.k)  # that's the only difference from pointnet, along with layer sizes
        return x
