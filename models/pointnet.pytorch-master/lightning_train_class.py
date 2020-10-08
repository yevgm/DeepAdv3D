from __future__ import print_function
import torch.nn as nn
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
import pointnet.model as mod
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers


class PointNetCls_light(pl.LightningModule):

    def __init__(self, feature_transform=False):
        super(PointNetCls_light, self).__init__()
        self.classes = 16
        self.feature_transform = feature_transform
        self.feat = mod.PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, self.classes)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

    def prepare_data(self):
        self.batchsize = 32
        self.num_points = 2500
        self.workers = 4
        self.nepoch = 50
        self.dataset_type = 'shapenet'
        self.dataset = '/home/jack/OneDrive/Studies/Undergrad_Project/Project_kickstart/Geometric Deep Learning Ramp/pointnet.pytorch-master/shapenetcore_partanno_segmentation_benchmark_v0'
        self.model = '' # model path
        self.outf = 'cls' # output folder
        self.feature_transform = False

        self.train_dataset = ShapeNetDataset(
            root=self.dataset,
            classification=True,
            npoints=self.num_points)

        self.test_dataset = ShapeNetDataset(
            root=self.dataset,
            classification=True,
            split='test',
            npoints=self.num_points,
            data_augmentation=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                            batch_size=self.batchsize,
                                            shuffle=True,
                                            num_workers=int(self.workers))
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                            batch_size=self.batchsize,
                                            num_workers=int(self.workers))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]

    def cross_entropy_loss(self, pred, target):
        return F.nll_loss(pred, target)

    def training_step(self, batch, batch_idx):
        points, target = batch
        target = target[:, 0]
        points = points.transpose(2, 1)
        points = points.to(self.device)
        pred, trans, trans_feat = self.forward(points)

        loss = self.cross_entropy_loss(pred, target)
        if self.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).sum()
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs, 'correct': correct.item() / self.batchsize}

    def validation_step(self, batch, batch_idx):
            j, data = next(enumerate(self.val_dataloader(), 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points = points.to(self.device)
            pred, _, _ = self.forward(points)
            pred = pred.cpu()
            loss = self.cross_entropy_loss(pred, target)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).sum()
            return {'loss': loss , 'correct': correct.item() / self.batchsize}

# train
tb_logger = pl_loggers.TensorBoardLogger('logs/')
model = PointNetCls_light()
trainer = pl.Trainer(logger=tb_logger, max_epochs=5,gpus=-1)

trainer.fit(model)