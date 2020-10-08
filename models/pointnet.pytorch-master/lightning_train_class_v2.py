from __future__ import print_function
import torch.nn as nn
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.autograd import Variable
import numpy as np
import os

if __name__ == "__main__":

    class STN3d(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(3, 64, 1)
            self.conv2 = torch.nn.Conv1d(64, 128, 1)
            self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 9)
            self.relu = nn.ReLU()

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)


        def forward(self, x):
            batchsize = x.size()[0]
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)

            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
            x = self.fc3(x)

            iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
            iden = iden.type_as(x)
            x = x + iden
            x = x.view(-1, 3, 3)
            return x


    class STNkd(pl.LightningModule):
        def __init__(self, k=64):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(k, 64, 1)
            self.conv2 = torch.nn.Conv1d(64, 128, 1)
            self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, k*k)
            self.relu = nn.ReLU()

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

            self.k = k

        def forward(self, x):
            batchsize = x.size()[0]
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)

            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
            x = self.fc3(x)

            iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
            iden.type_as(x)
            x = x + iden
            x = x.view(-1, self.k, self.k)
            return x

    class PointNetfeat(pl.LightningModule):
        def __init__(self, global_feat = True, feature_transform = False):
            super().__init__()
            self.stn = STN3d()
            self.conv1 = torch.nn.Conv1d(3, 64, 1)
            self.conv2 = torch.nn.Conv1d(64, 128, 1)
            self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.global_feat = global_feat
            self.feature_transform = feature_transform
            if self.feature_transform:
                self.fstn = STNkd(k=64)

        def forward(self, x):
            n_pts = x.size()[2]
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = F.relu(self.bn1(self.conv1(x)))

            if self.feature_transform:
                trans_feat = self.fstn(x)
                x = x.transpose(2,1)
                x = torch.bmm(x, trans_feat)
                x = x.transpose(2,1)
            else:
                trans_feat = None

            pointfeat = x
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans, trans_feat
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
                return torch.cat([x, pointfeat], 1), trans, trans_feat

    class PointNetDenseCls(pl.LightningModule):
        def __init__(self, k = 2, feature_transform=False):
            super().__init__()
            self.k = k
            self.feature_transform=feature_transform
            self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
            self.conv1 = torch.nn.Conv1d(1088, 512, 1)
            self.conv2 = torch.nn.Conv1d(512, 256, 1)
            self.conv3 = torch.nn.Conv1d(256, 128, 1)
            self.conv4 = torch.nn.Conv1d(128, self.k, 1)
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)

        def forward(self, x):
            batchsize = x.size()[0]
            n_pts = x.size()[2]
            x, trans, trans_feat = self.feat(x)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.conv4(x)
            x = x.transpose(2,1).contiguous()
            x = F.log_softmax(x.view(-1,self.k), dim=-1)
            x = x.view(batchsize, n_pts, self.k)
            return x, trans, trans_feat

    def feature_transform_regularizer(trans):
        d = trans.size()[1]
        batchsize = trans.size()[0]
        I = torch.eye(d)[None, :, :]
        I = I.type_as(trans)
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
        return loss

    class PointNetCls_light(pl.LightningModule):

        def __init__(self):
            super(PointNetCls_light, self).__init__()
            self.feature_transform = False
            self.classes = 16
            self.feat = PointNetfeat(global_feat=True, feature_transform=self.feature_transform)
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
            self.batchsize = 1
            self.num_points = 2500
            self.workers = 4
            self.nepoch = 3
            self.dataset_type = 'shapenet'

            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.dataset = os.path.join(dir_path,'shapenetcore_partanno_segmentation_benchmark_v0')
            self.outf = 'cls' # output folder


            self.train_dataset = ShapeNetDataset(
                root=self.dataset,
                classification=True,
                npoints=self.num_points)

            self.val_dataset = ShapeNetDataset(
                root=self.dataset,
                classification=True,
                split='val',
                npoints=self.num_points,
                data_augmentation=False)

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
            return torch.utils.data.DataLoader(self.val_dataset,
                                               batch_size=self.batchsize,
                                               num_workers=int(self.workers))

        def test_dataloader(self):
            return  torch.utils.data.DataLoader(self.test_dataset,
                                                batch_size=self.batchsize,
                                                shuffle=False,
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
            acc = correct.item() / self.batchsize
            logs = {'loss': loss, 'train_acc': acc}
            return {'loss': loss, 'log': logs, 'train_acc': acc}

        def validation_step(self, batch, batch_idx):
            points, target = batch # data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points = points.to(self.device)
            pred, _, _ = self.forward(points)
            pred = pred.cpu()
            loss = self.cross_entropy_loss(pred, target)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).sum()
            acc = correct.item() / self.batchsize
            print(" Validation step accuracy is ", acc)
            test_log = {'val_acc': acc, 'val_loss': loss}
            return {'val_loss': loss, 'val_acc': acc, 'log': test_log}

        def validation_end(self, outputs):
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_acc = np.stack([x['val_acc'] for x in outputs]).mean()
            end_log = {'val_loss': avg_loss,'val_acc':val_acc}
            return {'val_loss': avg_loss, 'val_acc':val_acc, 'log': end_log }

        def test_step(self, batch, batch_idx):
            points, target = batch
            target = target[:, 0]
            points = points.transpose(2, 1)
            points = points.to(self.device)
            pred, _, _ = self.forward(points)
            pred = pred.cpu()
            loss = self.cross_entropy_loss(pred, target)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).sum()
            acc = correct.item() / self.batchsize
            test_log = {'test_acc': acc,'test_loss':loss}
            return {'test_loss': loss, 'test_acc': acc, 'log': test_log}

        def test_end(self, outputs):
            avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
            test_acc = np.stack([x['test_acc'] for x in outputs]).mean()
            end_log = {'test_loss_mean': avg_loss, 'test_acc_mean': test_acc}
            print('\nAvg test loss: ',avg_loss,'\n','Avg test acc: ', test_acc)
            return {'test_loss_mean': avg_loss, 'test_acc_mean': test_acc, 'log': end_log}

    # train
    # to open tensorboard, run this command in terminal and open the browser:
    # tensorboard --logdir ./logs/

    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    # model = PointNetCls_light()
    trainer = pl.Trainer(logger=tb_logger, max_epochs=2, log_save_interval=20, fast_dev_run=False)#,gpus=-1)
    # trainer.fit(model) # train

    # testing /home/jack/logs/default/version_12
    model = PointNetCls_light.load_from_checkpoint(r'/home/jack/logs/default/version_12/checkpoints/epoch=38.ckpt')
    #trainer.test(test_dataloader=test_dataloader)
    trainer.test(model) # test