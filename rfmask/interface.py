'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-24 21:57:44
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-25 00:47:12
 # @ Description: Pytorch-lightning model definition, controlling training strategy and train/val/test dataset.
 '''

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import pytorch_lightning as pl

from .models import RFMask as RFMask_
from .models import RFPose2DMask as RFPose2DMask_
from .models import iou

class RFMask(pl.LightningModule):
    """Pytorch-lightning interface of RFMask, 
       containing training/val/test procedure.

    """

    def __init__(self, learning_rate=1e-3, batch_size=12, dual=True, 
                    fuse='multi-head', threshold=0.2):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = RFMask_(dual=dual, fuse=fuse)
        self.batch_size = batch_size
        self.thresh = threshold
        self.num_workers = 0
        if dual:
            self.loss_keys = ['loss_classifier_h', 'loss_box_reg_h', 'loss_classifier_v',
                        'loss_box_reg_v', 'loss_mask', 'loss_h_obj', 'loss_h_rpn_box',
                        'loss_v_obj', 'loss_v_rpn_box']
        else:
            self.loss_keys = ['loss_classifier_h', 'loss_box_reg_h', 
                        'loss_mask', 'loss_h_obj', 'loss_h_rpn_box']
        
    def forward(self, hor, ver, params, targets=None):
        result, losses = self.model(hor, ver, params, targets)
        return result, losses

    def mask_iou(self, preds, targets):
        results = []
        for p in preds:
            p = p['masks'].detach()
            if p.shape[0] == 0:
                p = torch.zeros((1, 1, 1248 // 2, 1640 // 2), device=p.device)
            p = p.max(dim=0)[0].squeeze()
            p[p < self.thresh] = 0
            results.append(p)
        gts = [x['masks'].max(dim=0)[0] for x in targets]
        results = torch.stack(results)
        gts = torch.stack(gts)
        biou = iou(results, gts)
        return biou
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        hor, ver, params, targets = batch
        result, losses = self.model(hor, ver, params)
        for r in result:
            for k in r.keys():
                r[k] = r[k].cpu()
        
        for k, v in losses.items():
            losses[k] = v.cpu()
        return result, losses

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        result, losses = self.model(*batch)
        loss = 0
        for k in self.loss_keys:
            loss = loss + losses[k]
        # loss = sum(losses.values())
        # loss = losses['loss_mask'] + sum([float(losses[x]) for x in set(self.loss_keys) - set(['loss_mask'])])
        
        # batch_miou = self.mask_iou(result, batch[-1]).mean().item()

        # Logging to TensorBoard by default
        self.log('train/loss', loss.cpu().detach(), batch_size=self.batch_size, sync_dist=True)
        self.log('train/mask_loss', losses['loss_mask'].cpu().detach(), batch_size=self.batch_size, sync_dist=True)
        # self.log('train/iou', batch_miou, batch_size=self.batch_size, sync_dist=True)
        for k in self.loss_keys:
            self.log('train_other/' + k, losses[k].cpu().detach(), batch_size=self.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        result, losses = self.model(*batch)
        loss = 0
        for k in self.loss_keys:
            loss = loss + losses[k]
        # loss = sum(losses.values())
        # loss = losses['loss_mask'] + sum([float(losses[x]) for x in set(self.loss_keys) - set(['loss_mask'])])

        # batch_miou = self.mask_iou(result, batch[-1]).mean().item()
        
        self.log('val/loss', loss.cpu().detach(), batch_size=self.batch_size, sync_dist=True)
        self.log('val/mask_loss', losses['loss_mask'].cpu().detach(), batch_size=self.batch_size, sync_dist=True)
        # self.log('val/iou', batch_miou, batch_size=self.batch_size, sync_dist=True)
        
        for k in self.loss_keys:
            self.log('val_other/' + k, losses[k].cpu().detach(), batch_size=self.batch_size, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 60, gamma = 0.5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40)
        lr_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
        return lr_dict

class RFPose2DMask(pl.LightningModule):
    """Pytorch-lightning interface of RFPose2DMask, 
       containing training/val/test procedure.

    """

    def __init__(self, learning_rate=1e-3, batch_size=4) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.model = RFPose2DMask_()
        self.batch_size = batch_size

    def forward(self, hor, ver, mask=None):
        result, losses = self.model(hor, ver, mask)
        return result, losses
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        result, loss = self.model(*batch[:-1])
        result = result.cpu()
        return result, loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        result, loss = self.model(*batch)
        
        # Logging to TensorBoard by default
        self.log('loss', loss, batch_size=self.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        result, loss = self.model(*batch)
        self.log('val_loss', loss, batch_size=self.batch_size, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 60, gamma = 0.5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
        lr_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
        return lr_dict