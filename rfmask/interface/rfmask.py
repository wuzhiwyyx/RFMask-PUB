'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-24 21:57:44
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-25 00:47:12
 # @ Description: Pytorch-lightning model definition, controlling training strategy and train/val/test dataset.
 '''

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch import optim

from ..models import RFMask as RFMask_
from .utils import close_and_rename, close_writer, init_writer, iou


class RFMask(pl.LightningModule):
    """Pytorch-lightning interface of RFMask, 
       containing training/val/test procedure.

    """

    def __init__(self, learning_rate=1e-3, batch_size=12, dual=True, 
                    fuse='multi-head', exper=None, threshold=0.2, 
                    optim='adam', sched='cos',
                    save_pred=False, vis=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = RFMask_(dual=dual, fuse=fuse)
        self.batch_size = batch_size
        self.thresh = threshold
        self.num_workers = 0
        self.exper = self._get_name() if exper is None else exper
        self.optim, self.sched = optim, sched
        self.set_config(save_pred=save_pred, vis=vis)
        if dual:
            self.loss_keys = ['loss_classifier_h', 'loss_box_reg_h', 'loss_classifier_v',
                        'loss_box_reg_v', 'loss_mask', 'loss_h_obj', 'loss_h_rpn_box',
                        'loss_v_obj', 'loss_v_rpn_box']
        else:
            self.loss_keys = ['loss_classifier_h', 'loss_box_reg_h', 
                        'loss_mask', 'loss_h_obj', 'loss_h_rpn_box']

    def set_config(self, config=None, save_pred=False, vis=False):
        self.config = config
        self.save_pred = save_pred
        if vis:
            self.bundle = init_writer() if not hasattr(self, 'bundle') else self.bundle
        self.vis = vis
        
    def forward(self, hor, ver, params, targets=None):
        result, losses = self.model(hor, ver, params, targets)
        return result, losses
    
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

    def test_step(self, batch, batch_idx):
        result, losses = self.model(*batch[:-1])

        canvases, mious = batch_iou(result, batch[-1], threshold=self.thresh)

        if hasattr(self, 'bundle'):
            batch_vis(canvases, mious, self.bundle)

        return result, losses, mious

    def test_epoch_end(self, outputs):
        miou = torch.tensor([x[2] for x in outputs]).nanmean()
        self.metric_value = (miou.item() + self.metric_value) / 2 if hasattr(self, 'metric_value') \
                                else miou
        if self.save_pred:
            prediction = [x[:2] for x in outputs]
            self.prediction = prediction if not hasattr(self, 'prediction') else \
                                self.prediction + prediction

    def on_test_end(self):
        miou = self.metric_value
        logging.getLogger('pytorch_lightning').info(f'Mask IoU : {miou.item():.3f}')
        if hasattr(self, 'bundle'):
            close_and_rename(self.bundle, self.exper, miou)
        if self.save_pred and hasattr(self, 'prediction'):
            save_prediction(self.prediction)

    def configure_optimizers(self):
        optimizers = {
            'sgd': torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9),
            'adam': torch.optim.Adam(self.parameters(), lr=self.learning_rate),
            'nadam': torch.optim.NAdam(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        }
        optimizer = optimizers[self.optim]
        schedulers = {
            'steplr': optim.lr_scheduler.StepLR(optimizer, step_size = 60, gamma = 0.5),
            'cos': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40)
        }
        scheduler = schedulers[self.sched]
        
        lr_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
        return lr_dict


def batch_iou(batch_res, batch_targets, threshold):
    preds = []
    gts = []
    for res, target in zip(batch_res, batch_targets):
        pred_mask = res['masks']
        if pred_mask.shape[0] == 0:
            pred_mask = torch.zeros((1, 1, 624, 820), device=pred_mask.device)
        pred_mask = pred_mask.max(dim=0)[0].squeeze()

        gt_mask, _idx_ = torch.max(target['masks'], axis=0)
        preds.append(pred_mask)
        gts.append(gt_mask)

    preds = torch.stack(preds, dim=0) > threshold
    gts = torch.stack(gts, dim=0).bool()
    canvases = torch.cat([gts, preds], dim=-1).cpu().detach()
    mious = iou(preds, gts).cpu().detach()
    return canvases, mious


def all_iou(predictions, data_loader, threshold=0.2):
    data_iter = iter(data_loader)
    canvases, ious = [], []
    for pred, target in zip(predictions, data_iter):
        b_canvas, b_miou = batch_iou(pred, target, threshold)
        canvases.append(b_canvas)
        ious.append(b_miou)
    return canvases, ious


def batch_vis(batch_canvases, batch_mious, bundle):
    ax = bundle['figure'].axes[0]
    for canvas, miou in zip(batch_canvases, batch_mious):
        info = f'Mask IoU: {miou.item():.3f}'
        im = ax.matshow(canvas)
        txt = plt.text(820, 624 // 12, info, fontsize=20, 
                    bbox={'alpha':0},
                    horizontalalignment='center', color='white')
        # plt.draw()
        bundle['writer'].grab_frame()
        im.remove()
        txt.remove()

def save_prediction(predictions):
    if predictions is None:
        return
    with Path('prediction.pkl').open('wb') as f:
        pickle.dump(predictions, f)
    logging.getLogger('pytorch_lightning').info('Prediction saved in prediction.pkl')

def all_vis(canvases, mious, exper):
    bundle = init_writer()
    for canvas, miou in zip(canvases, mious):
        batch_vis(canvas, miou, bundle)
    close_and_rename(bundle, exper, miou)

def postprocess(predictions, data_loader, threshold=0.2, exper=None):
    canvases, ious = all_iou(predictions, data_loader, threshold=threshold)
    if exper:
        all_vis(canvases, ious, exper)
    miou = torch.tensor(ious).nanmean()
    return miou
