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
from torchvision.transforms import InterpolationMode, Resize

from ..rfpose2dmask import RFPose2DMask as RFPose2DMask_
from .utils import close_and_rename, close_writer, init_writer, iou


class RFPose2DMask(pl.LightningModule):
    """Pytorch-lightning interface of RFPose2DMask, 
       containing training/val/test procedure.

    """

    def __init__(self, learning_rate=1e-3, batch_size=4, threshold=0.2,
                    exper=None, optim='adamw', sched='cos',
                    save_pred=False, vis=False) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.model = RFPose2DMask_()
        self.batch_size = batch_size
        self.thresh = threshold
        self.exper = self._get_name() if exper is None else exper
        self.optim, self.sched = optim, sched
        self.set_config(save_pred=save_pred, vis=vis)

    def set_config(self, config=None, save_pred=False, vis=False):
        self.config = config
        self.save_pred = save_pred
        if vis:
            self.bundle = init_writer() if not hasattr(self, 'bundle') else self.bundle
        self.vis = vis

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
        self.log('train/loss', loss, batch_size=self.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        result, loss = self.model(*batch)
        self.log('val/mask_loss', loss, batch_size=self.batch_size, sync_dist=True)
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
            'adamw': torch.optim.AdamW(self.parameters(), lr=self.learning_rate),
        }
        optimizer = optimizers[self.optim]
        schedulers = {
            'steplr': optim.lr_scheduler.StepLR(optimizer, step_size = 60, gamma = 0.5),
            'cos': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
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
        pred_mask = res[len(res) // 2 - 1:len(res) // 2]
        pred_mask = Resize((624, 820), interpolation=InterpolationMode.BILINEAR)(pred_mask)
        pred_mask = pred_mask.squeeze(0)
        gt_mask = target[len(target) // 2 - 1:len(target) // 2]
        gt_mask = Resize((624, 820), interpolation=InterpolationMode.NEAREST)(gt_mask)
        gt_mask = gt_mask.squeeze(0)
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
