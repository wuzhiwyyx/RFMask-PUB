'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-24 22:18:15
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-25 00:59:36
 # @ Description: Postprocessing prediction results.
 '''

import numpy as np
import torch
import cv2
from tqdm import tqdm
from .evaluation import box_iou

def postprocess(model_name, prediction, thresh=0.2):
    post = {
        'RFMask': rfmask_postprocess,
        'RFMask_single': rfmask_postprocess,
        'RFPose2DMask': rfpose2dmask_postprocess
    }
    return post[model_name](prediction, thresh)


def rfmask_postprocess(prediction, thresh=0.2):
    results = []
    pbar = tqdm(total=len(prediction), desc='Postprocessing')
    for i, batch_pred in enumerate(prediction):
        batch_pred, loss = batch_pred
        for pred in batch_pred:
            if pred['masks'].shape[0] == 0:
                pred['masks'] = torch.zeros((1, 1, 624, 820))
            mask = pred['masks'].max(dim=0)[0].squeeze()
            mask = mask.cpu().float().numpy()
            mask[mask < thresh] = 0
            mask[mask > thresh] = 1
            results.extend([mask])
        pbar.update(1)
    pbar.close()
    return results

def rfpose2dmask_postprocess(prediction, thresh=0.2):
    results = []
    pbar = tqdm(total=len(prediction), desc='Postprocessing')
    for i, batch_pred in enumerate(prediction):
        batch_pred, loss = batch_pred
        for pred in batch_pred:
            mask = pred[len(pred) // 2 - 1]
            mask = mask.cpu().float().numpy()
            mask = cv2.resize(mask, (820, 624), None, None, interpolation=cv2.INTER_LINEAR)
            mask[mask < thresh] = 0
            mask[mask > thresh] = 1
            results.extend([mask])
        pbar.update(1)
    pbar.close()
    return results


def iou(mask1, mask2):
    return np.sum(np.logical_and(mask1, mask2)) / np.sum(np.logical_or(mask1, mask2))

def calc_avg_iou(model_name, dataset, preds):
    iou = {
        'RFMask': rfmask_iou,
        'RFMask_single': rfmask_iou,
        'RFPose2DMask': rfpose2dmask_iou
    }
    return iou[model_name](dataset, preds)

def rfmask_iou(dataset, preds):
    results = []
    pbar = tqdm(total=len(dataset), desc='Calculating IoU')
    for d, p in zip(dataset, preds):

        gt = d[4].numpy()
        gt = np.max(gt, axis=0)
        gt = cv2.resize(gt, (820, 624), None, None, interpolation=cv2.INTER_NEAREST)
        results.append(iou(p, gt))
        pbar.update(1)
    pbar.close()
    return results, np.nanmean(results)

def rfpose2dmask_iou(dataset, preds):
    results = []
    pbar = tqdm(total=len(dataset), desc='Calculating IoU')
    for d, p in zip(dataset, preds):

        gt = d[2].numpy()
        gt = gt[len(gt) // 2 - 1]
        gt = cv2.resize(gt, (820, 624), None, None, interpolation=cv2.INTER_NEAREST)
        p = cv2.resize(p, (820, 624), None, None, interpolation=cv2.INTER_NEAREST)
        results.append(iou(p, gt))
        pbar.update(1)
    pbar.close()
    return results, np.nanmean(results)