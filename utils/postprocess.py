'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-24 22:18:15
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-25 00:59:36
 # @ Description: Postprocessing prediction results.
 '''

import numpy as np
import torch
from tqdm import tqdm
from .evaluation import box_iou

def postprocess(prediction):
    results = []
    pbar = tqdm(total=len(prediction), desc='Postprocessing')
    for i, pred in enumerate(prediction):
        category, trace, loss = pred
        category, trace = category.float().softmax(-1), trace.float()
        all_idx = category.argmax(-1).sum(-1)
        idx = torch.where(all_idx != 0)
        category = category[idx].max(dim=-1, keepdim=True)[0]
        trace = trace[idx]
        result = torch.cat([category, trace], dim=-1)
        results.append(result)
        pbar.update(1)
    pbar.close()
    return results


def calc_avg_iou(dataset, preds, frame_idx=0):
    results = []
    pbar = tqdm(total=len(dataset), desc='Calculating IoU')
    for d, p in zip(dataset, preds):
        # box shape: nobj, seq_len, 4
        # p shape: pnobj, seq_len, 1 + 4 (prob, x0, y0, x1, y1)
        hor, box, cls = d
        bb = box[:, frame_idx] # gt of first frame
        bb = bb[torch.where(bb.sum(axis=1) != 0)]
        pp = p[:, frame_idx, 1:]
        if pp.shape[0] == 0:
            pp = torch.ones((1, 4), device=pp.device)
        ious = box_iou(bb, pp)[0]
        ious = ious.max(dim=1)[0]
        results.append(ious.mean().unsqueeze(0))
        pbar.update(1)
    pbar.close()
    results = torch.cat(results)
    return results, np.nanmean(results)