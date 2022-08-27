'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 15:40:50
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 17:26:32
 # @ Description: HIBER dataset loader.
 '''


from matplotlib.pyplot import box
from torch.utils import data
import os
import numpy as np
import pickle
import torch
from torch.utils.data.dataloader import default_collate
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import itertools
import lmdb
import HIBERTools as hiber

class HIBERDataset(data.Dataset):
    def __init__(self, root, transform=None, mode='train', categories=['WALK'], views=[6]):
        self.root = root
        self.mode = mode
        self.views = views
        self.channels = 10
        self.hiber_dataset = hiber.HIBERDataset(root, categories, mode) 
        self.keys = self.__get_data_keys__()
        self.transform = transform
        self.env = lmdb.open(root, readonly=True, lock=False, readahead=False, meminit=False)
    
    def __len__(self):
        return len(self.keys)
    
    def __get_data_keys__(self):
        keys = self.hiber_dataset.get_lmdb_keys()
        # deal with sequence borders
        lower_bound = self.channels // 2 // 2
        upper_bound = 589 - self.channels // 2 // 2 + 1

        filtered = []
        for ks in keys:
            filtered_ks = [x for x in ks if x.startswith('h') 
                                and int(x.split('_')[-1]) >= lower_bound 
                                and int(x.split('_')[-1]) < upper_bound]
            if filtered_ks:
                filtered.append(filtered_ks)
        return np.array(filtered, dtype=object)


    def iou(self, boxes0, boxes1, eps=1e-7):
        # boxes0 shape: A, 4
        # boxes1 shape: B, 4
        # return shape: A, B
        A = boxes0.shape[0]
        B = boxes1.shape[0]

        xy_max = np.minimum(boxes0[:, np.newaxis, 2:].repeat(B, axis=1),
                            np.broadcast_to(boxes1[:, 2:], (A, B, 2)))
        xy_min = np.maximum(boxes0[:, np.newaxis, :2].repeat(B, axis=1),
                            np.broadcast_to(boxes1[:, :2], (A, B, 2)))
        # intersection
        inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
        inter = inter[:, :, 0]*inter[:, :, 1]
        # area of each box
        area_0 = ((boxes0[:, 2]-boxes0[:, 0])*(
            boxes0[:, 3] - boxes0[:, 1]))[:, np.newaxis].repeat(B, axis=1)
        area_1 = ((boxes1[:, 2] - boxes1[:, 0])*(
            boxes1[:, 3] - boxes1[:, 1]))[np.newaxis, :].repeat(A, axis=0)
        return inter/(area_0+area_1-inter+eps)
    
    def match_trace(self, boxes_):
        # boxes shape: nobj, 4, seq_len / 2
        boxes = boxes_.copy()
        nobj = boxes.shape[0]
        ntimes = boxes.shape[-1]
        if ntimes < 2:
            return boxes
        boxes = np.transpose(boxes, (2, 0, 1))
        boxes = np.ascontiguousarray(boxes)
        pre = boxes[:-1, :, :].reshape(-1, 4)
        post = boxes[1:, :, :].reshape(-1, 4)
        ious = self.iou(pre, post)
        diag = [np.split(row_blk, post.shape[0] // nobj, axis=1)[i]
                for i, row_blk in enumerate(np.split(ious, pre.shape[0] // nobj))]
        diag = np.array(diag)
        max_idxs = np.argmax(diag, axis=-1)
        order = np.arange(nobj)
        for tt in range(1, ntimes):
            max_idxs[tt-1] = max_idxs[tt-1][order]
            boxes[tt, :, :,] = boxes[tt, max_idxs[tt-1], :]
            order = max_idxs[tt-1]
        boxes = np.transpose(boxes, (1, 2, 0))
        boxes = np.ascontiguousarray(boxes)
        return boxes

    def xyxy2xywh(self, box, norm=False):
        # box shape : (nperson, 4, seq_len)
        box[:, [2, 3], :] = box[:, [2, 3], :] - box[:, [0, 1], :]
        box[:, [0, 1], :] = box[:, [2, 3], :] * 0.5 + box[:, [0, 1], :]
        if norm:
            box = box / np.array([200, 160, 200, 160]).reshape(1, 4, 1)
        return box
    
    def __getitem__(self, index):
        key = self.keys[index]
        start = int(key[0].split('_')[-1]) - self.channels // 2 // 2
        end = int(key[0].split('_')[-1]) + self.channels // 2 // 2 + 1
        keys = [[key[0].replace(key[0].split('_')[-1], f'{x:04d}'), key[1].replace(key[1].split('_')[-1], f'{x:04d}')] 
                    for x in range(start, end)]
        with self.env.begin(write=False) as txn:
            data_items = []
            for ks in keys:
                data_item = []
                for k in ks:
                    buf = txn.get(k.encode('ascii'))
                    data = np.frombuffer(buf, dtype=np.float64)
                    data_item.append(data)
                data_item[0] = data_item[0].reshape(160, 200, 2)
                data_item[1] = data_item[1].reshape(-1, 4)
                data_items.append(data_item)
            data_items
        rfs = [x[0] for x in data_items]
        boxes = [x[1] for x in data_items]
        rfs = np.concatenate(rfs, axis=-1)
        boxes = np.stack(boxes, axis=-1)
        boxes = self.match_trace(boxes)
        boxes = self.xyxy2xywh(boxes, norm=True)
        if self.transform:
            rfs, boxes, category = self.transform(rfs, boxes, category=True)
        return rfs, boxes, category
    
class HiberTrans():
    def norm(self, arr):
        seq_len = arr.shape[-1]
        mean_value = np.mean(arr.reshape(-1, seq_len), axis=0)
        std_value = np.std(arr.reshape(-1, seq_len), axis=0)
        arr = (arr - mean_value.reshape(1, 1, seq_len))
        arr = arr / std_value.reshape(1, 1, seq_len)
        return arr
    
    def __call__(self, rf, box, category=False):
        rf = torch.from_numpy(self.norm(rf))
        rf = rf.permute(2, 0, 1).float().contiguous()
        
        # box shape: nperson, 4, seq_len
        box = torch.from_numpy(box)

        box = box.permute(0, 2, 1).float().contiguous()
        _ = (box.flatten(0, 1).sum(-1) != 0).long().view(*box.shape[:2], 1)
        category = None if not category else _
        return rf, box, category

def hiber_collate(batch):
    if len(batch) == 0:
        return None
    else:
        result = {}
        rfs = default_collate([b[0] for b in batch])
        boxes = [b[1] for b in batch]
        categories = [b[2] for b in batch]
        targets = [{'labels': c.long(), 'boxes': b} for b, c in zip(boxes, categories)]
        return rfs, targets