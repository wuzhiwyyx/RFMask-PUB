'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 15:40:50
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 17:26:32
 # @ Description: HIBER dataset loader.
 '''


from copy import copy
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
    """Load HIBER Dataset from lmdb format."""

    def __init__(self, root, transform=None, mode='train', 
                    categories=['WALK'], views=[6], channels=12):
        self.root = root
        self.mode = mode
        self.views = views
        self.channels = channels
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

        filtered = [[int(y) for y in x[0][2:].split('_')] for x in keys]
        filtered = np.array(filtered)
        filtered = filtered[np.where(filtered[:, -1] >= lower_bound)]
        filtered = filtered[np.where(filtered[:, -1] < upper_bound)]
        filtered = filtered[np.where(np.isin(filtered[:, 0], self.views))]
        return filtered

    def __getitem__(self, index):
        v, g, f = self.keys[index]
        start = f - self.channels // 2 // 2
        end = f + self.channels // 2 // 2
        keys = [[f'h_{v:02d}_{g:02d}_{x:04d}', f'v_{v:02d}_{g:02d}_{x:04d}',
            f'hb_{v:02d}_{g:02d}_{x:04d}', f'hb_{v:02d}_{g:02d}_{x:04d}',
            f'm_{v:02d}_{g:02d}_{x:04d}']
                 for x in range(start, end)]
        with self.env.begin(write=False) as txn:
            data_items = []
            for ks in keys:
                data_item = []
                for k in ks:
                    buf = txn.get(k.encode('ascii'))
                    if k.startswith('m'):
                        data = np.frombuffer(buf, dtype=bool)
                    else:
                        data = np.frombuffer(buf, dtype=np.float64)
                    data_item.append(data)
                data_item[0] = data_item[0].reshape(160, 200, 2)
                data_item[1] = data_item[1].reshape(160, 200, 2)
                data_item[2] = data_item[2].reshape(-1, 4)
                data_item[3] = data_item[3].reshape(-1, 4)
                data_item[4] = data_item[4].reshape(-1, 1248, 1640)
                # Deal with zero mask results situation.
                data_item[4] = np.concatenate([data_item[4], 
                    np.zeros((data_item[2].shape[0] - data_item[4].shape[0], 1248, 1640))], axis=0)
                assert data_item[4].shape[0]
                data_items.append(data_item)

        hors = np.concatenate([x[0] for x in data_items], axis=-1)
        vers = np.concatenate([x[1] for x in data_items], axis=-1)
        hboxes = data_items[f - start][2]
        vboxes = data_items[f - start][3]
        silhouettes = data_items[f - start][4]
        
        if self.transform:
            _ = self.transform(hors, vers, hboxes, vboxes, silhouettes, np.array(v), category=True)
            hors, vers, hboxes, vboxes, silhouettes, categories, v = _
        return hors, vers, hboxes, vboxes, silhouettes, categories, v
    
class HiberTrans():
    def norm(self, arr):
        seq_len = arr.shape[-1]
        mean_value = np.mean(arr.reshape(-1, seq_len), axis=0)
        std_value = np.std(arr.reshape(-1, seq_len), axis=0)
        arr = (arr - mean_value.reshape(1, 1, seq_len))
        arr = arr / std_value.reshape(1, 1, seq_len)
        return arr
    
    def __call__(self, hors, vers, hboxes, vboxes, silhouettes, v, category=False):
        hors = torch.from_numpy(self.norm(hors))
        hors = hors.permute(2, 0, 1).float().contiguous()
        vers = torch.from_numpy(self.norm(vers))
        vers = vers.permute(2, 0, 1).float().contiguous()
        
        # box shape: nperson, 4, seq_len
        hboxes = torch.from_numpy(hboxes.copy())
        vboxes = torch.from_numpy(vboxes.copy())

        hboxes = hboxes.float().contiguous()
        _ = (hboxes.sum(-1) != 0).long()
        categories = None if not category else _

        silhouettes = torch.from_numpy(silhouettes.copy()).float()
        v = torch.from_numpy(v).long()
        return hors, vers, hboxes, vboxes, silhouettes, categories, v

def hiber_collate(batch):
    # batch : list of [hors, vers, box, silhouettes, categories, v]
    if len(batch) == 0:
        return None
    else:
        result = {}
        hors = default_collate([b[0] for b in batch])
        vers = default_collate([b[1] for b in batch])
        params = default_collate([b[-1] for b in batch])
        targets = []
        for b in batch:
            target = {}
            target['hboxes'] = b[2].float()
            target['vboxes'] = b[3].float()
            target['masks'] = b[4].float()
            assert target['masks'].shape[0]
            target['labels'] = b[5]
            targets.append(target)
        return hors, vers, params, targets