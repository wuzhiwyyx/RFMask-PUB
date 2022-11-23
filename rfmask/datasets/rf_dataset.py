'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 15:40:50
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 17:26:32
 # @ Description:
 '''


import os
import pickle

import numpy as np
import torch
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from torchvision.models.detection.transform import GeneralizedRCNNTransform


class RFDataset(data.Dataset):
    """ HIBER dataset loader."""

    def __init__(self, root, transform=None, mode='train', views=[6], 
                 start=None, end=None, groups=[None, 13]):      
        """RFDataset object constructor

        Args:
            root (str): Path of the HIBER dataset
            transform (callable, optional): A callable object which convert numpy array to tensors. Defaults to None.
            mode (str, optional): Return train data if equal to 'train', else return validate data. Defaults to 'train'.
            views (list, optional): Included views of HIBER dataset, candidates are [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]. Defaults to [6,7].
            start (_type_, optional): Index of start frame in each group. Defaults to 0 when mode == 'train' or 500 when not.
            end (_type_, optional): Index of end frame in each group. Defaults to 400 when mode == 'train' or 580 when not.
            groups (list, optional): Included groups in each view. Defaults are automatic calculation.
        """
        self.root = root
        self.mode = mode
        self.views = views

        # Datast information dict, storing the data type of each group within each view.
        self.category = {
            'walk':{'1':[1,36], '2':[1,21], '3':[1,13], '4':[1,13], '5':[1,13],
                    '6':[1,13], '7':[1,13], '8':[1,13], '9':[1,13], '10':[4,16]},
            'multi':{'1':[77,89], '2':[65, 77], '3':[13,22], '4':[13,25], '5':[13,25],
                    '6':[13,25], '7':[13,25], '8':[13,25], '9':[13,25], '10':[16,28]},
            'styrofoam':{'1':[65,68], '2':[37, 45], '3':[22,30], '4':[25,33], '5':[25,33],
                    '6':[25,33], '7':[25,33], '8':[25,33], '9':[25,33], '10':[28,36]},
            'carton':{'1':[62,65], '2':[21, 29], '3':[30,38], '4':[33,41], '5':[33,41],
                    '6':[33,41], '7':[33,41], '8':[33,41], '9':[33,41], '10':[36,44]},
            'yoga':{'1':[0,0], '2':[29, 37], '3':[39,47], '4':[41,49], '5':[41,49],
                    '6':[41,49], '7':[41,49], '8':[41,49], '9':[41,49], '10':[44,52]},
            'dark':{'1':[0,0], '2':[0, 0], '3':[0,0], '4':[0,0], '5':[0,0],
                    '6':[0,0], '7':[0,0], '8':[61,71], '9':[61,72], '10':[64,74]},
            'action':{'1':[37,62], '2':[45,65], '3':[47,62], '4':[49,61], '5':[49,61],
                        '6':[49,63], '7':[49,61], '8':[49,61], '9':[49,61], '10':[52,64]}
        }
        if mode == 'train':
            self.start = 0
            self.end = 400
        else:
            self.start = 500
            self.end = 580
        self.start = start if not start is None else self.start
        self.end = end if not end is None else self.end
        
        self.offset = 6 # For alignment between video frame and radar frame.
        self.channels = 12 # Channel number of each sample.
        self.data = self._get_data()
        self.transform = transform
    
    def _get_data(self):
        """Fetch data 

        Returns:
            list: List of (radar_frame, annotation).
        """
        radar = self._get_radar()
        addition = self._get_addition()
        return list(zip(radar, addition))
        
    def _get_radar(self):
        radars = []
        s, e = 0 - self.channels // 4, self.channels // 2 - self.channels // 4
        for v in self.views:
            prefix = os.path.join(self.root, 'view%d' % v)
            rng_help = lambda x:list(range(x[0], x[1]))
            rng = lambda c, v:rng_help(self.category[c][str(v)])
            rngs = rng('walk', v) + rng('multi', v) + rng('action', v)
#             gs, ge = self.category['walk'][str(v)]
            for g in rngs:
                h = os.path.join(prefix, 'hor', '%04d' % g, 'r15_%04d_%04d.npy')
                v = os.path.join(prefix, 'ver', '%04d' % g, 'r139_%04d_%04d.npy')
                for frame in range(self.start, self.end):
                    hh = [h % (g, frame + x + self.offset) for x in range(s, e)]
                    vv = [v % (g, frame + x + self.offset) for x in range(s, e)]
                    radars.append((hh, vv))
        return radars
    
    
    def _get_addition(self):
        additions = []
        for v in self.views:
            prefix = os.path.join(self.root, 'view_all', '%02d' % v)
            rng_help = lambda x:list(range(x[0], x[1]))
            rng = lambda c, v:rng_help(self.category[c][str(v)])
            rngs = rng('walk', v)
            rngs = rng('walk', v) + rng('multi', v) + rng('action', v)
#             rngs = rng('walk', v) + rng('action', v)
#             gs, ge = self.category['walk'][str(v)]
            for g in rngs:
                addition = os.path.join(prefix, 'view%02d_%04d_' % (v, g) + '%04d.pkl')
                for frame in range(self.start, self.end):
                    ad = addition % frame
                    additions.append(ad)
        return additions
        
    
    def __len__(self):
        return len(self.data)
    
    
    def _load_pkl(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    
    def __getitem__(self, index):
        radar, addition = self.data[index]        
        hor = np.concatenate([np.load(x) for x in radar[0]], axis=2)
        ver = np.concatenate([np.load(x) for x in radar[1]], axis=2)
        hor = np.rot90(hor).astype(np.float32)
        ver = np.rot90(ver).astype(np.float32)
        addition = self._load_pkl(addition)
        addition['hbox'] = addition['hbox'].reshape(-1, 4)
        addition['vbox'] = addition['vbox'].reshape(-1, 4)
        addition['mask_box'] = addition['mask_box'].reshape(-1, 4)
        addition['mask'] = addition['mask'].reshape(-1, 624, 820)
        if self.transform:
            hor, ver, addition = self.transform(hor, ver, addition)
        
        param = {}
        for k in ['offset', 'M', 'P', 'D', 'K', 'r_mat']:
            param[k] = addition[k]
        target = {}
        for k in ['hbox', 'vbox', 'mask', 'mask_box']:
            target[k] = addition[k]
        return hor, ver, param, target


class RFTrans(object):
    def _normalize(self, arr):
        arr_min, arr_max = np.min(arr, axis=0), np.max(arr, axis=0)
        arr = (arr - arr_min) / arr_max
        return arr
    
    
    def __call__(self, hor, ver, addition):
        hor, ver = hor.transpose(2, 0, 1), ver.transpose(2, 0, 1)
#         hor, ver = self._normalize(hor), self._normalize(ver)
        hor = torch.from_numpy(hor)
        ver = torch.from_numpy(ver)
        for k, v in addition.items():
            addition[k] = torch.from_numpy(v)
        return hor, ver, addition
    
    
def rf_collate(batch):
    if len(batch) == 0:
        return None
    else:
        result = {}
        hors = []
        vers = []
        params = []
        targets = []
        for b in batch:
            hor, ver, param, tar = b
            hors.append(hor)
            vers.append(ver)
            for k, v in param.items():
                param[k] = param[k].float()
            params.append(param)
            target = {}
            target['hboxes'] = tar['hbox'].float()
            target['vboxes'] = tar['vbox'].float()
            target['mboxes'] = tar['mask_box'].float()
            #target['labels'] = torch.ones(1, dtype=torch.int64)
            
            target['labels'] = torch.ones(tar['hbox'].size(0), dtype=torch.int64)
            #target['masks'] = tar['mask']
            ori_mask = tar['mask']
            mask = torch.cat([ori_mask] * len(target['hboxes']), dim=0)
            target['masks'] = mask
            targets.append(target)
        hors = default_collate(hors)
        vers = default_collate(vers)
        return hors, vers, params, targets
