'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 15:40:50
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 17:26:32
 # @ Description:
 '''


from torch.utils import data
import lmdb
import numpy as np
import cv2
import torch
import HIBERTools as hiber

from torch.utils.data.dataloader import default_collate
from torchvision.models.detection.transform import GeneralizedRCNNTransform


class HIBERMaskDataset(data.Dataset):
    def __init__(self, root, transform=None, mode='train', 
                    categories=['WALK'], views=[6], channels=12, complex=True):
        """HIBER Dataset constructor.

        Args:
            root (str): LMDB file path.
            transform (callable, optional): Transform numpy data to torch.tensor. Defaults to None.
            mode (str, optional): Train/test/val split of dataset. Defaults to 'train'.
            categories (list, optional): Data categories, candidate categories are 
                                            ['WALK', 'ACTION', 'MULTI', 'OCCLUSION', 'DARK']. Defaults to ['WALK'].
            views (list, optional): Environments (views) to be used. Defaults to [6].
            channels (int, optional): Channel number (sequence length) of output RF frames sequences. Defaults to 12.
            complex (bool, optional): If true the shape of returned RF frame sequences is (channels, 2, 160, 200), 
                                            '2' means real and image channel of original complex RF frame, 
                                            else (channels, 160, 200). Defaults to False.
        """
        self.root = root
        self.mode = mode
        self.views = views
        self.channels = channels
        self.hiber_dataset = hiber.HIBERDataset(root, categories, mode) 
        self.keys = self.__get_data_keys__()
        self.transform = transform
        self.complex = complex
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
        keys = [[f'h_{v:02d}_{g:02d}_{x:04d}', f'v_{v:02d}_{g:02d}_{x:04d}', f'm_{v:02d}_{g:02d}_{x:04d}']
                 for x in range(start, end)]
        with self.env.begin(write=False) as txn:
            data_items = []
            for ks in keys:
                data_item = []
                for k in ks:
                    buf = txn.get(k.encode('ascii'))
                    if k.startswith('m'):
                        data = np.frombuffer(buf, dtype=bool)
                    elif k.startswith(('h_', 'v_')):
                        data = np.frombuffer(buf, dtype=np.complex128)
                    else:
                        data = np.frombuffer(buf, dtype=np.float64)
                    data_item.append(data)
                                
                data_item[0] = data_item[0].reshape(2, 160, 200)
                data_item[1] = data_item[1].reshape(2, 160, 200)

                if self.complex:
                    data_item[0] = data_item[0].view(dtype=np.float64)
                    data_item[1] = data_item[1].view(dtype=np.float64)
                    data_item[0] = data_item[0].reshape(2, 160, 200, 2)
                    data_item[1] = data_item[1].reshape(2, 160, 200, 2)
                    data_item[0] = data_item[0].transpose((0, 3, 1, 2))
                    data_item[1] = data_item[1].transpose((0, 3, 1, 2))
                else:
                    data_item[0] = np.abs(data_item[0])
                    data_item[1] = np.abs(data_item[1])
                data_item[2] = data_item[2].reshape(-1, 1248, 1640)
                # Deal with zero mask results situation.
                if data_item[2].shape[0] == 0:
                    data_item[2] = np.zeros((1, 1248, 1640))
                else:
                    data_item[2] = np.max(data_item[2], axis=0, keepdims=True)
                data_items.append(data_item)

        hors = np.concatenate([x[0] for x in data_items], axis=0)
        vers = np.concatenate([x[1] for x in data_items], axis=0)
        silhouettes = np.concatenate([x[2] for x in data_items], axis=0)
        
        if self.transform:
            _ = self.transform(hors, vers, silhouettes)
            hors, vers, silhouettes = _
        return hors, vers, silhouettes


class HIBERMaskTrans():
    def norm(self, arr):
        
        complex = False if len(arr.shape) == 3 else True
        if complex:
            arr = arr.reshape(-1, *arr.shape[2:])
        seq_len = arr.shape[0]
        mean_value = np.mean(arr.reshape(seq_len, -1), axis=-1)
        std_value = np.std(arr.reshape(seq_len, -1), axis=-1)
        arr = (arr - mean_value.reshape(seq_len, 1, 1))
        arr = arr / std_value.reshape(seq_len, 1, 1)
        if complex:
            arr = arr.reshape(arr.shape[0] // 2, 2, *arr.shape[1:])
        return arr
    
    def __call__(self, hors, vers, silhouettes):
        # h, w = 154, 218
        silhouettes = silhouettes.transpose((1, 2, 0))
        silhouettes = np.ascontiguousarray(silhouettes).astype(int)
        # silhouettes = cv2.resize(silhouettes, (218, 154), None, None, interpolation=cv2.INTER_NEAREST)
        # silhouettes[silhouettes < 0.5] = 0
        # silhouettes[silhouettes >= 0.5] = 1
        hors = torch.from_numpy(self.norm(hors))
        hors = hors.float().contiguous()
        vers = torch.from_numpy(self.norm(vers))
        vers = vers.float().contiguous()
        
        silhouettes = torch.from_numpy(silhouettes)
        hors = hors.permute(1, 0, 2, 3).float().contiguous()
        vers = vers.permute(1, 0, 2, 3).float().contiguous()
        silhouettes = silhouettes.permute(2, 0, 1).float().contiguous()
        return hors, vers, silhouettes

