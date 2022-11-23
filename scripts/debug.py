import sys

sys.path.append('.')
import random

import matplotlib.pyplot as plt
# import project
import torch
from matplotlib.patches import Rectangle
from tqdm import tqdm

# from rfmask import RFMask
from utils import ConfigDict, build_model, load_config, load_dataset

# torch.cuda.set_device(0)

# torch.backends.cudnn.benchmark = False

config = load_config('configs/rfmask.yaml')
# trainset, train_loader = load_dataset(config.train.trainset)

# hiber = {
#     'name':'hiber', 
#     'dataset':{
#         'root': 'data/hiber_train.lmdb',
#         'mode': 'train',
#         'categories': ['WALK', 'MULTI', 'ACTION'],
#         'complex': True
#     }, 
#     'loader': {'batch_size': 4, 'shuffle': False, 'num_workers': 0}
# }
# hiber = ConfigDict(hiber)
hiberset, hiber_loader = load_dataset(config.train.valset)
# # _h = hiberset[0]

# # _r = trainset[0]


# data_iter = iter(train_loader)
# data_item = next(data_iter)

h_iter = iter(hiber_loader)
h_item = next(h_iter)

model = build_model(config.train.model).cuda()
# h_item = [x.cuda() for x in h_item]

# model = model(*h_item[:])
# model

for i, h_item in enumerate(tqdm(hiber_loader)):
    data = []
    data.append(h_item[0].cuda())
    data.append(h_item[1].cuda())
    data.append(h_item[2].cuda())
    data.append([])
    for x in h_item[3]:
        for k in x.keys():
            x[k] = x[k].cuda()
        data[3].append(x)
    assert not 0 in [x['masks'].shape[0] for x in data[3]]
    # masks = [x['masks'].cpu() for x in data[3]]
    # hboxes = [x['hboxes'].cpu() for x in data[3]]
    # vboxes = [x['vboxes'].cpu() for x in data[3]]
    # v_props, proposals = project.calc_v_props(hboxes, [torch.tensor([6])], project=True, pad=25)
    # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # for i, (mask, prop) in enumerate(zip(masks, proposals)):
    #     for j, (m, p) in enumerate(zip(mask, prop)):
    #         x, y, x_, y_ = p
    #         rect = Rectangle((x, y), x_ - x, y_ - y, linewidth=2, edgecolor='w', facecolor='none')
    #         axes[j].imshow(m)
    #         axes[j].add_patch(rect)
    # plt.savefig('test1.jpg')
    # torch.cuda.empty_cache()
    # print([x.shape for x in data[:3]])
    # print([x.shape for y in data[3] for x in y.values()])
    # data
    # plt.imshow(data[0][0][5].cpu().detach().numpy())
    # plt.savefig('test1.jpg')
    _ = model(*data[:-1])
    # _
    
# print(f'output {len(_)}')
# print(config)


# dataset = RFDataset('/share2/home/wuzhi/USTCData/', transform=RFTrans())
# print(len(dataset))

# dataset[0]

# config = load_config('configs/config.yaml')
# trainset, train_loader = load_dataset(config.train.trainset)

# data_iter = iter(train_loader)
# data_item = next(data_iter)
# data_item

# from pathlib import Path
# import matplotlib.pyplot as plt
# import numpy as np

# prefix = Path('/share/data/HIBER/')

# file_path = Path('HIBER_TRAIN/WALK/HORIZONTAL_HEATMAPS/02_01/0000.npy')
# o_path = prefix / file_path
# n_path = prefix / 'COMPLEX' / file_path.with_stem('0006')

# old = np.load(o_path)
# new = np.load(n_path)

# new = new.view(dtype=np.complex128).squeeze(-1)

# plt.imshow(old[:, :, 0])
# plt.savefig('test1.jpg')
# plt.close()
# plt.imshow(np.abs(new[0]))
# plt.savefig('test2.jpg')
# plt.close()

# o = old[:, :, 0]
# n = np.abs(new[1])
# o
