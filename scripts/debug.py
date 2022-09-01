import sys
sys.path.append('.')
import random

from rfmask import RFMask
from utils import load_config, build_model, load_dataset, ConfigDict

import torch
from tqdm import tqdm
torch.cuda.set_device(0)

torch.backends.cudnn.benchmark = False

config = load_config('configs/config.yaml')
trainset, train_loader = load_dataset(config.train.trainset)

hiber = {
    'name':'hiber', 
    'dataset':{
        'root': 'data/hiber_train.lmdb',
        'mode': 'train',
        'categories': ['WALK', 'MULTI', 'ACTION'],
    }, 
    'loader': {'batch_size': 4, 'shuffle': False, 'num_workers': 0}
}
hiber = ConfigDict(hiber)
hiberset, hiber_loader = load_dataset(hiber)
# _h = hiberset[0]

# _r = trainset[0]


data_iter = iter(train_loader)
data_item = next(data_iter)

h_iter = iter(hiber_loader)
h_item = next(h_iter)

model = build_model(config.train.model).cuda()

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
    
    # torch.cuda.empty_cache()
    _ = model(*data[:])
    
print(f'output {len(_)}')
print(config)


# dataset = RFDataset('/share2/home/wuzhi/USTCData/', transform=RFTrans())
# print(len(dataset))

# dataset[0]

# config = load_config('configs/config.yaml')
# trainset, train_loader = load_dataset(config.train.trainset)

# data_iter = iter(train_loader)
# data_item = next(data_iter)
# data_item