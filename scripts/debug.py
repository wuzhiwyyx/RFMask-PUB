import sys
sys.path.append('.')
import random

from rfmask import RFMask
from rfmask.datasets import load_dataset
from utils import load_config, build_model

config = load_config('configs/config.yaml')
trainset, train_loader = load_dataset(config.train.trainset)

data_iter = iter(train_loader)
data_item = next(data_iter)

model = build_model(config.train.model)

_ = model(*data_item[:-1])
print(_)
print(config)


# dataset = RFDataset('/share2/home/wuzhi/USTCData/', transform=RFTrans())
# print(len(dataset))

# dataset[0]

# config = load_config('configs/config.yaml')
# trainset, train_loader = load_dataset(config.train.trainset)

# data_iter = iter(train_loader)
# data_item = next(data_iter)
# data_item