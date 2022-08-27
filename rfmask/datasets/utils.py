'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 15:52:34
 # @ Description: Dataset loader.
 '''

from torch.utils.data import DataLoader

from .hiber_dataset import HIBERDataset, HiberTrans, hiber_collate
from .rf_dataset import RFDataset, RFTrans, rf_collate

def load_rf_dataset(config):
    """Load rf dataset from file

    Args:
        config (dict): Dict object containing dataset initial parameters and dataloader initial parameters.

    Returns:
        (Dataset, DataLoader): _description_
    """
    dataset = RFDataset(**config.dataset, transform=RFTrans())
    loader = DataLoader(dataset, **config.loader, drop_last=True, collate_fn=rf_collate)
    return dataset, loader

def load_hiber_dataset(config):
    dataset = HIBERDataset(**config.dataset, transform=HiberTrans())
    loader = DataLoader(dataset, **config.loader, drop_last=True, collate_fn=hiber_collate)
    return dataset, loader