'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 18:11:52
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 18:12:54
 # @ Description: Collection of some useful functions for running the whole project.
 '''

from models.interface import RFMask, RFPose2DMask
from models.interface.rfmask import postprocess as rfmask_postproc
from models.interface.rfpose2dmask import postprocess as rfpose2dmask_postproc
from models.datasets import (load_hiber_dataset, load_hiber_mask_dataset,
                             load_rf_dataset)


def build_model(name, **kwargs):
    """Build model object

    Args:
        cfg (dict): Model name and initialization parameters.

    Returns:
        LightningModule: Pytorch-lightning modules.
    """
    models = {
        'RFMask' : RFMask,
        'RFMask_single': RFMask,
        'RFPose2DMask' : RFPose2DMask
    }
    return models[name](**kwargs)
    
def load_dataset(dataset, **kwargs):
    datasets = {
        'hiber' : load_hiber_dataset,
        'rf' : load_rf_dataset,
        'hiber_mask' : load_hiber_mask_dataset
    }
    return datasets[dataset](**kwargs)

def postprocess(name, **kwargs):
    procs = {
        'RFMask': rfmask_postproc,
        'RFPose2DMask': rfpose2dmask_postproc
    }
    return procs[name](**kwargs)

def data2cuda(data):
    """_summary_

    Args:
        data (list): list of [hors, vers, params, targets]

    Returns:
        Tensors: Horizontal radar frame Tensors.
        Tensors: Vertical radar frame Tensors.
        Dict: Parameters.
        Dict: Targets.
    """
    hors, vers = data[0].cuda(), data[1].cuda()
    params = []
    for d in data[2]:
        param = {}
        for k, v in d.items():
            param[k] = v.cuda()
        params.append(param)
    targets = []
    for d in data[3]:
        target = {}
        for k, v in d.items():
            target[k] = v.cuda()
        targets.append(target)
    return hors, vers, params, targets
