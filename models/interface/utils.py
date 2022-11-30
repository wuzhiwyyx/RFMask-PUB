'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-11-30 00:13:08
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-11-30 00:13:13
 # @ Description: Functions used by interfaces.
 '''


import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FFMpegWriter


def init_writer():
    bundle = {}
    writer = FFMpegWriter(fps=10)
    fig = plt.figure(dpi=100, figsize=(16.40, 6.24))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    bundle['figure'] = fig
    bundle['temp_name'] = Path('results') / 'temp.mp4'
    writer.setup(fig, outfile=bundle['temp_name'], dpi=100)
    bundle['writer'] = writer
    # bundle['backend'] =  mpl.get_backend() 
    # mpl.use("Agg")
    return bundle

def close_writer(bundle):
    if not bundle is None:
        bundle['writer'].finish()

def vis_filename(exper, miou, create=True):
    current_time = time.strftime("%Y-%m-%d-%H-%M")
    vis_file = f'{exper}_{miou.item():.3f}_{current_time}.mp4'
    out = Path('results') / vis_file
    if create:
        out.parent.mkdir(parents=True, exist_ok=True)
    return out

def close_and_rename(bundle, exper, miou, create=True):
    close_writer(bundle)
    out = vis_filename(exper, miou, create=create)
    Path(bundle['temp_name']).rename(out)
    logging.getLogger('pytorch_lightning').info(f'Visualized results saved in {out}.')

def iou(mask1, mask2):
    union = torch.logical_or(mask1, mask2)
    union = union.reshape(*union.shape[:-2], -1)
    union = union.sum(dim=-1)

    inter = torch.logical_and(mask1, mask2)
    inter = inter.reshape(*inter.shape[:-2], -1)
    inter = inter.sum(dim=-1)
    return inter / union
