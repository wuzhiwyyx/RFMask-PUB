'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 13:39:35
 # @ Description: RFMask main structure definition.
 '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.jit.annotations import Dict, List, Optional, Tuple
from torchvision.models.detection.roi_heads import expand_boxes, expand_masks
from torchvision.transforms import InterpolationMode, Resize

from .branch import RFPose2DDecoder, RFPose2DEncoder


class RFPose2DMask(nn.Module):
    """
    RFPose2d model definition. Modified version of MIT RFPose2d which replaces
    original keypoint heatmap with silhouette result.
    
    """

    def __init__(self, seq_len=12) -> None:
        super().__init__()
        self.hor = RFPose2DEncoder(seq_len=seq_len)
        self.ver = RFPose2DEncoder(seq_len=seq_len)
        self.decoder = RFPose2DDecoder(in_channels=128)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, hor, ver, mask=None):
        h_feature = self.hor(hor)
        v_feature = self.ver(ver)
        feature = torch.cat([h_feature, v_feature], axis=1)
        output = self.decoder(feature)
        output = output.squeeze(1)
        loss = None
        if not mask is None:
            mask = Resize(output.shape[-2:], interpolation=InterpolationMode.NEAREST)(mask)
            loss = self.loss(output, mask)
        return output, loss