'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 13:16:21
 # @ Description: Predictor definition.
 '''

from collections import OrderedDict

import torch.nn as nn


class RFMaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(RFMaskRCNNPredictor, self).__init__(OrderedDict([
            ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu5", nn.ReLU(inplace=True)),
            ("conv6_mask", nn.ConvTranspose2d(dim_reduced, dim_reduced // 2, 2, 2, 0)),
            ("relu6", nn.ReLU(inplace=True)),
            ("conv7_mask", nn.ConvTranspose2d(dim_reduced // 2, dim_reduced // 4, 2, 2, 0)),
            ("relu7", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", nn.Conv2d(dim_reduced // 4, num_classes, 1, 1, 0)),
        ]))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
