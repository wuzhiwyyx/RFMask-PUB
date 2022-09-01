'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 13:13:24
 # @ Description: Branch definition, containing Encoder module.
 '''

import torch.nn as nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from collections import OrderedDict

from .backbone import resnet_fpn_backbone, misc_nn_ops
from .utils import RFTransform
from .rpn import RFRPN

class Encoder(nn.Module):
    def __init__(self, in_channels=6, min_size=160, max_size=200):
        """Backbone feature extractor. We tried two kinds of backbone structures. 
           Eight convolution layers or resnet18. In the end, we choose to use 'resnet18'.

        Args:
            in_channels (int, optional): Channel number of input data. Defaults to 6.
            backbone (str, optional): Which backbone to use, candidate values are ['convs', 'resnet18']. Defaults to 'resnet'.
            min_size (int, optional): Height of input data. Defaults to 160.
            max_size (int, optional): Width of input data. Defaults to 200.
        """
        nn.Module.__init__(self)
        backbone = 'resnet18'
        self.backbone = backbone
        assert backbone in ['convs', 'resnet18'], \
            f'backbone should be one of ["convs", "resnet18"], but get {backbone} instead.'
        if backbone == 'convs':
            self.conv = nn.Sequential(
                nn.Conv2d(6, 1, kernel_size=1, stride=1, padding=0, padding_mode='replicate', bias=False),
                nn.BatchNorm2d(1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True)
            )
            self.out_channels = 256
        elif backbone == 'resnet18':
            self.conv = resnet_fpn_backbone('resnet18', pretrained=False, 
                                            trainable_layers=5, norm_layer=nn.BatchNorm2d)
            self.conv.body.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.out_channels = self.conv.out_channels
        self.transform = RFTransform(min_size, max_size)

    def forward(self, x, target=None):
        # Transform input data according to 'min_size' and 'max_size'
        x, target = self.transform(x, target)
        features = self.conv(x.tensors)
        # Uniform return value format
        if self.backbone == 'convs':
            return OrderedDict([('0', features)]), x, target
        return features, x, target

class Branch(nn.Module):
    def __init__(self, backbone):
        """Branch model definition. Branch is responsible for extract feature and perform RPN.

        Args:
            backbone (Encoder): Encoder class instance.
        """
        nn.Module.__init__(self)
        self.backbone = backbone
        self.out_channels = backbone.out_channels
        self.rpn = RFRPN(self.out_channels)
    
    def forward(self, images, targets=None):
        # Extract features and perform RPN.
        features, images, targets = self.backbone(images, targets)
        proposals, proposal_losses = self.rpn(images, features, targets)
        return features, proposals, proposal_losses, images, targets