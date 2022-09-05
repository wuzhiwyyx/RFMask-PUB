'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 13:13:24
 # @ Description: Branch definition, containing Encoder module.
 '''

from turtle import forward
import torch.nn as nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from collections import OrderedDict

from .backbone import resnet_fpn_backbone, misc_nn_ops
from .utils import RFTransform
from .rpn import RFRPN

class Encoder(nn.Module):
    """Backbone feature extractor. We tried two kinds of backbone structures. 
       Eight convolution layers or resnet18. In the end, we choose to use 'resnet18'.
    """

    def __init__(self, in_channels=6, min_size=160, max_size=200):
        """
        Backbone feature constructor. 

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
    """Branch model definition. 
       Branch is responsible for extract feature and perform RPN
    """

    def __init__(self, backbone):
        """
        Branch model constructor.
        
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

class RFPose2DEncoder(nn.Module):
    """RFPose2DEncoder definition."""

    def __init__(self, in_channels=2, out_channels=64, 
                    mid_channels=64, layer_num=10, seq_len=12) -> None:
        """
        RFPose2DEncoder constructor.

        Args:
            in_channels (int, optional): Input data channel. Defaults to 2.
            out_channels (int, optional): Output feaature channel. Defaults to 64.
            mid_channels (int, optional): Module width of hidden layers. Defaults to 64.
            layer_num (int, optional): Total number of layers. Defaults to 10.
            seq_len (int, optional): Input sequence length. Defaults to 12.
        """
        super().__init__()
        layers = []
        for i in range(layer_num // 2):
            if i == 0:
                if seq_len < 9:
                    layers.extend(self.conv3d_layer(in_channels, mid_channels, k1=(3, 5, 5), p1=(1, 2, 2)))
                else:
                    layers.extend(self.conv3d_layer(in_channels, mid_channels))
            elif i == layer_num // 2 - 1:
                layers.extend(self.conv3d_layer(mid_channels, out_channels, s1=(2, 2, 2)))
            else:
                layers.extend(self.conv3d_layer(mid_channels, mid_channels))
        self.conv = nn.Sequential(*layers)

    def conv3d_layer(self, in_channels, out_channels, 
                     k1=(9, 5, 5), s1=(1, 2, 2), 
                     k2=(9, 5, 5), s2=(1, 1, 1), 
                     p1=(4, 2, 2), p2=(4, 2, 2)):
        res = [
            nn.Conv3d(in_channels , out_channels, k1, s1, p1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, k2, s2, p2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ]
        return res
    
    def forward(self, x):
        return self.conv(x)

class RFPose2DDecoder(nn.Module):
    """RFPose2DDecoder definition."""

    def __init__(self, in_channels=32, out_channels=1) -> None:
        """RFPose2DDecoder constructor.

        Args:
            in_channels (int, optional): Input feature channel. Defaults to 32.
            out_channels (int, optional): Output result channel. Defaults to 1.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 64, (3, 6, 6), (1, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),
            nn.ConvTranspose3d(64, 32, (3, 6, 6), (1, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),
            nn.ConvTranspose3d(32, 16, (3, 6, 6), (1, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),            
            nn.ConvTranspose3d(16, out_channels, (3, 6, 6), (1, 4, 4), padding=(1, 4, 4)),
#             nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x