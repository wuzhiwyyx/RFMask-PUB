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

from .backbone import resnet_fpn_backbone, misc_nn_ops
from .utils import RFTransform
from .rpn import RFRPN

class Encoder(nn.Module):
    def __init__(self, in_channels=6, direction='hor', min_size=160, max_size=200):
        nn.Module.__init__(self)
#         self.conv = nn.Sequential(
#             nn.Conv2d(6, 1, kernel_size=1, stride=1, padding=0, padding_mode='replicate', bias=False),
#             nn.BatchNorm2d(1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(inplace=True)
#         )
        self.conv = resnet_fpn_backbone('resnet18', pretrained=False, 
                                        trainable_layers=5, norm_layer=misc_nn_ops.BatchNorm2d)
        self.conv.body.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.out_channels = self.conv.out_channels
#         self.out_channels = 256
        self.direction = direction
        self.transform = RFTransform(min_size, max_size)

    def forward(self, x, target=None):
        # transform会根据min/max size改变target
        x, target = self.transform(x, target)
        features = self.conv(x.tensors)
#         return OrderedDict([('0', features)]), x, target
        return features, x, target

class Branch(nn.Module):
    def __init__(self, backbone):
        nn.Module.__init__(self)
        self.backbone = backbone
        self.out_channels = backbone.out_channels
#         self.rpn = self._make_rpn_layer()
        self.rpn = RFRPN(self.out_channels)
        
    def _make_rpn_layer(self):
#         anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        anchor_sizes = ((8, 16, 32, 64, 128),) * 5
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        out_channels = self.backbone.out_channels
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n_train = 1000
        rpn_pre_nms_top_n_test = 500
        rpn_post_nms_top_n_train = 1000
        rpn_post_nms_top_n_test = 500
        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        return rpn
    
    def forward(self, images, targets=None):
        features, images, targets = self.backbone(images, targets)
        proposals, proposal_losses = self.rpn(images, features, targets)
        return features, proposals, proposal_losses, images, targets