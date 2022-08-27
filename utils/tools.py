'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 18:11:52
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 18:12:54
 # @ Description: Collection of some useful functions.
 '''

from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


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


def resnet_fpn_backbone(
    backbone_name,
    pretrained,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None
):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 64
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
