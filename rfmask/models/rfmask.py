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
from torchvision.models.detection.roi_heads import expand_masks, expand_boxes

from torch.jit.annotations import Optional, List, Dict, Tuple

from .branch import Branch, Encoder
from .roi_heads import RFRoIHeads
from .utils import _onnx_paste_masks_in_image_loop

class RFMask(nn.Module):
    """RFMask model definition."""

    def __init__(self, num_classes=2):
        """RFMask model constructor.

        Args:
            num_classes (int, optional): Number of categories to classifiy. Defaults to 2.
        """
        nn.Module.__init__(self)
        # Horizontal and vertical branches have the same structure.
        self.hbranch = Branch(Encoder())
        self.vbranch = Branch(Encoder())
        self.num_classes = num_classes
        self.mask_size = (624, 820)
        self.roi_heads = RFRoIHeads(self.hbranch.out_channels, num_classes, (624, 820))

    def prepare_targets(self, targets, key='hboxes'):
        """Reorganize label structure.

        Args:
            targets (dict): Containing box and silhouette labels.
            key (str, optional): Assign the value of targets[key] to targets['boxes']. Defaults to 'hboxes'.

        Returns:
            dict: Reorganized dict.
        """
        if targets is None:
            return targets
        for target in targets:
            target['boxes'] = target[key]
        return targets
    
    def paste_mask_in_image(self, mask, box, im_h, im_w):
        # type: (Tensor, Tensor, int, int) -> Tensor
        TO_REMOVE = 1
        w = int(box[2] - box[0] + TO_REMOVE)
        h = int(box[3] - box[1] + TO_REMOVE)
        #w = max(w, 1)
        #h = max(h, 1)
        if w > 2 * im_w or h > 2 * im_h:
            return torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
        
        w = max(w, 1) if w < im_w else im_w
        h = max(h, 1) if h < im_h else im_h

        # Set shape to [batchxCxHxW]
        mask = mask.expand((1, 1, -1, -1))

        # Resize mask
        mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        mask = mask[0][0]

        im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
        x_0 = max(box[0], 0)
        x_1 = min(box[2] + 1, im_w)
        y_0 = max(box[1], 0)
        y_1 = min(box[3] + 1, im_h)
        _ = mask[
            (y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])
        ]
        if im_mask[y_0:y_1, x_0:x_1].numel() == _.numel():
            im_mask[y_0:y_1, x_0:x_1] = _
        return im_mask
    
    def paste_masks_in_image(self, masks, boxes, img_shape, padding=1):
        # type: (Tensor, Tensor, Tuple[int, int], int) -> Tensor
        masks, scale = expand_masks(masks, padding=padding)
        boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
        im_h, im_w = img_shape

        if torchvision._is_tracing():
            return _onnx_paste_masks_in_image_loop(masks, boxes,
                                                   torch.scalar_tensor(im_h, dtype=torch.int64),
                                                   torch.scalar_tensor(im_w, dtype=torch.int64))[:, None]
        res = [
            self.paste_mask_in_image(m[0], b, im_h, im_w)
            for m, b in zip(masks, boxes)
        ]
        if len(res) > 0:
            ret = torch.stack(res, dim=0)[:, None]
        else:
            ret = masks.new_empty((0, 1, im_h, im_w))
        return ret
        
    def forward(self, hor, ver, params, targets=None):
        """Main inference function of RFMask model.

        Args:
            hor (FloatTensor): Horizontal RF frames.
            ver (FloatTensor): Vertical RF frames.
            params (IntTensor): View (Environment) index of each data in a batch which is used to load camera parameters.
            targets (dict, optional): Label dict. Defaults to None.
        
        Returns:
            result (list[dict]): Containing predicted results. list of {'boxes':boxes, 'labels':labels, 'scores':scores}.
            losses (dict): Losses
        """
        # We use the off-the-shelf RPN model implemented by torchvision official code
        # which could be found in torchvision.models.detection.rpn.
        # RPN in torchvision automatically loads 'boxes' key in label dict as ground-truth.
        targets = self.prepare_targets(targets, 'hboxes')
        h_features, h_proposals, h_prop_losses, hor, h_target = self.hbranch(hor, targets)
        targets = self.prepare_targets(targets, 'vboxes')
        v_features, v_proposals, v_prop_losses, ver, v_target = self.vbranch(ver, targets)
        
        # Pack rpn outputs into tuples and feed them into roi_heads
        h_bundle = (h_features, h_proposals, hor.image_sizes)
        v_bundle = (v_features, v_proposals, ver.image_sizes)
        result, losses = self.roi_heads(h_bundle, v_bundle, params, targets)

        # If targets is passed then loss value will be returned, else losses is None.
        if not targets is None:
            rpn_loss = {}
            rpn_loss['loss_h_obj'] = h_prop_losses['loss_objectness']
            rpn_loss['loss_h_rpn_box'] = h_prop_losses['loss_rpn_box_reg']
            rpn_loss['loss_v_obj'] = v_prop_losses['loss_objectness']
            rpn_loss['loss_v_rpn_box'] = v_prop_losses['loss_rpn_box_reg']
            losses.update(rpn_loss)
            return result, losses
        else:
            for i, pred in enumerate(result):
                masks = pred["masks"]
                boxes = pred['boxes']
                o_im_s = self.mask_size
                masks = self.paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            return result, losses