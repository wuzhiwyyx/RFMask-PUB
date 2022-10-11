import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
# from torchvision.models.detection.roi_heads import maskrcnn_loss, maskrcnn_inference
from torchvision.models.detection.roi_heads import maskrcnn_inference
from torch.jit.annotations import Optional, List, Dict, Tuple

from .predictor import RFMaskRCNNPredictor
from .utils import calc_v_props, maskrcnn_loss

class RFRoIHeads(RoIHeads):
    """RFRoIHeads is the core part of RFMask, it is responsible for regressing 2D detection
       results, calculating 3D bounding boxes, cropping roi features, fuse features and 
       generate final results.

       RFRoIHeads is inherited from RoIHeads implemented by torchvision. Original codes can
       be found in torchvision.models.detection.roi_heads
    """
    def __init__(self, out_channels=256, dual=True, fuse='multi-head'):
        """RFRoIHeads constructor.

        Args:
            out_channels (int, optional): Feature channels outputed by Branch. Defaults to 256.
            fuse (str, optional): Feature fusion method, candicate method are ['multi-head0, 'cat'']. Defaults to multi-head fusion.
        """
        # Box regression parameters
        box_fg_iou_thresh, box_bg_iou_thresh = 0.5, 0.5
        box_batch_size_per_image, box_positive_fraction = 512, 0.25
        bbox_reg_weights = None
        box_score_thresh, box_nms_thresh, box_detections_per_img = 0.05, 0., 100
        
        # Build roiheads
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0'], output_size=14, sampling_ratio=2)
        box_head = TwoMLPHead(
            out_channels * box_roi_pool.output_size[0] ** 2, 1024)
        self.num_cls = 2
        box_predictor = FastRCNNPredictor(1024, self.num_cls)
        super(RFRoIHeads, self).__init__(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img
        )

        self.vbox_head = TwoMLPHead(
            out_channels * box_roi_pool.output_size[0] ** 2, 1024)
        self.vbox_predictor = FastRCNNPredictor(1024, self.num_cls)
        self.compress = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 14 * 14, 2000),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2000, 14 * 14),
        )
        self.mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0'], output_size=14, sampling_ratio=2)

        # initialize fuse method
        assert fuse in ['multi-head', 'cat'], f'candidate fusion methods are ["multi-head", "cat"], but got {fuse}.'
        head_in_feature = 1 if dual else 64
        if dual and fuse == 'multi-head':
            head_in_feature = 1
            self.fuse = self.multi_head_fusion
        elif dual and fuse == 'cat':
            head_in_feature = 128
            self.fuse = self.concatenate_fusion
        else:
            head_in_feature = 64

        self.mask_head = MaskRCNNHeads(head_in_feature, (64,), 1)
        self.mask_predictor = RFMaskRCNNPredictor(64, 32, self.num_cls)
        # self.mask_size = (624, 820)

        self.trans = self.trans_enc(d_model=64*14, num_layers=3)
        self.multi_head = nn.MultiheadAttention(64*14, 8)
        
    def trans_enc(self, d_model, nheads=8, num_layers=2):
        """Build transformer encoder.

        Args:
            d_model (int): Set 'd_model' of transformer.
            nheads (int, optional): Number of transformer heads. Defaults to 8.
            num_layers (int, optional): Number of layers. Defaults to 2.

        Returns:
            dec (nn.Module): TransformerEncoder object.
        """
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nheads)
        enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        return enc
    
    def trans_dec(self, d_model, nheads=8, num_layers=2):
        """Build transformer decoder.

        Args:
            d_model (int): Set 'd_model' of transformer.
            nheads (int, optional): Number of transformer heads. Defaults to 8.
            num_layers (int, optional): Number of layers. Defaults to 2.

        Returns:
            dec (nn.Module): TransformerDecoder object.
        """
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nheads)
        dec = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        return dec

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets,     # type: Optional[List[Dict[str, Tensor]]]
                                key='boxes'
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        """This function is copied from torchvision.models.detection.roi_heads.

        Args:
            proposals (list): Proposals generated by RPN.
            targets (dict): Ground-truth labels.
            key (str): Indicating which key in targets contains ground-truth boxes.

        Returns:
            proposals (list): Selected proposals.
            matched_idxs (list): Indexes of selected proposals.
            labels (list): Labels of each selected input (with in a batch).
            regression_targets (tensor): Encoded box results for regressing.
            sampled_inds (list): Remove ignored samples indexes, retain positive/negtive samples indexes.
        """
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t[key].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, sampled_inds

    def concatenate_fusion(self, h_feats, v_feats, pool_size=14):
        """Fuse feature by MLP.

        Args:
            h_feats (tensor): Horizontal roi features.
            v_feats (tensor): Vertical roi features.
            pool_size (int, optional): Pooled shape. Defaults to 14.
        """
        feats =  torch.cat([h_feats, v_feats], axis=-3)
        return feats
        b, c = h_feats.size()[:2]
        feats = torch.stack([h_feats, v_feats], dim=-3)
        feats = feats.view(-1, *feats.size()[-3:])
        feats = self.compress(feats)
        feats = feats.view(b, c, pool_size, pool_size)
        return feats
    
    def multi_head_fusion(self, h_feats, v_feats, pool_size=14):
        """Multi-Head fusion module.

        Args:
            h_feats (tensor): Horizontal RoI features. Shape [proposal_num, 64, 14, 14]
            v_feats (tensor): Vertical RoI features. Shape [proposal_num, 64, 14, 14]
            pool_size (int, optional): Pooled shape. Defaults to 14.

        Returns:
            att_weights (tensor): Attention weights calculcated by transformer.
        """
        n, c, h, w = h_feats.shape
        feats = torch.cat([h_feats, v_feats], dim=-1).view(n, c * h, 2 * w)
        feats = feats.permute(2, 0, 1)
        feats = self.trans(feats)
        att_output, att_weights = self.multi_head(feats, feats, feats)

        att_weights = att_weights[:, w:, :w].unsqueeze(1)
        return att_weights
    
    def box_reg(self, bundle, targets, key='hboxes'):
        """Regress target boundingboxes in each branch.

        Args:
            bundle (tuple): Containing features, proposals and original input shape.
            targets (dict): Label dict.
            key (str, optional): Indicating which key in targets stores boundingbox ground-truth. Defaults to 'hboxes'.

        Returns:
            result (list[dict]): Containing predicted results. list of {'boxes':boxes, 'labels':labels, 'scores':scores}.
            losses (ditc): Losses.
            props (list): RPN proposals.
            matched_idxs (list): Selected proposals.
            labels (list): Selected proposal indexes.
        """
        feats, props, img_sizes = bundle
        assert key in ['hboxes', 'vboxes']
        if key == 'hboxes':
            box_head, box_predictor = self.box_head, self.box_predictor
        elif key == 'vboxes':
            box_head, box_predictor = self.vbox_head, self.vbox_predictor
        
        if not targets is None:
            _ = self.select_training_samples(props, targets, key)
            props, matched_idxs, labels, regression_targets, sampled_inds = _
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
        
        box_feats = self.box_roi_pool(feats, props, img_sizes)
        box_feats = box_head(box_feats)
        class_logits, box_regression = box_predictor(box_feats)
        
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        
        if not targets is None:
            assert regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier_"+key[0]: loss_classifier,
                "loss_box_reg_"+key[0]: loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, props, img_sizes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        return result, losses, props, matched_idxs, labels
    
    def forward(self, h_bundle, v_bundle, params, targets=None):
        """Main forward function of RFRoIHeads

        Args:
            h_bundle (tuple): Horizontal features, proposals and original input shape.
            v_bundle (tuple): Vertical features, proposals and original input shape.
            params (tensor): Indicating which view (environment) current data belongs to.
            targets (dict, optional): Labels dict. Defaults to None.

        Returns:
            result (list[dict]): Containing predicted results. list of {'boxes':boxes, 'labels':labels, 'scores':scores}.
            losses (dict): Losses.
        """
        dual = not v_bundle is None

        # Unpack branch outputs.
        h_feats, h_props, h_img_sizes = h_bundle
        if dual:
            v_feats, v_props, v_img_sizes = v_bundle
        
        # Perform backbone on each horizontal/vertical input data, respectively.
        h_ = self.box_reg(h_bundle, targets, key='hboxes')
        result, losses, h_props, matched_idxs, labels = h_
        if dual:
            v_ = self.box_reg(v_bundle, targets, key='vboxes')
            v_losses = v_[1]

        # Prepare positive boxes
        h_preds = [p["boxes"] for p in result]
        if not targets is None:
            assert matched_idxs is not None
            # during training, only focus on positive boxes
            num_images = len(h_props)
            h_preds = []
            pos_matched_idxs = []
            for img_id in range(num_images):
                pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                h_preds.append(h_props[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
        else:
            pos_matched_idxs = None
        
        # import matplotlib.pyplot as plt
        # from matplotlib.patches import Rectangle
        # import numpy as np

        # Calculate vertical boxes according to horizontal boxes,
        # meanwhile undo offset and combine box-pairs into 3D bounding boxes.
        # Project 3D bounding boxes into result image plane
        v_preds, m_props = calc_v_props(h_preds, params)

        # Deal with nan brought by float16
        # for i in range(len(m_props)):
        #     selected = m_props[i].sum(dim=1).isnan()
        #     m_props[i][selected] = torch.tensor([0, 0, 10, 10], dtype=m_props[i].dtype, device=m_props[i].device)
        #     selected = ~m_props[i].sum(dim=1).isnan()
        #     if pos_matched_idxs[i].numel() != 0:
        #         pos_matched_idxs[i] = pos_matched_idxs[i][selected]
        #         m_props[i] = m_props[i][selected]
        #         h_preds[i] = h_preds[i][selected]
        #         v_preds[i] = v_preds[i][selected]
        #         result[i]["labels"] = result[i]["labels"][selected]
        #     else:
        #         pos_matched_idxs[i]
        #         m_props[i] = torch.zeros((0, 4), device=h_feats['0'].device)

        for i, res in enumerate(result):
            res['boxes'] = m_props[i]
            
        # Crop roi features

        hm_feats = self.mask_roi_pool(h_feats, h_preds, h_img_sizes) # (N, 64, 14, 14)
        if dual:
            vm_feats = self.mask_roi_pool(v_feats, v_preds, v_img_sizes) # (N, 64, 14, 14)

            # Fuse feature by our proposed Multi-Head Fusion module
            mask_features = self.fuse(hm_feats, vm_feats)
        else:
            mask_features = hm_feats

        # Generate silhouette results.
        mask_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_features)

        loss_mask = {}
        if not targets is None:
            assert targets is not None
            assert pos_matched_idxs is not None
            assert mask_logits is not None

            gt_masks = [t["masks"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            
            rcnn_loss_mask = maskrcnn_loss(
                mask_logits, m_props,
                gt_masks, gt_labels, pos_matched_idxs)
            loss_mask = {
                "loss_mask": rcnn_loss_mask
            }
        else:
            labels = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(mask_logits, labels)
            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

        if dual:
            losses.update(v_losses)
        losses.update(loss_mask)
        return result, losses