import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.models.detection.roi_heads import maskrcnn_loss, maskrcnn_inference
from torch.jit.annotations import Optional, List, Dict, Tuple

from .predictor import RFMaskRCNNPredictor

class RFRoIHeads(RoIHeads):
    def __init__(self, out_channels=256, num_classes=2, mask_size=(624, 820)):
        # box回归基本参数
        box_fg_iou_thresh, box_bg_iou_thresh = 0.5, 0.5
        box_batch_size_per_image, box_positive_fraction = 512, 0.25
        bbox_reg_weights = None
        box_score_thresh, box_nms_thresh, box_detections_per_img = 0.05, 0., 100
        
        # 构建RoIHead
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0'], output_size=14, sampling_ratio=2)
        box_head = TwoMLPHead(
            out_channels * box_roi_pool.output_size[0] ** 2, 1024)
        self.num_cls = num_classes
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
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(1000, 500),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2000, 14 * 14),
#             nn.Conv2d(14, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0'], output_size=14, sampling_ratio=2)
        self.mask_head = MaskRCNNHeads(1, (64,), 1)
        self.mask_predictor = RFMaskRCNNPredictor(64, 32, num_classes)
        self.mask_size = mask_size
#         self.h_trans = self.trans_enc(d_model=64)
#         self.v_trans = self.trans_enc(d_model=64)
#         self.fuse_trans = self.trans_dec(d_model=64)
        self.trans = self.trans_enc(d_model=64*14, num_layers=3)
        self.multi_head = nn.MultiheadAttention(64*14, 8)
        
    def trans_enc(self, d_model, nheads=8, num_layers=2):
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nheads)
        enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        return enc
    
    def trans_dec(self, d_model, nheads=8, num_layers=2):
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nheads)
        dec = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        return dec
#     def prepare_targets(self, targets, key='hboxes'):
#         if targets is None:
#             return targets
#         for target in targets:
#             target['boxes'] = target[key]
#         return targets
        
    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets,     # type: Optional[List[Dict[str, Tensor]]]
                                key='boxes'
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
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
    
    def project_3d_to_pixel(self, points, M, P, D, K):
        pts = torch.ones((1, points.shape[0]), device=points.device)
        pts = torch.cat([points.t(), pts], dim=0)
        cc = torch.mm(M, pts)
        pc = (cc[:2] / cc[2]).t()
        fx, fy, cx, cy, Tx, Ty = P[0,0], P[1,1], P[0,2], P[1,2], P[0,3], P[1,3]         
        uv_rect_x, uv_rect_y = pc[:, 0], pc[:, 1]
        xp, yp = (uv_rect_x - cx - Tx) / fx, (uv_rect_y - cy - Ty) / fy
        r2 = xp * xp + yp * yp
        r4 = r2 * r2
        r6 = r4 * r2
        a1 = 2 * xp * yp
        k1, k2, p1, p2, k3 = D
        barrel = 1 + k1 * r2 + k2 * r4 + k3 * r6
        xpp = xp * barrel + p1 * a1 + p2 * (r2 + 2 * (xp * xp))
        ypp = yp * barrel + p1 * (r2 + 2 * (yp * yp)) + p2 * a1
        kfx, kcx, kfy, kcy = K[0, 0], K[0, 2], K[1, 1], K[1, 2]
        u = xpp * kfx + kcx
        v = ypp * kfy + kcy
        pts = torch.stack([u, v], dim=0).t()
        pts = pts / 2
        return pts
    
    
    def _calc_v_props(self, h_proposals, params, project=True, pad=25):
        v_props = []
        for prop, param in zip(h_proposals, params):
            x_off, y_off, z_off, r_off, z_min, z_max = param['offset']
            v = prop.clone()
            for i in [1, 3]:
                _ = (160 - prop[:, i]) ** 2 + (prop[:, 0]/2 + prop[:, 2]/2 - r_off) ** 2
                v[:, i] = 160 - _ ** 0.5
            v[:, 0] = z_min
            v[:, 2] = z_max
            v_props.append(v)
        if not project:
            return v_props
        
#         def project_3d_to_pixel(points, M, P, D, K):
#             pts = torch.ones((1, points.shape[0]), device=points.device)
#             pts = torch.cat([points.t(), pts], dim=0)
#             cc = torch.mm(M, pts)
#             pc = (cc[:2] / cc[2]).t()
#             fx, fy, cx, cy, Tx, Ty = P[0,0], P[1,1], P[0,2], P[1,2], P[0,3], P[1,3]         
#             uv_rect_x, uv_rect_y = pc[:, 0], pc[:, 1]
#             xp, yp = (uv_rect_x - cx - Tx) / fx, (uv_rect_y - cy - Ty) / fy
#             r2 = xp * xp + yp * yp
#             r4 = r2 * r2
#             r6 = r4 * r2
#             a1 = 2 * xp * yp
#             k1, k2, p1, p2, k3 = D
#             barrel = 1 + k1 * r2 + k2 * r4 + k3 * r6
#             xpp = xp * barrel + p1 * a1 + p2 * (r2 + 2 * (xp * xp))
#             ypp = yp * barrel + p1 * (r2 + 2 * (yp * yp)) + p2 * a1
#             kfx, kcx, kfy, kcy = K[0, 0], K[0, 2], K[1, 1], K[1, 2]
#             u = xpp * kfx + kcx
#             v = ypp * kfy + kcy
#             pts = torch.stack([u, v], dim=0).t()
#             pts = pts / 2
#             return pts
            
#         fig = plt.figure(figsize=(16, 4))
#         axes = fig.subplots(1, 4)
#         for i in range(4):
#             canvas = np.zeros((160, 200, 3), dtype=np.uint8)
#             for p in h_proposals[i][:1]:
#                 print(tuple(p[:2].int()), tuple(p[2:].int()))
#                 canvas = cv2.rectangle(canvas, tuple(p[:2].int().numpy()), tuple(p[2:].int().numpy()), (255, 255, 255), 1)
#             axes[i].imshow(canvas)
#             axes[i].set_axis_off()
        
#         fig = plt.figure(figsize=(16, 4))
#         axes = fig.subplots(1, 4)
#         for i in range(4):
#             canvas = np.zeros((160, 200, 3), dtype=np.uint8)
#             for p in v_props[i][:1]:
#                 print(tuple(p[:2].int()), tuple(p[2:].int()))
#                 canvas = cv2.rectangle(canvas, tuple(p[:2].int().numpy()), tuple(p[2:].int().numpy()), (255, 255, 255), 1)
#             axes[i].imshow(canvas)
#             axes[i].set_axis_off()
        
            
        # 求3D box映射回到0号相机平面的boundingbox
#         offset = {'1-10':(100+35, 8), '16-30':(50+35, 2)}
#         x_offset, y_offset = 100, -2
#         z_offset = 98
        proposals = []
        for h, v, param in zip(h_proposals, v_props, params):
            x_off, y_off, z_off, r_off, z_min, z_max = param['offset']
            r_mat = param['r_mat']
            M, P, D, K = param['M'], param['P'], param['D'], param['K']
            
            if h.numel() == 0:
                proposals.append(torch.zeros((1, 4), device=h.device))
                continue
            co = torch.zeros((h.size(0), 6), device=h.device)
            pad = 5
            co[:, [0, 1]] = (h[:, [0, 2]] - x_off) * 0.05
            co[:, [2, 3]] = (y_off - h[:, [1, 3]]) * 0.05
            co[:, [4, 5]] = (z_off - v[:, [0, 2]]) * 0.05
#             print(co)
            _ = [0,2,4,0,2,5,0,3,4,0,3,5,1,2,4,1,2,5,1,3,4,1,3,5]
            coor3d = co[:, _]
            coor3d = coor3d.view(-1, 3)
#             r_mat_T = torch.from_numpy(np.load('/home/zhi/ssd/view10/r_mat_T.npy')).to(h.device).float()
            coor3d = torch.mm(coor3d, r_mat.t())
            # 利用相机参数变换到图像平面
            coor2d = self.project_3d_to_pixel(coor3d, M, P, D, K)
            coor2d = coor2d.view(-1, 8, 2)
            pad = 25
            xymin = torch.min(coor2d, dim=1)[0] - pad
            xymax = torch.max(coor2d, dim=1)[0] + pad
            coor2d = torch.cat([xymin, xymax], dim=1)
            proposals.append(coor2d)
        # 可视化proposals投影结果
#         fig = plt.figure(figsize=(16, 4))
#         axes = fig.subplots(1, 4)
# #         print(len(proposals))
#         for i in range(4):
#             prop = proposals[i]
#             canvas = np.zeros((624, 820, 3), dtype=np.uint8)
#             print(prop.int().numpy()[:1, :].shape)
#             for p in prop.int().numpy()[:1, :]:
#                 print(p)
#                 canvas = cv2.rectangle(canvas, tuple(p[:2]), tuple(p[2:]), (255, 255, 255), 2)
#             axes[i].imshow(canvas)
#             axes[i].set_axis_off()
#         assert False
#             print(coor2d.size(), 'after project')
        return v_props, proposals

    def feature_fusion(self, h_feats, v_feats, pool_size=14):
        # 特征融合
#         _ = h_feats[0,:,:,:].detach().numpy()
#         _ = np.sum(_, axis=0)
#         hhh = _
#         plt.imshow(_)
#         plt.show()
#         _ = v_feats[0,:,:,:].detach().numpy()
#         _ = np.sum(_, axis=0)
#         vvv = _
#         plt.imshow(_)
#         plt.show()
        b, c = h_feats.size()[:2]
#         h_feats = h_feats.unsqueeze(-3)
#         v_feats = v_feats.unsqueeze(-1)
#         feats = h_feats + v_feats
        feats = torch.stack([h_feats, v_feats], dim=-3)
        feats = feats.view(-1, *feats.size()[-3:])
        feats = self.compress(feats)
        feats = feats.view(b, c, pool_size, pool_size)
        return feats
    
    def trans_feature_fusion(self, h_feats, v_feats, pool_size=14):
#         torch.Size([9, 64, 14, 14]) torch.Size([9, 64, 14, 14])
        n, c, h, w = h_feats.shape
        feats = torch.cat([h_feats, v_feats], dim=-1).view(n, c * h, 2 * w)
        feats = feats.permute(2, 0, 1)
        feats = self.trans(feats)
        att_output, att_weights = self.multi_head(feats, feats, feats)
#         print(att_weights.shape)
#         fig = plt.figure()
#         plt.imshow(att_weights.detach().max(dim=0)[0].numpy())
        att_weights = att_weights[:, w:, :w].unsqueeze(1)
        return att_weights
#         print(att_weights.shape)
#         assert False
        
#         v_feats = v_feats.view(n, c, h * w)
#         v_feats = v_feats.permute(2, 0, 1)
#         v_feats = self.v_trans(v_feats)
        
#         h_feats = h_feats.permute(1, 2, 0).view(n, c, h, w)
#         v_feats = v_feats.permute(1, 2, 0).view(n, c, h, w)
        
#         feats = torch.stack([h_feats, v_feats], dim=-3)
#         feats = feats.view(-1, *feats.size()[-3:])
#         feats = self.compress(feats)
#         feats = feats.view(n, c, pool_size, pool_size)
        
#         feats = torch.cat([h_feats, v_feats], dim=0) # (2 * h * w, n, c)
#         tgt = torch.zeros_like(feats)
#         print('tgt', tgt.size())
#         feats = self.fuse_trans(tgt, feats)
        
#         h_feats = h_feats.view(h * w, 1, n, c)
#         v_feats = v_feats.view(1, h * w, n, c)
#         h_feats = h_feats.repeat(1, h * w, 1, 1).view(h * w * h * w, n, c)
#         v_feats = v_feats.repeat(h * w, 1, 1, 1).view(h * w * h * w, n, c)
#         feats = torch.cat([h_feats, v_feats], dim=0)
#         tgt = torch.zeros((h * w, n, c), device=h_feats.device)
#         feats = self.fuse_trans(tgt, feats)
#         feats = feats.permute(1, 2, 0)
#         feats = feats.view(n, c, h, w)
#         return feats
    
    def box_reg(self, bundle, targets, key='hboxes'):
        feats, props, img_sizes = bundle
        assert key in ['hboxes', 'vboxes']
        if key == 'hboxes':
            box_head, box_predictor = self.box_head, self.box_predictor
        elif key == 'vboxes':
            box_head, box_predictor = self.vbox_head, self.vbox_predictor
        
        if not targets is None:
#         if self.training:
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
#         if self.training:
#             assert labels is not None and regression_targets is not None
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
        h_feats, h_props, h_img_sizes = h_bundle
        v_feats, v_props, v_img_sizes = v_bundle
        
        # 计算水平hboxes对应的vboxes GT
#         v_props_gt = self._calc_v_props([x['hboxes'] for x in targets], project=False)
#         for target, v in zip(targets, v_props_gt):
#             target['vboxes'] = v
#         # 假设目标固定高度，利用几何关系求解竖直雷达数据的候选框
#         # 同时组合成空间3D立方体，获得图像平面投影候选框
#         v_props = self._calc_v_props(h_props, project=False)
#         print(targets, [x.size() for x in h_props])
#         assert False
        h_ = self.box_reg(h_bundle, targets, key='hboxes')
        result, losses, h_props, matched_idxs, labels = h_
        v_ = self.box_reg(v_bundle, targets, key='vboxes')
        v_losses = v_[1]
#         result, losses, props, matched_idxs, labels = h_
#         assert False
        
#         canvas = np.zeros((160, 200, 3), dtype=np.uint8)
#         for p in result[0]['boxes']:
#             canvas = cv2.rectangle(canvas, tuple(p[:2]), tuple(p[2:]), (255, 255, 255), 2)
#         plt.imshow(canvas)
#         assert False
        # mask 分支计算
        h_preds = [p["boxes"] for p in result]
        if not targets is None:
#         if self.training:
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
            
        v_preds, m_props = self._calc_v_props(h_preds, params)
        for i, res in enumerate(result):
            res['boxes'] = m_props[i]
#         print(h_feats.keys(), [v.size() for k, v in h_feats.items()], v_feats.keys())
        hm_feats = self.mask_roi_pool(h_feats, h_preds, h_img_sizes)
        vm_feats = self.mask_roi_pool(v_feats, v_preds, v_img_sizes)
#         mask_features = self.feature_fusion(hm_feats, vm_feats)
        mask_features = self.trans_feature_fusion(hm_feats, vm_feats)
        mask_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_features)
        
        loss_mask = {}
        if not targets is None:
#         if self.training:
            assert targets is not None
            assert pos_matched_idxs is not None
            assert mask_logits is not None

            gt_masks = [t["masks"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
#             print('mask_logits', mask_logits.size())
#             print('gt_masks', [x.size() for x in gt_masks])
#             print('gt_labels', [x.shape[0] for x in gt_labels])
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
        losses.update(v_losses)
        losses.update(loss_mask)
        return result, losses