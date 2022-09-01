'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 13:38:22
 # @ Description: Additional useful definition.
 '''

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.boxes import box_area

from ..datasets import camera_parameter

class RFTransform(GeneralizedRCNNTransform):
    """Data transformation. Normalize input data."""

    def __init__(self, min_size, max_size):
        super(RFTransform, self).__init__(min_size, max_size, [0] * 6, [1] * 6)
        
    def normalize(self, image):
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        std, mean = torch.std_mean(image, dim=(1, 2), unbiased=True)
        return (image - mean[:, None, None]) / std[:, None, None]

def box_iou(boxes1, boxes2, eps=1e-7):
    """Calucate ious between two bunch of boxes.

    Args:
        boxes1 (tensor): First bunch of boxes.
        boxes2 (tensor): Second bunch of boxes.
        eps (float, optional): Avoid zero divide. Defaults to 1e-7.

    Returns:
        _type_: _description_
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + eps)
    return iou, union

def _onnx_paste_mask_in_image(mask, box, im_h, im_w):
    """Copied from torchvision.models.detection.roi_heads"""

    one = torch.ones(1, dtype=torch.int64)
    zero = torch.zeros(1, dtype=torch.int64)

    w = (box[2] - box[0] + one)
    h = (box[3] - box[1] + one)
    w = torch.max(torch.cat((w, one)))
    h = torch.max(torch.cat((h, one)))

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, mask.size(0), mask.size(1)))

    # Resize mask
    mask = F.interpolate(mask, size=(int(h), int(w)), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    x_0 = torch.max(torch.cat((box[0].unsqueeze(0), zero)))
    x_1 = torch.min(torch.cat((box[2].unsqueeze(0) + one, im_w.unsqueeze(0))))
    y_0 = torch.max(torch.cat((box[1].unsqueeze(0), zero)))
    y_1 = torch.min(torch.cat((box[3].unsqueeze(0) + one, im_h.unsqueeze(0))))

    unpaded_im_mask = mask[(y_0 - box[1]):(y_1 - box[1]),
                           (x_0 - box[0]):(x_1 - box[0])]

    # TODO : replace below with a dynamic padding when support is added in ONNX

    # pad y
    zeros_y0 = torch.zeros(y_0, unpaded_im_mask.size(1))
    zeros_y1 = torch.zeros(im_h - y_1, unpaded_im_mask.size(1))
    concat_0 = torch.cat((zeros_y0,
                          unpaded_im_mask.to(dtype=torch.float32),
                          zeros_y1), 0)[0:im_h, :]
    # pad x
    zeros_x0 = torch.zeros(concat_0.size(0), x_0)
    zeros_x1 = torch.zeros(concat_0.size(0), im_w - x_1)
    im_mask = torch.cat((zeros_x0,
                         concat_0,
                         zeros_x1), 1)[:, :im_w]
    return im_mask

@torch.jit._script_if_tracing
def _onnx_paste_masks_in_image_loop(masks, boxes, im_h, im_w):
    """Copied from torchvision.models.detection.roi_heads"""

    res_append = torch.zeros(0, im_h, im_w)
    for i in range(masks.size(0)):
        mask_res = _onnx_paste_mask_in_image(masks[i][0], boxes[i], im_h, im_w)
        mask_res = mask_res.unsqueeze(0)
        res_append = torch.cat((res_append, mask_res))
    return res_append

def project_3d_to_pixel(points, M, P, D, K):
    """Project 3D points into result image plane.

    Args:
        points (tensor): 3D points obtained in input radar coordinate.
        M (tensor): Camera paramter.
        P (tensor): Camera paramter.
        D (tensor): Camera paramter.
        K (tensor): Camera paramter.

    Returns:
        pts (tensor): Projected 2D points in result image plane.
    """
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

def calc_v_props(h_proposals, params, project=True, pad=25):
    """Calculate vertical proposals according to horizontal proposals.

    Args:
        h_proposals (list): Horizontal proposals.
        params (tensor): View (environment) index.
        project (bool, optional): If true, projected results will be returned, else only returns vertical proposals. Defaults to True.
        pad (int, optional): Paddings added into projected boxes. Defaults to 25.

    Returns:
        v_props (list): Vertical proposals.
        proposals (list): Projected results.
    """
    v_props = []
    for prop, param in zip(h_proposals, params):
        param = camera_parameter[f'view{param.int().item():02d}']
        for key, value in param.items():
            param[key] = value
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
    
    proposals = []
    for h, v, param in zip(h_proposals, v_props, params):
        param = camera_parameter[f'view{param.cpu().int().item():02d}']
        for key, value in param.items():
            param[key] = torch.from_numpy(np.array(value)).to(device=h.device, dtype=h.dtype) if isinstance(value, list) else value
        x_off, y_off, z_off, r_off, z_min, z_max = param['offset']
        r_mat = param['r_mat']
        M, P, D, K = param['M'], param['P'], param['D'], param['K']
        
        if h.numel() == 0:
            proposals.append(torch.zeros((1, 4), device=h.device))
            continue
        co = torch.zeros((h.size(0), 6), device=h.device)

        co[:, [0, 1]] = (h[:, [0, 2]] - x_off) * 0.05
        co[:, [2, 3]] = (y_off - h[:, [1, 3]]) * 0.05
        co[:, [4, 5]] = (z_off - v[:, [0, 2]]) * 0.05

        _ = [0,2,4,0,2,5,0,3,4,0,3,5,1,2,4,1,2,5,1,3,4,1,3,5]
        coor3d = co[:, _]
        coor3d = coor3d.view(-1, 3)
        coor3d = torch.mm(coor3d, r_mat.t())
        
        # Project 3D points into result image plane.
        coor2d = project_3d_to_pixel(coor3d, M, P, D, K)
        coor2d = coor2d.view(-1, 8, 2)

        xymin = torch.min(coor2d, dim=1)[0] - pad
        xymax = torch.max(coor2d, dim=1)[0] + pad
        coor2d = torch.cat([xymin, xymax], dim=1)
        proposals.append(coor2d)

    return v_props, proposals