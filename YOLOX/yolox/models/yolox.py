#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
import config_da as cfg_da


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        if cfg_da.use_domain_adaptation:
            fpn_outs, backbone_features = self.backbone(x)
        else:
            fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            if cfg_da.use_domain_adaptation:
                 loss_info, head_features, output_preds = self.head(fpn_outs, targets, x)
                 loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = loss_info
            else:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            if cfg_da.use_domain_adaptation:
                outputs['backbone_features'] = backbone_features
                outputs['head_features'] = head_features
                outputs['output_preds'] = output_preds
        else:
            outputs = self.head(fpn_outs)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
