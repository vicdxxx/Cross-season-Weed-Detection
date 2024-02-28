#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        #color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        color = _COLORS(cls_id)
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        #txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        txt_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        lw = max(round(sum(img.shape) / 2 * 0.003), 2)
        tf = max(lw - 1, 1)

        #txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        txt_size = cv2.getTextSize(text, font, fontScale=lw / 3, thickness=tf)[0]

        #cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=lw, lineType=cv2.LINE_AA)

        #txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        txt_bk_color = _COLORS(cls_id)
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        #cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, lw/3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

    return img


#_COLORS = np.array(
#    [
#        0.000, 0.447, 0.741,
#        0.850, 0.325, 0.098,
#        0.929, 0.694, 0.125,
#        0.494, 0.184, 0.556,
#        0.466, 0.674, 0.188,
#        0.301, 0.745, 0.933,
#        0.635, 0.078, 0.184,
#        0.300, 0.300, 0.300,
#        0.600, 0.600, 0.600,
#        1.000, 0.000, 0.000,
#        1.000, 0.500, 0.000,
#        0.749, 0.749, 0.000,
#        0.000, 1.000, 0.000,
#        0.000, 0.000, 1.000,
#        0.667, 0.000, 1.000,
#        0.333, 0.333, 0.000,
#        0.333, 0.667, 0.000,
#        0.333, 1.000, 0.000,
#        0.667, 0.333, 0.000,
#        0.667, 0.667, 0.000,
#        0.667, 1.000, 0.000,
#        1.000, 0.333, 0.000,
#        1.000, 0.667, 0.000,
#        1.000, 1.000, 0.000,
#        0.000, 0.333, 0.500,
#        0.000, 0.667, 0.500,
#        0.000, 1.000, 0.500,
#        0.333, 0.000, 0.500,
#        0.333, 0.333, 0.500,
#        0.333, 0.667, 0.500,
#        0.333, 1.000, 0.500,
#        0.667, 0.000, 0.500,
#        0.667, 0.333, 0.500,
#        0.667, 0.667, 0.500,
#        0.667, 1.000, 0.500,
#        1.000, 0.000, 0.500,
#        1.000, 0.333, 0.500,
#        1.000, 0.667, 0.500,
#        1.000, 1.000, 0.500,
#        0.000, 0.333, 1.000,
#        0.000, 0.667, 1.000,
#        0.000, 1.000, 1.000,
#        0.333, 0.000, 1.000,
#        0.333, 0.333, 1.000,
#        0.333, 0.667, 1.000,
#        0.333, 1.000, 1.000,
#        0.667, 0.000, 1.000,
#        0.667, 0.333, 1.000,
#        0.667, 0.667, 1.000,
#        0.667, 1.000, 1.000,
#        1.000, 0.000, 1.000,
#        1.000, 0.333, 1.000,
#        1.000, 0.667, 1.000,
#        0.333, 0.000, 0.000,
#        0.500, 0.000, 0.000,
#        0.667, 0.000, 0.000,
#        0.833, 0.000, 0.000,
#        1.000, 0.000, 0.000,
#        0.000, 0.167, 0.000,
#        0.000, 0.333, 0.000,
#        0.000, 0.500, 0.000,
#        0.000, 0.667, 0.000,
#        0.000, 0.833, 0.000,
#        0.000, 1.000, 0.000,
#        0.000, 0.000, 0.167,
#        0.000, 0.000, 0.333,
#        0.000, 0.000, 0.500,
#        0.000, 0.000, 0.667,
#        0.000, 0.000, 0.833,
#        0.000, 0.000, 1.000,
#        0.000, 0.000, 0.000,
#        0.143, 0.143, 0.143,
#        0.286, 0.286, 0.286,
#        0.429, 0.429, 0.429,
#        0.571, 0.571, 0.571,
#        0.714, 0.714, 0.714,
#        0.857, 0.857, 0.857,
#        0.000, 0.447, 0.741,
#        0.314, 0.717, 0.741,
#        0.50, 0.5, 0
#    ]
#).astype(np.float32).reshape(-1, 3)

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    
_COLORS = Colors()
