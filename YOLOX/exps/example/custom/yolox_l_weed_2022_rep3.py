#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        
        self.data_dir = r"D:\Dataset\WeedData\WeedData2022"
        self.train_ann = "instances_train2017_8cls.2.json"
        self.val_ann = "instances_val2017_8cls.2.json"
        self.test_ann = "instances_test2017_8cls.2.json"
        self.train_name = "train2017_8cls.2"
        self.val_name = "val2017_8cls.2"
        self.test_name = "test2017_8cls.2"

        self.num_classes = 12
        self.warmup_epochs = 3

        self.max_epoch = 40
        self.data_num_workers = 0
        self.eval_interval = 2
        self.save_history_ckpt = True
        self.input_size = (800, 800)
        self.test_size = (800, 800)

        #self.basic_lr_per_img = 0.01 / 8.0 #bad
        self.basic_lr_per_img = 0.01 / 128.0

        #self.multiscale_range = 1
        #self.mixup_prob = 0.2
        #self.degrees = 5.0
        #self.mosaic_scale = (0.5, 1.5)
        #self.shear = 1.0
        #self.no_aug_epochs = 20
        #self.mosaic_prob = 0.5

