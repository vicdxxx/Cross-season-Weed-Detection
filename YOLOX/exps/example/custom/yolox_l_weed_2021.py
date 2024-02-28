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
        
        self.data_dir = "datasets/CottonWeedDet12"
        self.train_ann = "instances_train2017_8cls.0.subsample.json"
        self.val_ann = "instances_val2017_8cls.0.subsample.json"
        self.test_ann = "instances_test2017_8cls.0.subsample.json"
        self.train_name = "train2017_8cls.0.subsample"
        self.val_name = "val2017_8cls.0.subsample"
        self.test_name = "test2017_8cls.0.subsample"

        self.num_classes = 12
        # 5 3
        self.warmup_epochs = 3
        #self.warmup_epochs = 0

        self.max_epoch = 40
        #self.max_epoch = 20
        self.data_num_workers = 0
        self.eval_interval = 2
        self.save_history_ckpt = True
        self.input_size = (800, 800)
        self.test_size = (800, 800)

        #self.basic_lr_per_img = 0.01 / 64.0
        #self.basic_lr_per_img = 0.01 / 128.0
        self.basic_lr_per_img = 0.01 / 256.0
        #self.no_aug_epochs = 10
