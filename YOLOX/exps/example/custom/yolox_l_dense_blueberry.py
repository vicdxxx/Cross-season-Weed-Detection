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
        
        self.data_dir = "datasets/BlueberryDenseDetection"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"

        self.num_classes = 4

        self.max_epoch = 100
        self.data_num_workers = 0
        self.eval_interval = 2
        self.save_history_ckpt = False
        self.input_size = (1920, 1920)
        self.test_size = (1920, 1920)
