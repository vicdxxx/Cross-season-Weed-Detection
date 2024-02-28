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
        self.data_dir_cross_season = "datasets/WeedData2022"
        self.train_ann = None
        self.val_ann = None
        self.test_ann = "instances_test2017_cross_season.json"
        self.test_name = "WeedPlantsXXX_8skus_in_2021"

        self.num_classes = 12

        self.max_epoch = 40
        self.data_num_workers = 0
        self.eval_interval = 2
        self.save_history_ckpt = True
        self.input_size = (800, 800)
        self.test_size = (800, 800)
