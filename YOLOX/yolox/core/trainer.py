#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import config_da as cfg_da
import da_I3Net_module as da_I3Net
from yolox.data import DataPrefetcher
from yolox.exp import Exp
import torch.nn as nn
import config_all as cfg

from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)

last_best_epoch = cfg.last_best_epoch 

class Trainer:
    def __init__(self, exp: Exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        logger.info("---> basic_lr_per_img: {}".format(self.exp.basic_lr_per_img))
        logger.info("---> no_aug_epochs: {}".format(self.exp.no_aug_epochs))
        logger.info("---> mixup_prob: {}".format(self.exp.mixup_prob))
        logger.info("---> shear: {}".format(self.exp.shear))
        logger.info("---> mosaic_scale: {}".format(self.exp.mosaic_scale))
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()
        epoch = self.epoch

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        #print('self.input_size:', self.input_size)
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        batch = targets

        nb = len(self.train_loader)  # number of batches
        img_per_epoch = nb * batch.shape[0]
        if cfg_da.use_domain_adaptation:
            inps_da_td, targets_da_td = self.prefetcher_da_td.next()
            inps_da_td = inps_da_td.to(self.data_type)
            inps_da_td, targets_da_td = self.exp.preprocess(inps_da_td, targets_da_td, self.input_size)

        data_end_time = time.time()
        
        if cfg_da.use_domain_adaptation:
            sources = list()
            loc = list()
            conf = list()
            fea_lists = []
            pre_lists = []

            sources_t = list()
            loc_t = list()
            conf_t = list()
            fea_lists_t = []
            pre_lists_t = []

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)
            if cfg_da.use_domain_adaptation:
                # [4, 6, 9, 12, 15, 18, 21]
                assume_tot_layer_num = 30
                itermediate_outputs = [None]*assume_tot_layer_num
                itermediate_outputs[4] = outputs['backbone_features'][0]
                itermediate_outputs[15] = outputs['head_features'][0]
                itermediate_outputs[18] = outputs['head_features'][1]
                itermediate_outputs[21] = outputs['head_features'][2]
                preds = outputs['output_preds']
                domain_l = self.netD_pixel(da_I3Net.grad_reverse(itermediate_outputs[4]))
                domain_g = self.netD(da_I3Net.grad_reverse(itermediate_outputs[15]))
                gcr_pre = self.conv_gcr(itermediate_outputs[15])
                feat1 = itermediate_outputs[15]

                source = self.L2Norm(itermediate_outputs[15])
                sources.append(source)
                sources.append(itermediate_outputs[18])
                sources.append(itermediate_outputs[21])

                loc.append(preds[0][:, :-self.model.nc])
                loc.append(preds[1][:, :-self.model.nc])
                loc.append(preds[2][:, :-self.model.nc])

                conf.append(preds[0][:, -self.model.nc:])
                conf.append(preds[1][:, -self.model.nc:])
                conf.append(preds[2][:, -self.model.nc:])

                for i_src, src in enumerate(sources):
                    loc[i_src] = loc[i_src].permute(0, 2, 3, 1).contiguous()
                    conf[i_src] = conf[i_src].permute(0, 2, 3, 1).contiguous()
                    if i_src > 0:
                        fea_list = da_I3Net.get_fea_list(src.permute(0, 2, 3, 1).contiguous(), conf[i_src], self.model.nc)
                        fea_lists.append(fea_list)
                        pre_lists.append(conf[i_src])
                    if i_src == 0:
                        feat2 = conf[i_src]
                        g_feat = da_I3Net.get_feature_vector(feat1, feat2.detach(), self.softmax, self.RandomLayer, self.model.nc)
                self.fea_lists = da_I3Net.Moving_average(fea_lists, self.fea_lists)
                loss_kl = torch.tensor(0)
            if cfg_da.use_domain_adaptation:
                outputs_t = self.model(inps_da_td, targets_da_td)
                itermediate_outputs_t = [None]*assume_tot_layer_num
                itermediate_outputs_t[4] = outputs_t['backbone_features'][0]
                itermediate_outputs_t[15] = outputs_t['head_features'][0]
                itermediate_outputs_t[18] = outputs_t['head_features'][1]
                itermediate_outputs_t[21] = outputs_t['head_features'][2]
                preds_t = outputs_t['output_preds']
                domain_l_t = self.netD_pixel(da_I3Net.grad_reverse(itermediate_outputs_t[4]))
                domain_g_t = self.netD(da_I3Net.grad_reverse(itermediate_outputs_t[15]))
                gcr_pre_t = self.conv_gcr(itermediate_outputs_t[15])
                feat1_t = itermediate_outputs_t[15]

                source_t = self.L2Norm(itermediate_outputs_t[15])
                sources_t.append(source_t)
                sources_t.append(itermediate_outputs_t[18])
                sources_t.append(itermediate_outputs_t[21])

                loc_t.append(preds_t[0][:, :-self.model.nc])
                loc_t.append(preds_t[1][:, :-self.model.nc])
                loc_t.append(preds_t[2][:, :-self.model.nc])

                conf_t.append(preds_t[0][:, -self.model.nc:])
                conf_t.append(preds_t[1][:, -self.model.nc:])
                conf_t.append(preds_t[2][:, -self.model.nc:])

                for i_src_t, src_t in enumerate(sources_t):
                    loc_t[i_src_t] = loc_t[i_src_t].permute(0, 2, 3, 1).contiguous()
                    conf_t[i_src_t] = conf_t[i_src_t].permute(0, 2, 3, 1).contiguous()
                    if i_src_t > 0:
                        fea_list_t = da_I3Net.get_fea_list(src_t.permute(0, 2, 3, 1).contiguous(), conf_t[i_src_t], self.model.nc)
                        fea_lists_t.append(fea_list_t)
                        pre_lists_t.append(conf_t[i_src_t])
                    if i_src_t == 0:
                        feat2_t = conf_t[i_src_t]
                        g_feat_t = da_I3Net.get_feature_vector(feat1_t, feat2_t.detach(), self.softmax, self.RandomLayer, self.model.nc)
                self.fea_lists_t = da_I3Net.Moving_average(fea_lists_t, self.fea_lists_t)
                loss_kl_t = da_I3Net.get_kl_loss(pre_lists_t, self.softmax, self.model.nc)
            if cfg_da.use_domain_adaptation:
                loss_kl_tot = loss_kl + loss_kl_t
                loss_kl_tot *= cfg_da.da_info['kl_weight']

                ind_max_cls = torch.argmax(gcr_pre_t.detach(), 1)
                for i in ind_max_cls:
                    self.new_state[i] += 1
                w1 = da_I3Net.dcbr_w1_weight(gcr_pre_t.sigmoid().detach())
                w2 = torch.exp(1 - self.old_state[ind_max_cls]/img_per_epoch)
                
                if epoch >= cfg_da.da_info['open_all_loss_epoch_idx'] and torch.sum(self.old_state) > 100:
                    weight = (w1+w2)*0.5 
                else:
                    weight = torch.ones(w1.size(0)).to(w1.device)  

                dloss_l = 0.5 * torch.mean(domain_l ** 2) * cfg_da.da_info['dloss_l_weight']
                dloss_g = 0.5 * da_I3Net.weight_ce_loss(domain_g, 0, torch.ones(domain_g.size(0)).to(domain_g.device)) * 0.1

                dloss_l_t = 0.5 * torch.mean((1-domain_l_t) ** 2) * cfg_da.da_info['dloss_l_weight']
                dloss_g_t = 0.5 * da_I3Net.weight_ce_loss(domain_g_t, 1, weight) * cfg_da.da_info['dcbr_weight'] 
                loss_gf = cfg_da.da_info['loss_gf_weight'] * torch.pow(g_feat-g_feat_t, 2.0).mean()
                cls_idxes = batch[:, :, 0].flatten()
                batch_idxes = []
                for i_sample in range(batch.shape[0]):
                    batch_idxes += [i_sample]*batch.shape[1]
                cls_onehot = da_I3Net.gt_classes2cls_onehot(len(gcr_pre), cls_idxes, batch_idxes, self.model.nc)
                cls_onehot = torch.from_numpy(cls_onehot).to(gcr_pre.device)
                loss_gcr = nn.BCEWithLogitsLoss()(gcr_pre, cls_onehot) * cfg_da.da_info['gcr_weight']

        loss = outputs["total_loss"]
        if cfg_da.use_domain_adaptation:
            loss_da = torch.zeros(8, device=self.device)
            loss_da[0] = dloss_l
            loss_da[1] = dloss_l_t
            loss_da[2] = dloss_g
            loss_da[3] = dloss_g_t
            loss_da[4] = loss_gf
            loss_da[5] = loss_gcr
            outputs["dloss_l"] = dloss_l
            outputs["dloss_l_t"] = dloss_l_t
            outputs["dloss_g"] = dloss_g
            outputs["dloss_g_t"] = dloss_g_t
            outputs["loss_gf"] = loss_gf
            outputs["loss_gcr"] = loss_gcr
            if epoch >= cfg_da.da_info['open_all_loss_epoch_idx'] and torch.sum(self.old_state) > 100:
                loss_gpa = da_I3Net.get_pa_losses(fea_lists, fea_lists_t) * cfg_da.da_info['pa_losses_weight']
                loss_da[6] = loss_gpa
                loss_da[7] = loss_kl_tot
                outputs["loss_gpa"] = loss_gpa
                outputs["loss_kl_tot"] = loss_kl_tot
                self.tblogger.add_scalar("train/loss_gpa", loss_gpa, self.epoch + 1)
                self.tblogger.add_scalar("train/loss_kl_tot", loss_kl_tot, self.epoch + 1)
                self.tblogger.add_scalar("train/loss_gpa", loss_gpa, self.epoch + 1)

            loss += loss_da.sum()
            outputs["total_loss"] = loss
            if self.rank == 0:
                if self.args.logger == "tensorboard":
                    self.tblogger.add_scalar("train/total_loss", loss, self.epoch + 1)
                    self.tblogger.add_scalar("train/iou_loss", outputs["iou_loss"], self.epoch + 1)
                    self.tblogger.add_scalar("train/l1_loss", outputs["l1_loss"], self.epoch + 1)
                    self.tblogger.add_scalar("train/conf_loss", outputs["conf_loss"], self.epoch + 1)
                    self.tblogger.add_scalar("train/cls_loss", outputs["cls_loss"], self.epoch + 1)

                    self.tblogger.add_scalar("train/dloss_l", dloss_l, self.epoch + 1)
                    self.tblogger.add_scalar("train/dloss_l_t", dloss_l_t, self.epoch + 1)
                    self.tblogger.add_scalar("train/dloss_g", dloss_g, self.epoch + 1)
                    self.tblogger.add_scalar("train/dloss_g_t", dloss_g_t, self.epoch + 1)
                    self.tblogger.add_scalar("train/loss_gf", loss_gf, self.epoch + 1)
                    self.tblogger.add_scalar("train/loss_gcr", loss_gcr, self.epoch + 1)
            #loss_items = torch.cat((loss_items, loss_da.detach()))

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        self.tblogger.add_scalar("train/lr", lr, self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

        if cfg_da.use_domain_adaptation:
            self.old_state = self.new_state

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)
        # value of epoch will be set in `resume_train`

        if cfg_da.use_domain_adaptation:
            self.L2Norm = da_I3Net.L2Norm(256, cfg_da.da_info['td_cls_num'])
            self.netD_pixel = da_I3Net.netD_pixel()
            self.netD = da_I3Net.netD()
            self.conv_gcr = da_I3Net.net_gcr_simple(256, cfg_da.da_info['td_cls_num'])
            self.RandomLayer = da_I3Net.RandomLayer([256, cfg_da.da_info['td_cls_num']], 1024)
            self.old_state = torch.zeros(self.exp.num_classes)
            self.new_state = torch.zeros(self.exp.num_classes)

            self.softmax = nn.Softmax(dim=-1)
            self.pa_list = cfg_da.da_info['pa_list'] 
            self.fea_lists = [[torch.tensor([]).to(self.device) for _ in range(cfg_da.da_info['td_cls_num'])] for _ in range(len(self.pa_list))]
            self.fea_lists_t = [[torch.tensor([]).to(self.device) for _ in range(cfg_da.da_info['td_cls_num'])] for _ in range(len(self.pa_list))]

        self.model = model
        if cfg_da.use_domain_adaptation:
            model_add_ad = [self.model, self.L2Norm, self.netD_pixel, self.netD, self.RandomLayer, self.conv_gcr]
            self.optimizer = self.exp.get_optimizer(self.args.batch_size, model_list=model_add_ad)
        else:
            # solver related init
            self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        self.model = self.resume_train(self.model)
        self.model = self.model.to(self.device)
        if cfg_da.use_domain_adaptation:
            self.model.nc = self.exp.num_classes
            cfg_da.da_info['td_cls_num'] = self.model.nc

        if cfg_da.use_domain_adaptation:
            self.L2Norm.train()
            self.netD_pixel.train()
            self.netD.train()
            self.conv_gcr.train()
            self.RandomLayer.train()

            self.L2Norm = self.L2Norm.to(self.device)
            self.netD_pixel = self.netD_pixel.to(self.device)
            self.netD = self.netD.to(self.device)
            self.conv_gcr = self.conv_gcr.to(self.device)
            self.RandomLayer = self.RandomLayer.to(self.device)

            self.old_state = self.old_state.to(self.device)
            self.new_state = self.new_state.to(self.device)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)
        if cfg_da.use_domain_adaptation:
            self.da_td_loader = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
                no_aug=True,
                cache_img=self.args.cache,
                for_domain_adaptation=True
            )
            logger.info("init td prefetcher, this might take one minute or less...")
            self.prefetcher_da_td = DataPrefetcher(self.da_td_loader)
            # max_iter means iters per epoch
            self.max_iter_da_td = len(self.train_loader)
            #self.loss_names_train = 'box_loss', 'cls_loss', 'dfl_loss', \
            #    'dloss_l', 'dloss_l_t', 'dloss_g', 'dloss_g_t', 'loss_gf', 'loss_gcr', 'loss_gpa', 'loss_kl_tot'

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(self.model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                # consider self.exp_da?
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(self.model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if self.epoch < cfg.save_epoch_after * self.max_epoch:
            return

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.5f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({
                        "train/lr": self.meter["lr"].latest
                    })
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        self.start_epoch = 0

        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            if cfg.resume_load_optimizer:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                for state_key in self.optimizer.state.keys():
                    self.optimizer.state[state_key]['momentum_buffer'] = self.optimizer.state[state_key]['momentum_buffer'].to(self.device)

            self.best_ap = ckpt.pop("best_ap", 0)
            # resume the training states variables
            if cfg.resume_load_epoch_num:
                start_epoch = (
                    self.args.start_epoch - 1
                    if self.args.start_epoch is not None
                    else ckpt["start_epoch"]
                )
                self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)
                model = load_ckpt(model, ckpt["model"])
            self.start_epoch = 0

        if cfg_da.use_domain_adaptation:
            if "model_L2Norm" in ckpt:
                self.L2Norm.load_state_dict(ckpt["model_L2Norm"])
                self.netD_pixel.load_state_dict(ckpt["model_netD_pixel"])
                self.netD.load_state_dict(ckpt["model_netD"])
                self.conv_gcr.load_state_dict(ckpt["model_conv_gcr"])
                self.RandomLayer.load_state_dict(ckpt["model_RandomLayer"])

            if 'new_state' in ckpt:
                self.old_state = ckpt["old_state"]
                self.new_state = ckpt["new_state"]
        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)
        global last_best_epoch
        if self.epoch == 0:
            last_best_epoch = self.epoch
        if update_best_ckpt:
            last_best_epoch = self.epoch

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "train/epoch": self.epoch + 1,
                })
                self.wandb_logger.log_images(predictions)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

        if last_best_epoch + cfg.epoch_num_wait_for_new_best <= self.epoch:
            print('last_best_epoch:', last_best_epoch)
            print('self.epoch:', self.epoch)
            exit(0)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            if cfg_da.use_domain_adaptation:
                ckpt_state = {
                    "start_epoch": self.epoch + 1,
                    'model': save_model.state_dict(),
                    'model_L2Norm': self.L2Norm.state_dict(),
                    'model_netD_pixel': self.netD_pixel.state_dict(),
                    'model_netD': self.netD.state_dict(),
                    'model_conv_gcr': self.conv_gcr.state_dict(),
                    'model_RandomLayer': self.RandomLayer.state_dict(),
                    'old_state': self.old_state,
                    'new_state': self.new_state,
                    'optimizer': self.optimizer.state_dict(),
                    "best_ap": self.best_ap,
                    "curr_ap": ap
                }
            else:
                ckpt_state = {
                    "start_epoch": self.epoch + 1,
                    "model": save_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_ap": self.best_ap,
                    "curr_ap": ap,
                }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "best_ap": self.best_ap,
                        "curr_ap": ap
                    }
                )
