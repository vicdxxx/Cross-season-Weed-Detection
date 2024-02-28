# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Train a model on a dataset

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco128.yaml imgsz=640 epochs=100 batch=16
"""
import os
from random import Random
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from tqdm import tqdm

from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, colorstr, emojis, yaml_save)
from ultralytics.yolo.utils.autobatch import check_train_batch_size
from ultralytics.yolo.utils.checks import check_file, check_imgsz, print_args
from ultralytics.yolo.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.yolo.utils.files import get_latest_run, increment_path
from ultralytics.yolo.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle,
                                                select_device, strip_optimizer)
import config_da as cfg_da
import da_I3Net_module as da_I3Net
import config as cfg_tot

class BaseTrainer:
    """
    BaseTrainer

    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        check_resume (method): Method to check if training should be resumed from a saved checkpoint.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to last checkpoint.
        best (Path): Path to best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.check_resume()
        self.validator = None
        self.model = None
        self.metrics = None
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        if hasattr(self.args, 'save_dir'):
            self.save_dir = Path(self.args.save_dir)
        else:
            self.save_dir = Path(
                increment_path(Path(project) / name, exist_ok=self.args.exist_ok if RANK in (-1, 0) else True))
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type == 'cpu':
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataloaders.
        self.model = self.args.model
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.endswith('.yaml') or self.args.task in ('detect', 'segment'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{self.args.data}' error ❌ {e}")) from e

        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """
        Appends the given callback.
        """
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """
        Overrides the existing callbacks with the given callback.
        """
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        # Allow device='', device=None on Multi-GPU systems to default to device=0
        if isinstance(self.args.device, int) or self.args.device:  # i.e. device=0 or device=[0,1,2,3]
            world_size = torch.cuda.device_count()
        elif torch.cuda.is_available():  # i.e. device=None or device=''
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and 'LOCAL_RANK' not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting rect=False")
                self.args.rect = False
            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'Running DDP command {cmd}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._do_train(world_size)

    def _setup_ddp(self, world_size):
        torch.cuda.set_device(RANK)
        self.device = torch.device('cuda', RANK)
        LOGGER.info(f'DDP settings: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        dist.init_process_group('nccl' if dist.is_nccl_available() else 'gloo', rank=RANK, world_size=world_size)

    def _setup_train(self, world_size):
        """
        Builds dataloaders and optimizer on correct rank process.
        """
        # Model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        if cfg_da.use_domain_adaptation:
            cfg_da.da_info['td_cls_num'] = self.model.nc

            self.L2Norm = da_I3Net.L2Norm(cfg_da.da_info['channel_num'] , cfg_da.da_info['td_cls_num'])
            self.netD_pixel = da_I3Net.netD_pixel()
            self.netD = da_I3Net.netD()
            self.conv_gcr = da_I3Net.net_gcr_simple(cfg_da.da_info['channel_num'] , cfg_da.da_info['td_cls_num'])
            self.RandomLayer = da_I3Net.RandomLayer([cfg_da.da_info['channel_num'] , cfg_da.da_info['td_cls_num']], 1024)

            self.old_state = torch.zeros(self.model.nc)
            self.new_state = torch.zeros(self.model.nc)

            if cfg_da.da_info['model_L2Norm'] is not None:
                assert cfg_da.da_info['model_L2Norm'] is not None
                assert cfg_da.da_info['model_netD_pixel'] is not None
                assert cfg_da.da_info['model_netD'] is not None
                assert cfg_da.da_info['model_conv_gcr'] is not None
                assert cfg_da.da_info['model_RandomLayer'] is not None
                #def intersect_dicts(da, db, exclude=()):
                #    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
                #    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}
                #csd = intersect_dicts(csd, cfg_da.da_info['model_L2Norm'].state_dict())  # intersect
                #self.L2Norm.load_state_dict(cfg_da.da_info['model_L2Norm'], strict=False)
                self.L2Norm = cfg_da.da_info['model_L2Norm']
                self.netD_pixel = cfg_da.da_info['model_netD_pixel']
                self.netD = cfg_da.da_info['model_netD']
                self.conv_gcr = cfg_da.da_info['model_conv_gcr']
                self.RandomLayer = cfg_da.da_info['model_RandomLayer']

            if cfg_da.da_info['new_state'] is not None:
                self.old_state = cfg_da.da_info['old_state']
                self.new_state = cfg_da.da_info['new_state']
            # no bg class

            self.old_state = self.old_state.to(self.device)
            self.new_state = self.new_state.to(self.device)

            self.L2Norm = self.L2Norm.to(self.device)
            self.netD_pixel = self.netD_pixel.to(self.device)
            self.netD = self.netD.to(self.device)
            self.conv_gcr = self.conv_gcr.to(self.device)
            self.RandomLayer = self.RandomLayer.to(self.device)

            self.softmax = nn.Softmax(dim=-1)
            self.pa_list = cfg_da.da_info['pa_list'] 
            self.fea_lists = [[torch.tensor([]).to(self.device) for _ in range(cfg_da.da_info['td_cls_num'])] for _ in range(len(self.pa_list))]
            self.fea_lists_t = [[torch.tensor([]).to(self.device) for _ in range(cfg_da.da_info['td_cls_num'])] for _ in range(len(self.pa_list))]

        # Check AMP: Automatic Mixed Precision
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            try:
                callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
                self.amp = torch.tensor(check_amp(self.model), device=self.device)
                if cfg_da.use_domain_adaptation:
                    self.amp_L2Norm = torch.tensor(check_amp(self.L2Norm), device=self.device)
                    self.amp_netD_pixel = torch.tensor(check_amp(self.netD_pixel), device=self.device)
                    self.amp_netD = torch.tensor(check_amp(self.netD), device=self.device)
                    self.amp_conv_gcr = torch.tensor(check_amp(self.conv_gcr), device=self.device)
                    self.amp_RandomLayer = torch.tensor(check_amp(self.RandomLayer), device=self.device)
                    self.amp = self.amp and self.amp_L2Norm and self.amp_netD_pixel and self.amp_netD and self.amp_conv_gcr and self.amp_RandomLayer
                callbacks.default_callbacks = callbacks_backup  # restore callbacks
            except Exception as e:
                self.amp = torch.tensor(False).to(self.device)
                print(e)
            print('self.amp:', self.amp)

        if RANK > -1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        # Batch size
        if self.batch_size == -1:
            if RANK == -1:  # single-GPU only, estimate best batch size
                self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)
            else:
                SyntaxError('batch=-1 to use AutoBatch is only available in Single-GPU training. '
                            'Please pass a valid batch size value for Multi-GPU DDP training, i.e. batch=16')

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay

        if cfg_da.use_domain_adaptation:
            #model_add_ad = nn.ModuleList([self.model, self.L2Norm, self.netD_pixel, self.netD, self.conv_gcr, self.RandomLayer])
            model_add_ad = [self.model, self.L2Norm, self.netD_pixel, self.netD, self.RandomLayer, self.conv_gcr]
            self.optimizer = self.build_optimizer(model=model_add_ad,
                                                  name=self.args.optimizer,
                                                  lr=self.args.lr0,
                                                  momentum=self.args.momentum,
                                                  decay=weight_decay)
        else:
            self.optimizer = self.build_optimizer(model=self.model,
                                                  name=self.args.optimizer,
                                                  lr=self.args.lr0,
                                                  momentum=self.args.momentum,
                                                  decay=weight_decay)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False

        # dataloaders
        batch_size = self.batch_size // world_size if world_size > 1 else self.batch_size
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if cfg_da.use_domain_adaptation:
            self.da_td_loader = self.get_dataloader(
                cfg_da.da_info['td_dataset_train'], batch_size=batch_size, rank=RANK, mode='train', domain_adaptation=True)
            self.loss_names_train = 'box_loss', 'cls_loss', 'dfl_loss', \
                'dloss_l', 'dloss_l_t', 'dloss_g', 'dloss_g_t', 'loss_gf', 'loss_gcr', 'loss_gpa', 'loss_kl_tot'
        else:
            self.loss_names_train = 'box_loss', 'cls_loss', 'dfl_loss'

        if RANK in (-1, 0):
            # batch_size = batch_size * 2 / batch_size
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
            self.ema = ModelEMA(self.model)
            if self.args.plots and not self.args.v5loader:
                self.plot_training_labels()
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)

        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        if cfg_da.use_domain_adaptation:
            nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        else:
            nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        img_per_epoch = nb * self.batch_size

        #self.metrics, self.fitness = self.validate()

        if cfg_da.use_domain_adaptation:
            da_td_batch_iterator = iter(self.da_td_loader)

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()

            if cfg_da.use_domain_adaptation:
                self.L2Norm.train()
                self.netD_pixel.train()
                self.netD.train()
                self.conv_gcr.train()
                self.RandomLayer.train()

            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string(self.loss_names_train))
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()

            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        if cfg_da.use_domain_adaptation:
                            if j > 2:
                                x['lr'] = 0
                                #break
                        x['lr'] = np.interp(ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                if cfg_da.use_domain_adaptation:
                    try:
                        batch_t = next(da_td_batch_iterator)
                    except StopIteration:
                        print('StopIteration: da_td_batch_iterator, reinitiateing')
                        da_td_batch_iterator = iter(self.da_td_loader)
                        batch_t = next(da_td_batch_iterator)

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

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    #print(batch['im_file'][0])
                    #cfg_tot.show_image_with_bbox(batch['img'][0], rgb2bgr=1)
                    preds = self.model(batch['img'])
                    if cfg_da.use_domain_adaptation:
                        # [4, 6, 9, 12, 15, 18, 21]
                        preds, itermediate_outputs = preds
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
                        batch_t = self.preprocess_batch(batch_t)
                        #print(batch_t['im_file'][0])
                        #cfg_tot.show_image_with_bbox(batch_t['img'][0], rgb2bgr=1)
                        preds_t = self.model(batch_t['img'])
                        preds_t, itermediate_outputs_t = preds_t
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
                        loss_gf = 38 * torch.pow(g_feat-g_feat_t, 2.0).mean()

                        cls_onehot = da_I3Net.gt_classes2cls_onehot(len(gcr_pre), batch['cls'], batch['batch_idx'], self.model.nc)
                        cls_onehot = torch.from_numpy(cls_onehot).to(gcr_pre.device)
                        loss_gcr = nn.BCEWithLogitsLoss()(gcr_pre, cls_onehot) * cfg_da.da_info['gcr_weight']
                    # loss.sum() * batch_size, loss.detach()
                    self.loss, self.loss_items = self.criterion(preds, batch)

                    if cfg_da.use_domain_adaptation:
                        loss_da = torch.zeros(8, device=self.device)
                        loss_da[0] = dloss_l
                        loss_da[1] = dloss_l_t
                        loss_da[2] = dloss_g
                        loss_da[3] = dloss_g_t
                        loss_da[4] = loss_gf
                        loss_da[5] = loss_gcr
                        if epoch >= cfg_da.da_info['open_all_loss_epoch_idx'] and torch.sum(self.old_state) > 100:
                            loss_gpa = da_I3Net.get_pa_losses(fea_lists, fea_lists_t) * cfg_da.da_info['pa_losses_weight']
                            loss_da[6] = loss_gpa
                            loss_da[7] = loss_kl_tot
                        self.loss += loss_da.sum()
                        self.loss_items = torch.cat((self.loss_items, loss_da.detach()))

                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

                if cfg_da.use_domain_adaptation:
                    self.old_state = self.new_state
                    #print(self.old_state.cpu().numpy())

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                try:
                    if self.args.val or final_epoch:
                        self.metrics, self.fitness = self.validate()
                    self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                except Exception as e:
                    print(e)

                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def save_model(self):
        if cfg_da.use_domain_adaptation:
            ckpt = {
                'epoch': self.epoch,
                'best_fitness': self.best_fitness,
                'model': deepcopy(de_parallel(self.model)).half(),
                'model_L2Norm': deepcopy(de_parallel(self.L2Norm)).half(),
                'model_netD_pixel': deepcopy(de_parallel(self.netD_pixel)).half(),
                'model_netD': deepcopy(de_parallel(self.netD)).half(),
                'model_conv_gcr': deepcopy(de_parallel(self.conv_gcr)).half(),
                'model_RandomLayer': deepcopy(de_parallel(self.RandomLayer)).half(),
                'old_state': deepcopy(de_parallel(self.old_state)).half(),
                'new_state': deepcopy(de_parallel(self.new_state)).half(),
                'ema': deepcopy(self.ema.ema).half(),
                'updates': self.ema.updates,
                'optimizer': self.optimizer.state_dict(),
                'train_args': vars(self.args),  # save as dict
                'date': datetime.now().isoformat(),
                'version': __version__}
        else:
            ckpt = {
                'epoch': self.epoch,
                'best_fitness': self.best_fitness,
                'model': deepcopy(de_parallel(self.model)).half(),
                'ema': deepcopy(self.ema.ema).half(),
                'updates': self.ema.updates,
                'optimizer': self.optimizer.state_dict(),
                'train_args': vars(self.args),  # save as dict
                'date': datetime.now().isoformat(),
                'version': __version__}

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
        del ckpt

    @staticmethod
    def get_dataset(data):
        """
        Get train, val path from data dict if it exists. Returns None if data format is not recognized.
        """
        return data['train'], data.get('val') or data.get('test')

    def setup_model(self):
        """
        load/create/download model for any task.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = ckpt['model'].yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        if cfg_da.use_domain_adaptation:
            torch.nn.utils.clip_grad_norm_(self.L2Norm.parameters(), max_norm=10.0)  # clip gradients
            torch.nn.utils.clip_grad_norm_(self.netD_pixel.parameters(), max_norm=10.0)  # clip gradients
            torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=10.0)  # clip gradients
            torch.nn.utils.clip_grad_norm_(self.conv_gcr.parameters(), max_norm=10.0)  # clip gradients
            torch.nn.utils.clip_grad_norm_(self.RandomLayer.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """
        Allows custom preprocessing model inputs and ground truths depending on task type.
        """
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
        try:
            metrics = self.validator(self)
            fitness = metrics.pop('fitness', -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
            if not self.best_fitness or self.best_fitness < fitness:
                self.best_fitness = fitness
        except Exception as e:
            print(e)
            fitness = 0
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        raise NotImplementedError('get_validator function not implemented in trainer')

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train', domain_adaptation=False):
        """
        Returns dataloader derived from torch.data.Dataloader.
        """
        raise NotImplementedError('get_dataloader function not implemented in trainer')

    def criterion(self, preds, batch):
        """
        Returns loss and individual loss items as Tensor.
        """
        raise NotImplementedError('criterion function not implemented in trainer')

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        return {'loss': loss_items} if loss_items is not None else ['loss']

    def set_model_attributes(self):
        """
        To set or update model parameters before training.
        """
        self.model.names = self.data['names']

    def build_targets(self, preds, targets):
        pass

    def progress_string(self, loss_names):
        return ''

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        pass

    def plot_training_labels(self):
        pass

    def save_metrics(self, metrics):
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
        with open(self.csv, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch] + vals)).rstrip(',') + '\n')

    def plot_metrics(self):
        pass

    def final_eval(self):
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f'\nValidating {f}...')
                    self.metrics = self.validator(model=f)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')

    def check_resume(self):
        resume = self.args.resume
        if resume:
            try:
                last = Path(
                    check_file(resume) if isinstance(resume, (str,
                                                              Path)) and Path(resume).exists() else get_latest_run())
                self.args = get_cfg(attempt_load_weights(last).args)
                self.args.model, resume = str(last), True  # reinstate
            except Exception as e:
                raise FileNotFoundError('Resume checkpoint not found. Please pass a valid checkpoint to resume from, '
                                        "i.e. 'yolo train resume model=path/to/last.pt'") from e
        self.resume = resume

    def resume_training(self, ckpt):
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt['epoch'] + 1
        if ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
            best_fitness = ckpt['best_fitness']
        if self.ema and ckpt.get('ema'):
            self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
            self.ema.updates = ckpt['updates']
        if self.resume:
            assert start_epoch > 0, \
                f'{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n' \
                f"Start a new training without --resume, i.e. 'yolo task=... mode=train model={self.args.model}'"
            LOGGER.info(
                f'Resuming training from {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs')
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
            self.epochs += ckpt['epoch']  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            LOGGER.info('Closing dataloader mosaic')
            if hasattr(self.train_loader.dataset, 'mosaic'):
                self.train_loader.dataset.mosaic = False
            if hasattr(self.train_loader.dataset, 'close_mosaic'):
                self.train_loader.dataset.close_mosaic(hyp=self.args)

    @staticmethod
    def build_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
        """
        Builds an optimizer with the specified parameters and parameter groups.

        Args:
            model (nn.Module): model to optimize
            name (str): name of the optimizer to use
            lr (float): learning rate
            momentum (float): momentum
            decay (float): weight decay

        Returns:
            optimizer (torch.optim.Optimizer): the built optimizer
        """
        g = [], [], []  # optimizer parameter groups
        ad = [], [], []  # optimizer parameter groups
        adnet_gcr = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        if cfg_da.use_domain_adaptation:
            for i_module in range(len(model)):
                if i_module == 0:
                    for v in model[i_module].modules():
                        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                            g[2].append(v.bias)
                        if isinstance(v, bn):  # weight (no decay)
                            g[1].append(v.weight)
                        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                            g[0].append(v.weight)
                elif i_module == -1:
                    for v in model[i_module].modules():
                        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                            adnet_gcr[2].append(v.bias)
                        if isinstance(v, bn):  # weight (no decay)
                            adnet_gcr[1].append(v.weight)
                        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                            adnet_gcr[0].append(v.weight)
                else:
                    for v in model[i_module].modules():
                        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                            ad[2].append(v.bias)
                        if isinstance(v, bn):  # weight (no decay)
                            ad[1].append(v.weight)
                        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                            ad[0].append(v.weight)
        else:
            for v in model.modules():
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                    g[2].append(v.bias)
                if isinstance(v, bn):  # weight (no decay)
                    g[1].append(v.weight)
                elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                    g[0].append(v.weight)

        if name == 'Adam':
            optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented.')
        if cfg_da.use_domain_adaptation:
            optimizer.add_param_group({'params': g[0], 'weight_decay': decay, 'lr': lr, 'momentum': momentum})  # add g0 with weight_decay
            optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0, 'lr': lr, 'momentum': momentum})  # add g1 (BatchNorm2d weights)

            optimizer.add_param_group({'params': ad[2], 'weight_decay': 0.0, 'lr': cfg_da.da_info['lr'], 'momentum': momentum, 'nesterov': True})
            optimizer.add_param_group({'params': ad[0], 'weight_decay': decay, 'lr': cfg_da.da_info['lr'], 'momentum': momentum, 'nesterov': True})
            optimizer.add_param_group({'params': ad[1], 'weight_decay': 0.0, 'lr': cfg_da.da_info['lr'], 'momentum': momentum, 'nesterov': True})

            optimizer.add_param_group({'params': adnet_gcr[2], 'weight_decay': 0.0, 'lr': cfg_da.da_info['net_gcr_lr'], 'momentum': momentum, 'nesterov': True})
            optimizer.add_param_group({'params': adnet_gcr[0], 'weight_decay': decay, 'lr': cfg_da.da_info['net_gcr_lr'], 'momentum': momentum, 'nesterov': True})
            optimizer.add_param_group({'params': adnet_gcr[1], 'weight_decay': 0.0, 'lr': cfg_da.da_info['net_gcr_lr'], 'momentum': momentum, 'nesterov': True})
        else:
            optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
            optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                    f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias')
        return optimizer


def check_amp(model):
    """
    This function checks the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLOv8 model.
    If the checks fail, it means there are anomalies with AMP on the system that may cause NaN losses or zero-mAP
    results, so AMP will be disabled during training.

    Args:
        model (nn.Module): A YOLOv8 model instance.

    Returns:
        bool: Returns True if the AMP functionality works correctly with YOLOv8 model, else False.

    Raises:
        AssertionError: If the AMP checks fail, indicating anomalies with the AMP functionality on the system.
    """
    device = next(model.parameters()).device  # get model device
    if device.type in ('cpu', 'mps'):
        return False  # AMP only used on CUDA devices

    def amp_allclose(m, im):
        # All close FP32 vs AMP results
        a = m(im, device=device, verbose=False)[0].boxes.boxes  # FP32 inference
        with torch.cuda.amp.autocast(True):
            b = m(im, device=device, verbose=False)[0].boxes.boxes  # AMP inference
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # close to 0.5 absolute tolerance

    f = ROOT / 'assets/bus.jpg'  # image to check
    im = f if f.exists() else 'https://ultralytics.com/images/bus.jpg' if ONLINE else np.ones((640, 640, 3))
    prefix = colorstr('AMP: ')
    LOGGER.info(f'{prefix}running Automatic Mixed Precision (AMP) checks with YOLOv8n...')
    try:
        from ultralytics import YOLO
        assert amp_allclose(YOLO('yolov8n.pt'), im)
        LOGGER.info(f'{prefix}checks passed ✅')
    except ConnectionError:
        LOGGER.warning(f"{prefix}checks skipped ⚠️, offline and unable to download YOLOv8n. Setting 'amp=True'.")
    except AssertionError:
        LOGGER.warning(f'{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to '
                       f'NaN losses or zero-mAP results, so AMP will be disabled during training.')
        return False
    return True
