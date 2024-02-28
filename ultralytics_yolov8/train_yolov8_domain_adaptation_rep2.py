"""
batch_size=16
im_size=800
"""
import config as cfg
cfg.is_val = False

import sys
import time
import torch
import os
import platform
from os.path import join
import config_da as cfg_da

cfg_da.use_domain_adaptation = 1

import shutil

sys_name = platform.system()

if sys_name == "Windows":
    pass
else:
    #sys.path.insert(0, '/mnt/e/PHD/WeedDetection/ultralytics')
    pass

from ultralytics import YOLO
import config_da as cfg_da
import da_I3Net_module as da_I3Net

print('start')
# yolov8s_weed/yolov8l_weed
cfg_file_name = 'yolov8l_weed.yaml'
# best/yolov8s/yolov8l
model_name = 'yolov8l.pt'

if sys_name == "Windows":
    """
    activate pytorch
    cd E:\Repo\Biang\Graphics\WeedDetection
    e:
    python train_yolov8.py
    """
    config_file_dir = r'D:\BoyangDeng\Detection\ultralytics_yolov8\ultralytics\yolo\data\datasets'
    
    # E:\PHD\WeedDetection\CottonWeedDet12
    # E:\PHD\WeedDetection\ultralytics\runs\detect
    #cfg = os.path.join(r"E:\PHD\WeedDetection\ultralytics\ultralytics\models\v8", cfg_file_name)
    #cfg = os.path.join(r"D:\BoyangDeng\WeedDetection\ultralytics\ultralytics\models\v8", cfg_file_name)
    # weed_2022 weed_2022_val_2021 weed_2021_test_all
    #data_dir = r'E:\PHD\WeedDetection\ultralytics\ultralytics\yolo\data\datasets\weed_2021.yaml'
    #data_dir = r'D:\BoyangDeng\WeedDetection\ultralytics\ultralytics\yolo\data\datasets\weed_2021.yaml'
    # data_dir = r'D:\BoyangDeng\Detection\ultralytics_yolov8\ultralytics\yolo\data\datasets\weed_2022.yaml'
    data_dir = join(config_file_dir, 'weed_2022_rep2.yaml')
    
    workers = 0
else:
    """
    conda activate pytorch
    cd /mnt/e/Repo/Biang/Graphics/WeedDetection
    python train_yolov8.py
    windows/linux may cannot share dataset xxx.cache
    Logging results to /mnt/e/Repo/Biang/runs/detect/train
    E:\Repo\Biang\runs\detect
    """
    cfg = os.path.join("/mnt/e/PHD/WeedDetection/ultralytics/ultralytics/models/v8", cfg_file_name)
    data_dir = '/mnt/e/PHD/WeedDetection/ultralytics/ultralytics/yolo/data/datasets/weed_2021_linux.yaml'
    workers = 2


"""
4 trainings
lr0=0.01/0.005/0.001/0.0001
lrf=0.01/0.01/0.01/0.1
warmup_epochs=3/3/3/1
warmup_bias_lr=0.1/0.01/0.01/0.001
optimizer='SGD'/'SGD'/'SGD'/'Adam'

yolov8s/yolov8l
imgsz=800/800
batch=16/8
"""

cfg_da.use_domain_adaptation = True

lr0s = [0.01, 0.001]
lrfs = [0.01, 0.01]
warmup_epochs = [3, 0]
warmup_bias_lrs = [0.1, 0.01]
optimizers = ['SGD', 'SGD']
model_names = ['yolov8l.pt', 'last.pt']
epochs = [20, 20]

# lr0s = [0.01]
# lrfs = [0.01]
# warmup_epochs = [3]
# warmup_bias_lrs = [0.1]
# optimizers = ['SGD']
# model_names = ['yolov8l.pt']
# epochs = [40]
if sys_name == "Windows":
    model_dirs = [
        # D:\BoyangDeng\WeedDetection\ultralytics
        # D:\BoyangDeng\ultralytics_yolov8
        # r'.\pretrained',
        # r'E:\PHD\WeedDetection\ultralytics\pretrained',
        # r'E:\PHD\WeedDetection\ultralytics\pretrained',
        r'D:\BoyangDeng\Detection\ultralytics_yolov8\pretrained',
        r'D:\BoyangDeng\Detection\ultralytics_yolov8/runs/detect/train_rep2_da_2022/weights',
        
        # r'.\runs\detect\train\weights'
    ]
else:
    model_dirs = ['/mnt/e/PHD/WeedDetection/ultralytics/pretrained']

save_dirs = [
    './runs/detect/train_rep2_da_2022',
    './runs/detect/train2_rep2_da_2022',
]

# 10/5
close_mosaics = [0, 10]
imgsz = 800

run_idx = 0
# for lr0, lrf, warmup_epoch, warmup_bias_lr, optimizer, model_name, model_dir in zip(
#         lr0s, lrfs, warmup_epochs, warmup_bias_lrs, optimizers, model_names, model_dirs):
while run_idx <= 1:
    #if run_idx == 0:
    #    run_idx += 1
    #    continue

    close_mosaic =  close_mosaics[run_idx]
    lr0 =  lr0s[run_idx]
    lrf =  lrfs[run_idx]
    warmup_epoch =  warmup_epochs[run_idx]
    warmup_bias_lr =  warmup_bias_lrs[run_idx]
    optimizer =  optimizers[run_idx]
    model_name =  model_names[run_idx]
    model_dir =  model_dirs[run_idx]
    epoch =  epochs[run_idx]
    if run_idx >= 1:
        cfg_da.da_info['open_all_loss_epoch_idx'] = 0
        cfg_da.da_info['lr'] = 0.01
        cfg_da.da_info['net_gcr_lr'] = 0.1

    print('lr0:', lr0)
    print('lrf:', lrf)
    print('warmup_epoch:', warmup_epoch)
    print('warmup_bias_lr:', warmup_bias_lr)
    print('optimizer:', optimizer)
    print('model_name:', model_name)
    print('save_dir:', save_dirs[run_idx])
    if os.path.exists(save_dirs[run_idx]):
        shutil.rmtree(save_dirs[run_idx])
    t0 = time.time()
    model_path = os.path.join(model_dir, model_name)
    print('pretained model_path:', model_path)
    # model = YOLO(cfg)  # build a new model from scratch
    assert os.path.exists(model_path)
    model = YOLO(model_path)
    # 8/4/2
    batch = 8
    print('epoch:', epochs[run_idx])
    print('batch:', batch)

    if cfg_da.use_domain_adaptation:
        #cfg_da.da_info['open_all_loss_epoch_idx'] += warmup_epoch
        print('da open_all_loss_epoch_idx:', cfg_da.da_info['open_all_loss_epoch_idx'])
        print('da lr:', cfg_da.da_info['lr'])
        print('da net_gcr_lr:', cfg_da.da_info['net_gcr_lr'])

    results = model.train(data=data_dir,
                          epochs=epochs[run_idx],
                          imgsz=imgsz,
                          workers=workers,
                          lr0=lr0, lrf=lrf,
                          warmup_bias_lr=warmup_bias_lr, warmup_epochs=warmup_epoch,
                          optimizer=optimizer, batch=batch,
                          close_mosaic=close_mosaic,
                          save_dir=save_dirs[run_idx]
                          )  # train the model
    t1 = time.time()
    print("used time (minute):", (t1 - t0) / 60.0)
    # used time (minute): 164.90711260239283
    del model
    torch.cuda.empty_cache()
    # time.sleep(60)
    run_idx += 1
#results = model.val()
#results = model("https://ultralytics.com/images/bus.jpg")

# Export the model
# model.export(format="onnx")
