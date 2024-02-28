"""
batch_size=16
im_size=800
"""
import config as cfg
cfg.is_val = False

import config_da as cfg_da
from ultralytics import YOLO
import sys
import time
import torch
import os
import platform
import shutil
import config as cfg
from os.path import join

cfg_da.use_domain_adaptation = 0

sys_name = platform.system()

target_name = cfg.target_name

# self station
PC_name = 'station'

for i in range(1, len(sys.argv)):
    print('argument:', i, 'value:', sys.argv[i])
if len(sys.argv) > 1:
    PC_name = sys.argv[1]

single_cls = False
max_det = cfg.max_det

if sys_name == "Windows":
    # sys.path.insert(0, r'D:\BoyangDeng\ultralytics_yolov8')
    # sys.path.insert(0, r'E:\PHD\WeedDetection\ultralytics_old\ultralytics')
    pass
else:
    sys.path.insert(0, '/mnt/e/PHD/WeedDetection/ultralytics')


if target_name == 'weed2021' or target_name == 'weed2022':
    # yolov8s_weed/yolov8l_weed
    cfg_file_name = 'yolov8s_weed.yaml'
else:
    cfg_file_name = None
# best/yolov8s/yolov8l
# model_name = 'yolov8l.pt'

if sys_name == "Windows":
    """
    activate pytorch
    cd E:\Repo\Biang\Graphics\WeedDetection
    e:
    python train_yolov8.py
    """
    # E:\PHD\WeedDetection\CottonWeedDet12
    # E:\PHD\WeedDetection\ultralytics\runs\detect
    if cfg_file_name is not None:
        if PC_name == 'self':
            cfg_model = os.path.join(r"E:\PHD\WeedDetection\ultralytics\ultralytics\models\v8", cfg_file_name)

    config_file_dir = r'D:\BoyangDeng\Detection\ultralytics_yolov8\ultralytics\yolo\data\datasets'
    if target_name == 'weed2021':
        # weed_2021_test_all
        data_dir = join(config_file_dir, 'weed_2021.yaml')
    elif target_name == 'weed2022':
        # weed_2022_val_2021
        data_dir = join(config_file_dir, 'weed_2022.yaml')
    elif target_name == 'weed2023':
        data_dir = join(config_file_dir, 'weed_2023.yaml')
    elif target_name == 'weed_all':
        data_dir = join(config_file_dir, 'weed_all.yaml')
    elif target_name == 'weed_lambsquarters':
        data_dir = join(config_file_dir, 'weed_lambsquarters.yaml')
    elif target_name == 'weed_stable_diffusion':
        data_dir = join(config_file_dir, 'weed_stable_diffusion.yaml')
        
    elif target_name == 'blueberry_dense':
        data_dir = join(config_file_dir, 'dense_blueberry.yaml')
    elif target_name == 'cottonboll_dense':
        data_dir = join(config_file_dir, 'dense_cotton_boll.yaml')
    elif target_name == 'COCO':
        data_dir = join(config_file_dir, '7sku_original_im_to_generation_train.yaml')
    elif target_name == 'weed2021_3sku':
        data_dir = join(config_file_dir, 'weed_3sku_synthetic_im_train.yaml')

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
    cfg_model = os.path.join("/mnt/e/PHD/WeedDetection/ultralytics/ultralytics/models/v8", cfg_file_name)

    model_dir = '/mnt/e/PHD/WeedDetection/ultralytics/pretrained'

    if target_name == 'weed2021':
        data_dir = '/mnt/e/PHD/WeedDetection/ultralytics/ultralytics/yolo/data/datasets/weed_2021_linux.yaml'
    elif target_name == 'weed2022':
        data_dir = '/mnt/e/PHD/WeedDetection/ultralytics/ultralytics/yolo/data/datasets/weed_2022_linux.yaml'

    workers = 2
    # train_num=3
    # model_dir = '/mnt/e/Repo/Biang/runs/detect/train'+str(train_num)+'/weights'
    # model_dir = '/mnt/e/PHD/WeedDetection/ultralytics/runs/detect/train'+str(train_num)+'/weights'


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
if target_name == 'weed2021' or target_name == 'weed2022':
    # lr0s = [0.01, 0.005, 0.001, 0.0001]
    lr0s = [0.01]
    # lrfs = [0.01, 0.01, 0.01, 0.1]
    lrfs = [0.01]
    # warmup_epochs = [3, 3, 3, 1]
    # warmup_epochs = [3, 0, 0, 0]
    warmup_epochs = [3]
    # warmup_epochs = [0]*4
    # warmup_bias_lrs = [0.1, 0.01, 0.01, 0.001]
    warmup_bias_lrs = [0.1]
    # optimizers = ['SGD', 'SGD', 'SGD', 'Adam']
    optimizers = ['SGD']
    # model_names = ['yolov8l.pt', 'best.pt', 'best.pt', 'best.pt']
    model_names = ['yolov8l.pt']
    # epochs = [10] * 4
    epochs = [40]
    # epochs = [100] 
    # 10/5
    close_mosaic = 10
    imgsz = 800
    # 8/4/2
    batch = 8
    if sys_name == "Windows":
        if PC_name == 'self':
            model_dirs = [
                r'E:\PHD\WeedDetection\ultralytics\pretrained',
                # r'E:\PHD\WeedDetection\ultralytics\runs\detect\train\weights',
                # r'E:\PHD\WeedDetection\ultralytics\runs\detect\train' + str(2) + '\weights',
                # r'E:\PHD\WeedDetection\ultralytics\runs\detect\train' + str(3) + '\weights',
            ]
        else:
            model_dirs = [
                r'D:\BoyangDeng\Detection\ultralytics_yolov8\pretrained',
                # r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\train\weights',
                # r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\train' + str(2) + '\weights',
                # r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\train' + str(3) + '\weights',
            ]
    else:
        if PC_name == 'self':
            model_dirs = ['/mnt/e/PHD/WeedDetection/ultralytics/pretrained',
                          # '/mnt/e/PHD/WeedDetection/ultralytics/runs/detect/train/weights',
                          # '/mnt/e/PHD/WeedDetection/ultralytics/runs/detect/train'+str(2)+'\weights',
                          # '/mnt/e/PHD/WeedDetection/ultralytics/runs/detect/train'+str(3)+'\weights',
                          '/mnt/e/Repo/Biang/runs/detect/train/weights',
                          '/mnt/e/Repo/Biang/runs/detect/train' + str(2) + '/weights',
                          '/mnt/e/Repo/Biang/runs/detect/train' + str(3) + '/weights',
                          ]
        else:
            assert 0
    save_dirs = [
        './runs/detect/weed_rep5_2022',
        # './runs/detect/train2',
        # './runs/detect/train3',
        # './runs/detect/train4',
    ]


if target_name == 'blueberry_dense' or target_name == 'cottonboll_dense':
    lr0s = [0.01, 0.001]
    lrfs = [0.01, 0.01]
    # warmup_epochs = [3, 3, ]
    warmup_epochs = [3, 0, ]
    warmup_bias_lrs = [0.1, 0.1]
    optimizers = ['SGD', 'SGD']
    model_names = ['yolov8l.pt', 'best.pt']
    # model_names = ['yolov8x.pt', 'best.pt']
    # model_names = ['best.pt', 'best.pt']
    epochs = [100] * 2
    close_mosaic = 50
    # epochs = [50] * 2
    # close_mosaic = 25

    # not good for dense scene?
    # copy_past = 0.3
    # mixup = 0.3
    # mosaic = 0.3

    # 1920 3520
    imgsz = 3520
    # imgsz = 800
    # imgsz = 1920
    # imgsz = 256
    if cfg.train_roi:
        imgsz = 448

    single_cls = False
    max_det = 5000
    # max_det = 3000
    # max_det = 2000
    # max_det = 100
    if cfg.train_roi:
        max_det = 500

    # 4/1/32/512/
    batch = 1
    if cfg.train_roi:
        batch = 100

    if PC_name == 'self':
        model_dirs = [
            r'E:\PHD\WeedDetection\ultralytics\pretrained',
            # r'E:\PHD\WeedDetection\ultralytics\runs\detect\dense_blueberry\sample120\replication1_2_skus\train2\weights',
            r'E:\PHD\WeedDetection\ultralytics\runs\detect\train\weights',
        ]
    else:
        model_dirs = [
            r'D:\BoyangDeng\ultralytics_yolov8\pretrained',
            # r'D:\BoyangDeng\WeedDetection\ultralytics\runs\detect\dense_blueberry\sample120\replication1_\train2\weights',
            r'D:\BoyangDeng\ultralytics_yolov8\runs\detect\train\weights'
        ]
    save_dirs = [
        './runs/detect/train',
        './runs/detect/train2',
    ]


if target_name == 'weed2023':
    lr0s = [0.01, 0.001]
    lrfs = [0.01, 0.01]
    warmup_epochs = [3, 0, ]
    warmup_bias_lrs = [0.1, 0.1]
    optimizers = ['SGD', 'SGD']
    # model_names = ['yolov8l.pt', 'best.pt']
    model_names = ['yolov8s.pt', 'best.pt']
    epochs = [20] * 2
    close_mosaic = 10
    imgsz = 1920
    max_det = 1000
    batch = 16
    if PC_name == 'self':
        pass
    else:
        model_dirs = [
            r'D:\BoyangDeng\Detection\ultralytics_yolov8\pretrained',
            r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\train\weights'
        ]
    save_dirs = [
        './runs/detect/train',
        './runs/detect/train2',
    ]


if target_name == 'weed_all':
    # lr0s = [0.01, 0.001]
    # lrfs = [0.01, 0.01]
    lr0s = [0.01,]
    lrfs = [0.01,]
    # warmup_epochs = [3, 0, ]
    warmup_epochs = [3,]
    # warmup_bias_lrs = [0.1, 0.1]
    warmup_bias_lrs = [0.1, ]
    optimizers = ['SGD', 'SGD']
    model_names = ['yolov8l.pt', 'best.pt']
    # model_names = ['yolov8x.pt', 'best.pt']
    epochs = [48]
    # epochs = [20]*2
    close_mosaic = 10
    # imgsz = 1920
    # imgsz = 1560
    # imgsz = 960
    imgsz = 640
    max_det = 1000
    # batch = 16
    # batch = 8
    batch = 2
    if PC_name == 'self':
        pass
    else:
        model_dirs = [
            r'D:\BoyangDeng\Detection\ultralytics_yolov8\pretrained',
            # r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\train\weights'
        ]
    save_dirs = [
        './runs/detect/real_and_synthetic_MorningGlory',
        # './runs/detect/train2',
    ]


if target_name == 'weed_lambsquarters':
    # lr0s = [0.01, 0.001]
    lr0s = [0.01,]
    # lrfs = [0.01, 0.01]
    lrfs = [0.01, ]
    # warmup_epochs = [3, 0, ]
    warmup_epochs = [3, ]
    # warmup_bias_lrs = [0.1, 0.1]
    warmup_bias_lrs = [0.1,]
    # optimizers = ['SGD', 'SGD']
    optimizers = ['SGD']
    # model_names = ['yolov8l.pt', 'best.pt']
    model_names = ['yolov8l.pt']
    # model_names = ['yolov8s.pt', 'best.pt']
    # epochs = [20] * 2
    # epochs = [40] 
    epochs = [24] 
    close_mosaic = 10
    imgsz = 800
    max_det = 1000
    # batch = 16
    batch = 8
    if PC_name == 'self':
        pass
    else:
        model_dirs = [
            r'D:\BoyangDeng\Detection\ultralytics_yolov8\pretrained',
            # r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\train\weights'
        ]
    save_dirs = [
        './runs/detect/train2017_lambsquaters_real_and_generation',
        # './runs/detect/train',
        # './runs/detect/train2',
    ]

if target_name == 'weed_stable_diffusion':
    lr0s = [0.01, 0.001]
    lrfs = [0.01, 0.01]
    warmup_epochs = [3, 0, ]
    warmup_bias_lrs = [0.1, 0.1]
    optimizers = ['SGD', 'SGD']
    model_names = ['yolov8l.pt', 'best.pt']
    # model_names = ['yolov8s.pt', 'best.pt']
    epochs = [20] * 2
    close_mosaic = 10
    imgsz = 800
    max_det = 1000
    # batch = 16
    batch = 8
    if PC_name == 'self':
        pass
    else:
        model_dirs = [
            r'D:\BoyangDeng\Detection\ultralytics_yolov8\pretrained',
            r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\train\weights'
        ]
    save_dirs = [
        './runs/detect/train',
        './runs/detect/train2',
    ]

if target_name == 'COCO':
    # lr0s = [0.01, 0.001]
    lr0s = [0.001,]
    # lrfs = [0.01, 0.01]
    lrfs = [0.01, ]
    # warmup_epochs = [3, 0, ]
    warmup_epochs = [3, ]
    # warmup_bias_lrs = [0.1, 0.1]
    warmup_bias_lrs = [0.1,]
    # optimizers = ['SGD', 'SGD']
    optimizers = ['SGD']
    # model_names = ['yolov8l.pt', 'best.pt']
    model_names = ['yolov8l.pt']
    # model_names = ['yolov8s.pt', 'best.pt']
    # epochs = [20] * 2
    epochs = [12] 
    close_mosaic = 3
    imgsz = 800
    max_det = 1000
    # batch = 16
    batch = 8
    if PC_name == 'self':
        pass
    else:
        model_dirs = [
            r'D:\BoyangDeng\Detection\ultralytics_yolov8\pretrained',
            # r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\train\weights'
        ]
    save_dirs = [
        './runs/detect/7sku_original_im_to_generation_train_person',
        # './runs/detect/train',
        # './runs/detect/train2',
    ]
    
if target_name == 'weed2021_3sku':
    # lr0s = [0.01, 0.001]
    lr0s = [0.001,]
    # lrfs = [0.01, 0.01]
    lrfs = [0.01, ]
    # warmup_epochs = [3, 0, ]
    warmup_epochs = [3, ]
    # warmup_bias_lrs = [0.1, 0.1]
    warmup_bias_lrs = [0.1,]
    # optimizers = ['SGD', 'SGD']
    optimizers = ['SGD']
    # model_names = ['yolov8l.pt', 'best.pt']
    model_names = ['yolov8l.pt']
    # model_names = ['yolov8s.pt', 'best.pt']
    # epochs = [20] * 2
    epochs = [12] 
    close_mosaic = 3
    imgsz = 800
    max_det = 1000
    # batch = 16
    batch = 8
    if PC_name == 'self':
        pass
    else:
        model_dirs = [
            r'D:\BoyangDeng\Detection\ultralytics_yolov8\pretrained',
            # r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\train\weights'
        ]
    save_dirs = [
        './runs/detect/pose_person_link',
        # './runs/detect/train',
        # './runs/detect/train2',
    ]
  
print('cfg.train_roi:', cfg.train_roi)
run_idx = 0
# for lr0, lrf, warmup_epoch, warmup_bias_lr, optimizer, model_name, model_dir, epoch in zip(
#         lr0s, lrfs, warmup_epochs, warmup_bias_lrs, optimizers, model_names, model_dirs, epochs):
while run_idx <=0:
    # if run_idx != 1:
    # if run_idx < 1:
    #    run_idx += 1
    #    continue
    lr0 =  lr0s[run_idx]
    lrf =  lrfs[run_idx]
    warmup_epoch =  warmup_epochs[run_idx]
    warmup_bias_lr =  warmup_bias_lrs[run_idx]
    optimizer =  optimizers[run_idx]
    model_name =  model_names[run_idx]
    model_dir =  model_dirs[run_idx]
    epoch =  epochs[run_idx]
   
    # if run_idx >= 1:
    #     cfg_da.da_info['open_all_loss_epoch_idx'] = 0
    #     cfg_da.da_info['lr'] = 0.01
    #     cfg_da.da_info['net_gcr_lr'] = 0.1

    print('lr0:', lr0)
    print('lrf:', lrf)
    print('warmup_epoch:', warmup_epoch)
    print('warmup_bias_lr:', warmup_bias_lr)
    print('optimizer:', optimizer)
    print('model_name:', model_name)
    print('epoch:', epoch)
    print('save_dir:', save_dirs[run_idx])
    if os.path.exists(save_dirs[run_idx]):
        shutil.rmtree(save_dirs[run_idx])

    t0 = time.time()
    model_path = os.path.join(model_dir, model_name)
    print('pretained model_path:', model_path)
    # model = YOLO(cfg_model)  # build a new model from scratch
    assert os.path.exists(model_path)
    model = YOLO(model_path)
    print('batch:', batch)
    print('single_cls:', single_cls)
    print('max_det:', max_det)
    print('imgsz:', imgsz)
    results = model.train(data=data_dir,
                          epochs=epoch,
                          imgsz=imgsz,
                          workers=workers,
                          lr0=lr0, lrf=lrf,
                          warmup_bias_lr=warmup_bias_lr, warmup_epochs=warmup_epoch,
                          optimizer=optimizer, batch=batch,
                          close_mosaic=close_mosaic,
                          save_dir=save_dirs[run_idx],
                          single_cls=single_cls,
                          max_det=max_det,
                          )  # train the model
    t1 = time.time()
    print("used time (minute):", (t1 - t0) / 60.0)
    # used time (minute): 164.90711260239283
    del model
    torch.cuda.empty_cache()
    # time.sleep(60)
    run_idx += 1
# results = model.val()
# results = model("https://ultralytics.com/images/bus.jpg")

# Export the model
# model.export(format="onnx")

if 0:
    cfg_root_dir = r'E:\PHD\WeedDetection\ultralytics\ultralytics\yolo\data\datasets'
    cfg_names = ['weed_2021_test.yaml', 'weed_2022_test.yaml']
    train_num = 4
    model_dir = r'E:\PHD\WeedDetection\ultralytics\runs\detect\train' + str(train_num) + '\weights'
    model_path = os.path.join(model_dir, 'best.pt')

    model = YOLO(model_path)

    for cfg_name in cfg_names:
        data_dir = os.path.join(cfg_root_dir, cfg_name)
        print(data_dir)
        results = model.val(data=data_dir, workers=0, batch=16)

cfg.notify_by_email()


