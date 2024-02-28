conda activate pytorch
cd /mnt/e/PHD/WeedDetection/YOLOX
cd /mnt/d/BoyangDeng/WeedDetection/YOLOX

pip3 install -v -e .

demo
python tools/demo.py image -f exps/default/yolox_l.py -c pretrained/yolox_l.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
# CottonWeedDet12 WeedData2022
# E:\PHD\WeedDetection\YOLOX\datasets\CottonWeedDet12 E:\PHD\WeedDetection\CottonWeedDet12
# D:\BoyangDeng\WeedDetection\YOLOX\datasets\WeedData2022 D:\Dataset\WeedData\WeedData2022
# train2017_8cls train2017_8cls.0 train2017_8cls.0.subsample
# val2017_8cls val2017_8cls.0 val2017_8cls.0.subsample
# test2017_8cls test2017_8cls.0 test2017_8cls.0.subsample
# E:\PHD\WeedDetection\WeedData2022\annotations
# E:\PHD\WeedDetection\dataset\WeedData2022\annotations
# instances_train2017 instances_val2017 instances_test2017

yolox_l_weed.py
train
# -b 16 / 4
GPU 11G
python tools/train.py -f exps/custom/yolox_l_weed_2022.py -d 1 -b 4 --fp16 -o -c pretrained/yolox_l.pth --cache (OOM)
python tools/train.py -f exps/custom/yolox_l_weed_2022.py -d 1 -b 4 --fp16 -o -c pretrained/yolox_l.pth
python tools/train_domain_adaptation.py -f exps/custom/yolox_l_weed_2022.py -d 1 -b 4 --fp16 -o -c pretrained/yolox_l.pth
GPU 49G
python tools/train.py -f exps/custom/yolox_l_weed_2022.py -d 1 -b 16 --fp16 -o -c pretrained/yolox_l.pth --cache
python tools/train_domain_adaptation.py -f exps/custom/yolox_l_weed_2022_rep3.py --start_epoch 30 --resume YOLOX_outputs/yolox_l_weed_2022_rep3/best_ckpt.pth -d 1 -b 1 --fp16 -o -c pretrained/yolox_l.pth

python tools/train_domain_adaptation.py -f exps/custom/yolox_l_weed_2022.py -d 1 -b 8 --fp16 -o -c pretrained/yolox_l.pth  --cache
python tools/train_domain_adaptation_rep2.py -f exps/custom/yolox_l_weed_2021_rep2.py -d 1 -b 8 --fp16 -o -c pretrained/yolox_l.pth --cache
python tools/train_domain_adaptation_rep3.py -f exps/custom/yolox_l_weed_2022_rep3.py -d 1 -b 8 --fp16 -o -c pretrained/yolox_l.pth --cache

python tools/train.py -f exps/custom/yolox_l_dense_blueberry.py -d 1 -b 2 --fp16 -o -c pretrained/yolox_l.pth --cache
python tools/train.py -f exps/custom/yolox_l_dense_cottonboll.py -d 1 -b 2 --fp16 -o -c pretrained/yolox_l.pth --cache

test
instances_test2017.json
instances_test2017_cross_season.json

mklink D:\Dataset\WeedData\CottonWeedDet12\annotations\instances_test2017_cross_season.json D:\Dataset\WeedData\WeedData2022\annotations\instances_WeedPlantsXXX_8skus_in_2021.json

mklink D:\Dataset\WeedData\WeedData2022\annotations\instances_test2017_cross_season.json D:\Dataset\WeedData\CottonWeedDet12\annotations\instances_WeedPlantsXXX_8skus_in_2021.json

# yolox_l_weed_2021 yolox_l_weed_2021_rep2 yolox_l_weed_2021_rep3 yolox_l_weed_2021_test_cross_season
# yolox_l_weed_2022 yolox_l_weed_2022_rep2 yolox_l_weed_2022_rep3 yolox_l_weed_2022_test_cross_season
# yolox_l_weed/data2021_subsample/replication1
# yolox_l_weed_add_I3Net/yolox_l_weed_2022/replication1
# /mnt/d/BoyangDeng
# D:\BoyangDeng
python tools/eval.py -f exps/custom/yolox_l_weed_2022_test_cross_season.py -c D:/BoyangDeng/YOLOX/YOLOX_outputs/yolox_l_weed_add_I3Net/yolox_l_weed_2022/replication3_4/best_ckpt.pth -b 64 -d 1 --conf 0.001 --fp16 --fuse --test
python tools/eval.py -f exps/custom/yolox_l_weed_2022_rep3.py -c /mnt/d/BoyangDeng/WeedDetection/YOLOX/YOLOX_outputs/yolox_l_weed_2022_rep3/best_ckpt.pth -b 64 -d 1 --conf 0.001 --fp16 --fuse --test

# yolox_l_dense_blueberry yolox_l_dense_blueberry/replication1
# yolox_l_dense_cottonboll yolox_l_dense_cottonboll/replication1
python tools/eval.py -f exps/custom/yolox_l_dense_blueberry.py -c /mnt/d/BoyangDeng/WeedDetection/YOLOX/YOLOX_outputs/yolox_l_dense_blueberry/replication1/best_ckpt.pth -b 2 -d 1 --conf 0.001 --fp16 --fuse --test

demo
# dataset
# assets/dog.jpg D:\Dataset\WeedData\replication1_test_set\2021
# /mnt/d/Dataset/WeedData/replication1_test_set/2021
# model
# yolox_l_weed_2021.py yolox_l_weed_2021/best_ckpt.pth
# yolox_l_weed_2022.py yolox_l_weed_2022/best_ckpt.pth
# save to YOLOX_outputs\xxx\vis_res\xxx
# 2021_yolox_detected_model_2021
# 2022_yolox_detected_model_2021
# 2022_yolox_detected_model_2022
# 2021_yolox_detected_model_2022
python tools/demo.py image -f exps/custom/yolox_l_weed_2021.py -c /mnt/d/BoyangDeng/WeedDetection/YOLOX/YOLOX_outputs/yolox_l_weed_2021/best_ckpt.pth --path /mnt/d/Dataset/WeedData/replication1_test_set/2022 --conf 0.25 --nms 0.45 --tsize 800 --save_result --device gpu


yolox/exp/yolox_base.py

self.exp.no_aug_epochs


output_pred
[reg_output, obj_output.sigmoid(), cls_output.sigmoid()]
torch.Size([8, 17, xxx, xxx])


self.no_aug_epochs = 25
self.mixup_prob = 0.0
self.shear = 0.0
self.mosaic_scale = (0.5, 1.5)