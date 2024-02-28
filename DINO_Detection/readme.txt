"""
conda activate pytorch
cd /mnt/e/PHD/WeedDetection/DINO/


cd models/dino/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
"""

coco_transformer.py

DINO_train_swin_weed.sh

DINO_eval.sh

export CUDA_VISIBLE_DEVICES=0 && python main.py --output_dir logs/DINO/SWIN_MS4_weed_2021_rep1 -c config/DINO/DINO_4scale_swin_weed.py --coco_path /mnt/d/Dataset/WeedData/CottonWeedDet12 --options dn_scalar=900 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 backbone_dir=backbone

python main.py --output_dir logs/DINO/SWIN-MS4-dense_blueberry_images -c config/DINO/DINO_4scale_swin_dense_detection.py --coco_path /mnt/d/BoyangDeng/BlueberryDenseDetection --eval --resume logs/DINO/SWIN-MS4-dense_blueberry/checkpoint_best_regular.pth --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 backbone_dir=backbone

python main_test.py --output_dir logs/DINO/SWIN-MS4-dense_blueberry_replication1_test -c config/DINO/DINO_4scale_swin_dense_detection.py --coco_path /mnt/d/BoyangDeng/BlueberryDenseDetection --eval --resume logs/DINO/SWIN-MS4-dense_blueberry_replication1/checkpoint.pth --options dn_scalar=900 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 backbone_dir=backbone

save_results

demo.py
COCOVisualizer

maxDets?
COCOeval
ious = np.zeros((len(dts), len(gts)))

coco.py
build


mklink D:\Dataset\WeedData\CottonWeedDet12\annotations\instances_test2017.json D:\Dataset\WeedData\CottonWeedDet12\annotations\instances_test2017_8cls.0.subsample.json