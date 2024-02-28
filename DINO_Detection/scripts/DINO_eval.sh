coco_path=$1
checkpoint=$2
python main.py --output_dir logs/DINO/R50-MS4-%j -c config/DINO/DINO_4scale.py --coco_path $coco_path  --eval --resume $checkpoint --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0

python main.py --output_dir logs/DINO/SWIN-MS4-dense_blueberry_images -c config/DINO/DINO_4scale_swin_dense_detection.py --coco_path /mnt/d/BoyangDeng/BlueberryDenseDetection  --eval --resume logs/DINO/SWIN-MS4-dense_blueberry/checkpoint.pth --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 backbone_dir=backbone

python main_test.py --output_dir logs/DINO/SWIN-MS4-dense_blueberry_images -c config/DINO/DINO_4scale_swin_dense_detection.py --coco_path /mnt/d/BoyangDeng/BlueberryDenseDetection  --eval --resume logs/DINO/SWIN-MS4-dense_blueberry/checkpoint.pth --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 backbone_dir=backbone

save_results

demo.py
COCOVisualizer