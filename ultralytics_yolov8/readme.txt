YOLOv8 innovation and improvement points:

1. Backbone . The idea of ​​CSP is still used , but the C3 module in YOLOv5 is replaced by the C2f module to achieve further lightweight, and YOLOv8 still uses the SPPF module used in YOLOv5 and other architectures;

2. PAN-FPN . There is no doubt that YOLOv8 still uses the idea of ​​​​PAN, but by comparing the structure diagrams of YOLOv5 and YOLOv8, we can see that YOLOv8 deletes the convolution structure in the PAN-FPN upsampling stage in YOLOv5, and also replaces the C3 module with C2f module

3. Decoupled-Head . Did you smell something different? Yes, YOLOv8 went to Decoupled-Head;

4. Anchor-Free . YOLOv8 abandoned the previous Anchor-Base and used the idea of ​​Anchor-Free ;

5. Loss function . YOLOv8 uses VFL Loss as classification loss and DFL Loss+CIOU Loss as detection loss;

6. Sample matching . YOLOv8 abandoned the previous IOU matching or unilateral ratio allocation, but used the Task-Aligned Assigner matching method.

D:\BoyangDeng\Detection\ultralytics_yolov8\ultralytics\yolo\cfg
D:\BoyangDeng\Detection\ultralytics_yolov8\ultralytics\yolo\data\datasets

# D:\Dataset\WeedData E:\PHD\WeedDetection
# D:\BoyangDeng\CottonBollDenseDetection
# D:\BoyangDeng\BlueberryDenseDetection
# BlueberryDenseDetection CottonBollDenseDetection
# train2017 train2017_rep2 train2017_rep3 train2017_8cls
# train2017 train2017_rep2 train2017_rep3
# train2017.0 train2017.1 train2017.2 train2017_8cls.0 train2017_8cls.0.subsample
mklink D:\BoyangDeng\CottonBollDenseDetection\train2017 D:\BoyangDeng\CottonBollDenseDetection\train2017.0 /D
mklink D:\BoyangDeng\CottonBollDenseDetection\val2017 D:\BoyangDeng\CottonBollDenseDetection\val2017.0 /D
mklink D:\BoyangDeng\CottonBollDenseDetection\test2017 D:\BoyangDeng\CottonBollDenseDetection\test2017.0 /D


ultralytics\yolo\cfg\default.yaml
save_period
save_json: True
save_hybrid: False
save_txt: True 

hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
'2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')

E:\PHD\WeedDetection\ultralytics\ultralytics\yolo\utils\__init__.py
DEFAULT_CFG_DICT['plot_all'] = True

detections (array[N, 6]), x1, y1, x2, y2, conf, class
labels (array[M, 5]), class, x1, y1, x2, y2

self.args.close_mosaic

self.args.single_cls
AutoShape -> max_det = 5000
max_det: 5000
mosaic: 0.3  # image mosaic (probability)
mixup: 0.3  # image mixup (probability)
copy_paste: 0.3  # segment copy-paste (probability)

TaskAlignedAssigner

ultralytics\yolo\v8\detect\train.py
DetectionTrainer
Loss
make_anchors
assigner

ultralytics\yolo\engine\trainer.py
get_dataloader

ultralytics\nn\tasks.py
DetectionModel

ultralytics\yolo\data\dataset.py
YOLODataset
build_transforms
get_labels

ultralytics\yolo\data\augment.py
class Mosaic(BaseMixTransform):

dense detection
iou: 0.7 -> higher

update_metrics
len(batch['cls'])
torch.sum(correct_bboxes_mAP50)
batch['im_file']
batch['img'].shape

ultralytics\yolo\utils\plotting.py
box_label

ultralytics\yolo\v8\detect\val.py
accumulate_finds/accumulate_GT

ultralytics\yolo\utils\metrics.py
ap_per_class

plot_images

dense blueberry mAP50=0.915
conf:  0.05 # object confidence threshold for detection (default 0.25 predict, 0.001 val)
iou: 0.5  # intersection over union (IoU) threshold for NMS

stream_inference
if self.args.verbose:
LOGGER.info(f'{s}{self.dt[1].dt * 1E3:.1f}ms')

# fix bug, add this func
#def __setstate__(self, attr):
#    pass

D:\BoyangDeng\Detection\ultralytics_yolov8\ultralytics\yolo\data
base.py -> load_image

color_classify


D:\BoyangDeng\Detection\ultralytics_yolov8\ultralytics\yolo\utils
metrics.py