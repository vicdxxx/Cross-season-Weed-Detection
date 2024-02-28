import os, sys
import torch, json
import numpy as np
from tqdm import tqdm

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops
import config_all as cfg
from config.DINO.coco_transformer import data_aug_scales, data_aug_max_size, data_aug_scales2_resize, data_aug_scales2_crop, data_aug_scale_overlap
id2name=cfg.id2name
 # change the path of the model config file
#model_config_path = "config/DINO/DINO_4scale_swin_weed.py"
model_config_path = "/mnt/d/BoyangDeng/WeedDetection/DINO/logs/DINO/SWIN-MS4-dense_blueberry/config_cfg.py"

# DINO/data2021/replication1
# DINO/data2021_subsample/replication1
# DINO/data2022/replication1
model_checkpoint_path = "/mnt/d/BoyangDeng/WeedDetection/DINO//logs/DINO/SWIN-MS4-dense_blueberry/checkpoint_best_regular.pth" # change the path of the model checkpoint

# 2021 2022
# 2021_yolox_detected_model_2021
# 2022_yolox_detected_model_2021
# 2022_yolox_detected_model_2022
# 2021_yolox_detected_model_2022
#r'D:\Dataset\WeedData\replication1_test_set\'
savedir = '/mnt/d/BoyangDeng/BlueberryDenseDetection/test2017_detected'
im_dir = '/mnt/d/BoyangDeng/BlueberryDenseDetection/test2017'
if not os.path.exists(savedir):
    os.mkdir(savedir)ß

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def list_dir(path, list_name, extension):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name, extension)
        else:
            if file_path.endswith(extension):
                list_name.append(file_path)
    return list_name

args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

#param_num = get_n_params(model)
#print('param_num:', param_num)
#exit(0)

from PIL import Image
import datasets.transforms as T 

# load coco names
#with open('util/coco_id2name.json') as f:
#    id2name = json.load(f)
#    id2name = {int(k):v for k,v in id2name.items()}

def detect_image(image, savedir, im_name):
    # transform images
    transform = T.Compose([
        #T.RandomResize([800], max_size=1333),
        T.RandomResize([max(data_aug_scales)], max_size=data_aug_max_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image, _ = transform(image, None)


    # predict images
    output = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

    # visualize outputs
    thershold = 0.3 # set a thershold

    vslzr = COCOVisualizer()

    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold
    #print(labels)
    box_label = [id2name[int(item)] for item in labels[select_mask]]
    pred_dict = {
        'boxes': boxes[select_mask],
        'size': torch.Tensor([image.shape[1], image.shape[2]]),
        'box_label': box_label
    }
    vslzr.visualize(image, pred_dict, savedir=savedir, dpi=800, show_in_console=False, im_name=im_name)

im_paths = list_dir(im_dir, [], '.jpg')
for im_path in tqdm(im_paths):
    im_name = os.path.basename(im_path)
    image = Image.open(im_path).convert("RGB") # load image
    detect_image(image, savedir, im_name)