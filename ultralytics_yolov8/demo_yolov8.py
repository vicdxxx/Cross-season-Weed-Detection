"""
batch_size=16
im_size=800
"""
# with open('flag.txt','r') as f:
#    x=f.read()
#    if x =='':
#        x=0
#    if int(x)>0:
#        exit()
# with open('flag.txt','w') as f:
#    x=int(x)+1
#    f.write(str(x))
from ultralytics import YOLO
import config_da as cfg_da
import da_I3Net_module as da_I3Net
import cv2
import yaml
from tqdm import tqdm
import sys
import os
import platform
sys_name = platform.system()
if sys_name == "Windows":
    #sys.path.insert(0, r'E:\PHD\WeedDetection\ultralytics')
    sys.path.insert(0, r'E:\PHD\WeedDetection\ultralytics_old\ultralytics')
else:
    sys.path.insert(0, '/mnt/e/PHD/WeedDetection/ultralytics')


cfg_da.use_domain_adaptation = False


def list_dir(path, list_name, extension):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name, extension)
        else:
            if file_path.endswith(extension):
                list_name.append(file_path)
    return list_name


"""
conda activate pytorch
cd /mnt/e/Repo/Biang/Graphics/WeedDetection
python test_yolov8.py
windows/linux may cannot share dataset xxx.cache
Logging results to /mnt/e/Repo/Biang/runs/detect/train
E:\Repo\Biang\runs\detect
"""
dst_video_path = r'E:\Repo\Biang_Doc\Doc\Topic\WeedDetection\WeedDetectionFormal\detected\replication1_test_set\2022_yolov8_detected_model_2021.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(dst_video_path, fourcc, 2, (3024, 4032), True)

#model_dir = r'E:\PHD\WeedDetection\ultralytics\runs\detect\data2021\replication1\train4\weights'
model_dir = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\data2022_add_I3Net\train2\weights'
#model_dir = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\data2021_subsample_add_I3Net\train2_rep3\weights'
#model_dir = r'D:\BoyangDeng\WeedDetection\ultralytics\runs\detect\data2021_subsample_add_I3Net\train2\weights'
#model_dir = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\weed2023\yolov8l\rep2\train2\weights'

#src_dir = r'E:\Repo\Biang_Doc\Doc\Topic\WeedDetection\WeedDetectionFormal\detected\replication1_test_set\2022'
src_dir = r'D:\BoyangDeng\test\test2017'
dst_dir = r'D:\BoyangDeng\test'
# if not os.path.exists(dst_dir):
#    os.makedirs(dst_dir)

im_paths = list_dir(src_dir, [], '.jpg')
# best/last
model_path = os.path.join(model_dir, 'best.pt')
print('model_path:', model_path)

#data_dir = r'E:\PHD\WeedDetection\ultralytics\ultralytics\yolo\data\datasets\weed_2022_test_video.yaml'
data_dir = r'D:\BoyangDeng\Detection\ultralytics_yolov8\ultralytics\yolo\data\datasets\weed_demo.yaml'
print('data config:', os.path.split(data_dir)[1])
with open(data_dir, 'r', encoding='UTF-8') as file:
    data_info = yaml.safe_load(file)

model = YOLO(model_path)
model.model.names = data_info['names']

# batch=16/1
for im_path in tqdm(im_paths):
    im_name = os.path.basename(im_path)
    res = model(im_path)
    res_plotted = res[0].plot()
    #cv2.imshow("result", res_plotted)
    dst_path = os.path.join(dst_dir, im_name)
    cv2.imwrite(dst_path, res_plotted)
    #videoWriter.write(res_plotted)
videoWriter.release()
# Export the model
# model.export(format="onnx")
