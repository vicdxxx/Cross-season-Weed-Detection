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
import sys
import os
import platform
sys_name = platform.system()
if sys_name == "Windows":
    #sys.path.insert(0, r'E:\PHD\WeedDetection\ultralytics')
    sys.path.insert(0, r'E:\PHD\WeedDetection\ultralytics_old\ultralytics')
else:
    sys.path.insert(0, '/mnt/e/PHD/WeedDetection/ultralytics')

import cv2
import da_I3Net_module as da_I3Net
import config_da as cfg_da
from ultralytics import YOLO
import yaml


cfg_da.use_domain_adaptation = False


if sys_name == "Windows":
    cfg = r"E:\PHD\WeedDetection\ultralytics\ultralytics\models\v8\yolov8l_weed.yaml"
    train_num = 4
    model_dir = r'E:\PHD\WeedDetection\ultralytics\runs\detect\train' + str(train_num) + '\weights'
    if train_num == 0:
        model_dir = r'E:\PHD\WeedDetection\ultralytics\runs\detect\train' + '\weights'
    else:
        model_dir = r'E:\PHD\WeedDetection\ultralytics\runs\detect\train' + str(train_num) + '\weights'
    # weed_2022 weed_2022_test weed_2021_test_all weed_2022_test _video
    # dense_blueberry_test dense_cotton_boll_test
    data_dir = r'E:\PHD\WeedDetection\ultralytics\ultralytics\yolo\data\datasets\weed_2022_test_video.yaml'
else:
    # yolov8l_weed yolov8s_weed
    cfg = "/mnt/e/PHD/WeedDetection/ultralytics/ultralytics/models/v8/yolov8l_weed.yaml"
    train_num = 4
    #model_dir = '/mnt/e/PHD/WeedDetection/ultralytics/runs/detect/train'+str(train_num)+'/weights'

    #model_dir = '/mnt/e/Repo/Biang/runs/detect/train'+str(train_num)+'/weights'
    model_dir = '/mnt/e/Repo/Biang/runs/detect/data2021_subsample/replication1/train' + str(train_num) + '/weights'

    #model_dir = '/mnt/e/PHD/WeedDetection/ultralytics/runs/detect/train'+str(train_num)+'/weights'
    # data_dir='/mnt/e/PHD/WeedDetection/ultralytics/ultralytics/yolo/data/datasets/weed_test_linux.yaml'

    # weed_test_dataset_2021_linux weed_test_dataset_2022_linux
    # weed_test_dataset_2021_linux_all weed_test_dataset_2022_linux_all
    data_dir = '/mnt/e/PHD/WeedDetection/ultralytics/ultralytics/yolo/data/datasets/weed_test_dataset_2021_linux.yaml'

"""
conda activate pytorch
cd /mnt/e/Repo/Biang/Graphics/WeedDetection
python test_yolov8.py
windows/linux may cannot share dataset xxx.cache
Logging results to /mnt/e/Repo/Biang/runs/detect/train
E:\Repo\Biang\runs\detect
"""

# data2021_subsample data2022
model_dir = r'E:\PHD\WeedDetection\ultralytics\runs\detect\data2021\replication1\train4\weights'

# model = YOLO(cfg)  # build a new model from scratch

print('data config:', os.path.split(data_dir)[1])
with open(data_dir, 'r', encoding='UTF-8') as file:
    data_info = yaml.safe_load(file)
# best/last
model_path = os.path.join(model_dir, 'best.pt')
print('model_path:', model_path)

model = YOLO(model_path)
model.model.names = data_info['names']
# Open the video file
video_path = r"E:\PHD\WeedDetection\WeedVideoes\20220827_iPhoneSE_YL_v04.mp4"
dst_path = r"E:\PHD\WeedDetection\WeedVideoes\20220827_iPhoneSE_YL_v04_model_2021_detected.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(dst_path, fourcc, 30, (1080, 1920), True)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        videoWriter.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
videoWriter.release()
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# Export the model
# model.export(format="onnx")
