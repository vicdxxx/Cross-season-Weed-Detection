from torch import no_grad


label_idx_names = {}
merge_labels = True
# weed2021  weed2022 weed_all weed_lambsquarters weed_stable_diffusion
# blueberry_dense cottonboll_dense weed2023 COCO weed2021_3sku
target_name = 'weed_all'
merge_labels_dict = {}
if target_name == 'blueberry_dense':
    # blueberry
    # {
    # "Unblue_visible": 0,
    # "Blue_occluded": 1,
    # "Unblue_occluded": 2,
    # "Blue_visible": 3
    # }
    merge_labels_dict = {2: 0, 1: 3}
    immature_label_idx = 0
    mature_label_idx = 3

elif target_name == 'cottonboll_dense':
    # cottonboll
    # {
    # "Opened": 0,
    # "Unopened": 1,
    # "PartiallyOpened": 2
    # }
    merge_labels_dict = {2: 0}
    # merge_labels_dict = {2: 1}
    immature_label_idx = 1
    mature_label_idx = 0

# 3520
max_size = 900
# 5000 1000
max_det = 1000

#consider_exif = False

train_roi = False
# True False
is_val = True
set_rectangle_resort_ims = False

verbose = True

not_show_label = False and is_val
use_large_label_size = True
#label rect_shape self.batch_shapes
# not work dur to model constrait input size
no_pad_in_plot_im = False
conf_show_percentage = True
use_file_name_as_result_name = True
plot_all_eval_images = False and is_val

plot_correct_with_box_size = False
plot_wrong_with_box_size = False

show_recal_per_im = False

use_EM_Merger = False and is_val

plot_maturity_rate_and_counting = False and is_val
# default conf=0.25
plot_bbox_conf = 0.25

# default.yaml
pred_bbox_conf = 0.25
use_val_bbox_conf = False and is_val
val_bbox_conf = 0.05
nms_iou = 0.5
# strict for occlusion estimation?
# nms_iou = 0.8

use_color_filter_postprocess = False and is_val
current_im = None
current_im_origin = None
current_im_name = None
current_im_im_base_name = None

test_use_letter_box = False and is_val

test_use_nms_agnostic_for_overlaps = False
test_use_roi_model = False and is_val
roi_model = None
# 256 448 416?
roi_new_shape = 448
use_new_boxes = False and test_use_roi_model
use_update_boxes = False and test_use_roi_model
show_uncorrected_detections = False and is_val
use_nms_after_roi = False and test_use_roi_model
preds_original = None
compare_original_prediction = False and test_use_roi_model
update_box_idxes = []
new_box_idxes = []
update_box_precision_per_im = {}
new_box_precision_per_im = {}
update_box_tot_box_num = 0
update_box_tot_correct_box_num = 0
update_box_tot_precision = 0
new_box_tot_box_num = 0
new_box_tot_correct_box_num = 0
new_box_tot_precision = 0

not_consider_boundary_bboxes = False and is_val
if not_consider_boundary_bboxes:
    boundary_margin = 5


def notify_by_email():
    import win32com.client as win32
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = 'toseedrina@gmail.com'
    mail.Subject = 'Training Finished'
    mail.Body = ''
    mail.HTMLBody = '<h2>Training Finished</h2>'  # this field is optional
    mail.Send()

# cfg.show_image_simple(roi_im[:, :, ::-1])
# cfg.show_image_simple(obj)


def show_image_simple(image, wait=True, name="x"):
    import numpy as np
    import cv2 as cv
    # img = image.cpu().detach().numpy().copy()
    if isinstance(image, np.ndarray):
        features_np = image.copy()
    else:
        # import paddle
        # if isinstance(features, paddle.Tensor):
        #    features_np = features.numpy().copy()
        import torch
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy().copy()

    print(image.min(), image.max())
    img = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    img = img.astype(np.uint8)
    cv.namedWindow(name, 0)
    cv.imshow(name, img)
    cv.waitKey()
    # image = (image-image.min())/(image.max()-image.min())
    # from matplotlib import pyplot as plt
    # plt.figure()
    # print(image.min(), image.max())
    # plt.imshow(image)
    # plt.show()


"""
target bbox in im
cfg.show_image_with_bbox(img_show, targets=one_im_pred[i_pred:i_pred+1], font_scale=1.5, font_thickness=3, name='target bbox in im')

target bbox in roi
cfg.show_image_with_bbox(np.transpose(roi_lettered_torch.cpu().numpy()[0], (1, 2, 0))[:,:,::-1], targets=original_pred_roi, font_scale=0.8, font_thickness=2, name='target bbox in roi')

roi_preds in roi
cfg.show_image_with_bbox(np.transpose(roi_lettered_torch.cpu().numpy()[0], (1, 2, 0))[:,:,::-1], targets=roi_preds, font_scale=0.5, font_thickness=1, name='roi_preds in roi')

global_preds in roi
cfg.show_image_with_bbox(img_show, targets=in_region_one_im_pred, font_scale=0.5, font_thickness=2, name='global_preds in roi')

the newest new box
cfg.show_image_with_bbox(np.transpose(roi_lettered_torch.cpu().numpy()[0], (1, 2, 0))[:,:,::-1], targets=new_boxes, font_scale=0.8, font_thickness=2, name='the newest new box')

all new_boxes
cfg.show_image_with_bbox(img_show, targets=one_im_new_boxes, font_scale=0.8, font_thickness=2, name='new_boxes'+im_base_name, save =True, show=False)

one_im_update_boxes
cfg.show_image_with_bbox(img_show, targets=one_im_update_boxes, font_scale=0.8, font_thickness=2, name='update_boxes'+im_base_name, save =True, show=False)

plot uncorrected bboxes?
"""

def show_image_with_bbox(x, targets=None, colors=None, label_idx=5, conf_idx=4, use_cxcywh=False, font_scale=0.5, font_thickness=2, name='x', save=False, show=True, rgb2bgr=False):
    import cv2
    import os
    import numpy as np

    tot_bbox_num = 0
    tot_im_num = 0

    tot_im_num += 1

    if isinstance(x, np.ndarray):
        y = x.copy()
    else:
        import torch
        if isinstance(x, torch.Tensor):
            if x.shape[1] == 3:
                x_new = x.permute(0, 2, 3, 1)[0]
                y = x_new.cpu().detach().numpy().copy()
            else:
                y = x.cpu().detach().numpy().copy()
    if len(y.shape) == 4 and y.shape[0] == 1:
        y = y[0]
    if y.max() <= 1.0:
        z = np.zeros(y.shape, dtype=np.float32)
        z = cv2.normalize(y, z, alpha=0, beta=255.0, norm_type=cv2.NORM_MINMAX)
    else:
        z = y
    z = z.astype(np.uint8)
    # print("image range: {} -> {}".format(z.min(), z.max()))
    z = np.ascontiguousarray(z, dtype=np.uint8)
    if targets is not None:
        if type(targets) == list:
            targets_ = targets
        else:
            targets_ = targets.cpu().numpy()
            targets_y = targets_[:, label_idx]

        bbox_num = len(targets_)
        # print("bbox_num: {}".format(bbox_num))
        tot_bbox_num += bbox_num
        for i_y in range(bbox_num):
            sku_id = int(targets_[i_y, label_idx])
            conf = round(targets_[i_y, conf_idx], 3)

            bbox_start_idx = 0
            bbox_end_idx = 4

            x0, y0, x1, y1 = targets_[i_y, bbox_start_idx:bbox_end_idx]

            if colors is not None:
                color = tuple(colors[sku_id])
            else:
                color = (0, 255, 255)
            thickness = font_thickness
            font_scale = font_scale
            cv2.rectangle(z, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)
            cv2.putText(z, str(sku_id), (int(x0), int(y0) + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            cv2.putText(z, str(conf), (int(x0) + 20, int(y0) + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    # print("tot_bbox_num: {}".format(tot_bbox_num))
    if tot_im_num > 1:
        mean_bbox_num = tot_bbox_num / tot_im_num
        print("mean_bbox_num: {}".format(mean_bbox_num))
    if show:
        if len(z.shape) == 2:
            z = cv2.applyColorMap(z, cv2.COLORMAP_JET)
        elif z.shape[0] == 3:
            z = np.transpose(z, (1, 2, 0))
        cv2.namedWindow(name, 0)
        print(z.shape)
        if rgb2bgr:
            z = z[:, :, ::-1]
        cv2.imshow(name, z)
        cv2.waitKey()
    if save:
        save_dir = r'D:\BoyangDeng\test'
        if not name.endswith('.jpg'):
            name = name + '.jpg'
        save_path = os.path.join(save_dir, name)
        cv2.imwrite(save_path, z)


def log_xalg(*info, log_path=None, show=True, end=None, overwrite=False):
    if show:
        if end is not None:
            print(*info, end=end)
        else:
            print(*info)
    if log_path:
        if overwrite:
            f_log = open(log_path, 'w')
        else:
            f_log = open(log_path, 'a')
        print(*info, file=f_log)
        f_log.close()


if compare_original_prediction:
    log_xalg('', log_path=r'D:\BoyangDeng\test\_update_box_precision_per_im.txt', overwrite=True)
    log_xalg('', log_path=r'D:\BoyangDeng\test\_new_box_precision_per_im.txt', overwrite=True)
    log_xalg('', log_path=r'D:\BoyangDeng\test\_original_precision_per_im.txt', overwrite=True)
    log_xalg('', log_path=r'D:\BoyangDeng\test\_optimized_precision_per_im.txt', overwrite=True)
