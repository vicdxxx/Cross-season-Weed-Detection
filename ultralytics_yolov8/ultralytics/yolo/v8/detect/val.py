# Ultralytics YOLO 🚀, GPL-3.0 license

import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pathlib import Path

import numpy as np
import torch
import cv2
from ultralytics.yolo.data.augment import LetterBox, LetterBoxPadRightBottom
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, colorstr, ops
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.metrics import ConfusionMatrix, DetMetrics, box_iou, box_iou_by_box2
from ultralytics.yolo.utils.plotting import output_to_target, plot_images
from ultralytics.yolo.utils.torch_utils import de_parallel
import config as cfg
from tqdm import tqdm
if cfg.use_EM_Merger:
    from EMMerger import EmMerger

tot_bbox_num = 0
tot_revised_num_blue_to_non = 0
tot_revised_num_non_to_blue = 0

accumulate_finds = 0
accumulate_GT = 0
accumulate_sample_num = 0

accumulate_finds1 = 0
accumulate_GT1 = 0
accumulate_sample_num1 = 0

if cfg.plot_maturity_rate_and_counting:
    predict_mature_nums = []
    predict_immature_nums = []
    gt_mature_nums = []
    gt_immature_nums = []

    predict_mature_ratios = []
    gt_mature_ratios = []

    predict_countings = []
    gt_countings = []


class DetectionValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None):
        super().__init__(dataloader, save_dir, pbar, args)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        if cfg.test_use_letter_box:
            self.letter_box = LetterBox(new_shape=(3520, 3520))
        if cfg.test_use_roi_model:
            self.letter_box_pad_right_bottom = LetterBoxPadRightBottom(new_shape=(cfg.roi_new_shape, cfg.roi_new_shape))

    def preprocess(self, batch):
        if cfg.test_use_letter_box:
            im = batch['img'].cpu().numpy()[0].transpose(1, 2, 0)
            im_lettered = self.letter_box(image=im)
            batch['img'] = torch.tensor(im_lettered.transpose(2, 0, 1))[None, :, :, :]
            # revise annotations?
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(self.device)

        nb = len(batch['img'])
        self.lb = [torch.cat([batch['cls'], batch['bboxes']], dim=-1)[batch['batch_idx'] == i]
                   for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling

        return batch

    def init_metrics(self, model):
        val = self.data.get(self.args.split, '')  # validation path
        self.is_coco = isinstance(val, str) and val.endswith(f'coco{os.sep}val2017.txt')  # is COCO dataset
        self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self):
        return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)')

    def postprocess(self, preds):
        #print('\nself.args.max_det:', self.args.max_det)
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=self.args.single_cls,
                                        max_det=self.args.max_det)

        if cfg.test_use_nms_agnostic_for_overlaps:
            preds_torch = None
            for pred in preds:
                if preds_torch is None:
                    preds_torch = pred[None, :]
                else:
                    preds_torch = torch.concat([preds_torch, pred[None, :]], 0)
            preds = ops.non_max_suppression_1(preds_torch,
                                            self.args.conf,
                                            0.91,
                                            labels=None,
                                            agnostic=True,
                                            max_det=self.args.max_det)

        if cfg.use_color_filter_postprocess:
            img = np.transpose(cfg.current_im.cpu().numpy()[0], (1, 2, 0))
            img = img[:, :, ::-1]
            preds = self.color_classify(img, preds, 'blue')
        if cfg.test_use_roi_model:
            #roi_include_im_names = ['MSState_YL_Cropped_0054']
            #for i_im, one_im_pred in tqdm(enumerate(preds)):
            #    if cfg.current_im_im_base_name not in roi_include_im_names:
            #        return preds
            
            img = np.transpose(cfg.current_im.cpu().numpy()[0], (1, 2, 0))
            img = img[:, :, ::-1]
            
            cfg.preds_original = []
            for pred in preds: # torch.where(preds[0]!=cfg.preds_original[0])
                cfg.preds_original.append(pred.clone())

            preds, update_box_idxes, new_box_idxes = self.detect_on_roi(cfg.current_im, img, preds)
            cfg.update_box_idxes = update_box_idxes
            cfg.new_box_idxes = new_box_idxes
            # filter out duplicate bboxes (same sku) after corrected

            if cfg.use_nms_after_roi:
                preds_torch = None
                for pred in preds:
                    if preds_torch is None:
                        preds_torch = pred[None, :]
                    else:
                        preds_torch = torch.concat([preds_torch, pred[None, :]], 0)
                preds = ops.non_max_suppression_1(preds_torch,
                                                self.args.conf,
                                                0.91,
                                                labels=None,
                                                agnostic=False,
                                                max_det=self.args.max_det)
        return preds

    def detect_on_roi(self, current_im, img_show, preds):
        results = []
        h, w = img_show.shape[:2]

        new_box_idxes = []
        update_box_idxes = []

        for i_im, one_im_pred in tqdm(enumerate(preds)):
            print('im_base_name:', cfg.current_im_im_base_name)

            one_im_new_boxes = []
            one_im_update_boxes = []
            one_im_update_boxes_dict = {'unblue->blue':0, 'blue->unblue':0}
            update_boxes_skip_im_names = []
            #update_boxes_include_im_names = ['MSState_YL_Cropped_0054', 'MSState_YL_Cropped_0095','MSState_YL_Cropped_0093', 'MSState_YL_Cropped_0122']
            #update_boxes_include_im_names = ['MSState_YL_Cropped_0037']

            for i_pred, pred in enumerate(one_im_pred):
                x0 = max(int(pred[0]), 0)
                y0 = max(int(pred[1]), 0)
                x1 = min(int(pred[2]), w)
                y1 = min(int(pred[3]), h)
                conf = pred[4]
                conf_t = 0.3
                if conf > conf_t:
                    continue
                #print('conf:', conf)
                box_w = x1 - x0
                box_h = y1 - y0
                target_size = cfg.roi_new_shape
                if box_w > target_size and box_h > target_size:
                    continue
                if box_w <= 0 and box_h <= 0:
                    continue
                pad_w = (target_size - box_w) // 2
                pad_h = (target_size - box_h) // 2
                x0_ = max(int(x0 - pad_w), 0)
                y0_ = max(int(y0 - pad_h), 0)
                x1_ = min(int(x1 + pad_w), w)
                y1_ = min(int(y1 + pad_h), h)

                x0_roi = pad_w
                y0_roi = pad_h
                x1_roi = pad_w + box_w
                y1_roi = pad_h + box_h

                roi_im = current_im[i_im, :, y0_:y1_, x0_:x1_]
                roi_im = roi_im.cpu().numpy().transpose(1, 2, 0)
                roi_lettered = self.letter_box_pad_right_bottom(image=roi_im)
                roi_lettered_torch = torch.tensor(roi_lettered.transpose(2, 0, 1))[None, :, :, :]
                roi_preds = cfg.roi_model.predict(roi_lettered_torch)
                res = roi_preds[0].boxes
                roi_preds = res.boxes

                roi_region_psudo_box = torch.Tensor([x0_, y0_, x1_, y1_])[None, :].to(roi_preds.device)
                roi_region_iou = box_iou_by_box2(roi_region_psudo_box, one_im_pred[:, :4])
                box_in_region_iou_t = 0.1
                box_in_region = roi_region_iou > box_in_region_iou_t
                valid_box_idxes = torch.where(roi_region_iou > box_in_region_iou_t)
                roi_region_iou_valid = roi_region_iou[box_in_region]
                in_region_one_im_pred = one_im_pred[valid_box_idxes[1]]

                in_region_one_im_pred_roi = torch.zeros_like(in_region_one_im_pred).to(roi_preds.device)
                in_region_one_im_pred_roi[:, 0] = in_region_one_im_pred[:, 0] - x0_
                in_region_one_im_pred_roi[:, 1]  = in_region_one_im_pred[:, 1] - y0_
                in_region_one_im_pred_roi[:, 2] = in_region_one_im_pred[:, 2] - x0_
                in_region_one_im_pred_roi[:, 3] = in_region_one_im_pred[:, 3] - y0_
                in_region_one_im_pred_roi[:, 4] = in_region_one_im_pred[:, 4]
                in_region_one_im_pred_roi[:, 5] = in_region_one_im_pred[:, 5]
                global_roi_match_iou = box_iou_by_box2(in_region_one_im_pred_roi[:, :4], roi_preds[:, :4])

                roi_preds_max = torch.max(global_roi_match_iou, 0)
                #roi_preds_max.indices
                # 0.5
                overlap_iou_t = 0.3
                valid_box_idxes = roi_preds_max.values < overlap_iou_t
                new_boxes = roi_preds[valid_box_idxes]
                # 0.75
                new_box_conf_t = 0.7
                for new_box in new_boxes:
                    new_box[0] = max(int(new_box[0]), 0)
                    new_box[1] = max(int(new_box[1]), 0)
                    new_box[2] = min(int(new_box[2]), target_size)
                    new_box[3] = min(int(new_box[3]), target_size)
                    if new_box[2]<box_w or new_box[0]>target_size-box_w:
                        continue
                    if new_box[3]<box_h or new_box[1]>target_size-box_h:
                        continue
                    new_box[0] += x0_
                    new_box[1] += y0_
                    new_box[2] += x0_
                    new_box[3] += y0_
                    if new_box[4] > new_box_conf_t:
                        if len(one_im_new_boxes) > 0:
                            new_box_exist_ious = box_iou(new_box[:4][None, :], one_im_new_boxes[:, :4])
                            new_box_exist_iou = torch.max(new_box_exist_ious[0], 0)
                            if new_box_exist_iou.values > 0.7:
                                continue
                        
                        if len(one_im_new_boxes) == 0:
                            one_im_new_boxes = new_box[None, :]
                        else:
                            one_im_new_boxes = torch.concat([one_im_new_boxes, new_box[None, :]], 0)
                
                original_pred_roi = torch.Tensor([x0_roi, y0_roi, x1_roi, y1_roi, pred[4], pred[5]])[None, :]
                original_pred_roi_box = torch.Tensor([x0_roi, y0_roi, x1_roi, y1_roi])[None, :].to(roi_preds.device)
                iou = box_iou(original_pred_roi_box, roi_preds[:, :4])
                iou_t = 0.7
                if cfg.use_update_boxes:
                    #if cfg.current_im_im_base_name not in update_boxes_include_im_names:
                    #    continue
                    if 0 not in iou.shape and iou.max() > iou_t:
                        roi_conf = roi_preds[iou.argmax()][4]
                        roi_sku = roi_preds[iou.argmax()][5]
                        if roi_sku != pred[5]:
                            margin = 0.3
                            if roi_conf > pred[4] + margin:
                                x0, y0, x1, y1 = preds[i_im][i_pred][:4].cpu().numpy()
                                # unblue -> blue often
                                # blue -> unblue rare
                                if roi_sku == 3:
                                    color_classify_res = self.color_classify_one_box_unmature(img_show, [x0, y0, x1, y1])
                                    if color_classify_res == 'blue' and pred[5] == 3:
                                        continue
                                    elif color_classify_res == 'unknown' and pred[5] == 0:
                                        continue
                                    one_im_update_boxes_dict['unblue->blue']+=1
                                else:
                                    one_im_update_boxes_dict['blue->unblue']+=1
                                print('i_pred:', i_pred, 'conf:', pred[4].cpu().numpy(), 'sku:', pred[5].cpu().numpy(), 'roi_conf:', roi_conf.cpu().numpy(), 'roi_sku:', roi_sku.cpu().numpy())

                                update_box_idxes.append(i_pred)

                                preds[i_im][i_pred][4] = roi_conf
                                #preds[i_im][i_pred][4] = 0.99
                                preds[i_im][i_pred][5] = roi_sku
                                if len(one_im_update_boxes) == 0:
                                    one_im_update_boxes = preds[i_im][i_pred][None, :]
                                else:
                                    one_im_update_boxes = torch.concat([one_im_update_boxes, preds[i_im][i_pred][None, :]], 0)
                else:
                    # preds[i_im][i_pred][4] = 0
                    pass
            if len(one_im_new_boxes) > 0 and cfg.use_new_boxes:
                print('new bboxes number:', len(one_im_new_boxes))

                for i_new in range(len(one_im_new_boxes)):
                    new_box_idxes.append(len(preds[i_im])+i_new)

                preds[i_im] = torch.concat([preds[i_im], one_im_new_boxes], 0)
                cfg.show_image_with_bbox(img_show, targets=one_im_new_boxes, font_scale=0.8, font_thickness=2, name=cfg.current_im_im_base_name+'_new_boxes', save=True, show=False)
            if len(one_im_update_boxes) > 0:
                print('updated bboxes number:', len(one_im_update_boxes))
                print(one_im_update_boxes_dict)
                cfg.show_image_with_bbox(img_show, targets=one_im_update_boxes, font_scale=0.8, font_thickness=2, name=cfg.current_im_im_base_name+'_update_boxes', save=True, show=False)
            # xc = preds[i_im][:, 4] > 0.001
            # preds[i_im] = preds[i_im][xc] 
            # cfg.current_im_name
        return preds, update_box_idxes, new_box_idxes

    def color_classify(self, img, preds, color):
        results = []
        h, w = img.shape[:2]

        # remove_im_list = ['MSState_YL_Cropped_0118.jpg']

        global tot_bbox_num
        global tot_revised_num_non_to_blue
        global tot_revised_num_blue_to_non

        for i_im, one_im_pred in enumerate(preds):
            idxes_need_keep = []
            revised_num_blue_to_non = 0
            revised_num_non_to_blue = 0
            print('cfg.current_im_name:', cfg.current_im_name)
            # im_name =  os.path.basename(cfg.current_im_name[i_im])
            # if im_name in remove_im_list:
            #    continue

            tot_bbox_num += len(one_im_pred)
            for i_pred, pred in enumerate(one_im_pred):
                x0 = max(int(pred[0]), 0)
                y0 = max(int(pred[1]), 0)
                x1 = min(int(pred[2]), w)
                y1 = min(int(pred[3]), h)
                conf = pred[4]
                if conf > 0.9:
                    # idxes_need_keep += [i_pred]
                    continue
                # if pred[-1] == 3:
                #    if color == 'blue':
                #        res = self.color_classify_one_box_blue(img, [x0, y0, x1, y1])
                #    else:
                #        assert False
                #    if res == 'empty':
                #        continue
                #    idxes_need_keep += [i_pred]
                #    pred_color = res
                #    if pred_color == 'blue':
                #        preds[i_im][i_pred][-1] = 3
                #    else:
                #        preds[i_im][i_pred][-1] = 0
                #        revised_num_blue_to_non += 1

                if pred[-1] == 0:
                    if color == 'blue':
                        res = self.color_classify_one_box_unmature(img, [x0, y0, x1, y1])
                    else:
                        assert False
                    if res == 'empty':
                        continue
                    # idxes_need_keep += [i_pred]
                    pred_color = res
                    if pred_color == 'blue':
                        preds[i_im][i_pred][-1] = 3
                        revised_num_non_to_blue += 1
                    else:
                        preds[i_im][i_pred][-1] = 0
            # preds[i_im] = preds[i_im][idxes_need_keep]

            print('revised_num_blue_to_non:', revised_num_blue_to_non)
            print('revised_num_non_to_blue:', revised_num_non_to_blue)

            tot_revised_num_blue_to_non += revised_num_blue_to_non
            tot_revised_num_non_to_blue += revised_num_non_to_blue

        print('tot_revised_num_blue_to_non:', tot_revised_num_blue_to_non)
        print('tot_revised_num_non_to_blue:', tot_revised_num_non_to_blue)
        print('tot_bbox_num:', tot_bbox_num)
        print('ratio blue_to_non:', round(float(tot_revised_num_blue_to_non) / float(tot_bbox_num), 5))
        print('ratio non_to_blue:', round(float(tot_revised_num_non_to_blue) / float(tot_bbox_num), 5))
        return preds

    def color_classify_one_box_blue(self, img, box):
        lower_blue = np.array([20])
        # lower_blue = np.array([45])
        # lower_blue = np.array([70])
        # lower_blue = np.array([80])
        # upper_blue = np.array([130])
        upper_blue = np.array([150])

        x0, y0, x1, y1 = box
        obj_ = img[y0:y1, x0:x1, :].copy()
        if len(obj_) == 0:
            return 'empty'
        h = obj_.shape[0]
        w = obj_.shape[1]
        area = h * w

        obj_hsv = cv2.cvtColor(obj_, cv2.COLOR_BGR2HSV)
        (H, S, V) = cv2.split(obj_hsv)

        output = obj_hsv.copy()
        output1 = obj_hsv.copy()

        mask_blue = cv2.inRange(H, lower_blue, upper_blue)
        output = cv2.bitwise_and(obj_, obj_, mask=mask_blue)
        ret, output_binary = cv2.threshold(mask_blue, 0.5, 1, cv2.THRESH_BINARY)

        blue_output = np.sum(output_binary)
        ratio = blue_output / float(area)
        # print('ratio: ', ratio)

        # consider occluded by leaves
        if ratio > 0.15:
            pred_color = 'blue'
        else:
            pred_color = 'unknown'
        return pred_color

    def color_classify_one_box_unmature(self, img, box):
        lower_range_2 = np.array([90])
        upper_range_2 = np.array([160])

        lower_range_0 = np.array([0])
        #upper_range_0 = np.array([90])
        upper_range_0 = np.array([100])
        #lower_range_1 = np.array([135])
        lower_range_1 = np.array([120])
        upper_range_1 = np.array([180])
        x0, y0, x1, y1 = box
        x0 = round(x0)
        y0 = round(y0)
        x1 = round(x1)
        y1 = round(y1)
        obj = img[y0:y1, x0:x1, :]
        if len(obj) == 0:
            return 'empty'
        h = obj.shape[0]
        w = obj.shape[1]
        area = h * w

        obj_mean_filtered = cv2.pyrMeanShiftFiltering(obj, 5, 10)

        obj_hsv = cv2.cvtColor(obj_mean_filtered, cv2.COLOR_BGR2HSV)
        (H, S, V) = cv2.split(obj_hsv)

        mask_0 = cv2.inRange(H, lower_range_0, upper_range_0)
        mask_1 = cv2.inRange(H, lower_range_1, upper_range_1)
        output_0 = cv2.bitwise_and(obj, obj, mask=mask_0)
        output_1 = cv2.bitwise_and(obj, obj, mask=mask_1)

        mask_2 = cv2.inRange(H, lower_range_2, upper_range_2)
        output_2 = cv2.bitwise_and(obj, obj, mask=mask_2)

        ret, output_binary_0 = cv2.threshold(mask_0, 0.5, 1, cv2.THRESH_BINARY)
        ret, output_binary_1 = cv2.threshold(mask_1, 0.5, 1, cv2.THRESH_BINARY)
        tot_output_binary = output_binary_0+output_binary_1
        ret, tot_output_binary = cv2.threshold(tot_output_binary, 0.5, 1, cv2.THRESH_BINARY)

        ret, output_binary_2 = cv2.threshold(mask_2, 0.5, 1, cv2.THRESH_BINARY)
        unmature_area = cv2.bitwise_and(output_binary_2, tot_output_binary, mask=mask_2)

        ##cfg.show_image_simple(obj)
        ##cv2.threshold(obj[:,:,0],0,255,cv2.THRESH_OTSU)[1]
        ##cv2.pyrMeanShiftFiltering(obj, 5, 10)
        #from scipy import ndimage
        ##image_max = ndimage.maximum_filter(obj, size=10, mode='constant')
        #from skimage.feature import peak_local_max
        #from skimage.morphology import watershed
        #from skimage.segmentation import watershed
        #D = ndimage.distance_transform_edt(tot_output_binary)
        #localMax = peak_local_max(D, indices=False, min_distance=20, labels=tot_output_binary)
        #markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        #labels = watershed(-D, markers, mask=tot_output_binary)
        ##print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

        #tot_output = np.sum(output_binary1 + output_binary)
        #ratio = tot_output / float(area)

        ratio = unmature_area.sum()/output_binary_2.sum()

        #if ratio > 0.15:
        if ratio > 0.35:
            pred_color = 'unknown'
        else:
            pred_color = 'blue'
        return pred_color

    def update_metrics(self, preds, batch):
        # Metrics
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]

            if cfg.not_consider_boundary_bboxes:
                height, width = batch['img'].shape[2:]
                valid_idxes = []
                for i_pred in range(len(pred)):
                    x1, y1, x2, y2 = pred[i_pred][:4]
                    if x1 < cfg.boundary_margin or x2 > width - cfg.boundary_margin:
                        pass
                    elif y1 < cfg.boundary_margin or y2 > height - cfg.boundary_margin:
                        pass
                    else:
                        valid_idxes.append(i_pred)
                pred = pred[valid_idxes]
                origin_bbox = ops.xywh2xyxy(bbox) * torch.tensor((width, height, width, height), device=self.device)
                valid_idxes = []
                for i_bbox in range(len(origin_bbox)):
                    x1, y1, x2, y2 = origin_bbox[i_bbox]
                    if x1 < cfg.boundary_margin or x2 > width - cfg.boundary_margin:
                        pass
                    elif y1 < cfg.boundary_margin or y2 > height - cfg.boundary_margin:
                        pass
                    else:
                        valid_idxes.append(i_bbox)
                bbox = bbox[valid_idxes]
                cls = cls[valid_idxes]

            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0

            if cfg.use_EM_Merger:
                image_name = batch['im_file'][si]
                pred_num = pred.shape[0]
                results = torch.zeros(pred_num, 7, dtype=pred.dtype, device=self.device)
                results[:, :4] = pred[:, :4]
                results[:, 4] = pred[:, 4]
                results[:, 5] = pred[:, 4]
                results[:, 6] = pred[:, 5]
                filtered_data = None
                # filtered_data = EmMerger.merge_detections(batch['img'][si], image_name, results)

                try:
                    filtered_data = EmMerger.merge_detections(batch['img'][si], image_name, results)
                except Exception as e:
                    print(e)
                    print(image_name)

                if filtered_data is not None:
                    filtered_boxes = []
                    filtered_scores = []
                    filtered_labels = []
                    pred_new = []
                    for _, detection in filtered_data.iterrows():
                        box = np.asarray([detection['x1'], detection['y1'], detection['x2'], detection['y2']])
                        filtered_boxes.append(box)
                        filtered_scores.append(detection['confidence'])
                        filtered_labels.append('{0:.2f}'.format(detection['hard_score']))
                        pred_new.append([detection['x1'], detection['y1'], detection['x2'], detection['y2'], detection['confidence'], detection['cls']])
                    pred = torch.Tensor(pred_new).to(pred.device)
                    nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
                    correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init

            predn = pred.clone()
            ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape, ratio_pad=batch['ratio_pad'][si])  # native-space pred

            if cfg.compare_original_prediction:
                predn_original = cfg.preds_original[si].clone()
                ops.scale_boxes(batch['img'][si].shape[1:], predn_original[:, :4], shape, ratio_pad=batch['ratio_pad'][si])  # native-space pred

            correct_bboxes_mAP50 = None
            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device)  # target boxes
                ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape, ratio_pad=batch['ratio_pad'][si])  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)

                if cfg.compare_original_prediction:
                    correct_bboxes_original = self._process_batch(predn_original, labelsn, pre_name='_original')
                    correct_bboxes_mAP50_original = correct_bboxes_original[:, 0]
                    recall_1 = round(torch.sum(correct_bboxes_mAP50_original).cpu().numpy() / len(cls), 2)
                    global accumulate_finds1
                    global accumulate_GT1
                    global accumulate_sample_num1
                    accumulate_sample_num1 += 1
                    accumulate_finds1 += torch.sum(correct_bboxes_mAP50_original).cpu().numpy()
                    accumulate_GT1 += len(cls)

                    cfg.log_xalg("tot recall:", round(accumulate_finds1 / accumulate_GT1, 2), "cur recall:", recall_1, 'gt box num:', len(cls), 'pred box num:', len(correct_bboxes_mAP50_original),
                          'corret box num:', torch.sum(correct_bboxes_mAP50_original).cpu().numpy(), 'precision 50%:', round(torch.sum(correct_bboxes_mAP50_original).cpu().numpy()/max(len(correct_bboxes_mAP50_original), 1), 4), batch['img'].shape, os.path.basename(batch['im_file'][si]), log_path=r'D:\BoyangDeng\test\_original_precision_per_im.txt', overwrite=False, show=True)

                    correct_bboxes_update = self._process_batch(predn, labelsn, pre_name='_optimized_update')
                    correct_bboxes_new = self._process_batch(predn, labelsn, pre_name='_optimized_new')
                    if len(correct_bboxes_update) > 0:
                        print('update tot', len(correct_bboxes_update[:, 0]), 'correct:', torch.sum(correct_bboxes_update[:, 0]), 'uncorrect:', len(correct_bboxes_update[:, 0])-torch.sum(correct_bboxes_update[:, 0]))
                        cfg.update_box_precision_per_im[cfg.current_im_im_base_name] = {
                            'tot': len(correct_bboxes_update[:, 0]),
                            'precision': torch.round(torch.sum(correct_bboxes_update[:, 0])/len(correct_bboxes_update[:, 0]), decimals=4).cpu().numpy()[()],
                        }
                        cfg.update_box_tot_box_num += len(correct_bboxes_update[:, 0])
                        cfg.update_box_tot_correct_box_num += torch.sum(correct_bboxes_update[:, 0]).cpu().numpy()[()]
                        cfg.update_box_tot_precision = round(cfg.update_box_tot_correct_box_num/cfg.update_box_tot_box_num, 4)
                        cfg.update_box_precision_per_im['overall'] = {'tot': cfg.update_box_tot_box_num, 'precision': cfg.update_box_tot_precision}

                        cfg.log_xalg(cfg.update_box_precision_per_im, log_path=r'D:\BoyangDeng\test\_update_box_precision_per_im.txt', overwrite=True, show=False)
                    if len(correct_bboxes_new) > 0:
                        print('new tot', len(correct_bboxes_new[:, 0]), 'correct:', torch.sum(correct_bboxes_new[:, 0]), 'uncorrect:', len(correct_bboxes_new[:, 0])-torch.sum(correct_bboxes_new[:, 0]))
                        cfg.new_box_precision_per_im[cfg.current_im_im_base_name] = {
                            'tot': len(correct_bboxes_new[:, 0]),
                            'precision': torch.round(torch.sum(correct_bboxes_new[:, 0])/len(correct_bboxes_new[:, 0]), decimals=4).cpu().numpy()[()],
                        }
                        cfg.new_box_tot_box_num += len(correct_bboxes_new[:, 0])
                        cfg.new_box_tot_correct_box_num += torch.sum(correct_bboxes_new[:, 0]).cpu().numpy()[()]
                        cfg.new_box_tot_precision = round(cfg.new_box_tot_correct_box_num/cfg.new_box_tot_box_num, 4)
                        cfg.new_box_precision_per_im['overall'] = {'tot': cfg.new_box_tot_box_num, 'precision': cfg.new_box_tot_precision}

                        cfg.log_xalg(cfg.new_box_precision_per_im, log_path=r'D:\BoyangDeng\test\_new_box_precision_per_im.txt', overwrite=True, show=False)

                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
                    correct_bboxes_mAP50 = correct_bboxes[:, 0]
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)
            if not self.training:
                if correct_bboxes_mAP50 is not None:
                    recall_ = round(torch.sum(correct_bboxes_mAP50).cpu().numpy() / len(cls), 2)
                    global accumulate_finds
                    global accumulate_GT
                    global accumulate_sample_num
                    accumulate_sample_num += 1
                    accumulate_finds += torch.sum(correct_bboxes_mAP50).cpu().numpy()
                    accumulate_GT += len(cls)
                    if cfg.show_recal_per_im:
                        cfg.log_xalg("tot recall:", round(accumulate_finds / accumulate_GT, 2), "cur recall:", recall_, 'gt box num:', len(cls), 'pred box num:', len(correct_bboxes_mAP50),
                            'corret box num:', torch.sum(correct_bboxes_mAP50).cpu().numpy(), 'precision 50%:', round(torch.sum(correct_bboxes_mAP50).cpu().numpy()/max(len(correct_bboxes_mAP50), 1), 4), batch['img'].shape, os.path.basename(batch['im_file'][si]), log_path=r'D:\BoyangDeng\test\_optimized_precision_per_im.txt', overwrite=False, show=True)

                if cfg.plot_maturity_rate_and_counting:
                    import matplotlib.pyplot as plt
                    import pandas as pd
                    from sklearn.metrics import r2_score
                    save_dir = r'./result_figures'
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    fig = plt.figure()

                    global predict_mature_nums
                    global predict_immature_nums
                    global gt_mature_nums
                    global gt_immature_nums

                    global predict_mature_ratios
                    global gt_mature_ratios

                    valid_predn_idxes = predn[:, 4] > cfg.plot_bbox_conf
                    valid_predn = predn[valid_predn_idxes]
                    immature_idxes = valid_predn[:, 5] == cfg.immature_label_idx
                    mature_idxes = valid_predn[:, 5] == cfg.mature_label_idx
                    immature_num = int(torch.sum(immature_idxes).cpu().numpy())
                    mature_num = int(torch.sum(mature_idxes).cpu().numpy())
                    immature_idxes_gt = labelsn[:, 0] == cfg.immature_label_idx
                    mature_idxes_gt = labelsn[:, 0] == cfg.mature_label_idx
                    immature_num_gt = int(torch.sum(immature_idxes_gt).cpu().numpy())
                    mature_num_gt = int(torch.sum(mature_idxes_gt).cpu().numpy())

                    predict_immature_nums += [immature_num]
                    predict_mature_nums += [mature_num]
                    gt_immature_nums += [immature_num_gt]
                    gt_mature_nums += [mature_num_gt]

                    predict_mature_ratios += [mature_num / (mature_num + immature_num)]
                    gt_mature_ratios += [mature_num_gt / (mature_num_gt + immature_num_gt)]
                    ideal_x = np.linspace(0, 1, 100)

                    global predict_countings
                    global gt_countings
                    predict_countings += [immature_num + mature_num]
                    gt_countings += [immature_num_gt + mature_num_gt]
                    if len(predict_mature_ratios) >= 3:

                        theta_mature_ratios = np.polyfit(gt_mature_ratios, predict_mature_ratios, 1)
                        fit_line_mature_ratios = theta_mature_ratios[1] + theta_mature_ratios[0] * np.array(gt_mature_ratios)
                        r2_score_mature_ratios = r2_score(gt_mature_ratios, predict_mature_ratios)

                        plt.title("Maturity estimation, testing: r-squared = {:.3f}".format(r2_score_mature_ratios))
                        plt.plot(gt_mature_ratios, predict_mature_ratios, 'r*', label='Data')
                        plt.plot(ideal_x, ideal_x, 'gray', label='Y=X')
                        plt.plot(gt_mature_ratios, fit_line_mature_ratios, 'green', label='Fit')
                        plt.tick_params(axis='both', which='major', labelsize=14)
                        plt.xlabel('GT maturity ratio', fontsize=14)
                        plt.ylabel('Predicted maturity ratio', fontsize=14)
                        plt.legend(prop={'size': 14})
                        fig.tight_layout()
                        # plt.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.03)
                        plt.rcParams['figure.dpi'] = 800
                        plt.rcParams['savefig.dpi'] = 800
                        save_path = os.path.join(save_dir, "maturity_ratio_" + str(accumulate_sample_num))
                        fig.savefig(save_path)

                        fig = plt.figure()

                        theta_countings = np.polyfit(gt_countings, predict_countings, 1)
                        fit_line_countings = theta_countings[1] + theta_countings[0] * np.array(gt_countings)
                        r2_score_countings = r2_score(gt_countings, predict_countings)

                        plt.title("Counting, testing: r-squared = {:.3f}".format(r2_score_countings))
                        plt.plot(gt_countings, predict_countings, 'r*', label='Data')
                        plt.plot(ideal_x, ideal_x, 'gray', label='Y=X')
                        plt.plot(gt_countings, fit_line_countings, 'green', label='Fit')
                        plt.tick_params(axis='both', which='major', labelsize=14)
                        plt.xlabel('GT fruit number', fontsize=14)
                        plt.ylabel('Predicted fruit number', fontsize=14)
                        plt.legend(prop={'size': 14})
                        fig.tight_layout()
                        plt.rcParams['figure.dpi'] = 800
                        plt.rcParams['savefig.dpi'] = 800
                        save_path = os.path.join(save_dir, "counting_" + str(accumulate_sample_num))
                        fig.savefig(save_path)

                        dict = {
                            'predict_mature_nums': predict_mature_nums, 'predict_immature_nums': predict_immature_nums,
                            'gt_mature_nums': gt_mature_nums, 'gt_immature_nums': gt_immature_nums,
                            'predict_mature_ratios': predict_mature_ratios, 'gt_mature_ratios': gt_mature_ratios,
                            'mature_ratios_fit_line_scale': [theta_mature_ratios[0]] * accumulate_sample_num,
                            'mature_ratios_fit_line_bias': [theta_mature_ratios[1]] * accumulate_sample_num,
                            'countings_fit_line_scale': [theta_countings[0]] * accumulate_sample_num,
                            'countings_fit_line_bias': [theta_countings[1]] * accumulate_sample_num,
                            'mature_ratios_r2_score': [r2_score_mature_ratios] * accumulate_sample_num,
                            'countings_r2_score': [r2_score_countings] * accumulate_sample_num,
                            'predict_countings': predict_countings, 'gt_countings': gt_countings,
                        }
                        df = pd.DataFrame(dict)
                        save_path = os.path.join(save_dir, "plot_maturity_rate_and_counting_" + str(accumulate_sample_num) + '.csv')
                        df.to_csv(save_path)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)
        self.nt_per_class = np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):
        pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels')

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        if self.args.plots:
            self.confusion_matrix.plot(save_dir=self.save_dir, names=list(self.names.values()))

    def _process_batch(self, detections, labels, pre_name=''):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        optimized_detections = []
        if 'optimized' in pre_name:
            if 'update' in  pre_name:
                update_box_idxes = cfg.update_box_idxes 
                for update_box_idx in update_box_idxes:
                    if len(optimized_detections) == 0:
                        optimized_detections = detections[update_box_idx][None, :]
                    else:
                        optimized_detections = torch.concat([optimized_detections, detections[update_box_idx][None, :]], 0)
            if 'new' in  pre_name:
                new_box_idxes = cfg.new_box_idxes 
                for new_box_idx in new_box_idxes:
                    if len(optimized_detections) == 0:
                        optimized_detections = detections[new_box_idx][None, :]
                    else:
                        optimized_detections = torch.concat([optimized_detections, detections[new_box_idx][None, :]], 0)

            detections = optimized_detections
        if len(detections) == 0:
            correct = np.zeros((0, self.iouv.shape[0])).astype(bool)
            return torch.tensor(correct, dtype=torch.bool, device=labels.device)

        iou = box_iou(labels[:, 1:], detections[:, :4])
        #if cfg.use_update_boxes:
        #    for update_box_idx in cfg.update_box_idxes:
        #        iou[iou[:, update_box_idx]>0.2, update_box_idx] += 0.3

        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(self.iouv)):
            x = torch.where((iou >= self.iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    #if cfg.use_update_boxes:
                    #    for update_box_idx in cfg.update_box_idxes:
                    #        det_idxes = matches[:, 1]
                    #        det_idxes = matches[:, 1] == update_box_idx
                    #        if len(det_idxes) > 0:
                    #            matches[det_idxes][:, 2] += 0.2

                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        if cfg.show_uncorrected_detections:
            one_im_corrected_boxes = []
            one_im_uncorrected_boxes = []
            for i_det in range(len(correct)):
                box = detections[i_det, :]
                if correct[i_det][0] == True:
                    if len(one_im_corrected_boxes) == 0:
                        one_im_corrected_boxes = box[None, :]
                    else:
                        one_im_corrected_boxes = torch.concat([one_im_corrected_boxes, box[None, :]], 0)
                else:
                    if len(one_im_uncorrected_boxes) == 0:
                        one_im_uncorrected_boxes = box[None, :]
                    else:
                        one_im_uncorrected_boxes = torch.concat([one_im_uncorrected_boxes, box[None, :]], 0)
            if len(one_im_corrected_boxes) > 0:
                cfg.show_image_with_bbox(cfg.current_im_origin, targets=one_im_corrected_boxes, font_scale=0.8, font_thickness=2, name=cfg.current_im_im_base_name+pre_name+'_corrected_boxes', save=True, show=False)
            if len(one_im_uncorrected_boxes) > 0:
                cfg.show_image_with_bbox(cfg.current_im_origin, targets=one_im_uncorrected_boxes, font_scale=0.8, font_thickness=2, name=cfg.current_im_im_base_name+pre_name+'_corrected_un_boxes', save=True, show=False)
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)

    def get_dataloader(self, dataset_path, batch_size):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return create_dataloader(path=dataset_path,
                                 imgsz=self.args.imgsz,
                                 batch_size=batch_size,
                                 stride=gs,
                                 hyp=vars(self.args),
                                 cache=False,
                                 pad=0.5,
                                 rect=self.args.rect,
                                 workers=self.args.workers,
                                 prefix=colorstr(f'{self.args.mode}: '),
                                 shuffle=False,
                                 seed=self.args.seed)[0] if self.args.v5loader else \
            build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, names=self.data['names'],
                             mode='val')[0]

    def plot_val_samples(self, batch, ni):
        plot_images(batch['img'],
                    batch['batch_idx'],
                    batch['cls'].squeeze(-1),
                    batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_labels.jpg',
                    names=self.names)

    def plot_predictions(self, batch, preds, ni):
        plot_images(batch['img'],
                    *output_to_target(preds, max_det=cfg.max_det),
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names)  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def pred_to_json(self, predn, filename):
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)})

    def eval_json(self, stats):
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data['path'] / 'annotations/instances_val2017.json'  # annotations
            pred_json = self.save_dir / 'predictions.json'  # predictions
            LOGGER.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements('pycocotools>=2.0.6')
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f'{x} file not found'
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, 'bbox')
                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = eval.stats[:2]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f'pycocotools unable to run: {e}')
        return stats


def val(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'

    args = dict(model=model, data=data)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).val(**args)
    else:
        validator = DetectionValidator(args=args)
        validator(model=args['model'])


if __name__ == '__main__':
    val()
