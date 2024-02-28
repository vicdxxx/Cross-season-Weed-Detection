import os

use_domain_adaptation = True
da_info = {}
if use_domain_adaptation:
    # CottonWeedDet12 WeedData2022 BlueberryDenseDetection_sliced_1920
    da_info['td_dataset'] = 'CottonWeedDet12'

    #da_info['root_dir'] = r'E:\PHD\WeedDetection'
    da_info['root_dir'] = r'D:\Dataset\WeedData'
    #da_info['root_dir'] = r'D:\BoyangDeng\BlueberryDenseDetection\PaddleDetection\dataset'

    da_info['td_dataset_dir'] = os.path.join(da_info['root_dir'], da_info['td_dataset'])

    da_info['td_dataset_train'] = os.path.join(da_info['td_dataset_dir'], 'train2017_8cls')
    da_info['td_dataset_val'] = os.path.join(da_info['td_dataset_dir'], 'val2017_8cls')
    da_info['td_dataset_test'] = os.path.join(da_info['td_dataset_dir'], 'test2017_8cls')
    da_info['td_dataset_all'] = os.path.join(da_info['td_dataset_dir'], 'WeedPlantsXXX_8skus_in_2021')

    #da_info['td_dataset_train'] = os.path.join(da_info['td_dataset_dir'], 'instances_test2017.0_images_1920_01')
    #da_info['td_dataset_val'] = os.path.join(da_info['td_dataset_dir'], 'instances_test2017.0_images_1920_01')
    #da_info['td_dataset_test'] = os.path.join(da_info['td_dataset_dir'], 'instances_test2017.0_images_1920_01')

    # cfg_da.da_info['td_cls_num'] = self.model.nc
    da_info['td_cls_num'] = 12
    #da_info['td_cls_num'] = 4

    da_info['pa_list'] = [18, 21]
    da_info['dloss_l_weight'] = 1.0
    da_info['kl_weight'] = 0.1
    da_info['gcr_weight'] = 1.0
    da_info['dcbr_weight'] = 0.05
    da_info['pa_losses_weight'] = 0.1
    da_info['open_all_loss_epoch_idx'] = 3
    da_info['lr'] = 0.001
    da_info['net_gcr_lr'] = 0.1
    # da_info['lr'] = 0.0
    # da_info['net_gcr_lr'] = 0.00
    
    da_info['model_L2Norm'] = None
    da_info['model_netD_pixel'] = None
    da_info['model_netD'] = None
    da_info['model_conv_gcr'] = None
    da_info['model_RandomLayer'] = None
    da_info['old_state'] = None
    da_info['new_state'] = None
    #256 320
    da_info['channel_num'] = 256
    da_info['inner_channel_num'] = da_info['channel_num']
    #da_info['inner_channel_num'] = da_info['channel_num']//3
