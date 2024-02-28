import os

use_domain_adaptation = True
da_info = {}
if use_domain_adaptation:
    # CottonWeedDet12 WeedData2022
    da_info['td_dataset'] = r'D:\Dataset\WeedData\CottonWeedDet12'
    #r'E:\PHD\WeedDetection'
    #r'D:\Dataset\WeedData'
    da_info['root_dir'] = 'datasets'
    da_info['td_dataset_dir'] = os.path.join(da_info['root_dir'], da_info['td_dataset'])
    da_info['td_dataset_json'] = 'instances_train2017_8cls.0.subsample.json'
    da_info['td_dataset_name'] = 'train2017_8cls.0.subsample'
    da_info['td_dataset_train'] = os.path.join(da_info['td_dataset_dir'], 'train2017_8cls')
    da_info['td_dataset_val'] = os.path.join(da_info['td_dataset_dir'], 'val2017_8cls')
    da_info['td_dataset_test'] = os.path.join(da_info['td_dataset_dir'], 'test2017_8cls')
    da_info['td_dataset_all'] = os.path.join(da_info['td_dataset_dir'], 'WeedPlantsXXX_8skus_in_2021')
    # cfg_da.da_info['td_cls_num'] = self.model.nc
    da_info['td_cls_num'] = 12
    da_info['pa_list'] = [0, 1] # two pose
    # 1 10
    weight_multiple = 3
    da_info['dloss_l_weight'] = 1.0 * weight_multiple
    da_info['kl_weight'] = 0.1 * weight_multiple
    da_info['gcr_weight'] = 1.0 * weight_multiple
    da_info['dcbr_weight'] = 0.05 * weight_multiple
    da_info['pa_losses_weight'] = 0.1 * weight_multiple
    da_info['loss_gf_weight'] = 38 * weight_multiple*10

    da_info['open_all_loss_epoch_idx'] = 3
    # 0.001 0.01 0.1
    da_info['lr'] = 0.1
    da_info['net_gcr_lr'] = 0.1

    da_info['model_L2Norm'] = None
    da_info['model_netD_pixel'] = None
    da_info['model_netD'] = None
    da_info['model_conv_gcr'] = None
    da_info['model_RandomLayer'] = None
    da_info['old_state'] = None
    da_info['new_state'] = None
