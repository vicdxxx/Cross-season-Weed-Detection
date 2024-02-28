#data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
#data_aug_max_size = 1333
#data_aug_scales2_resize = [400, 500, 600]
#data_aug_scales2_crop = [384, 600]

#data_aug_scales = [2520, 2620, 2720, 2820, 2920, 3020, 3120, 3220, 3320, 3420, 3520]
#data_aug_max_size = 3520
#data_aug_scales2_resize = [3020, 3120, 3220]
#data_aug_scales2_crop = [2720, 3220]

data_aug_scales_max = 2560
data_aug_scales = [data_aug_scales_max-1000, data_aug_scales_max-900, data_aug_scales_max-800, data_aug_scales_max-700, data_aug_scales_max-600, data_aug_scales_max-500, data_aug_scales_max-400, data_aug_scales_max-300, data_aug_scales_max-200, data_aug_scales_max-100, data_aug_scales_max]
data_aug_max_size = data_aug_scales_max
data_aug_scales2_resize = [data_aug_scales_max-500, data_aug_scales_max-400, data_aug_scales_max-300]
data_aug_scales2_crop = [data_aug_scales_max-600, data_aug_scales_max-300]

data_aug_scale_overlap = None

