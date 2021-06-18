import os

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

def get_experiment_config(net_name, bockbone_selector='A', up_selector='A'):

    _C = {}
    _C['device'] = 'cuda'
    _C['workers'] = 4

    # -----------------------------------------------------------------------------
    # dataset
    # -----------------------------------------------------------------------------
    dataset = {}
    dataset['root_path'] = 'data/glaucoma'
    dataset['name'] = 'glaucoma_retinanet'
    dataset['train_file'] = 'train_list_wj_20201017'
    dataset['valid_file'] = 'valid_list_wj_20201017'
    dataset['test_file'] = 'test_list_wj_20201017'
    _C['dataset'] = dataset

    # -----------------------------------------------------------------------------
    # data parameters
    # -----------------------------------------------------------------------------
    data_params = {}
    data_params['aspect_ratio_group_factor'] = -1
    data_params['crop_width']      = 512
    data_params['normalizer_mode'] = 'zscore'
    data_params['crop_ob_prob']    = 0.75
    data_params['crop_size']       = (256, 256)
    data_params['data_visual_aug'] = {}
    data_params['data_visual_aug']['aug'] = True
    data_params['data_visual_aug']['contrast_range']    = (0.9, 1.1)
    data_params['data_visual_aug']['brightness_range']  = (-0.1, 0.1)
    data_params['data_visual_aug']['hue_range']         = (-0.05, 0.05)
    data_params['data_visual_aug']['saturation_range']  = (0.95, 1.05)
    data_params['data_transform_aug'] = {}
    data_params['data_transform_aug']['aug'] = True
    data_params['data_transform_aug']['min_rotation']     = 0#-0.1
    data_params['data_transform_aug']['max_rotation']     = 0#0.1
    data_params['data_transform_aug']['min_translation']  = (-0.05, -0.05)
    data_params['data_transform_aug']['max_translation']  = (0.05, 0.05)
    data_params['data_transform_aug']['min_shear']        = 0#-0.1
    data_params['data_transform_aug']['max_shear']        = 0#0.1
    data_params['data_transform_aug']['min_scaling']      = (0.95, 0.95)
    data_params['data_transform_aug']['max_scaling']      = (1.05, 1.05)
    data_params['data_transform_aug']['flip_x_chance']    = 0.5
    data_params['data_transform_aug']['flip_y_chance']    = 0.5
    _C['data_params'] = data_params

    # -----------------------------------------------------------------------------
    # anchor parameters + NMS parameters
    # -----------------------------------------------------------------------------
    detector_params = {}
    detector_params['anchor_sizes'] = ((10,), (20,), (40,), (80,))
    N = len(detector_params['anchor_sizes'])
    detector_params['anchor_aspect_ratios']  = ((0.5, 1.0, 2.0),) * N
    detector_params['anchor_scales']  = ((2**0, 2**(1/3), 2**(2/3)),) * N
    detector_params['box_low_iou_thresh'] = 0.4
    detector_params['box_high_iou_thresh'] = 0.5
    detector_params['box_stat_mean'] = [0,0,0,0]
    detector_params['box_stat_std'] = [0.2,0.2,0.2,0.2]
    detector_params['nms_score_threshold']  = 0.2
    detector_params['nms_iou_threshold']    = 0.4
    detector_params['detections_per_class'] = 1
    detector_params['feature_size'] = 256 # param for head
    detector_params['num_conv'] = 4 # param for head
    detector_params['prior'] = 0.01
    detector_params['num_classes'] = 2
    if net_name == 'retinanet_mil':
        detector_params['detach'] = True
        detector_params['pseudo_generator'] = 1
    _C['detector_params'] = detector_params

    # -----------------------------------------------------------------------------
    # network parameters for retinanet
    # -----------------------------------------------------------------------------
    backbone_cfgs = {'A': [32, 32, 'M', 64, 64, 'M', 
                           128, 128, 'M', 256, 256, 'M', 256, 256, 'M', 512],
                     'B': [48, 48, 'M', 96, 96, 'M', 
                           192, 192, 'M', 384, 384, 'M', 384, 384, 'M', 768],
                     'C': [64, 64, 'M', 128, 128, 'M', 
                           256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 1024],
                     'D': [64, 64, 'M', 128, 128, 'M', 
                           256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                     'E': [64, 64, 'M', 128, 128, 'M', 
                           128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512]}
    cfg_ups = {'A': [[64, 64], [64, 64], [128, 128], [256, 256], [256]],
               'B': [[96, 96], [96, 96], [192, 192], [384, 384], [768]],
               'C': [[128, 128], [128, 128], [256, 256], [512, 512], [1024]]}
    detection_losses = [('FocalLoss', {'alpha':0.25, 'gamma':2.0}, 1),
                        ('SmoothL1Loss', {'sigma': 3.0}, 1)]
    segmentation_losses = [('MILUnarySigmoidLoss',{'mode':'all', 
            'focal_params':{'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}},1),
            ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
    net_params = {}
    backbone_params = {}
    backbone_params['input_dim'] = 3
    backbone_params['backbone_cfg'] = backbone_cfgs[bockbone_selector]
    if net_name == 'retinanet':
        backbone_params['model_name'] = 'vgg_fpn_det'
        backbone_params['model_version'] = 'Backbone_v1'
        backbone_params['seg_num_classes'] = 2
        backbone_params['fpn_out_channels'] = 256 # 128
        backbone_params['pyramid_levels'] = [2,3,4,5]
        assert detector_params['feature_size'] == backbone_params['fpn_out_channels']
        assert len(detector_params['anchor_sizes']) == len(backbone_params['pyramid_levels'])
        assert detector_params['feature_size'] == backbone_params['fpn_out_channels']
        assert len(detector_params['anchor_sizes']) == len(backbone_params['pyramid_levels'])
        net_params['softmax'] = False
        net_params['losses'] = {'detection_losses': detection_losses}
    elif 'unet_mil' in net_name:
        backbone_params['model_name'] = 'unet_vgg'
        backbone_params['model_version'] = 'Backbone_UNet_v1'
        backbone_params['seg_num_classes'] = 2
        backbone_params['softmax'] = False
        backbone_params['cfg_up'] = cfg_ups[up_selector]
        backbone_params['cfg_seg'] = [64, 32]
        backbone_params['nb_features'] = 1
        backbone_params['nb_output'] = 1
        backbone_params['seg_prior'] = 0.01
        net_params['losses'] = {'segmentation_losses': segmentation_losses}
    elif 'unet_det_mil' in net_name:
        backbone_params['model_name'] = 'vgg_unet_det'
        backbone_params['model_version'] = 'MIL_UNet_det'
        backbone_params['seg_num_classes'] = 2
        backbone_params['softmax'] = False
        backbone_params['cfg_up'] = cfg_ups[up_selector]
        backbone_params['cfg_seg'] = [64, 32]
        backbone_params['nb_features'] = 1
        backbone_params['nb_output'] = 1
        backbone_params['seg_prior'] = 0.01
        # backbone_params['num_det_levels'] = 4
        backbone_params['det_level_in_seg'] = [-4, -3, -2, -1]
        backbone_params['num_conv'] = 4
        backbone_params['feature_size'] = 256
        backbone_params['head_mode'] = 'normal'
        net_params['obj_sizes'] = [1, 8]
        net_params['detection_sample_selection'] = {'method': 'all'}
        #{'method': 'iou', 'iou_th': 0.6, 'use_seg': True, 'sigma': 2.0}
        net_params['nms_iou_threshold'] = 0.4
        net_params['detections_per_class'] = 1
        net_params['box_normalizer'] = None
        net_params['losses'] = {'detection_losses': [('SmoothL1Loss', {'sigma': 3.0}, 1)],
                                'segmentation_losses': segmentation_losses}
    net_params['backbone_params'] = backbone_params
    _C['net_params'] = net_params

    # -----------------------------------------------------------------------------
    # model training
    # -----------------------------------------------------------------------------
    train_params = {}
    train_params['batch_size']   = 12
    train_params['epochs']       = 150
    train_params['epoch_per_save'] = 1
    train_params['start_epoch']  = 0
    train_params['resume']       = ''
    train_params['test_only']    = False
    train_params['lr']           = 1e-3
    train_params['clipnorm']	 = 0.001
    train_params['momentum']     = 0.9
    train_params['weight_decay'] = 1e-4
    train_params['lr_step_size'] = 8
    train_params['lr_steps']     = [16, 22]
    train_params['lr_gamma']     = 0.1
    train_params['factor']       = 0.1
    train_params['patience']     = 4
    train_params['print_freq']   = 50
    train_params['optimizer']    = 'Adam'#'SGD'
    train_params['betas'] = (0.9, 0.999)
    _C['train_params'] = train_params



    # -----------------------------------------------------------------------------
    # model saving
    # -----------------------------------------------------------------------------
    save_params = {}
    save_params['dir_save_root']   = 'results'
    save_params['dir_save']        = os.path.join('results', dataset['name'])
    save_params['experiment_name'] = None
    _C['save_params'] = save_params

    return _C

