import datetime
import os, sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt

import torch
from torchvision_wj.models.detection.default_retinanet_backbone import * 
from torchvision_wj.models.detection.retinanet import Retinanet
from torchvision_wj.models.segmentation.default_unet_net_vgg import *
from torchvision_wj.models.segwithbox.unetwithbox import UNetWithBox
from torchvision_wj.models.segwithbox.default_unetwithbox_det_vgg import *
from torchvision_wj.models.segwithbox.unetwithbox_det import UNetWithBox_det
from torchvision_wj.datasets.transform import random_transform_generator
from torchvision_wj.datasets.image import random_visual_effect_generator
from torchvision_wj.utils.losses import *
from torchvision_wj.utils.early_stop import EarlyStopping
from torchvision_wj.utils import config, utils
from torchvision_wj.utils.glaucoma_utils import get_glaucoma
from torchvision_wj.utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from torchvision_wj.utils.engine import train_one_epoch, validate_loss
import torchvision_wj.utils.transforms as T
from torchvision_wj._C_glaucoma import get_experiment_config
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_exp', default=0, type=int,
                        help='the index of experiments')
    # distributed training parameters
    parser.add_argument('--world-size', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print(args)
    n_exp = args.n_exp
    detection_losses = [('FocalLoss', {'alpha':0.25, 'gamma':2.0}, 1),
                        ('SmoothL1Loss', {'sigma': 3.0}, 1)]
    segmentation_losses = [('MILUnarySigmoidLoss',{'mode':'all', 
            'focal_params':{'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}},1),
            ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
    if n_exp == 0:
        net_name = 'retinanet'
        # retinanet
        _C = get_experiment_config(net_name, bockbone_selector='B')
        cfg = {'workers': 8,
               'train_params': {'lr': 1e-3, 'batch_size': 40},
               'save_params': {'experiment_name': 'retinanet_B'}}
        _C1 = config.config_updates(_C, cfg)
        _C_array = [_C1]
    elif n_exp == 1:
        # FSIS
        net_name = 'unet_mil'
        seg_loss = [('CrossEntropyLoss', {'mode': 'all'}, 1)]
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'net_params': {'losses': {'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 24},
               'save_params': {'experiment_name': 'unet_BB_supervised'}}
        _C1 = config.config_updates(_C, cfg)
        _C_array = [_C1]
    elif n_exp == 2:
        # WSIS
        # grid search for theta in positive bags
        net_name = 'unet_mil'
        # theta = 0
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'train_params': {'lr': 1e-3, 'batch_size': 24},
               'save_params': {'experiment_name': 'unet_mil_BB'}}
        _C1 = config.config_updates(_C, cfg)
        # theta = (-40,40,10)
        seg_loss = [('MILUnaryParallelSigmoidLoss',{'angle_params':(-40,41,10), 
                    'mode':'focal', 'focal_params':{'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}},1),
                     ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'train_params': {'lr': 1e-3, 'batch_size': 24},
              'losses': {'segmentation_losses': seg_loss},
              'save_params': {'experiment_name': 'unet_mil_BB_10'}}
        _C2 = config.config_updates(_C, cfg)
        # theta = (-40,40,20)
        seg_loss = [('MILUnaryParallelSigmoidLoss',{'angle_params':(-40,41,20), 
                    'mode':'focal', 'focal_params':{'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}},1),
                     ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'train_params': {'lr': 1e-3, 'batch_size': 24},
              'losses': {'segmentation_losses': seg_loss},
              'save_params': {'experiment_name': 'unet_mil_BB_20'}}
        _C3 = config.config_updates(_C, cfg)
        # theta = (-60,60,30)
        seg_loss = [('MILUnaryParallelSigmoidLoss',{'angle_params':(-60,61,30), 
                    'mode':'focal', 'focal_params':{'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}},1),
                     ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'train_params': {'lr': 1e-3, 'batch_size': 24},
              'losses': {'segmentation_losses': seg_loss},
              'save_params': {'experiment_name': 'unet_mil_BB_30'}}
        _C4 = config.config_updates(_C, cfg)
        _C_array = [_C1, _C2, _C3, _C4]
    elif n_exp == 3:
        # WSIS, with optimal theta = (-40,40,10)
        # grid seach for alpha in smooth maximum approximation
        net_name = 'unet_mil'
        # alpha-softmax approximation, alpha = 6
        seg_loss = [('MILUnaryParallelApproxSigmoidLoss', {'angle_params':(-40,41,10), 
                     'mode':'focal', 'method':'expsumr', 'gpower':6},1),
                    ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'train_params': {'lr': 1e-3, 'batch_size': 24},
              'losses': {'segmentation_losses': seg_loss},
              'save_params': {'experiment_name': 'unet_mil_BB_10_expsumr=6'}}
        _C1 = config.config_updates(_C, cfg)
        # alpha-softmax approximation, alpha = 8
        seg_loss = [('MILUnaryParallelApproxSigmoidLoss', {'angle_params':(-40,41,10), 
                     'mode':'focal', 'method':'expsumr', 'gpower':8},1),
                    ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'train_params': {'lr': 1e-3, 'batch_size': 24},
              'losses': {'segmentation_losses': seg_loss},
              'save_params': {'experiment_name': 'unet_mil_BB_10_expsumr=8'}}
        _C2 = config.config_updates(_C, cfg)
        # alpha-softmax approximation, alpha = 10
        seg_loss = [('MILUnaryParallelApproxSigmoidLoss', {'angle_params':(-40,41,10), 
                     'mode':'focal', 'method':'expsumr', 'gpower':10},1),
                    ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'train_params': {'lr': 1e-3, 'batch_size': 24},
              'losses': {'segmentation_losses': seg_loss},
              'save_params': {'experiment_name': 'unet_mil_BB_10_expsumr=10'}}
        _C3 = config.config_updates(_C, cfg)
        # alpha-quasimax approximation, alpha = 6
        seg_loss = [('MILUnaryParallelApproxSigmoidLoss', {'angle_params':(-40,41,10), 
                     'mode':'focal', 'method':'explogs', 'gpower':6},1),
                    ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'train_params': {'lr': 1e-3, 'batch_size': 24},
              'losses': {'segmentation_losses': seg_loss},
              'save_params': {'experiment_name': 'unet_mil_BB_10_explogs=6'}}
        _C4 = config.config_updates(_C, cfg)
        # alpha-quasimax approximation, alpha = 8
        seg_loss = [('MILUnaryParallelApproxSigmoidLoss', {'angle_params':(-40,41,10), 
                     'mode':'focal', 'method':'explogs', 'gpower':8},1),
                    ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'train_params': {'lr': 1e-3, 'batch_size': 24},
              'losses': {'segmentation_losses': seg_loss},
              'save_params': {'experiment_name': 'unet_mil_BB_10_explogs=8'}}
        _C5 = config.config_updates(_C, cfg)
        # alpha-quasimax approximation, alpha = 10
        seg_loss = [('MILUnaryParallelApproxSigmoidLoss', {'angle_params':(-40,41,10), 
                     'mode':'focal', 'method':'explogs', 'gpower':10},1),
                    ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'train_params': {'lr': 1e-3, 'batch_size': 24},
              'losses': {'segmentation_losses': seg_loss},
              'save_params': {'experiment_name': 'unet_mil_BB_10_explogs=10'}}
        _C6 = config.config_updates(_C, cfg)
        _C_array = [_C1, _C2, _C3, _C4, _C5, _C6]
    elif n_exp == 4:
        # CDRNet, grid search for sigma in smooth L1 loss
        # optimal theta = (-40, 40, 10) according to WSIS
        # optimal alpha = 8 for both smooth maximum approximation functions according to WSIS
        net_name = 'unet_det_mil'
        seg_loss = [('MILUnaryParallelApproxSigmoidLoss', {'angle_params':(-40,41,10), 
                     'mode':'focal', 'method':'expsumr', 'gpower':8},1),
                    ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        # sigma = 3
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'all'},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 3.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=3'}}
        _C1 = config.config_updates(_C, cfg)
        # sigma = 4
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'all'},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 4.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=4'}}
        _C2 = config.config_updates(_C, cfg)
        # sigma = 5
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'all'},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 5.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=5'}}
        _C3 = config.config_updates(_C, cfg)
        # sigma = 6
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'all'},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 6.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6'}}
        _C4 = config.config_updates(_C, cfg)
        # sigma = 7
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'all'},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 7.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=7'}}
        _C5 = config.config_updates(_C, cfg)
        _C_array = [_C1, _C2, _C3, _C4, _C5]
    elif n_exp == 5:
        # CDRNet, grid search for T
        # optimal theta = (-40, 40, 10) according to WSIS
        # optimal alpha = 8 for both smooth maximum approximation functions according to WSIS
        # optimal sigma = 6
        net_name = 'unet_det_mil'
        seg_loss = [('MILUnaryParallelApproxSigmoidLoss', {'angle_params':(-40,41,10), 
                     'mode':'focal', 'method':'explogs', 'gpower':8},1),
                    ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        det_loss = [('SmoothL1Loss', {'sigma': 6.0}, 1)]
        # alpha-softmax approximation, T = 0.5
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.5, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': det_loss,
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=6_iou_T=0.5'}}
        _C1 = config.config_updates(_C, cfg)
        # alpha-softmax approximation, T = 0.6
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.6, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': det_loss,
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=6_iou_T=0.6'}}
        _C2 = config.config_updates(_C, cfg)
        # alpha-softmax approximation, T = 0.7
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.7, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': det_loss,
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=6_iou_T=0.7'}}
        _C3 = config.config_updates(_C, cfg)        

        seg_loss = [('MILUnaryParallelApproxSigmoidLoss', {'angle_params':(-40,41,10), 
                     'mode':'focal', 'method':'expsumr', 'gpower':8},1),
                    ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        det_loss = [('SmoothL1Loss', {'sigma': 6.0}, 1)]
        # alpha-quasimax approximation, T = 0.5
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.5, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': det_loss,
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6_iou_T=0.5'}}
        _C4 = config.config_updates(_C, cfg)
        # alpha-quasimax approximation, T = 0.6
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.6, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': det_loss,
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6_iou_T=0.6'}}
        _C5 = config.config_updates(_C, cfg)
        # alpha-quasimax approximation, T = 0.7
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.7, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': det_loss,
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6_iou_T=0.7'}}
        _C6 = config.config_updates(_C, cfg)
        _C_array = [_C1, _C2, _C3, _C4, _C5, _C6]
    elif n_exp == 6:
        # CDRNet, robuestness to sigma
        # optimal theta = (-40, 40, 10) according to WSIS
        # optimal alpha = 8 for both smooth approximation functions according to WSIS
        # optimal T = 0.6 for alpha-softmax approximation, T = 0.5 for alpha-quasimax approximation
        net_name = 'unet_det_mil'
        seg_loss = [('MILUnaryParallelApproxSigmoidLoss', {'angle_params':(-40,41,10), 
                     'mode':'focal', 'method':'expsumr', 'gpower':8},1),
                    ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        # alpha-softmax approximation, sigma=3
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.6, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 3.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=3_iou_T=0.6'}}
        _C1 = config.config_updates(_C, cfg)
        # alpha-softmax approximation, sigma=4
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.6, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 4.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=4_iou_T=0.6'}}
        _C2 = config.config_updates(_C, cfg)
        # alpha-softmax approximation, sigma=5
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.6, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 5.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=5_iou_T=0.6'}}
        _C3 = config.config_updates(_C, cfg)
        # alpha-softmax approximation, sigma = 7
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.6, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 7.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=7_iou_T=0.6'}}
        _C4 = config.config_updates(_C, cfg)
        # alpha-softmax approximation, sigma = 8
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.6, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 8.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=8_iou_T=0.6'}}
        _C5 = config.config_updates(_C, cfg)
        _C_array = [_C5]
    elif n_exp == 7:
        # CDRNet, robuestness to sigma
        # optimal theta = (-40, 40, 10) according to WSIS
        # optimal alpha = 8 for both smooth approximation functions according to WSIS
        # optimal T = 0.6 for alpha-softmax approximation, T = 0.5 for alpha-quasimax approximation
        net_name = 'unet_det_mil'
        seg_loss = [('MILUnaryParallelApproxSigmoidLoss', {'angle_params':(-40,41,10), 
                     'mode':'focal', 'method':'explogs', 'gpower':8},1),
                    ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]
        # alpha-quasimax approximation, sigma = 3
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.5, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 3.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=3_iou_T=0.5'}}
        _C1 = config.config_updates(_C, cfg)
        # alpha-quasimax approximation, sigma = 4
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.5, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 4.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=4_iou_T=0.5'}}
        _C2 = config.config_updates(_C, cfg)
        # alpha-quasimax approximation, sigma = 5
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.5, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 5.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=5_iou_T=0.5'}}
        _C3 = config.config_updates(_C, cfg)
        # alpha-quasimax approximation, sigma = 7
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.5, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 7.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=7_iou_T=0.5'}}
        _C4 = config.config_updates(_C, cfg)
        # alpha-quasimax approximation, sigma = 8
        _C = get_experiment_config(net_name, bockbone_selector='B', up_selector='B')
        cfg = {'detector_params': {'detach': False},
               'net_params': { 'backbone_params': {'nb_features': 1, 'nb_output': 1,
               'det_level_in_seg': [0], 'num_conv': 6, 'head_mode': 'dilated'},
               'detection_sample_selection': {'method': 'iou', 'iou_th': 0.5, 'use_seg': False, 'sigma': 2.0},
               'box_normalizer': [40, 70],#[40.4914, 69.5429],
               'losses': {'detection_losses': [('SmoothL1Loss', {'sigma': 8.0}, 1)],
                          'segmentation_losses': seg_loss}},
               'train_params': {'lr': 1e-3, 'batch_size': 8},
               'save_params': {'experiment_name': 'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=8_iou_T=0.5'}}
        _C5 = config.config_updates(_C, cfg)
        _C_array = [_C1, _C2, _C3, _C4, _C5] 

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    for _C_used in _C_array:
        assert _C_used['save_params']['experiment_name'] is not None, "experiment_name has to be set"

        train_params       = _C_used['train_params']
        data_params        = _C_used['data_params']
        net_params         = _C_used['net_params']
        detector_params    = _C_used['detector_params']
        dataset_params     = _C_used['dataset']
        save_params        = _C_used['save_params']
        data_visual_aug    = data_params['data_visual_aug']
        data_transform_aug = data_params['data_transform_aug']

        output_dir = os.path.join(save_params['dir_save'],save_params['experiment_name'])
        os.makedirs(output_dir, exist_ok=True)
        if not train_params['test_only']:
            config.save_config_file(os.path.join(output_dir,'config.yaml'), _C_used)
            print("saving files to {:s}".format(output_dir))

        device = torch.device(_C_used['device'])
        epoch_per_save = train_params['epoch_per_save']

        def get_transform(train):
            transforms = []
            transforms.append(T.ToTensor())
            transforms.append(T.Resize(width=data_params['crop_width']))
            transforms.append(T.Normalizer(mode=data_params['normalizer_mode']))
            if train&(data_params['crop_ob_prob']>0):
                transforms.append(T.RandomCrop(prob=data_params['crop_ob_prob'], 
                                               crop_size=data_params['crop_size'], 
                                               weak=True))
            return T.Compose(transforms)

        if data_transform_aug['aug']:
            transform_generator = random_transform_generator(
                min_rotation=data_transform_aug['min_rotation'],
                max_rotation=data_transform_aug['max_rotation'],
                min_translation=data_transform_aug['min_translation'],
                max_translation=data_transform_aug['max_translation'],
                min_shear=data_transform_aug['min_shear'],
                max_shear=data_transform_aug['max_shear'],
                min_scaling=data_transform_aug['min_scaling'],
                max_scaling=data_transform_aug['max_scaling'],
                flip_x_chance=data_transform_aug['flip_x_chance'],
                flip_y_chance=data_transform_aug['flip_y_chance'],
                )
        else:
            transform_generator = None
        if data_visual_aug['aug']:
            visual_effect_generator = random_visual_effect_generator(
                contrast_range=data_visual_aug['contrast_range'],
                brightness_range=data_visual_aug['brightness_range'],
                hue_range=data_visual_aug['hue_range'],
                saturation_range=data_visual_aug['saturation_range']
                )
        else:
            visual_effect_generator = None
        print('---data augmentation---')
        print('transform: ',transform_generator)
        print('visual: ',visual_effect_generator)

        # Data loading code
        print("Loading data")
        dataset      = get_glaucoma(root=dataset_params['root_path'], csv_file=dataset_params['train_file'], 
                                    transforms=get_transform(train=True),
                                    transform_generator=transform_generator,
                                    visual_effect_generator=visual_effect_generator)
        dataset_test = get_glaucoma(root=dataset_params['root_path'], csv_file=dataset_params['valid_file'], 
                                    transforms=get_transform(train=False),
                                    transform_generator=None, visual_effect_generator=None)
        
        print("Creating data loaders")
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        if data_params['aspect_ratio_group_factor'] >= 0:
            group_ids = create_aspect_ratio_groups(dataset, k=data_params['aspect_ratio_group_factor'])
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, train_params['batch_size'])
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, train_params['batch_size'], drop_last=True)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=_C_used['workers'],
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=test_sampler, num_workers=_C_used['workers'],
            collate_fn=utils.collate_fn)

        print("Creating model with parameters: {}".format(net_params))
        losses, loss_weights = dict(), dict()
        for task_name, task_losses_list in net_params['losses'].items():
            task_losses, task_loss_weights = [], []
            for loss in task_losses_list:
                task_losses.append(eval(loss[0])(**loss[1]))
                task_loss_weights.append(loss[2])
            losses[task_name] = task_losses
            loss_weights[task_name] = task_loss_weights
        backbone_params = net_params['backbone_params']
        backbone_arch = backbone_params.pop('model_name')
        backbone = eval(backbone_arch)(**backbone_params)
        if (net_name == 'retinanet'):
            model = Retinanet(backbone, losses['detection_losses'], 
                            loss_weights['detection_losses'], **detector_params) 
        elif net_name == 'unet_mil':
            model = UNetWithBox(backbone, losses['segmentation_losses'], 
                            loss_weights['segmentation_losses'], softmax=backbone_params['softmax'])
        elif net_name == 'unet_det_mil':
            model = UNetWithBox_det(backbone, losses, loss_weights, 
                            softmax=backbone_params['softmax'],
                            obj_sizes=net_params['obj_sizes'], 
                            detection_sample_selection=net_params['detection_sample_selection'],
                            box_normalizer=net_params['box_normalizer'],
                            nms_score_threshold=detector_params['nms_score_threshold'],
                            nms_iou_threshold=detector_params['nms_iou_threshold'],
                            detections_per_class=detector_params['detections_per_class'])
        model.to(device)

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                find_unused_parameters=True)
            model_without_ddp = model.module

        nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('# trainable parameters: {}'.format(nb_params))

        params = [p for p in model.parameters() if p.requires_grad]
        if train_params['optimizer']=='SGD':
            optimizer = torch.optim.SGD(params, lr=train_params['lr'], 
                                        momentum=train_params['momentum'], weight_decay=train_params['weight_decay'])
        elif train_params['optimizer']=='Adam':
            optimizer = torch.optim.Adam(params, lr=train_params['lr'], betas=train_params['betas'])

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                                  factor=train_params['factor'], 
                                                                  patience=train_params['patience'])
        early_stop = EarlyStopping(patience=2*train_params['patience'])

        if train_params['resume']:
            print('resuming model {}'.format(train_params['resume']))
            checkpoint = torch.load(os.path.join(output_dir, train_params['resume']), map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            train_params['start_epoch'] = checkpoint['epoch'] + 1

        model.training = True
        if train_params['test_only']:
            val_metric_logger = validate_loss(model, data_loader_test, device)
        else:
            print("Start training")
            start_time = time.time()
            min_val_loss = 1e10
            summary = {'epoch':[]}
            for epoch in range(train_params['start_epoch'], train_params['epochs']):
                if args.distributed:
                    train_sampler.set_epoch(epoch)

                ## add extra detection losses at epoch = 20
                if 'extra_detection_losses' in losses.keys():
                    if epoch == 20:
                        losses['detection_losses'] += losses['extra_detection_losses']
                        loss_weights['detection_losses'] += loss_weights['extra_detection_losses']
                        model.losses_func = losses
                        model.loss_weights = loss_weights

                metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, 
                                train_params['clipnorm'], train_params['print_freq'])

                epoch_save_start = 20
                if (epoch+1)%epoch_per_save==0:
                    # evaluate after every epoch
                    val_metric_logger = validate_loss(model, data_loader_test, device)
                    val_loss = val_metric_logger.meters['val_loss'].global_avg
                    if (epoch >= epoch_save_start) & (min_val_loss>val_metric_logger.meters['val_loss'].global_avg):
                        min_val_loss = val_loss
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'args': args,
                            'epoch': epoch},
                            os.path.join(output_dir, 'model_{:02d}.pth'.format(epoch)))

                    nb_save = (epoch+1) // epoch_per_save
                    # collect the results and save
                    summary['epoch'].append(epoch)
                    for name, meter in metric_logger.meters.items():
                        if name=='lr':
                            v = meter.global_avg
                        else:
                            v = float(np.around(meter.global_avg,8))
                        if name not in summary.keys():
                            if nb_save != 1:
                                summary[name] = [0]*(nb_save-1) + [v]  
                            else:  
                                summary[name] = [v]
                        else:
                            summary[name].append(v)
                    for name, meter in val_metric_logger.meters.items():
                        v = float(np.around(meter.global_avg,8))
                        if name not in summary.keys():
                            if nb_save != 1:
                                summary[name] = [0]*(nb_save-1) + [v]  
                            else:  
                                summary[name] = [v]
                        else:
                            summary[name].append(v)
                    summary_save = pd.DataFrame(summary)
                    summary_save.to_csv(os.path.join(output_dir,'summary.csv'), index=False)

                    # update lr scheduler
                    lr_scheduler.step(val_loss)

                    # early stop check
                    if epoch >= epoch_save_start:
                        if early_stop.step(val_loss):
                            print('Early stop at epoch = {}'.format(epoch))
                            break

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))

            ## plot training and validation loss
            plt.figure()
            plt.plot(summary_save['epoch'],summary_save['loss'],'-ro', label='train')
            plt.plot(summary_save['epoch'],summary_save['val_loss'],'-g+', label='valid')
            plt.legend(loc=0)
            plt.savefig(os.path.join(output_dir,'loss.jpg'))
            time.sleep(2)
            print("Training done")
