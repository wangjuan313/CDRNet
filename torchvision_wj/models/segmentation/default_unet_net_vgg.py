from torch import nn
from ..vgg_backbone import vgg_backbone
from torchvision_wj.models.segmentation.backbone_unet import *

__all__ = ["unet_vgg"]

def unet_vgg(input_dim, backbone_cfg, cfg_up, cfg_seg, nb_features=1, nb_output=1,
        seg_num_classes=2, model_version='Backbone_UNet_v1', seg_prior=0.01, softmax=False):

    backbone, return_layers, in_channels_list = vgg_backbone(input_dim, backbone_cfg)
    model = eval(model_version)(backbone, return_layers, in_channels_list, 
                        cfg_up, cfg_seg, nb_features, nb_output,
                        seg_num_classes, seg_prior, softmax)

    return model