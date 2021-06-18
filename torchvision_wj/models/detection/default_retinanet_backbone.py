import torch
from torch import nn
from.feature_pyramid_network import FeaturePyramidNetwork
from .._utils import IntermediateLayerGetter
from ..vgg_backbone import vgg_backbone, vgg_backbone_v2
from torchvision_wj.models.segmentation.utils import Up
import math
import numpy as np

class Backbone_UNet_v1(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, 
                 cfg_up, num_classes=2, 
                 fpn_out_channels=256, pyramid_levels=[2,3,4,5]):
        super(Backbone_UNet_v1, self).__init__()
        self.num_classes    = num_classes
        self.stages         = len(in_channels_list)
        backbone_out_channels = backbone.out_channels

        cfg_up_invert = cfg_up[::-1]
        print(in_channels_list)
        self._initialize_weights()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        nb_params = sum(p.numel() for p in self.body.parameters() if p.requires_grad)
        print('# trainable parameters in backbone: {}'.format(nb_params))

        self.ups = nn.ModuleList()
        up_channels1 = [in_channels_list[-1]]+[cfg[-1] for cfg in cfg_up][::-1][:self.stages-2]
        up_channels2 = in_channels_list[:self.stages-1][::-1]
        up_channels = [up1+up2 for up1, up2 in zip(up_channels1, up_channels2)]
        print("** up channel", up_channels1)
        print("** down channel", up_channels2)
        print("** decoder channels: ", up_channels)
        
        for k in range(self.stages-1):
            up = Up(cfg_up_invert[k], up_channels[k])
            self.ups.append(up)
        self.ups = self.ups[:len(pyramid_levels)-1]
        nb_params = []
        for model in self.ups:
            nb_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('# trainable parameters in decoder network: {}'.format(np.sum(nb_params)))

        # fpn_in_channels_list = [cfg[-1] for cfg in cfg_up][:4]
        fpn_in_channels_list = [cfg[-1] for cfg in cfg_up]+[backbone_out_channels]
        fpn_in_channels_list = fpn_in_channels_list[-4:]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=fpn_in_channels_list,
            out_channels=fpn_out_channels,
            pyramid_levels=pyramid_levels,
        )
        self.out_channels = fpn_out_channels
        nb_params = sum(p.numel() for p in self.fpn.parameters() if p.requires_grad)
        print('# trainable parameters in fpn: {}'.format(nb_params))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, in_channels, batch_norm=True):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, images):
        x = self.body(images)
        x = list(x.values())[::-1]
        y = [x[0]]
        for k, up_block in enumerate(self.ups):
            y.append(up_block(y[-1], x[k+1]))

        y = y[::-1]
        det_feats = self.fpn(y)
        return det_feats

def vgg_unet_fpn_det(input_dim, backbone_cfg, seg_num_classes=2, model_version='Backbone_UNet_v1',
        cfg_up=[[128, 128], [128, 128], [256, 256], [512, 512], [512, 512]],
        fpn_out_channels=256, pyramid_levels=[2,3,4,5]):

    backbone, return_layers, in_channels_list = vgg_backbone(input_dim, backbone_cfg)
    model = eval(model_version)(backbone, return_layers, in_channels_list, 
                        cfg_up, seg_num_classes, fpn_out_channels, pyramid_levels)

    return model


class Backbone_v1(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, 
                 num_classes=2, fpn_out_channels=256, pyramid_levels=[2,3,4,5]):
        super(Backbone_v1, self).__init__()
        self.num_classes    = num_classes   
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        nb_params = sum(p.numel() for p in self.body.parameters() if p.requires_grad)
        print('# trainable parameters in backbone: {}'.format(nb_params))

        fpn_in_channels_list = in_channels_list[-4:]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=fpn_in_channels_list,
            out_channels=fpn_out_channels,
            pyramid_levels=pyramid_levels,
        )
        self.out_channels = fpn_out_channels
        nb_params = sum(p.numel() for p in self.fpn.parameters() if p.requires_grad)
        print('# trainable parameters in fpn: {}'.format(nb_params))

    def forward(self, images):
        x = self.body(images)
        y = list(x.values())
        det_feats = self.fpn(y[-4:])
        return det_feats

def vgg_fpn_det(input_dim, backbone_cfg, seg_num_classes=2, model_version='Backbone_v1',
        fpn_out_channels=256, pyramid_levels=[2,3,4,5]):

    backbone, return_layers, in_channels_list = vgg_backbone(input_dim, backbone_cfg)
    model = eval(model_version)(backbone, return_layers, in_channels_list, 
                        seg_num_classes, fpn_out_channels, pyramid_levels)
    return model

def vgg_fpn_det_v2(input_dim, backbone_cfg, seg_num_classes=2, model_version='Backbone_v1',
        fpn_out_channels=256, pyramid_levels=[2,3,4,5]):

    backbone, return_layers, in_channels_list = vgg_backbone_v2(input_dim, backbone_cfg)
    model = eval(model_version)(backbone, return_layers, in_channels_list, 
                        seg_num_classes, fpn_out_channels, pyramid_levels)
    return model