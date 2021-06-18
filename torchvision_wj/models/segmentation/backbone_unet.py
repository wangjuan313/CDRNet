import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from torchvision_wj.models.detection.feature_pyramid_network import FeaturePyramidNetwork
from torchvision_wj.models._utils import IntermediateLayerGetter
from .utils import Up
import numpy as np

__all__ = ['Backbone_UNet_v1','Backbone_UNet_v2']


class Backbone_UNet_v1(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, 
                 cfg_up, cfg_seg=[128,64], nb_features=1, nb_output=1, num_classes=2, 
                 prior=0.01, softmax=False):
        super(Backbone_UNet_v1, self).__init__()
        self.nb_features    = nb_features
        self.nb_output      = nb_output
        self.num_classes    = num_classes
        self.stages         = len(in_channels_list)
        assert self.nb_output<=self.nb_features

        cfg_up_invert = cfg_up[::-1]
        up_channels = [cfg[-1] for cfg in cfg_up] + [in_channels_list[-1]]
        self.seg_feats = nn.ModuleList()
        self.outs = nn.ModuleList()
        for k in range(self.nb_features):
            seg_feat = self.make_layers(cfg_seg, in_channels=up_channels[k])
            out = nn.Conv2d(cfg_seg[-1], num_classes, kernel_size=1)
            self.seg_feats.append(seg_feat)
            self.outs.append(out)
            
        nb_params = []
        for model in self.seg_feats:
            nb_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('# trainable parameters in segmentation features: {}'.format(np.sum(nb_params)))

        print(in_channels_list)

        nb_params = []
        for model in self.outs:
            nb_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('# trainable parameters in segmentation outputs: {}'.format(np.sum(nb_params)))

        if softmax:
            self.pred_func = nn.Softmax(dim=1)
        else:
            self.pred_func = nn.Sigmoid()

        self._initialize_weights()

        # initialization of segmentation output
        bias_value = -(math.log((1 - prior) / prior))
        for out in self.outs:
            out.bias.data.fill_(bias_value)

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        nb_params = sum(p.numel() for p in self.body.parameters() if p.requires_grad)
        print('# trainable parameters in backbone: {}'.format(nb_params))

        self.ups = nn.ModuleList()
        up_channels1 = [in_channels_list[-1]]+[cfg[-1] for cfg in cfg_up][::-1][:self.stages-2]
        up_channels2 = in_channels_list[:self.stages-1][::-1]
        up_channels = [up1+up2 for up1, up2 in zip(up_channels1, up_channels2)]
        print("**",up_channels1)
        print("**",up_channels2)
        print("**up channels: ", up_channels)
        
        for k in range(self.stages-1):
            up = Up(cfg_up_invert[k], up_channels[k])
            self.ups.append(up)
        nb_params = []
        for model in self.ups:
            nb_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('# trainable parameters in decoder network: {}'.format(np.sum(nb_params)))

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

    def forward(self, images, return_layers=None):
        x = self.body(images)
        x = list(x.values())[::-1]
        y = [x[0]]
        for k, up_block in enumerate(self.ups):
            y.append(up_block(y[-1], x[k+1]))

        y = y[::-1]
        seg_preds = []
        for k, (seg_feat, out) in enumerate(zip(self.seg_feats, self.outs)):
            seg_preds.append(self.pred_func(out(seg_feat(y[k]))))
        assert len(seg_preds)==self.nb_features
        # print("<<< predictions: ", [v.shape for v in seg_preds])
        return seg_preds[:self.nb_output]

class Backbone_UNet_v2(Backbone_UNet_v1):
    def __init__(self, backbone, return_layers, in_channels_list, 
                 cfg_up, cfg_seg=[128,64], nb_features=1, nb_output=1, num_classes=2, 
                 prior=0.01, softmax=False):
        super(Backbone_UNet_v2, self).__init__(backbone, return_layers, in_channels_list, 
                                cfg_up, cfg_seg, nb_features, nb_output, num_classes, prior, softmax)

        self.outs = nn.ModuleList()
        for k in range(self.nb_features):
            out_feat = cfg_seg[-1]+num_classes
            if k==self.nb_features-1:
                out_feat = cfg_seg[-1]
            out = nn.Conv2d(out_feat, num_classes, kernel_size=1)
            self.outs.append(out)
        self.outs = self.outs[::-1]

        self.up_pred = self.pred_upsampling(2)

        bias_value = -(math.log((1 - prior) / prior))
        for modules in self.outs:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, bias_value)
        
    def pred_upsampling(self, scale_factor):
        layers = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        return layers

    def forward(self, images, return_layers=None):
        x = self.body(images)
        x = list(x.values())[::-1]
        y = [x[0]]
        for k, up_block in enumerate(self.ups):
            y.append(up_block(y[-1], x[k+1]))

        y = y[::-1]
        z = []
        for k, seg_feat in enumerate(self.seg_feats):
            z.insert(0, seg_feat(y[k]))

        seg_preds, logits = [], []
        for k, out in enumerate(self.outs):
            if k==0:
                logit = out(z[k])
            else:
                logit = out(torch.cat([z[k], self.up_pred(logits[-1])], dim=1))
            logits.append(logit)
            seg_preds.append(self.pred_func(logit))
        seg_preds = seg_preds[::-1]    
        assert len(seg_preds)==self.nb_features
        return seg_preds[:self.nb_output]