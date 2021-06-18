from collections import OrderedDict
import torch
from torch import nn
from torch.nn.functional import interpolate
from .._utils import IntermediateLayerGetter
from ..vgg_backbone import vgg_backbone
from torchvision_wj.models.segmentation.utils import Up
import math
import numpy as np

class HeadSingle(nn.Module):
    def __init__(self, in_channels, num_classes, num_conv=4, feature_size=256, head_mode='normal'):
        super(HeadSingle, self).__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.num_conv = num_conv
        self.head_mode = head_mode
        assert self.head_mode in ['normal', 'dilated', 'separable']
        if head_mode == 'normal':
            self.bbox_feat = self._make_layers()
        elif head_mode == 'dilated':
            self.bbox_feat = self._make_dilated_layers()
        elif head_mode == 'separable':
            self.bbox_feat = self._make_separable_layers()
        self.bbox_pred = nn.Conv2d(feature_size, 4*num_classes, kernel_size=3, padding=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def _make_layers(self):
        in_channels  = self.in_channels
        feature_size = self.feature_size
        num_conv     = self.num_conv
        layers = OrderedDict()
        layers['conv1'] = nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1)
        layers['relu1'] = nn.ReLU()
        for k in range(1,num_conv):
            layers['conv'+str(k+1)] = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
            layers['relu'+str(k+1)] = nn.ReLU()
        return nn.Sequential(layers)

    def _make_dilated_layers(self):
        in_channels  = self.in_channels
        feature_size = self.feature_size
        num_conv     = self.num_conv
        layers = OrderedDict()
        layers['conv1'] = nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1)
        layers['relu1'] = nn.ReLU()
        layers['conv2'] = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        layers['relu2'] = nn.ReLU()
        for k in range(2,num_conv):
            layers['conv'+str(k+1)] = nn.Conv2d(feature_size, feature_size, kernel_size=3, dilation=2**(k-1), padding=2**(k-1))
            layers['relu'+str(k+1)] = nn.ReLU()
        return nn.Sequential(layers)
    
    def _make_separable_layers(self):
        in_channels  = self.in_channels
        feature_size = self.feature_size
        num_conv     = self.num_conv
        layers = OrderedDict()
        layers['conv1'] = nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1)
        layers['relu1'] = nn.ReLU()
        layers['conv2'] = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        layers['relu2'] = nn.ReLU()
        for k in range(2,num_conv):
            layers['conv'+str(k+1)+'_w'] = nn.Conv2d(feature_size, feature_size, kernel_size=(1,2*k+3), padding=(0,k+1))
            # layers['relu'+str(k+1)+'_w'] = nn.ReLU()
            layers['conv'+str(k+1)+'_h'] = nn.Conv2d(feature_size, feature_size, kernel_size=(2*k+3,1), padding=(k+1,0))
            layers['relu'+str(k+1)] = nn.ReLU()
        return nn.Sequential(layers)

    def forward(self, x):
        bbox_feat = self.bbox_feat(x)
        bbox_reg = self.bbox_pred(bbox_feat)
        return bbox_reg


class MIL_UNet_det(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, 
                 cfg_up, cfg_seg=[128,64], nb_features=6, nb_output=6, num_classes=2, 
                 prior=0.01, softmax=False, det_level_in_seg=[0,1,2,3], 
                 num_conv=4, feature_size=256, head_mode='normal'):
        super(MIL_UNet_det, self).__init__()
        self.nb_features    = nb_features
        self.nb_output      = nb_output
        self.num_classes    = num_classes
        self.stages         = len(in_channels_list)
        self.det_level_in_seg = det_level_in_seg
        assert self.nb_output <= self.nb_features
        assert len(self.det_level_in_seg) <= self.nb_output

        cfg_up_invert = cfg_up[::-1]
        up_channels = [cfg[-1] for cfg in cfg_up] + [in_channels_list[-1]]
        self.up_channels = up_channels
        self.out_channels = [up_channels[k] for k in self.det_level_in_seg]
        # print("<<<<<<< out_channels", self.out_channels)
        # print(">>>>>>> cfg_seg", cfg_seg)
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
        print("** up channel", up_channels1)
        print("** down channel", up_channels2)
        print("** decoder channels: ", up_channels)
        
        for k in range(self.stages-1):
            up = Up(cfg_up_invert[k], up_channels[k])
            self.ups.append(up)
        nb_params = []
        for model in self.ups:
            nb_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('# trainable parameters in decoder network: {}'.format(np.sum(nb_params)))

        print("<<<<<<<<<<<<<<", self.out_channels)
        self.heads_list = nn.ModuleList()
        for k in range(len(self.det_level_in_seg)):
            single_head = HeadSingle(self.out_channels[k], self.num_classes, num_conv, feature_size, head_mode)
            self.heads_list.append(single_head)
        nb_params = []
        for model in self.heads_list:
            nb_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('# trainable parameters in heads: {}'.format(nb_params))

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
        feats = []
        for k, seg_feat in enumerate(self.seg_feats):
            feats.append(seg_feat(y[k]))
        
        det_feats = [y[k] for k in self.det_level_in_seg]
        det_preds = []
        for feat, head in zip(det_feats, self.heads_list):
            pred = head(feat)
            pred = pred.reshape(pred.shape[0], self.num_classes, 4, pred.shape[-2], pred.shape[-1])
            det_preds.append(pred)

        seg_preds = []
        for k, out in enumerate(self.outs):
            seg_preds.append(self.pred_func(out(feats[k])))
        assert len(seg_preds)==self.nb_features

        # print("<<< seg_preds: ", [v.shape for v in seg_preds])
        # print("<<< det_feats: ", [v.shape for v in det_feats])
        # print("<<< det_preds: ", [v.shape for v in det_preds])
        return seg_preds[:self.nb_output], det_preds



class MIL_UNet_det_v2(MIL_UNet_det):
    def __init__(self, backbone, return_layers, in_channels_list, 
                 cfg_up, cfg_seg=[128,64], nb_features=1, nb_output=1, num_classes=2, 
                 prior=0.01, softmax=False, num_det_levels=4, num_conv=4, feature_size=256):
        super(MIL_UNet_det_v2, self).__init__(backbone, return_layers, in_channels_list, 
                 cfg_up, cfg_seg, nb_features, nb_output, num_classes, 
                 prior, softmax, num_det_levels, num_conv, feature_size)

        self.out_channels = cfg_seg[-1]
        self.heads_list = nn.ModuleList()
        for k in range(num_det_levels):
            single_head = HeadSingle(self.out_channels[k], num_conv, feature_size)
            self.heads_list.append(single_head)
        nb_params = []
        for model in self.heads_list:
            nb_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('# trainable parameters in heads: {}'.format(nb_params))

    def forward(self, images):
        x = self.body(images)
        x = list(x.values())[::-1]
        y = [x[0]]
        for k, up_block in enumerate(self.ups):
            y.append(up_block(y[-1], x[k+1]))

        y = y[::-1]
        feats = []
        for k, seg_feat in enumerate(self.seg_feats):
            feats.append(seg_feat(y[k]))
        
        det_feats = feats[-4:]
        det_preds = []
        for feat, head in zip(det_feats, self.heads_list):
            det_preds.append(head(feat))

        seg_preds = []
        for k, out in enumerate(self.outs):
            seg_preds.append(self.pred_func(out(feats[k])))
        assert len(seg_preds)==self.nb_features

        print("<<< seg_preds: ", [v.shape for v in seg_preds])
        print("<<< det_feats: ", [v.shape for v in det_feats])
        print("<<< det_preds: ", [v.shape for v in det_preds])
        return seg_preds[:self.nb_output], det_preds


def vgg_unet_det(input_dim, backbone_cfg, seg_num_classes=2, softmax=False, 
        model_version='MIL_UNet_det',
        cfg_up=[[128, 128], [128, 128], [256, 256], [512, 512], [512, 512]],
        cfg_seg=[128, 64], nb_features=1, nb_output=1, seg_prior=0.5,
        det_level_in_seg=[-4,-3,-2,-1], num_conv=4, feature_size=256, head_mode='normal'):

    backbone, return_layers, in_channels_list = vgg_backbone(input_dim, backbone_cfg)
    model = eval(model_version)(backbone, return_layers, 
                        in_channels_list, cfg_up, cfg_seg, nb_features, 
                        nb_output, seg_num_classes, seg_prior, softmax,
                        det_level_in_seg, num_conv, feature_size, head_mode)
    return model
