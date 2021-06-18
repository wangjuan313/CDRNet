import torch
from torch import nn

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.cfg         = cfg
        self.in_channels = in_channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = self.make_layers()
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, batch_norm=True):
        in_channels = self.in_channels
        layers = []
        for v in self.cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_Light(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.cfg         = cfg
        self.in_channels = in_channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = self.make_layers()
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, batch_norm=True):
        in_channels = self.in_channels
        layers = []
        for v in self.cfg:
            conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
            conv2 = nn.Conv2d(in_channels, v, kernel_size=1, padding=0)
            if batch_norm:
                layers += [conv1, nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                           conv2, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1, nn.ReLU(inplace=True), conv2, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2 is None:
            x = x1
        else:
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)