import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels=256,
                 pyramid_levels=[3,4,5,6,7]):
        super(FeaturePyramidNetwork, self).__init__()
        
        self.pyramid_levels = pyramid_levels
        if 2 in pyramid_levels:
            assert len(in_channels_list)==4, "in_channels_list must be length of 4"
        if len(in_channels_list)==3:
            C3_size, C4_size, C5_size = in_channels_list 
            C2_size = C3_size
        elif len(in_channels_list)==4:
            C2_size, C3_size, C4_size, C5_size = in_channels_list        

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, out_channels, kernel_size=1, stride=1, padding=0)
        # self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, out_channels, kernel_size=1, stride=1, padding=0)
        # self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, out_channels, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if 2 in self.pyramid_levels:
            # add P3 elementwise to C2
            self.P2_1 = nn.Conv2d(C2_size, out_channels, kernel_size=1, stride=1, padding=0)
            self.P2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if 6 in self.pyramid_levels:
            # P6 is obtained via a 3x3 stride-2 conv on C5
            self.P6 = nn.Conv2d(C5_size, out_channels, kernel_size=3, stride=2, padding=1)

        if 7 in self.pyramid_levels:
            # P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6
            self.P7_1 = nn.ReLU()
            self.P7_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs):
        # names = list(inputs.keys())
        if isinstance(inputs, list):
            C2, C3, C4, C5 = inputs
        else:    
            C2, C3, C4, C5 = list(inputs.values())

        # upsample C5 to get P5
        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, size=C4.shape[-2:], mode="nearest")
        P5_x = self.P5_2(P5_x)

        # add P5 elementwise to C4
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, size=C3.shape[-2:], mode="nearest")
        P4_x = self.P4_2(P4_x)

        # add P4 elementwise to C3
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        if 2 in self.pyramid_levels:
            P3_upsampled_x = F.interpolate(P3_x, size=C2.shape[-2:], mode="nearest")
        P3_x = self.P3_2(P3_x)
        
        output_layers = [P3_x, P4_x, P5_x]
        names         = ['P3','P4', 'P5']
        
        if 2 in self.pyramid_levels:
            P2_x = self.P2_1(C2)
            P2_x = P2_x + P3_upsampled_x
            P2_x = self.P2_2(P2_x)
            output_layers = [P2_x] + output_layers
            names         = ['P2'] + names

        # P6 is obtained via a 3x3 stride-2 conv on C5
        if 6 in self.pyramid_levels:
            P6_x = self.P6(C5)
            output_layers = output_layers + [P6_x]
            names         = names + ['P6']
        
        # P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6
        if 7 in self.pyramid_levels:
            if 6 not in self.pyramid_levels:
                raise ValueError("P6 is required to use P7")
            P7_x = self.P7_1(P6_x)
            P7_x = self.P7_2(P7_x)
            output_layers = output_layers + [P7_x]
            names         = names + ['P7']

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, output_layers)])
        return out