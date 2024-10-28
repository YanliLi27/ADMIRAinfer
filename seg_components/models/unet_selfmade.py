import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class PreDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DualUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, 
                 width:int=4, softmax_flag:bool=True, 
                 bilinear=False, init:Literal[True, False, 'kaiming_norm', 'glorot_uniform']=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        w= width
        
        self.inc = (DoubleConv(n_channels, 16*w))  # x [b, 1, l, w]-> x1 [b, 16*, l, w]
        self.down1 = (PreDown(16*w, 32*w))  # x1  [b, 16*, l, w] -> [b, 16*, l/2, w/2] -> x2 [b, 32*, l/2, w/2]
        self.down2 = (PreDown(32*w, 64*w))  # x2  [b, 32*, l/2, w/2] -> [b, 32*, l/4, w/4] -> x3 [b, 64*, l/4, w/4]
        self.down3 = (PreDown(64*w, 128*w))  # x3  [b, 64*, l/4, w/4] -> [b, 64*, l/8, w/8] -> x4 [b, 128*, l/8, w/8]
        factor = 2 if bilinear else 1
        self.down4 = (PreDown(128*w, 128*w//factor))  # x4  [b, 128*, l/8, w/8] -> [b, 128*, l/16, w/16] -> x5 [b, 128*, l/16, w/16]
        self.bottom = (DoubleConv(128*w // factor, 256*w // factor))  # x5 [b, 128*, l/16, w/16] --> x5 [b, 256*, l/16, w/16]
        self.up1 = DualUp(256*w // factor, 128*w//factor, bilinear)
        # x5 [b, 256*, l/16, w/16] --> x5 [b, 128*, l/8, w/8]
        # x5 + x4 -- [b, 2x128*, 1/8, w/8]  --> x [b, 128*, l/8, w/8]
        self.up2 = (DualUp(128*w//factor, 64*w // factor, bilinear))  
        # x [b, 128*, l/8, w/8] --> x [b, 64*, l/4, w/4]
        # x + x3 -- [b, 2x64*, l/4, w/4]  --> x [b, 64*, l/4, w/4]
        self.up3 = (DualUp(64*w//factor, 32*w // factor, bilinear))
        self.up4 = (DualUp(32*w//factor, 16*w//factor, bilinear))
        self.outc = (OutConv(16*w//factor, n_classes))   # [batch, n_classes, W, L]
        
        self.softmax = nn.Softmax(dim=1) if softmax_flag else None  # [batch, n_classes, W, L]   --> y [batch, I*n_classesm, W, L] one-hot

        if init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if init=='kaiming':
                        nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x1 = self.inc(x)  # x1 [b, 16*, l, w]
        x2 = self.down1(x1)  # x1  [b, 16*, l, w] -> [b, 16*, l/2, w/2] -> x2 [b, 32*, l/2, w/2]
        x3 = self.down2(x2)  # x2  [b, 32*, l, w] -> [b, 32*, l/4, w/4] -> x3 [b, 64*, l/4, w/4]
        x4 = self.down3(x3)  # x3  [b, 64*, l/4, w/4] -> [b, 64*, l/8, w/8] -> x4 [b, 128*, l/8, w/8]
        x5 = self.down4(x4)  # x4  [b, 128*, l/8, w/8] -> [b, 128*, l/16, w/16] -> x5 [b, 128*, l/16, w/16]
        x5 = self.bottom(x5)  # x5 [b, 128*, l/16, w/16] -> [b, 256*, l/16, w/16]
        x = self.up1(x5, x4)  
        # x5 [b, 256*, l/16, w/16] --> x5 [b, 128*, l/8, w/8]
        # x5 + x4 -- [b, 2x128*, 1/8, w/8]  --> x [b, 128*, l/8, w/8]
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)  # [b, 16*, l, w]
        x = self.outc(x)  # [b, n_class, l, w]
        x = self.softmax(x) if self.softmax is not None else x
        return x

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.bottom = torch.utils.checkpoint(self.bottom)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)