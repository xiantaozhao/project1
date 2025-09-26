# src/model/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm="none"):
        super().__init__()
        layers = [conv3x3(in_ch, out_ch), nn.ReLU(inplace=True),
                  conv3x3(out_ch, out_ch), nn.ReLU(inplace=True)]
        if norm == "bn":
            layers = [conv3x3(in_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                      conv3x3(out_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        elif norm == "in":
            layers = [conv3x3(in_ch, out_ch), nn.InstanceNorm2d(out_ch, affine=True), nn.ReLU(inplace=True),
                      conv3x3(out_ch, out_ch), nn.InstanceNorm2d(out_ch, affine=True), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm="none"):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, norm)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, norm="none"):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch, norm)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, norm)
        self.bilinear = bilinear

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad to match skip
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=48, depth=4, bilinear=True, norm="none"):
        super().__init__()
        assert depth in (3,4,5), "depth 仅支持 3/4/5"
        chs = [base_ch * (2**i) for i in range(depth)]
        self.inc = DoubleConv(in_ch, chs[0], norm)
        self.downs = nn.ModuleList([Down(chs[i], chs[i+1], norm) for i in range(depth-1)])
        self.ups = nn.ModuleList([])
        for i in range(depth-1, 0, -1):
            self.ups.append(Up(chs[i] + chs[i-1], chs[i-1], bilinear, norm))
        self.outc = nn.Conv2d(chs[0], out_ch, kernel_size=1)

    def forward(self, x):
        xs = [self.inc(x)]
        for d in self.downs:
            xs.append(d(xs[-1]))
        y = xs[-1]
        for i, up in enumerate(self.ups):
            y = up(y, xs[-2 - i])
        return self.outc(y)
