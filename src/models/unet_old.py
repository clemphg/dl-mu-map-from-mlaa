"""Hwang et al., 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """2 3D convolution layers with rectified linear units and batch normalization"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_filters=12):
        super(UNet, self).__init__()
        
        # initial merging layer
        self.initial = nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1)
        
        # contracting path (ensure the number of filters double at each step)
        self.enc1 = ConvBlock(base_filters, base_filters)
        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # bottleneck
        self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8)
        
        # expanding path
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters)
        
        # output layer
        self.out_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.initial(x)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        out = self.out_conv(d1)
        return out


# Example usage
if __name__ == "__main__":

    model = UNet()

    x = torch.randn(8, 2, 32, 32, 32)  # batch size of 8, 2 input channels, 32x32x32 volume

    y = model(x)
    print(y.shape)  # output shape: (8, 1, 32, 32, 32)
