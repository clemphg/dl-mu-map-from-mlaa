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


class DownBlock(nn.Module):
    """Resolution reduction was performed by 2x2x2 convolution kernels with stride 2"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_filters=64):
        super(UNet, self).__init__()
        
        # initial merging layer
        self.initial = nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1)
        
        # contracting path (ensure the number of filters double at each step)
        self.enc1 = ConvBlock(base_filters, base_filters)
        self.down1 = DownBlock(base_filters, base_filters)

        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.down2 = DownBlock(base_filters * 2, base_filters * 2)

        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.down3 = DownBlock(base_filters * 4, base_filters * 4)

        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.down4 = DownBlock(base_filters * 8, base_filters * 8)
        
        # bottleneck
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16)
        self.dropout = nn.Dropout(p=0.15)

        # expanding path
        self.up4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_filters * 16, base_filters * 8)
        
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
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        
        b = self.dropout(self.bottleneck(self.down4(e4)))
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        out = self.out_conv(d1)
        return out


# Example usage
if __name__ == "__main__":

    model = UNet()

    x = torch.randn(8, 2, 32, 64, 64)  # batch size of 8, 2 input channels, 32x64x64 volume

    y = model(x)
    print("out shape", y.shape)  # output shape: (8, 1, 32, 64, 64)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Nb of params: {pytorch_total_params}\n")
