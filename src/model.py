import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelwiseAttention(nn.Module):
    """
    Channel-wise Attention Module (Squeeze-and-Excitation block)
    Computes global channel dependencies to re-weight spectral/bathymetric feature maps.
    """
    def __init__(self, in_channels: int):
        super(ChannelwiseAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        gap_val = self.gap(x).view(b, c)
        attn = self.sigmoid(self.fc(gap_val)).view(b, c, 1, 1)
        return x * attn


class ConvBlock(nn.Module):
    """
    Dual Convolution Block with Group Normalization and Channelwise Attention.
    Uses GroupNorm to maintain gradient stability across varying/small batch sizes.
    """
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super(ConvBlock, self).__init__()
        groups = num_groups if out_channels % num_groups == 0 else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        self.attn = ChannelwiseAttention(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        return self.attn(x)


class DecoderBlock(nn.Module):
    """
    Decoder Block with Bilinear Upsampling and Skip-Connection Concat.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_groups: int = 8):
        super(DecoderBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        groups = num_groups if (in_channels // 2) % num_groups == 0 else 1
        self.gn1x1 = nn.GroupNorm(groups, in_channels // 2)
        self.relu = nn.ReLU(inplace=True)

        fused_channels = (in_channels // 2) + skip_channels
        self.conv_block = ConvBlock(fused_channels, out_channels, num_groups=num_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.relu(self.gn1x1(self.conv1x1(x)))

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class BHUNet(nn.Module):
    """
    Benthic Habitat UNet (BH-UNet) with Squeeze-and-Excitation Attention.
    
    Args:
        in_channels (int): Input feature channels (e.g., 9 for 8-band + bathymetry).
        num_classes (int): Number of semantic habitat classes (e.g., 6 for ACA classes).
        base_channels (int): Base feature multiplier (default: 16).
    """
    def __init__(self, in_channels: int = 9, num_classes: int = 6, base_channels: int = 16):
        super(BHUNet, self).__init__()
        c = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(c * 8, c * 16)

        # Decoder with Skip Connections
        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = DecoderBlock(c * 2, c, c)

        # Output projection
        self.final_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.final_conv(d1)
