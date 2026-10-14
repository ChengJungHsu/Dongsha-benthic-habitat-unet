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

class DepthFeatureExtractor(nn.Module):
    """
    水深獨立降採樣特徵分支：
    將 (B, 1, 256, 256) 的水深平滑圖降採樣 4 次 (共 1/16 尺度，對應 16x16 網格)
    提取出高層語意地形特徵，不干擾淺層光譜。
    """
    def __init__(self, out_channels=32):
        super(DepthFeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 128x128
            nn.GroupNorm(1, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.GroupNorm(2, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1), # 32x32
            nn.GroupNorm(4, 24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, out_channels, kernel_size=3, stride=2, padding=1), # 16x16
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, depth):
        return self.net(depth)


class BHUNet_LateFusion(nn.Module):
    """
    延後特徵融合 (Late Fusion) BH-UNet：
    輸入格式保持不變：(B, 9, H, W)，其中 [:, 0:8] 為光譜，[:, 8:9] 為水深。
    無需修改任何資料加載器 (DataLoader) 或前處理產生的 .npz！
    """
    def __init__(self, num_classes=6, base_channels=16, depth_channels=32):
        super(BHUNet_LateFusion, self).__init__()
        c = base_channels # 16

        # 1. 光譜主幹編碼器 (專心處理 8 個光譜通道)
        self.enc1 = ConvBlock(8, c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(c, c * 2) # 32
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c * 2, c * 4) # 64
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(c * 4, c * 8) # 128
        self.pool4 = nn.MaxPool2d(2)

        # 2. 光譜 Bottleneck
        self.bottleneck = ConvBlock(c * 8, c * 16) # 256

        # 3. 水深獨立分支 (輸出 depth_channels=32)
        self.depth_branch = DepthFeatureExtractor(out_channels=depth_channels)

        # 4. 融合降維卷積：將 (256 + 32 = 288) 降維回到 256
        self.fuse_conv = nn.Sequential(
            nn.Conv2d((c * 16) + depth_channels, c * 16, kernel_size=1, bias=False),
            nn.GroupNorm(8, c * 16),
            nn.ReLU(inplace=True)
        )

        # 5. 解碼器 (結構與通道數完全保持原樣，與跳躍連接無縫相容)
        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = DecoderBlock(c * 2, c, c)

        self.final_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        # 於網路輸入端解耦：切分光譜與水深
        spec_x = x[:, :8, :, :]   # (B, 8, H, W)
        depth_x = x[:, 8:, :, :]  # (B, 1, H, W)

        # 步驟一：光譜特徵提取 (不受水深數值干擾)
        e1 = self.enc1(spec_x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4)) # (B, 256, H/16, W/16)

        # 步驟二：提取水深高階地形特徵
        d_feat = self.depth_branch(depth_x) # (B, 32, H/16, W/16)

        # 步驟三：在 Bottleneck 進行延後拼接融合
        fused_b = torch.cat([b, d_feat], dim=1) # (B, 288, H/16, W/16)
        fused_b = self.fuse_conv(fused_b)       # (B, 256, H/16, W/16)

        # 步驟四：正常解碼出圖
        d4 = self.dec4(fused_b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        logits = self.final_conv(d1)
        return logits