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

class DepthFeatureExtractor(nn.Module):
    """
    水深獨立降採樣特徵分支：
    將 (B, depth_in_channels, 256, 256) 的水深圖經過 4 次步長為 2 的卷積
    降採樣至 1/16 尺度 (16x16)，提煉出高階地形特徵。
    """
    def __init__(self, in_channels=1, out_channels=32):
        super(DepthFeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),   # 128x128
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
    通用型 BH-UNet 模型：支援純光譜、Early Fusion（全部一起算）與 Late Fusion（解耦融合）。
    
    參數說明：
    - op_channels: 光譜波段數 (例如 8 或 4)
    - depth_in_channels: 水深/地形通道數 (例如 1 或 0)
    - use_late_fusion: 
        * True : 啟用 Late Fusion（光譜走主幹，水深走獨立分支並於 Bottleneck 融合）
        * False: 啟用 Early Fusion（將光譜與水深共 9 通道直接作為 enc1 輸入，全部一起算）
    """
    def __init__(
        self,
        op_channels: int = 8,
        depth_in_channels: int = 1,
        use_late_fusion: bool = True,
        num_classes: int = 6,
        base_channels: int = 16,
        depth_channels: int = 32
    ):
        super(BHUNet_LateFusion, self).__init__()
        self.op_channels = op_channels
        self.depth_in_channels = depth_in_channels
        self.use_late_fusion = use_late_fusion
        c = base_channels # 16

        # -------------------------------------------------------------
        # 1. 判斷第一層編碼器 (enc1) 的輸入通道數
        # -------------------------------------------------------------
        if self.depth_in_channels > 0 and not self.use_late_fusion:
            # 【全部一起算 (Early Fusion)】：輸入為 8 + 1 = 9 通道
            first_in_channels = self.op_channels + self.depth_in_channels
        else:
            # 【Late Fusion】或【純光譜】：enc1 專注處理光譜通道 (8 通道)
            first_in_channels = self.op_channels

        self.enc1 = ConvBlock(first_in_channels, c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(c, c * 2) # 32
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c * 2, c * 4) # 64
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(c * 4, c * 8) # 128
        self.pool4 = nn.MaxPool2d(2)

        # -------------------------------------------------------------
        # 2. 光譜 Bottleneck
        # -------------------------------------------------------------
        self.bottleneck = ConvBlock(c * 8, c * 16) # 256

        # -------------------------------------------------------------
        # 3. 條件式建立水深分支 (僅在開啟 Late Fusion 且有水深時建立)
        # -------------------------------------------------------------
        if self.depth_in_channels > 0 and self.use_late_fusion:
            self.depth_branch = DepthFeatureExtractor(
                in_channels=depth_in_channels,
                out_channels=depth_channels
            )
            # Bottleneck 融合降維層 (256 + 32 -> 256)
            self.fuse_conv = nn.Sequential(
                nn.Conv2d((c * 16) + depth_channels, c * 16, kernel_size=1, bias=False),
                nn.GroupNorm(8, c * 16),
                nn.ReLU(inplace=True)
            )
        else:
            self.depth_branch = None
            self.fuse_conv = None

        # -------------------------------------------------------------
        # 4. 解碼器與跳躍連接 (所有模式共用相同維度)
        # -------------------------------------------------------------
        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = DecoderBlock(c * 2, c, c)

        self.final_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        # 模式 A：【全部一起算 (Early Fusion)】
        # x 為完整的 9 通道，直接整包送入編碼器
        if self.depth_in_channels > 0 and not self.use_late_fusion:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            e4 = self.enc4(self.pool3(e3))
            b = self.bottleneck(self.pool4(e4))

        # 模式 B：【Late Fusion】或【純光譜】
        else:
            spec_x = x[:, :self.op_channels, :, :]  # 前 8 通道為光譜
            e1 = self.enc1(spec_x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            e4 = self.enc4(self.pool3(e3))
            b = self.bottleneck(self.pool4(e4))     # (B, 256, H/16, W/16)

            # 若開啟 Late Fusion，在 Bottleneck 提取並融合水深特徵
            if self.depth_in_channels > 0 and self.use_late_fusion:
                depth_x = x[:, self.op_channels:, :, :]  # 第 9 通道為水深
                d_feat = self.depth_branch(depth_x)       # (B, depth_channels, H/16, W/16)
                fused_b = torch.cat([b, d_feat], dim=1)   # Concat
                b = self.fuse_conv(fused_b)               # 降為 256

        # 解碼器還原空間解析度出圖
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.final_conv(d1)

