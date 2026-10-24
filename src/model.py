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

##add DenseNet
class DenseLayer(nn.Module):
    """
    DenseNet 的基本合成層 (Bottleneck layer + 3x3 Convolution)
    採用 BN -> ReLU -> Conv(1x1) -> BN -> ReLU -> Conv(3x3) 的複合結構 (DenseNet-BC 風格)
    """
    def __init__(self, in_channels: int, growth_rate: int, num_groups: int = 8):
        super(DenseLayer, self).__init__()
        inter_channels = 4 * growth_rate
        groups1 = num_groups if inter_channels % num_groups == 0 else 1
        groups2 = num_groups if growth_rate % num_groups == 0 else 1

        self.bn1 = nn.GroupNorm(groups1, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)

        self.bn2 = nn.GroupNorm(groups2, inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """
    Dense Block: 內部包含多個 DenseLayer，層層密集拼接特徵圖。
    """
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int, num_groups: int = 8):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate, num_groups=num_groups))
            current_channels += growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """
    Transition Layer (過渡層): 負責壓縮通道數並進行 2x2 平均池化降採樣。
    """
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super(TransitionLayer, self).__init__()
        groups = num_groups if out_channels % num_groups == 0 else 1
        self.bn = nn.GroupNorm(groups, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.bn(x)))
        return self.pool(out)


class DenseDecoderBlock(nn.Module):
    """
    結合跳躍連接（Skip Connection）與雙線性上取樣的 Dense 解碼器模組。
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_groups: int = 8):
        super(DenseDecoderBlock, self).__init__()
        # 降維以融合 skip connection
        fused_in = in_channels + skip_channels
        groups = num_groups if out_channels % num_groups == 0 else 1

        self.conv1x1 = nn.Conv2d(fused_in, out_channels, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn_smooth = nn.GroupNorm(groups, out_channels)
        self.attn = ChannelwiseAttention(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.gn(self.conv1x1(x)))
        x = self.relu(self.gn_smooth(self.conv_smooth(x)))
        return self.attn(x)


# class DepthFeatureExtractor(nn.Module):
    # """
    # 水深獨立降採樣特徵分支 (與原 U-Net 架構完全相容)：
    # 將水深圖降採樣至 1/16 尺度，提煉高階地形特徵。
    # """
    # def __init__(self, in_channels=1, out_channels=32):
        # super(DepthFeatureExtractor, self).__init__()
        # self.net = nn.Sequential(
            # nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
            # nn.GroupNorm(1, 8),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            # nn.GroupNorm(2, 16),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            # nn.GroupNorm(4, 24),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(24, out_channels, kernel_size=3, stride=2, padding=1),
            # nn.GroupNorm(4, out_channels),
            # nn.ReLU(inplace=True)
        # )

    # def forward(self, depth):
        # return self.net(depth)


class BendingDenseNet_LateFusion(nn.Module):
    """
    基於 DenseNet 骨幹與 U-Shape 結構的底棲棲地語意分割模型 (支援 Late Fusion / Early Fusion)。
    類別名稱維持相容於原專案之呼叫介面。
    """
    def __init__(
        self,
        op_channels: int = 8,
        depth_in_channels: int = 1,
        use_late_fusion: bool = True,
        num_classes: int = 6,
        growth_rate: int = 16,
        depth_channels: int = 32
    ):
        super(BendingDenseNet_LateFusion, self).__init__()
        self.op_channels = op_channels
        self.depth_in_channels = depth_in_channels
        self.use_late_fusion = use_late_fusion

        # 1. 決定輸入通道數
        if self.depth_in_channels > 0 and not self.use_late_fusion:
            first_in_channels = self.op_channels + self.depth_in_channels
        else:
            first_in_channels = self.op_channels

        # 初始特徵提取
        self.init_conv = nn.Conv2d(first_in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.init_gn = nn.GroupNorm(8, 32)
        self.init_relu = nn.ReLU(inplace=True)

        # Encoder (Dense Blocks & Transition Layers)
        # Block 1 (32 -> 32 + 4*16 = 96)
        self.dense1 = DenseBlock(in_channels=32, growth_rate=growth_rate, num_layers=4, num_groups=8)
        self.trans1 = TransitionLayer(in_channels=96, out_channels=48, num_groups=8)

        # Block 2 (48 -> 48 + 4*16 = 112)
        self.dense2 = DenseBlock(in_channels=48, growth_rate=growth_rate, num_layers=4, num_groups=8)
        self.trans2 = TransitionLayer(in_channels=112, out_channels=64, num_groups=8)

        # Block 3 (64 -> 64 + 6*16 = 160)
        self.dense3 = DenseBlock(in_channels=64, growth_rate=growth_rate, num_layers=6, num_groups=8)
        self.trans3 = TransitionLayer(in_channels=160, out_channels=96, num_groups=8)

        # Bottleneck
        self.bottleneck = DenseBlock(in_channels=96, growth_rate=growth_rate, num_layers=6, num_groups=8)
        bottleneck_out_channels = 96 + 6 * growth_rate  # 96 + 96 = 192

        # 2. 條件式建立水深分支 (Late Fusion)
        if self.depth_in_channels > 0 and self.use_late_fusion:
            self.depth_branch = DepthFeatureExtractor(
                in_channels=depth_in_channels,
                out_channels=depth_channels
            )
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(bottleneck_out_channels + depth_channels, bottleneck_out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(8, bottleneck_out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.depth_branch = None
            self.fuse_conv = None

        # 3. Decoder
        self.dec3 = DenseDecoderBlock(in_channels=bottleneck_out_channels, skip_channels=160, out_channels=96)
        self.dec2 = DenseDecoderBlock(in_channels=96, skip_channels=112, out_channels=64)
        self.dec1 = DenseDecoderBlock(in_channels=64, skip_channels=96, out_channels=48)
        self.dec0 = DenseDecoderBlock(in_channels=48, skip_channels=96, out_channels=32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        if self.depth_in_channels > 0 and not self.use_late_fusion:
            # Early Fusion
            feat = self.init_relu(self.init_gn(self.init_conv(x)))
        else:
            # Late Fusion / 純光譜
            spec_x = x[:, :self.op_channels, :, :]
            feat = self.init_relu(self.init_gn(self.init_conv(spec_x)))

        # Encoder 前向傳遞並保留跳躍連接 (Skip Connections)
        e0 = feat  # 32 channels
        e1 = self.dense1(e0)  # 96 channels
        t1 = self.trans1(e1)  # 48 channels

        e2 = self.dense2(t1)  # 112 channels
        t2 = self.trans2(e2)  # 64 channels

        e3 = self.dense3(t2)  # 160 channels
        t3 = self.trans3(e3)  # 96 channels

        b = self.bottleneck(t3)  # 192 channels

        # Late Fusion 水深特徵融合
        if self.depth_in_channels > 0 and self.use_late_fusion:
            depth_x = x[:, self.op_channels:, :, :]
            d_feat = self.depth_branch(depth_x)
            fused_b = torch.cat([b, d_feat], dim=1)
            b = self.fuse_conv(fused_b)

        # Decoder 還原
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        return self.final_conv(d0)

BHUNet_LateFile_Alias = BendingDenseNet_LateFusion