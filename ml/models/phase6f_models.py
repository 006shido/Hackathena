"""
Phase 6F: Autoencoding Architectures for Aligned-Source Reconstruction Diagnostic
Conditions:
  - Condition A: Direct Bottleneck Autoencoder (No skips, Bilinear upsampling)
  - Condition B: Multi-Scale Skip Autoencoder (U-Net style, full skips at 64, 32, 16)
  - Condition C: 6E-Style Bottleneck Autoencoder (4 ResBlocks at 8x8, Nearest upsampling)
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# Standard Residual Block for Condition C
# -------------------------------------------------------------
class ConvResBlock2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=False)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.act2(self.norm2(self.conv2(out)) + res)
        return out


# -------------------------------------------------------------
# Condition A: Direct Bottleneck Autoencoder
# -------------------------------------------------------------
class ConditionA_BottleneckAutoencoder(nn.Module):
    """
    Direct bottleneck autoencoder (no skips).
    Uses bilinear upsampling + 3x3 conv to avoid transposed conv artifacts.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        # Encoder: 128 -> 64 -> 32 -> 16 -> 8
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 8, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder: 8 -> 16 -> 32 -> 64 -> 128
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out_head = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)    # [B, 64, 64, 64]
        e2 = self.enc2(e1)   # [B, 128, 32, 32]
        e3 = self.enc3(e2)   # [B, 256, 16, 16]
        z  = self.enc4(e3)   # [B, 512, 8, 8]

        d1 = F.interpolate(z, scale_factor=2.0, mode='bilinear', align_corners=False)
        d1 = self.dec1(d1)   # [B, 256, 16, 16]

        d2 = F.interpolate(d1, scale_factor=2.0, mode='bilinear', align_corners=False)
        d2 = self.dec2(d2)   # [B, 128, 32, 32]

        d3 = F.interpolate(d2, scale_factor=2.0, mode='bilinear', align_corners=False)
        d3 = self.dec3(d3)   # [B, 64, 64, 64]

        d4 = F.interpolate(d3, scale_factor=2.0, mode='bilinear', align_corners=False)
        d4 = self.dec4(d4)   # [B, 64, 128, 128]

        out = self.out_head(d4)
        return out


# -------------------------------------------------------------
# Condition B: Multi-Scale Skip Autoencoder (U-Net Style)
# -------------------------------------------------------------
class ConditionB_MultiScaleSkipAutoencoder(nn.Module):
    """
    Multi-scale skip connection autoencoder.
    Preserves spatial detail at 64x64, 32x32, and 16x16 via lateral skips.
    Supports modular disabling of individual skips for ablation.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 8, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder with Skip Concatenations:
        # Up 1 (16x16): 512 upsampled to 256 via dec1_conv, then + 256 skip = 512 channels
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1_fuse = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Up 2 (32x32): 256 upsampled to 128 + 128 skip = 256 channels
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2_fuse = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Up 3 (64x64): 128 upsampled to 64 + 64 skip = 128 channels
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec3_fuse = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Up 4 (128x128)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out_head = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(
        self,
        x: torch.Tensor,
        use_skip16: bool = True,
        use_skip32: bool = True,
        use_skip64: bool = True
    ) -> torch.Tensor:
        e1 = self.enc1(x)    # [B, 64, 64, 64]
        e2 = self.enc2(e1)   # [B, 128, 32, 32]
        e3 = self.enc3(e2)   # [B, 256, 16, 16]
        z  = self.enc4(e3)   # [B, 512, 8, 8]

        # Up 1: 8 -> 16
        d1 = F.interpolate(z, scale_factor=2.0, mode='bilinear', align_corners=False)
        d1 = self.dec1_conv(d1)
        skip16_feat = e3 if use_skip16 else torch.zeros_like(e3)
        d1 = self.dec1_fuse(torch.cat([d1, skip16_feat], dim=1))

        # Up 2: 16 -> 32
        d2 = F.interpolate(d1, scale_factor=2.0, mode='bilinear', align_corners=False)
        d2 = self.dec2_conv(d2)
        skip32_feat = e2 if use_skip32 else torch.zeros_like(e2)
        d2 = self.dec2_fuse(torch.cat([d2, skip32_feat], dim=1))

        # Up 3: 32 -> 64
        d3 = F.interpolate(d2, scale_factor=2.0, mode='bilinear', align_corners=False)
        d3 = self.dec3_conv(d3)
        skip64_feat = e1 if use_skip64 else torch.zeros_like(e1)
        d3 = self.dec3_fuse(torch.cat([d3, skip64_feat], dim=1))

        # Up 4: 64 -> 128
        d4 = F.interpolate(d3, scale_factor=2.0, mode='bilinear', align_corners=False)
        d4 = self.dec4(d4)

        out = self.out_head(d4)
        return out


# -------------------------------------------------------------
# Condition C: 6E-Style Bottleneck Autoencoder
# -------------------------------------------------------------
class ConditionC_6EStyleBottleneckAutoencoder(nn.Module):
    """
    Replicates the Phase 6E bottleneck processing:
    - 4 Residual blocks at the 8x8 bottleneck
    - Nearest-neighbor interpolation upsampling (matching 6E generator)
    - No skip connections
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 8, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 4 Bottleneck Residual Blocks at 8x8
        self.res1 = ConvResBlock2d(base_channels * 8)
        self.res2 = ConvResBlock2d(base_channels * 8)
        self.res3 = ConvResBlock2d(base_channels * 8)
        self.res4 = ConvResBlock2d(base_channels * 8)

        # Decoder using Nearest-Neighbor upsampling (replicating 6E)
        self.up1 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out_head = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)    # [B, 64, 64, 64]
        e2 = self.enc2(e1)   # [B, 128, 32, 32]
        e3 = self.enc3(e2)   # [B, 256, 16, 16]
        z  = self.enc4(e3)   # [B, 512, 8, 8]

        # 4 Bottleneck blocks
        z = self.res1(z)
        z = self.res2(z)
        z = self.res3(z)
        z = self.res4(z)

        # Upsampling with nearest neighbor
        d1 = F.interpolate(z, scale_factor=2.0, mode='nearest')
        d1 = self.up1(d1)

        d2 = F.interpolate(d1, scale_factor=2.0, mode='nearest')
        d2 = self.up2(d2)

        d3 = F.interpolate(d2, scale_factor=2.0, mode='nearest')
        d3 = self.up3(d3)

        d4 = F.interpolate(d3, scale_factor=2.0, mode='nearest')
        d4 = self.up4(d4)

        out = self.out_head(d4)
        return out
