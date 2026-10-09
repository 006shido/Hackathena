"""
Modular Neural Network Building Blocks for Hackathena Face-Swap Model
Contains:
  - Gaussian blur filter for low-frequency target context
  - TargetStructureEncoder: Hierarchical downsampler with non-affine InstanceNorm
  - AdaIN2d: Reusable Adaptive Instance Normalization module
  - AdaINResBlock2d: Residual block with AdaIN identity modulation
  - IdentityConditionedGenerator: Multi-scale decoder with dual RGB and Mask heads
  - SoftCompositor: Soft alpha-blending layer
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_blur_2d(
    x: torch.Tensor,
    kernel_size: int = 9,
    sigma: float = 3.0
) -> torch.Tensor:
    """
    Applies 2D Gaussian blur across spatial dimensions of input image tensor.
    Suppresses high-frequency facial texture details while preserving coarse illumination
    and spatial layout context. Low-frequency RGB is an appearance signal and is NOT
    guaranteed to be identity-free.

    Args:
        x: [B, C, H, W] float32 image tensor in [-1.0, 1.0]
        kernel_size: Odd integer filter diameter (default: 9)
        sigma: Standard deviation of Gaussian kernel (default: 3.0)

    Returns:
        blurred: [B, C, H, W] float32 image tensor
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    # Generate 1D Gaussian kernel
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Form 2D separable Gaussian kernel [1, 1, K, K]
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0)

    channels = x.shape[1]
    weight = kernel_2d.repeat(channels, 1, 1, 1)

    return F.conv2d(x, weight, padding=radius, groups=channels)


class AdaIN2d(nn.Module):
    """
    Adaptive Instance Normalization Layer for 2D Feature Maps.
    Normalizes feature maps per-channel per-instance, then scales and shifts
    using affine parameters derived from the 512-D source identity embedding.

    Formula:
        F_mod = (1 + gamma) * InstanceNorm(F) + beta
    """
    def __init__(self, num_features: int, embedding_dim: int = 512):
        super().__init__()
        self.num_features = num_features
        # InstanceNorm with affine=False strictly strips channel variance/mean
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        # 2-layer MLP projection mapping identity vector to scale and shift
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, num_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_features, num_features * 2)
        )

        # Initialize final linear layer to zero so initial modulation is identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, z_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial feature map [B, C, H, W]
            z_id: L2-normalized source identity embedding [B, 512]

        Returns:
            modulated: [B, C, H, W]
        """
        style = self.mlp(z_id)  # [B, 2 * C]
        gamma, beta = style.chunk(2, dim=1)  # [B, C], [B, C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]

        normalized = self.norm(x)
        return (1.0 + gamma) * normalized + beta


class AdaINResBlock2d(nn.Module):
    """
    Residual Block with dual AdaIN identity modulation.
    Maintains spatial layout while conditioning feature style on z_id.
    """
    def __init__(self, channels: int, embedding_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.adain1 = AdaIN2d(channels, embedding_dim)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.adain2 = AdaIN2d(channels, embedding_dim)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, z_id: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.adain1(out, z_id)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.adain2(out, z_id)
        out = self.act2(out + residual)
        return out


class TargetStructureEncoder(nn.Module):
    """
    Downsampling Convolutional Encoder for Target Structure.
    Extracts spatial layout, pose, and expression from:
      - Low-frequency target RGB (3 channels)
      - Precomputed structural landmark map (1 channel)
      Total input channels = 4.

    InstanceNorm layers use affine=False to avoid learning dataset-specific contrast scales.
    Strongly suppresses target identity information while preserving target pose, expression,
    lighting, and spatial layout.
    """
    def __init__(self, in_channels: int = 4, base_channels: int = 64):
        super().__init__()

        # 128x128 -> 64x64
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 64x64 -> 32x32
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 32x32 -> 16x16
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 16x16 -> 8x8
        self.layer4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 8, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x_tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_tgt: [B, 4, 128, 128] concatenated low-frequency RGB + landmark map

        Returns:
            f_tgt: Spatial bottleneck representation [B, 512, 8, 8]
        """
        f1 = self.layer1(x_tgt)  # [B, 64, 64, 64]
        f2 = self.layer2(f1)     # [B, 128, 32, 32]
        f3 = self.layer3(f2)     # [B, 256, 16, 16]
        f_tgt = self.layer4(f3)  # [B, 512, 8, 8]
        return f_tgt


class IdentityConditionedGenerator(nn.Module):
    """
    Generator Decoder conditioned on 512-D source identity vector z_id.
    Processes spatial bottleneck F_tgt through:
      - 4 AdaIN residual blocks at 8x8
      - 4 upsampling stages with AdaIN conditioning:
          8x8 -> 16x16 (512 -> 256)
          16x16 -> 32x32 (256 -> 128)
          32x32 -> 64x64 (128 -> 64)
          64x64 -> 128x128 (64 -> 64)
      - Dual output heads:
          RGB head (Conv2d + Tanh) -> I_swap [B, 3, 128, 128] in [-1.0, 1.0]
          Mask head (Conv2d + Sigmoid) -> M_pred [B, 1, 128, 128] in [0.0, 1.0]
    """
    def __init__(self, bottleneck_channels: int = 512, embedding_dim: int = 512):
        super().__init__()

        # 4 Bottleneck AdaIN Residual Blocks at 8x8
        self.res1 = AdaINResBlock2d(bottleneck_channels, embedding_dim)
        self.res2 = AdaINResBlock2d(bottleneck_channels, embedding_dim)
        self.res3 = AdaINResBlock2d(bottleneck_channels, embedding_dim)
        self.res4 = AdaINResBlock2d(bottleneck_channels, embedding_dim)

        # Upsampling Stage 1: 8x8 -> 16x16 (512 -> 256)
        self.conv_up1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.res_up1 = AdaINResBlock2d(256, embedding_dim)

        # Upsampling Stage 2: 16x16 -> 32x32 (256 -> 128)
        self.conv_up2 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.res_up2 = AdaINResBlock2d(128, embedding_dim)

        # Upsampling Stage 3: 32x32 -> 64x64 (128 -> 64)
        self.conv_up3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.res_up3 = AdaINResBlock2d(64, embedding_dim)

        # Upsampling Stage 4: 64x64 -> 128x128 (64 -> 64)
        self.conv_up4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.res_up4 = AdaINResBlock2d(64, embedding_dim)

        # Dual Output Heads
        self.rgb_head = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        f_tgt: torch.Tensor,
        z_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_tgt: Target spatial bottleneck features [B, 512, 8, 8]
            z_id: Source identity embedding vector [B, 512]

        Returns:
            i_swap: Synthesized face [B, 3, 128, 128] in [-1.0, 1.0]
            m_pred: Soft blending mask [B, 1, 128, 128] in [0.0, 1.0]
        """
        # Bottleneck processing
        x = self.res1(f_tgt, z_id)
        x = self.res2(x, z_id)
        x = self.res3(x, z_id)
        x = self.res4(x, z_id)

        # Up 1: 8x8 -> 16x16
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv_up1(x)
        x = self.res_up1(x, z_id)

        # Up 2: 16x16 -> 32x32
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv_up2(x)
        x = self.res_up2(x, z_id)

        # Up 3: 32x32 -> 64x64
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv_up3(x)
        x = self.res_up3(x, z_id)

        # Up 4: 64x64 -> 128x128
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv_up4(x)
        x = self.res_up4(x, z_id)

        i_swap = self.rgb_head(x)
        m_pred = self.mask_head(x)

        return i_swap, m_pred


class SoftCompositor(nn.Module):
    """
    Applies differentiable alpha-compositing between synthesized face and target background:
        I_composite = M_pred * I_swap + (1 - M_pred) * I_t

    Guarantees:
      - Pixels outside the composited mask region are physically preserved as I_t.
      - Pixels inside M_pred are subject to neural synthesis and soft transition blending.
    """
    def forward(
        self,
        i_swap: torch.Tensor,
        m_pred: torch.Tensor,
        i_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            i_swap: [B, 3, 128, 128] float32 in [-1.0, 1.0]
            m_pred: [B, 1, 128, 128] float32 in [0.0, 1.0]
            i_target: [B, 3, 128, 128] float32 in [-1.0, 1.0]

        Returns:
            i_composite: [B, 3, 128, 128] float32 in [-1.0, 1.0]
        """
        return m_pred * i_swap + (1.0 - m_pred) * i_target
