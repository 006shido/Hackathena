#!/usr/bin/env python3
"""
PatchGAN Discriminator for Stage 2 Adversarial Realism
Processes RGB face images/patches and classifies local NxN patches as real or fake.
Follows standard 70x70 receptive field PatchGAN architecture with Spectral Norm / InstanceNorm.
"""

import torch
import torch.nn as nn
from typing import Tuple


class PatchGANDiscriminator(nn.Module):
    """
    Multiscale / Patch-based Discriminator for Photorealistic Face Blending.
    Accepts 128x128 RGB images in range [-1.0, 1.0].
    Outputs 14x14 grid of patch validity logits.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            # Stage 1: 128x128 -> 64x64
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 2: 64x64 -> 32x32
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 3: 32x32 -> 16x16
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 4: 16x16 -> 15x15 (stride 1)
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Output head: 15x15 -> 14x14 patch logits
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input face image tensor [B, 3, 128, 128] in [-1.0, 1.0]
        Returns:
            logits: Patch logits [B, 1, 14, 14]
        """
        return self.net(x)


class AdversarialLoss(nn.Module):
    """
    Least-Squares GAN (LSGAN) loss for stable training.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def discriminator_loss(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        D loss: 0.5 * (E[(D(real) - 1)^2] + E[D(fake)^2])
        """
        loss_real = self.mse(d_real, torch.ones_like(d_real))
        loss_fake = self.mse(d_fake, torch.zeros_like(d_fake))
        total_d_loss = 0.5 * (loss_real + loss_fake)
        return total_d_loss, loss_real, loss_fake

    def generator_loss(self, d_fake: torch.Tensor) -> torch.Tensor:
        """
        G adversarial loss: E[(D(fake) - 1)^2]
        """
        return self.mse(d_fake, torch.ones_like(d_fake))
