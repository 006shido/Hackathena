#!/usr/bin/env python3
"""
Source Feature Encoder Prototype for Phase 6C Diagnostic.
Extracts multi-resolution source spatial feature representations to test whether
spatial source features contain identity-discriminative information missing from
the 512-D ArcFace recognition embedding.
"""

from typing import Dict
import torch
import torch.nn as nn


class SourceFeatureEncoder(nn.Module):
    """
    Convolutional encoder producing multi-scale source feature representations:
      - Stage 1: [B, 64, 64, 64]
      - Stage 2: [B, 128, 32, 32]
      - Stage 3: [B, 256, 16, 16]
      - Stage 4: [B, 512, 8, 8]
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        # 128x128 -> 64x64
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 64x64 -> 32x32
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 32x32 -> 16x16
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 16x16 -> 8x8
        self.stage4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f64 = self.stage1(x)    # [B, 64, 64, 64]
        f32 = self.stage2(f64)  # [B, 128, 32, 32]
        f16 = self.stage3(f32)  # [B, 256, 16, 16]
        f8 = self.stage4(f16)   # [B, 512, 8, 8]
        return {
            "f64": f64,
            "f32": f32,
            "f16": f16,
            "f8": f8
        }
