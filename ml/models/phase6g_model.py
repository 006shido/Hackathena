"""
Phase 6G: Correspondence-Aware Multi-Scale Face-Swap Model
Combines:
  1. 6D local piecewise-affine aligned source
  2. Multi-scale source encoder with lateral skips at 16x16, 32x32, and 64x64
  3. Target structure encoder (blurred RGB + landmark heatmap)
  4. ArcFace 512D identity vector conditioning via AdaIN
  5. Multi-scale decoder with lateral skip fusion
  6. Soft facial composition with anti-collapse supervision
"""

from typing import Optional, Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.modules import (
    AdaINResBlock2d,
    TargetStructureEncoder,
    SoftCompositor,
    gaussian_blur_2d
)
from ml.models.face_swap_model import ArcFaceIdentityExtractor


# -------------------------------------------------------------
# Multi-Scale Source Encoder
# -------------------------------------------------------------
class MultiScaleSourceEncoder(nn.Module):
    """
    Extracts multi-scale spatial representations from 6D aligned source RGB:
      128x128 -> 64x64  (64 ch)   -> s_64
      64x64   -> 32x32  (128 ch)  -> s_32
      32x32   -> 16x16  (256 ch)  -> s_16
      16x16   -> 8x8    (512 ch)  -> s_8
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 8, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x_src_aligned: torch.Tensor) -> Dict[str, torch.Tensor]:
        s64 = self.layer1(x_src_aligned)   # [B, 64, 64, 64]
        s32 = self.layer2(s64)             # [B, 128, 32, 32]
        s16 = self.layer3(s32)             # [B, 256, 16, 16]
        s8  = self.layer4(s16)             # [B, 512, 8, 8]
        return {
            "s64": s64,
            "s32": s32,
            "s16": s16,
            "s8": s8
        }


# -------------------------------------------------------------
# Multi-Scale Skip Generator Decoder
# -------------------------------------------------------------
class MultiScaleSkipGenerator(nn.Module):
    """
    Generator Decoder with lateral skip fusion from aligned source:
      - Bottleneck fusion at 8x8: target 512 ch + source 512 ch -> 512 ch
      - 4 AdaIN residual blocks at 8x8 modulated by ArcFace z_id
      - Up 1 (16x16): bilinear upsample, conv 512->256, fuse with s16 (256 ch), AdaIN res block
      - Up 2 (32x32): bilinear upsample, conv 256->128, fuse with s32 (128 ch), AdaIN res block
      - Up 3 (64x64): bilinear upsample, conv 128->64,  fuse with s64 (64 ch),  AdaIN res block
      - Up 4 (128x128): bilinear upsample, conv 64->64, AdaIN res block
      - Dual output heads:
          RGB head  -> I_swap [B, 3, 128, 128] in [-1.0, 1.0]
          Mask head -> M_pred [B, 1, 128, 128] in [0.0, 1.0]
    """
    def __init__(self, bottleneck_channels: int = 512, embedding_dim: int = 512):
        super().__init__()

        # Bottleneck 8x8 fusion: 512 target + 512 source -> 512
        self.fuse_8 = nn.Sequential(
            nn.Conv2d(bottleneck_channels * 2, bottleneck_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(bottleneck_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 4 Bottleneck AdaIN Residual Blocks at 8x8
        self.res1 = AdaINResBlock2d(bottleneck_channels, embedding_dim)
        self.res2 = AdaINResBlock2d(bottleneck_channels, embedding_dim)
        self.res3 = AdaINResBlock2d(bottleneck_channels, embedding_dim)
        self.res4 = AdaINResBlock2d(bottleneck_channels, embedding_dim)

        # Stage 1: 8x8 -> 16x16 (512 -> 256)
        self.conv_up1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.fuse_16 = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.res_up1 = AdaINResBlock2d(256, embedding_dim)

        # Stage 2: 16x16 -> 32x32 (256 -> 128)
        self.conv_up2 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.fuse_32 = nn.Sequential(
            nn.Conv2d(128 * 2, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.res_up2 = AdaINResBlock2d(128, embedding_dim)

        # Stage 3: 32x32 -> 64x64 (128 -> 64)
        self.conv_up3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.fuse_64 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.res_up3 = AdaINResBlock2d(64, embedding_dim)

        # Stage 4: 64x64 -> 128x128 (64 -> 64)
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
        z_id: torch.Tensor,
        source_skips: Optional[Dict[str, torch.Tensor]] = None,
        confidence_map: Optional[torch.Tensor] = None,
        disable_skips: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Bottleneck 8x8 fusion
        if source_skips is not None and not disable_skips and "s8" in source_skips:
            s8 = source_skips["s8"]
            if confidence_map is not None:
                c8 = F.interpolate(confidence_map, size=(8, 8), mode='bilinear', align_corners=False)
                s8 = s8 * c8
            x = self.fuse_8(torch.cat([f_tgt, s8], dim=1))
        else:
            zero_s8 = torch.zeros_like(f_tgt)
            x = self.fuse_8(torch.cat([f_tgt, zero_s8], dim=1))

        # 4 Bottleneck AdaIN ResBlocks
        x = self.res1(x, z_id)
        x = self.res2(x, z_id)
        x = self.res3(x, z_id)
        x = self.res4(x, z_id)

        # Stage 1: 8 -> 16
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.conv_up1(x)
        if source_skips is not None and not disable_skips and "s16" in source_skips:
            s16 = source_skips["s16"]
            if confidence_map is not None:
                c16 = F.interpolate(confidence_map, size=(16, 16), mode='bilinear', align_corners=False)
                s16 = s16 * c16
            x = self.fuse_16(torch.cat([x, s16], dim=1))
        else:
            x = self.fuse_16(torch.cat([x, torch.zeros_like(x)], dim=1))
        x = self.res_up1(x, z_id)

        # Stage 2: 16 -> 32
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.conv_up2(x)
        if source_skips is not None and not disable_skips and "s32" in source_skips:
            s32 = source_skips["s32"]
            if confidence_map is not None:
                c32 = F.interpolate(confidence_map, size=(32, 32), mode='bilinear', align_corners=False)
                s32 = s32 * c32
            x = self.fuse_32(torch.cat([x, s32], dim=1))
        else:
            x = self.fuse_32(torch.cat([x, torch.zeros_like(x)], dim=1))
        x = self.res_up2(x, z_id)

        # Stage 3: 32 -> 64 (MANDATORY 64x64 SKIP)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.conv_up3(x)
        if source_skips is not None and not disable_skips and "s64" in source_skips:
            s64 = source_skips["s64"]
            if confidence_map is not None:
                c64 = F.interpolate(confidence_map, size=(64, 64), mode='bilinear', align_corners=False)
                s64 = s64 * c64
            x = self.fuse_64(torch.cat([x, s64], dim=1))
        else:
            x = self.fuse_64(torch.cat([x, torch.zeros_like(x)], dim=1))
        x = self.res_up3(x, z_id)

        # Stage 4: 64 -> 128
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.conv_up4(x)
        x = self.res_up4(x, z_id)

        i_swap = self.rgb_head(x)
        m_pred = self.mask_head(x)
        return i_swap, m_pred


# -------------------------------------------------------------
# Complete Multi-Scale Correspondence Face-Swap Model
# -------------------------------------------------------------
class MultiScaleCorrespondenceFaceSwapModel(nn.Module):
    """
    Phase 6G Architecture:
      - Aligned source piecewise-affine multi-scale encoder
      - Target structure encoder
      - Multi-scale skip generator with AdaIN identity injection
      - Soft facial composition
      - Frozen official ArcFace
    """
    def __init__(
        self,
        base_channels: int = 64,
        bottleneck_channels: int = 512,
        embedding_dim: int = 512,
        blur_kernel_size: int = 9,
        blur_sigma: float = 3.0,
        arcface_checkpoint_path: Optional[str] = None
    ):
        super().__init__()
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma

        # 1. Target structure encoder
        self.target_encoder = TargetStructureEncoder(
            in_channels=4,
            base_channels=base_channels
        )

        # 2. Multi-scale source encoder
        self.source_encoder = MultiScaleSourceEncoder(
            in_channels=3,
            base_channels=base_channels
        )

        # 3. Multi-scale skip generator
        self.generator = MultiScaleSkipGenerator(
            bottleneck_channels=bottleneck_channels,
            embedding_dim=embedding_dim
        )

        # 4. Soft compositor
        self.compositor = SoftCompositor()

        # 5. Frozen ArcFace
        self.arcface: Optional[ArcFaceIdentityExtractor] = None
        if arcface_checkpoint_path is not None:
            self.arcface = ArcFaceIdentityExtractor(arcface_checkpoint_path)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.arcface is not None:
            self.arcface.eval()
        return self

    def prepare_target_input(
        self,
        i_target: torch.Tensor,
        l_target: torch.Tensor
    ) -> torch.Tensor:
        i_target_low = gaussian_blur_2d(
            i_target,
            kernel_size=self.blur_kernel_size,
            sigma=self.blur_sigma
        )
        return torch.cat([i_target_low, l_target], dim=1)

    def forward(
        self,
        i_target: torch.Tensor,
        l_target: torch.Tensor,
        i_source: torch.Tensor,
        aligned_source: Optional[torch.Tensor] = None,
        confidence_map: Optional[torch.Tensor] = None,
        disable_skips: bool = False
    ) -> Dict[str, torch.Tensor]:
        # 1. Source identity embedding from frozen ArcFace
        if self.arcface is None:
            raise RuntimeError("ArcFace identity extractor is not configured")
        z_id = self.arcface(i_source)

        # 2. Target structure encoding
        x_tgt = self.prepare_target_input(i_target, l_target)
        f_tgt = self.target_encoder(x_tgt)

        # 3. Multi-scale source encoding
        source_skips = None
        if aligned_source is not None and not disable_skips:
            source_skips = self.source_encoder(aligned_source)

        # 4. Multi-scale decode & synthesis
        i_swap, m_pred = self.generator(
            f_tgt=f_tgt,
            z_id=z_id,
            source_skips=source_skips,
            confidence_map=confidence_map,
            disable_skips=disable_skips
        )

        # 5. Soft composition
        i_composite = self.compositor(i_swap, m_pred, i_target)

        return {
            "i_swap": i_swap,
            "m_pred": m_pred,
            "i_composite": i_composite,
            "z_id": z_id,
            "f_tgt": f_tgt
        }

    def load_baseline_checkpoint(self, checkpoint_path: str):
        """Loads compatible baseline weights from Stage 2 best checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "generator_state_dict" in state_dict:
            gen_state = state_dict["generator_state_dict"]
        elif "model_state_dict" in state_dict:
            gen_state = state_dict["model_state_dict"]
        else:
            gen_state = state_dict

        model_dict = self.state_dict()
        loaded_keys = []
        for k, v in gen_state.items():
            clean_k = k[7:] if k.startswith("module.") else k
            if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
                model_dict[clean_k] = v
                loaded_keys.append(clean_k)

        self.load_state_dict(model_dict)
        print(f"Loaded {len(loaded_keys)} compatible weight tensors from baseline checkpoint: {checkpoint_path}", flush=True)
