"""
Phase 6E: Correspondence-Aware Face Swap Model
Integrates Aligned Source Spatial Encoder and Gated Residual Fusion with
the baseline Phase 6B.2 AdaIN Face Swap architecture.
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.face_swap_model import (
    AdaINFaceSwapModel,
    ArcFaceIdentityExtractor,
    TargetStructureEncoder,
    IdentityConditionedGenerator,
    SoftCompositor
)


class AlignedSourceEncoder(nn.Module):
    """
    Downsampling Convolutional Encoder for Aligned Source Image.
    Mirrors the target encoder structure but processes 3-channel aligned source RGB:
      128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
      Channels: 3 -> 64 -> 128 -> 256 -> 512
    Uses non-affine InstanceNorm2d to match target structure normalization.
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

    def forward(self, x_src_aligned: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_src_aligned: [B, 3, 128, 128] Aligned source image in target coordinates
        Returns:
            f_src: [B, 512, 8, 8] Spatial source bottleneck features
        """
        f1 = self.layer1(x_src_aligned)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f4


class GatedResidualFusion(nn.Module):
    """
    Spatially-varying gated residual fusion at the 8x8 bottleneck:
      gate = sigmoid(G([F_tgt, F_src_mod]))  # [B, 512, 8, 8]
      res  = P(F_src_mod)                   # [B, 512, 8, 8]
      fused = F_tgt + gate * res

    Initialized such that proj_source is zero-weighted at start,
    guaranteeing identical start to Phase 6B.2 baseline.
    """
    def __init__(self, channels: int = 512):
        super().__init__()
        # 1024 -> 512 spatial gate
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        # Residual projection
        self.proj_source = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

        # Zero-initialize projection to start identically to 6B.2 baseline
        nn.init.zeros_(self.proj_source.weight)
        nn.init.zeros_(self.proj_source.bias)

    def forward(
        self,
        f_tgt: torch.Tensor,
        f_src_mod: torch.Tensor,
        disable_pathway: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_tgt: Target bottleneck features [B, 512, 8, 8]
            f_src_mod: Confidence-modulated source features [B, 512, 8, 8]
            disable_pathway: If True, returns f_tgt with zero gate (for ablation)
        Returns:
            fused: [B, 512, 8, 8]
            gate: [B, 512, 8, 8]
        """
        if disable_pathway:
            gate = torch.zeros_like(f_tgt)
            return f_tgt, gate

        cat_feat = torch.cat([f_tgt, f_src_mod], dim=1)
        gate = self.gate_conv(cat_feat)
        res = self.proj_source(f_src_mod)
        fused = f_tgt + gate * res
        return fused, gate


class CorrespondenceAwareFaceSwapModel(nn.Module):
    """
    Phase 6E Neural Face-Swap Model with Correspondence-Aware Source Pathway.
    Integrates:
      1. Frozen official ArcFace -> global z_id [B, 512]
      2. TargetStructureEncoder -> F_tgt [B, 512, 8, 8]
      3. AlignedSourceEncoder -> F_src [B, 512, 8, 8]
      4. Geometric Confidence Map Modulation -> F_src_mod [B, 512, 8, 8]
      5. Gated Residual Fusion -> F_fused [B, 512, 8, 8]
      6. IdentityConditionedGenerator(F_fused, z_id) -> I_swap, M_pred
      7. SoftCompositor(I_swap, M_pred, I_target) -> I_composite
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

        # Core Stage 2 submodules
        self.target_encoder = TargetStructureEncoder(
            in_channels=4,
            base_channels=base_channels
        )
        self.generator = IdentityConditionedGenerator(
            bottleneck_channels=bottleneck_channels,
            embedding_dim=embedding_dim
        )
        self.compositor = SoftCompositor()

        # Phase 6E new submodules
        self.aligned_source_encoder = AlignedSourceEncoder(
            in_channels=3,
            base_channels=base_channels
        )
        self.fusion = GatedResidualFusion(channels=bottleneck_channels)

        # Frozen ArcFace
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
        from ml.models.modules import gaussian_blur_2d
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
        disable_aligned_source: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward execution pass.
        Args:
            i_target: Target RGB [B, 3, 128, 128]
            l_target: Target landmark map [B, 1, 128, 128]
            i_source: Original source RGB [B, 3, 128, 128]
            aligned_source: 6D Piecewise warped source [B, 3, 128, 128] (optional)
            confidence_map: Geometric confidence map [B, 1, 128, 128] (optional)
            disable_aligned_source: If True, bypasses source spatial pathway (ablation)
        """
        # 1. Global identity embedding from frozen ArcFace
        if self.arcface is None:
            raise RuntimeError("ArcFace identity extractor is not configured")
        z_id = self.arcface(i_source)

        # 2. Target structure encoding
        x_tgt = self.prepare_target_input(i_target, l_target)
        f_tgt = self.target_encoder(x_tgt)  # [B, 512, 8, 8]

        # 3. Aligned source spatial encoding & fusion
        if aligned_source is not None and not disable_aligned_source:
            f_src = self.aligned_source_encoder(aligned_source)  # [B, 512, 8, 8]
            if confidence_map is not None:
                conf_8 = F.interpolate(confidence_map, size=(8, 8), mode='bilinear', align_corners=False)
                f_src_mod = f_src * conf_8
            else:
                f_src_mod = f_src
            f_fused, gate = self.fusion(f_tgt, f_src_mod, disable_pathway=False)
        else:
            # Fallback or ablation mode
            f_fused = f_tgt
            gate = torch.zeros_like(f_tgt)

        # 4. Identity-conditioned synthesis
        i_swap, m_pred = self.generator(f_fused, z_id)

        # 5. Soft composition
        i_composite = self.compositor(i_swap, m_pred, i_target)

        return {
            "i_swap": i_swap,
            "m_pred": m_pred,
            "i_composite": i_composite,
            "z_id": z_id,
            "f_tgt": f_tgt,
            "f_fused": f_fused,
            "gate": gate
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

        # Filter and load matching keys
        model_dict = self.state_dict()
        loaded_keys = []
        for k, v in gen_state.items():
            # Strip 'module.' prefix if present
            clean_k = k[7:] if k.startswith("module.") else k
            if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
                model_dict[clean_k] = v
                loaded_keys.append(clean_k)

        self.load_state_dict(model_dict)
        print(f"Loaded {len(loaded_keys)} parameter tensors from baseline checkpoint '{checkpoint_path}'.")
