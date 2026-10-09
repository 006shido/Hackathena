"""
Trainable Neural Face-Swap Model Architecture for Hackathena
Implements the Phase 3A.1 approved architecture:
  - Source: ArcFace iResNet-50 extracting 512-D L2-normalized identity embedding z_id
  - Target Input: Low-frequency target RGB (GaussianBlur sigma=3.0) + Landmark Map L_t -> [B, 4, 128, 128]
  - Target Encoder: 4-layer InstanceNorm (affine=False) downsampler -> Spatial Bottleneck F_tgt [B, 512, 8, 8]
  - Generator: 4 AdaIN bottleneck blocks + 4 AdaIN upsample blocks
  - Dual Output Heads:
      RGB Head: Conv2d(64, 3) + Tanh -> I_swap [B, 3, 128, 128]
      Mask Head: Conv2d(64, 1) + Sigmoid -> M_pred [B, 1, 128, 128]
  - Compositor: I_composite = M_pred * I_swap + (1 - M_pred) * I_target
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.modules import (
    gaussian_blur_2d,
    TargetStructureEncoder,
    IdentityConditionedGenerator,
    SoftCompositor
)
from ml.models.iresnet import iresnet50


class ArcFaceIdentityExtractor(nn.Module):
    """
    Facial Identity Feature Extractor using frozen ArcFace iResNet-50.
    Input: [B, 3, 128, 128] RGB tensor normalized to [-1.0, 1.0]
    Output: 512-dimensional float32 vector, L2 normalized

    Strictly requires a real pretrained checkpoint. If the checkpoint is missing,
    raises FileNotFoundError. Does NOT download models or substitute a fake model.
    """
    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.backbone = iresnet50(num_features=512)
        self.checkpoint_path = checkpoint_path

        if checkpoint_path is not None:
            self.load_pretrained_weights(checkpoint_path)

        # Freeze backbone weights
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        # ArcFace backbone is frozen and must permanently remain in eval mode
        super().train(False)
        self.backbone.eval()
        return self

    def load_pretrained_weights(self, checkpoint_path: str):
        path = Path(checkpoint_path)
        if not path.is_file():
            # Check relative to repo root
            candidate = Path(__file__).resolve().parent.parent.parent / checkpoint_path
            if candidate.is_file():
                path = candidate
            else:
                raise FileNotFoundError(
                    f"ArcFace pretrained checkpoint not found at '{checkpoint_path}' or '{candidate}'.\n"
                    "Genuine identity extraction requires real pretrained ArcFace iResNet-50 weights. "
                    "Automatic downloading is disabled to preserve environment integrity. "
                    "Please configure a valid checkpoint path in ml/training/config.json."
                )

        state_dict = torch.load(str(path), map_location='cpu')
        # Handle state_dict key prefixes if needed
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.backbone.load_state_dict(clean_state_dict, strict=False)
        self.checkpoint_path = str(path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Source face tensor [B, 3, 128, 128] in [-1.0, 1.0]

        Returns:
            z_id: L2-normalized 512-D identity embedding [B, 512]
        """
        # ArcFace iResNet-50 input resolution is strictly 112x112
        if x.shape[2:] != (112, 112):
            x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)

        with torch.no_grad():
            embedding = self.backbone(x)

        return embedding


class AdaINFaceSwapModel(nn.Module):
    """
    Complete Neural Face-Swap Model Architecture.

    Flow:
      1. Source face I_s -> ArcFace -> z_id [B, 512]
      2. Target face I_t -> GaussianBlur(sigma=3.0) -> I_t_low [B, 3, 128, 128]
      3. Concat(I_t_low, L_t) -> X_tgt [B, 4, 128, 128]
      4. TargetStructureEncoder(X_tgt) -> F_tgt [B, 512, 8, 8]
      5. IdentityConditionedGenerator(F_tgt, z_id) -> I_swap [B, 3, 128, 128], M_pred [B, 1, 128, 128]
      6. SoftCompositor(I_swap, M_pred, I_t) -> I_composite [B, 3, 128, 128]
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

        # Submodules
        self.target_encoder = TargetStructureEncoder(
            in_channels=4,
            base_channels=base_channels
        )
        self.generator = IdentityConditionedGenerator(
            bottleneck_channels=bottleneck_channels,
            embedding_dim=embedding_dim
        )
        self.compositor = SoftCompositor()

        # ArcFace identity extractor (optional in constructor, can be attached or passed externally)
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
        """
        Creates the 4-channel target structural input tensor X_tgt:
          - Applies Gaussian blur (kernel=9, sigma=3.0) to target RGB.
          - Concatenates with precomputed structural landmark map L_t [B, 1, 128, 128].

        Args:
            i_target: [B, 3, 128, 128] float32 in [-1.0, 1.0]
            l_target: [B, 1, 128, 128] float32 in [0.0, 1.0]

        Returns:
            x_tgt: [B, 4, 128, 128]
        """
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
        z_id: Optional[torch.Tensor] = None,
        i_source: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for face swapping.

        Args:
            i_target: Target image [B, 3, 128, 128]
            l_target: Precomputed structural landmark map [B, 1, 128, 128]
            z_id: Precomputed source identity embedding [B, 512] (preferred for efficiency)
            i_source: Optional source image [B, 3, 128, 128] (if z_id is not precomputed)

        Returns:
            Dictionary containing:
              - 'i_swap': Raw synthesized face [B, 3, 128, 128]
              - 'm_pred': Predicted soft blending mask [B, 1, 128, 128]
              - 'i_composite': Soft blended output face [B, 3, 128, 128]
              - 'f_tgt': Target spatial bottleneck features [B, 512, 8, 8]
              - 'z_id': Source identity vector used [B, 512]
        """
        if z_id is None:
            if i_source is None:
                raise ValueError("Either 'z_id' or 'i_source' must be provided to forward().")
            if self.arcface is None:
                raise RuntimeError(
                    "Cannot extract identity from 'i_source': ArcFace extractor is not initialized. "
                    "Provide 'z_id' directly or initialize model with valid 'arcface_checkpoint_path'."
                )
            z_id = self.arcface(i_source)

        # 1. Prepare target structural input [B, 4, 128, 128]
        x_tgt = self.prepare_target_input(i_target, l_target)

        # 2. Downsample through target structure encoder -> [B, 512, 8, 8]
        f_tgt = self.target_encoder(x_tgt)

        # 3. Decode via AdaIN generator modulated by z_id
        i_swap, m_pred = self.generator(f_tgt, z_id)

        # 4. Soft compositing with target background
        i_composite = self.compositor(i_swap, m_pred, i_target)

        return {
            "i_swap": i_swap,
            "m_pred": m_pred,
            "i_composite": i_composite,
            "f_tgt": f_tgt,
            "z_id": z_id
        }


def get_face_swap_model(config_dict: Optional[Dict[str, Any]] = None) -> AdaINFaceSwapModel:
    """Factory function to instantiate the face swap model from configuration parameters."""
    cfg = config_dict or {}
    return AdaINFaceSwapModel(
        base_channels=cfg.get("base_channels", 64),
        bottleneck_channels=cfg.get("bottleneck_channels", 512),
        embedding_dim=cfg.get("embedding_dim", 512),
        blur_kernel_size=cfg.get("blur_kernel_size", 9),
        blur_sigma=cfg.get("blur_sigma", 3.0),
        arcface_checkpoint_path=cfg.get("arcface_checkpoint_path", None)
    )
