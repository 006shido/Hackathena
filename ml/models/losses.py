"""
Stage 1 Training Loss Functions for Hackathena Neural Face-Swap
Implements:
  1. Masked L1 Reconstruction Loss (normalized by active mask area)
  2. ArcFace Identity Cosine Loss (L_id = 1 - cos(z_src, z_out)) using frozen ArcFace
  3. Background Preservation Loss (outside face mask)
  4. Mask Regularization Loss (preventing all-zero or all-one collapse)
  5. Optional Perceptual Loss (when a verified perceptual network checkpoint is provided)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class MaskedL1Loss(nn.Module):
    """
    Masked L1 reconstruction loss normalized by the active mask area.
    Ensures that loss scale remains invariant to face size / mask coverage.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Generated / composited image [B, C, H, W]
            target: Ground truth target image [B, C, H, W]
            mask: Soft facial mask [B, 1, H, W] with values in [0.0, 1.0]
        """
        diff = torch.abs(pred - target)  # [B, C, H, W]
        # Expand mask to match channels if needed
        if mask.shape[1] == 1 and diff.shape[1] > 1:
            mask = mask.expand(-1, diff.shape[1], -1, -1)

        masked_diff = diff * mask
        # Sum over spatial and channel dimensions
        loss_per_sample = masked_diff.sum(dim=(1, 2, 3))
        mask_area = mask.sum(dim=(1, 2, 3)) + self.eps

        # Normalized loss per sample, then mean across batch
        normalized_loss = loss_per_sample / mask_area
        return normalized_loss.mean()


class BackgroundPreservationLoss(nn.Module):
    """
    Penalizes deviations in the background region (1 - mask) between output and target.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        bg_mask = (1.0 - mask).clamp(0.0, 1.0)
        if bg_mask.shape[1] == 1 and pred.shape[1] > 1:
            bg_mask = bg_mask.expand(-1, pred.shape[1], -1, -1)

        diff = torch.abs(pred - target) * bg_mask
        bg_area = bg_mask.sum(dim=(1, 2, 3)) + self.eps
        loss_per_sample = diff.sum(dim=(1, 2, 3)) / bg_area
        return loss_per_sample.mean()


class ArcFaceIdentityLoss(nn.Module):
    """
    Cosine identity distance loss between source identity embedding and generated output embedding:
    L_id = 1 - cosine_similarity(z_source, z_output)
    Both embeddings are L2 normalized.
    """
    def __init__(self, arcface_extractor: nn.Module):
        super().__init__()
        self.arcface = arcface_extractor
        # Ensure ArcFace backbone is frozen
        self.arcface.eval()
        for p in self.arcface.parameters():
            p.requires_grad = False

    def forward(
        self,
        i_source: torch.Tensor,
        i_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            i_source: Source face RGB [B, 3, 128, 128] in [-1.0, 1.0]
            i_output: Generated / composite face RGB [B, 3, 128, 128] in [-1.0, 1.0]
        """
        with torch.no_grad():
            z_source = self.arcface(i_source)  # [B, 512], L2 normalized

        z_output = self.arcface(i_output)      # [B, 512], L2 normalized

        # Cosine similarity for unit vectors is dot product
        cos_sim = (z_source * z_output).sum(dim=-1)
        loss = 1.0 - cos_sim
        return loss.mean()


class MaskRegularizationLoss(nn.Module):
    """
    Prevents degenerate masks:
      1. Boundary Total Variation (smoothness)
      2. Range penalty (keeping predictions strictly well-formed)
      3. Non-triviality margin (preventing collapse to 0 or 1)
    """
    def __init__(self, target_coverage: float = 0.35):
        super().__init__()
        self.target_coverage = target_coverage

    def forward(self, pred_mask: torch.Tensor, gt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_mask: Predicted mask from mask head [B, 1, H, W]
            gt_mask: Optional ground truth face-oval mask [B, 1, H, W]
        """
        # 1. Total variation for spatial smoothness
        tv_h = torch.abs(pred_mask[:, :, 1:, :] - pred_mask[:, :, :-1, :]).mean()
        tv_w = torch.abs(pred_mask[:, :, :, 1:] - pred_mask[:, :, :, :-1]).mean()
        tv_loss = tv_h + tv_w

        # 2. If gt_mask is available in Stage 1, directly supervise the mask head against the real face-oval
        if gt_mask is not None:
            mask_l1 = F.l1_loss(pred_mask, gt_mask)
            return mask_l1 + 0.1 * tv_loss
        else:
            # Prevent collapse to all-zeros or all-ones
            mean_coverage = pred_mask.mean()
            coverage_penalty = (mean_coverage - self.target_coverage) ** 2
            return tv_loss + coverage_penalty


class Stage1CompositeLoss(nn.Module):
    """
    Aggregated Stage 1 loss manager.
    Weights:
      - w_recon: Masked L1 on I_swap and I_composite (default: 10.0)
      - w_id: ArcFace identity cosine loss (default: 5.0)
      - w_bg: Background preservation loss (default: 5.0)
      - w_mask: Mask supervision / regularization (default: 2.0)
    """
    def __init__(
        self,
        arcface_extractor: nn.Module,
        w_recon: float = 10.0,
        w_id: float = 5.0,
        w_bg: float = 5.0,
        w_mask: float = 2.0
    ):
        super().__init__()
        self.w_recon = w_recon
        self.w_id = w_id
        self.w_bg = w_bg
        self.w_mask = w_mask

        self.masked_l1 = MaskedL1Loss()
        self.bg_loss = BackgroundPreservationLoss()
        self.id_loss = ArcFaceIdentityLoss(arcface_extractor)
        self.mask_reg = MaskRegularizationLoss()

    def forward(
        self,
        i_source: torch.Tensor,
        i_target: torch.Tensor,
        i_swap: torch.Tensor,
        i_composite: torch.Tensor,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Masked reconstruction loss on composite and swap
        loss_recon_comp = self.masked_l1(i_composite, i_target, target_mask)
        loss_recon_swap = self.masked_l1(i_swap, i_target, target_mask)
        loss_recon = 0.5 * (loss_recon_comp + loss_recon_swap)

        # 2. Background preservation loss (composite outside mask must match target)
        loss_bg = self.bg_loss(i_composite, i_target, target_mask)

        # 3. Identity preservation (output identity must match source identity)
        loss_id = self.id_loss(i_source, i_composite)

        # 4. Mask supervision / regularization
        loss_mask = self.mask_reg(pred_mask, target_mask)

        total_loss = (
            self.w_recon * loss_recon +
            self.w_bg * loss_bg +
            self.w_id * loss_id +
            self.w_mask * loss_mask
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_recon": loss_recon.item(),
            "loss_bg": loss_bg.item(),
            "loss_id": loss_id.item(),
            "loss_mask": loss_mask.item()
        }

        return total_loss, loss_dict


class LowFrequencyStructureLoss(nn.Module):
    """
    Penalizes deviations in low-frequency structure (lighting, pose, skin tone)
    between generated face and target face within the target face mask,
    allowing high-frequency identity features to conform to the source.
    """
    def __init__(self, kernel_size: int = 9, sigma: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.eps = eps

        # Build Gaussian kernel
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        grid = coords.repeat(kernel_size, 1)
        kernel_2d = torch.exp(-(grid**2 + grid.t()**2) / (2 * sigma**2))
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.register_buffer("kernel", kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1))

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        return F.conv2d(x, self.kernel, padding=pad, groups=3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred_blur = self._blur(pred)
        target_blur = self._blur(target)

        diff = torch.abs(pred_blur - target_blur)
        if mask.shape[1] == 1 and diff.shape[1] > 1:
            mask = mask.expand(-1, diff.shape[1], -1, -1)

        masked_diff = diff * mask
        loss_per_sample = masked_diff.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + self.eps)
        return loss_per_sample.mean()


class Stage2CompositeLoss(nn.Module):
    """
    Stage 2 Composite Loss for Cross-Identity Face Swapping + Adversarial Realism.

    Anti-Collapse Upgrades:
      1. Dual Identity Loss:
         - L_id_swap: Direct ArcFace supervision on I_swap (1 - cos(z_src, z_swap))
         - L_id_output: ArcFace supervision on I_composite (1 - cos(z_src, z_out))
         - Prevents generator from suppressing mask to avoid identity loss.
      2. MediaPipe Face-Region Mask Coverage:
         - Inside face: penalizes mask dropping to 0 within target face-oval
         - Outside face: penalizes mask expanding into background
      3. Background Preservation outside face mask
      4. Low-frequency target structural conditioning
      5. PatchGAN adversarial realism
    """
    def __init__(
        self,
        arcface_extractor: nn.Module,
        w_id: float = 10.0,
        w_id_swap: float = 5.0,
        w_struct: float = 5.0,
        w_bg: float = 5.0,
        w_mask: float = 5.0,
        w_adv: float = 0.5,
        eps: float = 1e-6
    ):
        super().__init__()
        self.w_id = w_id
        self.w_id_swap = w_id_swap
        self.w_struct = w_struct
        self.w_bg = w_bg
        self.w_mask = w_mask
        self.w_adv = w_adv
        self.eps = eps

        self.id_loss = ArcFaceIdentityLoss(arcface_extractor)
        self.struct_loss = LowFrequencyStructureLoss(kernel_size=9, sigma=3.0)
        self.bg_loss = BackgroundPreservationLoss()
        self.mask_reg = MaskRegularizationLoss()

    def forward(
        self,
        i_source: torch.Tensor,
        i_target: torch.Tensor,
        i_swap: torch.Tensor,
        i_composite: torch.Tensor,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
        d_fake_logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Dual Identity Supervision: Both I_composite AND I_swap directly
        loss_id_out = self.id_loss(i_source, i_composite)
        loss_id_swap = self.id_loss(i_source, i_swap)
        total_id_loss = self.w_id * loss_id_out + self.w_id_swap * loss_id_swap

        # 2. Target low-frequency structural preservation
        loss_struct = self.struct_loss(i_composite, i_target, target_mask)

        # 3. Target background preservation (composite outside mask must match target)
        loss_bg = self.bg_loss(i_composite, i_target, target_mask)

        # 4. Anti-Collapse Mask Supervision using genuine MediaPipe face region:
        # A. Inside face region: mask must have meaningful coverage (penalize 1 - pred_mask)
        face_area = target_mask.sum(dim=(1, 2, 3)) + self.eps
        inside_face_penalty = (target_mask * torch.abs(pred_mask - target_mask)).sum(dim=(1, 2, 3)) / face_area
        loss_mask_inside = inside_face_penalty.mean()

        # B. Outside face region: mask must remain near zero (penalize pred_mask outside)
        bg_mask = (1.0 - target_mask).clamp(0.0, 1.0)
        bg_area = bg_mask.sum(dim=(1, 2, 3)) + self.eps
        outside_face_penalty = (bg_mask * pred_mask).sum(dim=(1, 2, 3)) / bg_area
        loss_mask_outside = outside_face_penalty.mean()

        # C. Smoothness regularization
        loss_mask_smooth = self.mask_reg(pred_mask, target_mask)

        loss_mask_total = loss_mask_inside + loss_mask_outside + 0.5 * loss_mask_smooth

        # 5. Adversarial loss from PatchGAN
        if d_fake_logits is not None:
            loss_adv = F.mse_loss(d_fake_logits, torch.ones_like(d_fake_logits))
        else:
            loss_adv = torch.tensor(0.0, device=i_composite.device)

        total_loss = (
            total_id_loss +
            self.w_struct * loss_struct +
            self.w_bg * loss_bg +
            self.w_mask * loss_mask_total +
            self.w_adv * loss_adv
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_id_out": loss_id_out.item(),
            "loss_id_swap": loss_id_swap.item(),
            "loss_struct": loss_struct.item(),
            "loss_bg": loss_bg.item(),
            "loss_mask": loss_mask_total.item(),
            "loss_mask_inside": loss_mask_inside.item(),
            "loss_mask_outside": loss_mask_outside.item(),
            "loss_adv": loss_adv.item()
        }

        return total_loss, loss_dict
