"""Detection losses: modified focal loss (heatmap) + masked L1 (regression)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """CornerNet-style modified focal loss on heatmap logits.

    ``pred_logits`` and ``target`` both have shape ``(B, C, H, W)`` with
    ``target`` in ``[0, 1]`` (Gaussian peaks).
    """
    pred = pred_logits.sigmoid()
    pos_mask = target.eq(1.0)
    neg_mask = target.lt(1.0)

    neg_weights = torch.pow(1.0 - target, beta)

    pos_loss = torch.log(pred.clamp(min=1e-6)) * torch.pow(1.0 - pred, alpha) * pos_mask
    neg_loss = (
        torch.log((1.0 - pred).clamp(min=1e-6))
        * torch.pow(pred, alpha)
        * neg_weights
        * neg_mask
    )

    num_pos = pos_mask.float().sum().clamp_min(1.0)
    loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss


def reg_loss(
    pred_reg: torch.Tensor,
    target_reg: torch.Tensor,
    reg_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked L1 on regression maps. ``reg_mask`` is ``(B, H, W)`` or ``(H, W)``."""
    if reg_mask.dim() == 2:
        reg_mask = reg_mask.unsqueeze(0)
    mask = reg_mask.unsqueeze(1)  # (B, 1, H, W)
    diff = F.l1_loss(pred_reg, target_reg, reduction="none")
    return (diff * mask).sum() / mask.sum().clamp_min(1.0)


def detection_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    hm_weight: float = 1.0,
    reg_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Sum heatmap + regression losses. Adds batch dimension if missing."""
    pred_hm = outputs["heatmap"]
    pred_reg = outputs["reg"]

    tgt_hm = targets["heatmap"]
    tgt_reg = targets["reg"]
    reg_mask = targets["reg_mask"]

    if pred_hm.dim() == 3:
        pred_hm = pred_hm.unsqueeze(0)
        pred_reg = pred_reg.unsqueeze(0)
    if tgt_hm.dim() == 3:
        tgt_hm = tgt_hm.unsqueeze(0)
        tgt_reg = tgt_reg.unsqueeze(0)
        reg_mask = reg_mask.unsqueeze(0)

    loss_hm = focal_loss(pred_hm, tgt_hm)
    loss_reg = reg_loss(pred_reg, tgt_reg, reg_mask)
    total = hm_weight * loss_hm + reg_weight * loss_reg

    return {
        "loss": total,
        "loss_hm": loss_hm.detach(),
        "loss_reg": loss_reg.detach(),
    }
