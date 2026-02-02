"""
Loss functions for S-IB Adapter training.

Includes:
- Focal Loss: Handles class imbalance (many background patches)
- Dice Loss: Better for sparse masks (small foreground)
- Combined Loss: Weighted combination for robust training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


def sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Focal Loss for handling class imbalance.
    
    Reduces weight for easy negatives, focuses on hard examples.
    
    Args:
        pred: Predictions in range (0, 1) [B, N, 1]
        target: Ground truth binary [B, N, 1]
        alpha: Balancing factor for positive class
        gamma: Focusing parameter (higher = more focus on hard)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Focal loss value
    """
    # BCE component
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    
    # Focal weight: (1 - p_t)^gamma
    p_t = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    
    # Alpha weighting
    alpha_weight = alpha * target + (1 - alpha) * (1 - target)
    
    focal_loss = alpha_weight * focal_weight * bce
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    return focal_loss


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Dice Loss for binary mask prediction.
    
    Better than BCE for masks with small foreground areas.
    Uses Laplace smoothing to prevent division by zero.
    
    Args:
        pred: Predictions in range (0, 1) [B, N, 1]
        target: Ground truth binary [B, N, 1]
        smooth: Smoothing factor (1.0 = Laplace)
        reduction: 'mean' or 'none'
        
    Returns:
        Dice loss (1 - Dice coefficient)
    """
    # Flatten to [B, N]
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Dice coefficient
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1 - dice
    
    if reduction == 'mean':
        return loss.mean()
    return loss


def calc_sib_loss(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
    l1_weight: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Combined loss for S-IB Adapter training.
    
    Combines:
    - Focal Loss: Handles class imbalance (many bg patches)
    - Dice Loss: Handles sparse foreground (small objects)
    - L1 Sparsity: Encourages compact masks (Information Bottleneck)
    
    Args:
        pred_mask: Predicted mask [B, N, 1] in range (0, 1)
        gt_mask: Ground truth mask [B, N, 1] binary
        sample_weights: Per-sample weights [B] (GQA=1, IV-VQA=3)
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
        dice_weight: Weight for dice loss component
        focal_weight: Weight for focal loss component
        l1_weight: Weight for L1 sparsity (set 0 to disable)
        
    Returns:
        total_loss: Combined loss scalar
        loss_dict: Dictionary with individual loss components
    """
    B = pred_mask.size(0)
    
    if sample_weights is None:
        sample_weights = torch.ones(B, device=pred_mask.device)
    
    # 1. Focal Loss (class imbalance)
    focal = sigmoid_focal_loss(
        pred_mask, gt_mask,
        alpha=focal_alpha,
        gamma=focal_gamma,
        reduction='none',
    )
    focal = focal.mean(dim=(1, 2)) * sample_weights
    focal_loss = focal.mean()
    
    # 2. Dice Loss (sparse foreground)
    dice = dice_loss(pred_mask, gt_mask, smooth=1.0, reduction='none')
    dice = dice * sample_weights
    dice_loss_val = dice.mean()
    
    # 3. L1 Sparsity Regularization (optional)
    if l1_weight > 0:
        l1_reg = pred_mask.abs().mean(dim=(1, 2)) * l1_weight
        l1_loss = l1_reg.mean()
    else:
        l1_loss = torch.tensor(0.0, device=pred_mask.device)
    
    # Combined loss
    total_loss = (
        focal_weight * focal_loss 
        + dice_weight * dice_loss_val 
        + l1_loss
    )
    
    loss_dict = {
        'focal': focal_loss.detach(),
        'dice': dice_loss_val.detach(),
        'l1': l1_loss.detach(),
        'total': total_loss.detach(),
    }
    
    return total_loss, loss_dict


def compute_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) metric.
    
    IoU > 0.4-0.5 indicates the model has learned to focus on
    the right regions.
    
    Args:
        pred_mask: Predicted mask [B, N, 1]
        gt_mask: Ground truth mask [B, N, 1]
        threshold: Binarization threshold
        
    Returns:
        Mean IoU across batch
    """
    # Binarize prediction
    pred_bin = (pred_mask > threshold).float()
    
    # Flatten
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    gt_flat = gt_mask.view(gt_mask.size(0), -1)
    
    # Intersection and Union
    intersection = (pred_flat * gt_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - intersection
    
    # IoU with epsilon for numerical stability
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.mean()
