"""
S-IB Adapter: Spatial Information Bottleneck Adapter

A lightweight adapter that learns to generate causal masks for 
query-aware visual feature filtering.

Key features:
- Dynamic patch handling (works with any N)
- Leaky Bottleneck for cold-start protection
- Noise injection for background suppression
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class SIB_Adapter(nn.Module):
    """
    Spatial Information Bottleneck Adapter.
    
    Generates a binary mask to filter visual features based on text query,
    then projects the filtered features to LLM embedding space.
    
    Args:
        vis_dim: Vision encoder output dimension (e.g., 1152 for SigLIP)
        text_dim: Text embedding dimension (e.g., 4096 for Qwen)
        hidden_dim: Projector hidden dimension
        alpha_init: Initial value for leaky coefficient
    """
    
    def __init__(
        self,
        vis_dim: int = 1152,
        text_dim: int = 4096,
        hidden_dim: int = 4096,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        
        self.vis_dim = vis_dim
        self.text_dim = text_dim
        
        # 1. Text Mapper: Map text features to vision space for concatenation
        self.text_mapper = nn.Linear(text_dim, vis_dim)
        
        # 2. Mask Generator Network
        self.mask_net = nn.Sequential(
            nn.Linear(vis_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        
        # Initialize final layer bias to positive value for warm start
        # This ensures initial masks are close to 1 (pass-through)
        nn.init.constant_(self.mask_net[-2].bias, 2.0)
        
        # 3. Leaky Coefficient (learnable residual)
        # Prevents "blindness" during early training
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        
        # 4. Projector: Maps visual features to LLM space
        self.projector = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        vis_feats: torch.Tensor,
        text_feats: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional noise injection.
        
        Args:
            vis_feats: Visual features [B, N, D_v] (D_v = vis_dim)
            text_feats: Global text embedding [B, D_t] (D_t = text_dim)
            training: Whether in training mode (applies noise)
            
        Returns:
            pred_mask: Predicted foreground mask [B, N, 1]
            llm_inputs: Projected features for LLM [B, N, hidden_dim]
        """
        B, N, D = vis_feats.shape
        
        # --- A. Mask Generation ---
        # 1. Map text features to vision space and expand
        txt_mapped = self.text_mapper(text_feats)  # [B, D_v]
        txt_expanded = txt_mapped.unsqueeze(1).expand(-1, N, -1)  # [B, N, D_v]
        
        # 2. Concatenate visual and text features
        cat_feats = torch.cat([vis_feats, txt_expanded], dim=-1)  # [B, N, 2*D_v]
        
        # 3. Generate mask
        pred_mask = self.mask_net(cat_feats)  # [B, N, 1] in range (0, 1)
        
        # --- B. Leaky Information Bottleneck ---
        if training:
            # Generate noise with same statistics as visual features
            noise = torch.randn_like(vis_feats) * vis_feats.std(dim=1, keepdim=True)
            
            # Soft gating with leaky residual:
            # Z_clean = Z * M + alpha * Z + noise * (1 - M)
            # - Foreground (M→1): keeps original features
            # - Background (M→0): replaced with noise
            # - Alpha term: always preserves some signal (anti-blindness)
            vis_feats_ib = (
                vis_feats * pred_mask 
                + self.alpha * vis_feats 
                + noise * (1 - pred_mask)
            )
        else:
            # Inference: just apply soft mask with leaky residual
            vis_feats_ib = vis_feats * pred_mask + self.alpha * vis_feats
        
        # --- C. Projection to LLM space ---
        llm_inputs = self.projector(vis_feats_ib)
        
        return pred_mask, llm_inputs
    
    def load_projector_weights(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load pre-trained projector weights from original VLM.
        
        Args:
            state_dict: State dict from original VLM's projector/merger
        """
        self.projector.load_state_dict(state_dict)
    
    def get_mask_only(
        self,
        vis_feats: torch.Tensor,
        text_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get mask prediction without projection (for visualization/loss).
        
        Args:
            vis_feats: Visual features [B, N, D_v]
            text_feats: Text embedding [B, D_t]
            
        Returns:
            pred_mask: [B, N, 1]
        """
        B, N, D = vis_feats.shape
        txt_mapped = self.text_mapper(text_feats).unsqueeze(1).expand(-1, N, -1)
        cat_feats = torch.cat([vis_feats, txt_mapped], dim=-1)
        return self.mask_net(cat_feats)


def reshape_mask_to_grid(
    mask: torch.Tensor,
    grid_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Reshape flat mask to 2D grid for visualization.
    
    Args:
        mask: [B, N, 1] or [B, N]
        grid_size: Size of square grid (auto-computed if None)
        
    Returns:
        mask_2d: [B, 1, H, W]
    """
    if mask.dim() == 3:
        mask = mask.squeeze(-1)  # [B, N]
    
    B, N = mask.shape
    
    if grid_size is None:
        grid_size = int(math.sqrt(N))
        assert grid_size * grid_size == N, f"N={N} is not a perfect square"
    
    return mask.view(B, 1, grid_size, grid_size)


def interpolate_gt_mask(
    gt_mask: torch.Tensor,
    target_n: int,
) -> torch.Tensor:
    """
    Interpolate ground truth mask to match model's patch count.
    
    Args:
        gt_mask: Ground truth mask [B, 1, H, W] (e.g., 384x384)
        target_n: Target number of patches (e.g., 576 for 24x24)
        
    Returns:
        gt_flat: Flattened mask [B, N, 1]
    """
    grid_size = int(math.sqrt(target_n))
    
    # Resize using nearest neighbor to preserve binary values
    gt_resized = F.interpolate(
        gt_mask,
        size=(grid_size, grid_size),
        mode='nearest',
    )  # [B, 1, grid_size, grid_size]
    
    # Flatten to [B, N, 1]
    gt_flat = gt_resized.flatten(2).transpose(1, 2)  # [B, N, 1]
    
    return gt_flat
