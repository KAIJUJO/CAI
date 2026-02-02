"""
S-IB Adapter Trainer V2

Updated version that handles raw images from Dataset
and runs Vision Encoder on GPU (frozen).

Key Changes from V1:
- Accepts pixel_values instead of vis_feats
- Runs frozen Vision Encoder in forward pass
- Runs frozen Text Encoder for text embeddings
- Handles zero-mask samples properly
"""

import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from models.sib_adapter import SIB_Adapter
from models.losses import calc_sib_loss, compute_iou

logger = logging.getLogger(__name__)


class SIBTrainerV2:
    """
    Trainer V2 for S-IB Adapter with integrated feature extraction.
    
    This version:
    - Receives raw images from DataLoader
    - Runs frozen Vision Encoder on GPU
    - Runs frozen Text Encoder for question embeddings
    - Handles dynamic mask interpolation
    
    Args:
        adapter: SIB_Adapter instance
        vision_encoder: Frozen vision encoder (e.g., SigLIP)
        text_encoder: Frozen text encoder (LLM embedding layer)
        train_loader: Training DataLoader
        val_loader: Optional validation DataLoader
        device: Training device
        lr_adapter: Learning rate for mask_net
        lr_projector: Learning rate for projector
        warmup_ratio: Warmup ratio for scheduler
        max_epochs: Maximum training epochs
        grad_accumulation: Gradient accumulation steps
        use_amp: Use automatic mixed precision
    """
    
    def __init__(
        self,
        adapter: SIB_Adapter,
        vision_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        text_tokenizer: Optional[Any] = None,
        train_loader: DataLoader = None,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda',
        lr_adapter: float = 1e-4,
        lr_projector: float = 1e-5,
        warmup_ratio: float = 0.05,
        max_epochs: int = 5,
        grad_accumulation: int = 1,
        use_amp: bool = True,
    ):
        self.device = torch.device(device)
        
        # Models
        self.adapter = adapter.to(self.device)
        self.vision_encoder = vision_encoder.to(self.device) if vision_encoder else None
        self.text_encoder = text_encoder.to(self.device) if text_encoder else None
        self.text_tokenizer = text_tokenizer
        
        # Freeze encoders
        if self.vision_encoder:
            self.vision_encoder.eval()
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        
        if self.text_encoder:
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.grad_accumulation = grad_accumulation
        self.use_amp = use_amp
        
        # Setup optimizer with different LRs
        self.optimizer = self._setup_optimizer(lr_adapter, lr_projector)
        
        # Setup scheduler
        if train_loader:
            total_steps = len(train_loader) * max_epochs // grad_accumulation
            warmup_steps = int(total_steps * warmup_ratio)
            self.scheduler = self._setup_scheduler(total_steps, warmup_steps)
        else:
            self.scheduler = None
        
        # AMP scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Metrics
        self.history = {
            'train_loss': [],
            'train_iou': [],
            'val_loss': [],
            'val_iou': [],
        }
        
        self.best_iou = 0.0
        self.global_step = 0
    
    def _setup_optimizer(
        self,
        lr_adapter: float,
        lr_projector: float,
    ) -> optim.Optimizer:
        """Setup AdamW with different LRs for adapter and projector."""
        param_groups = [
            {
                'params': [
                    p for n, p in self.adapter.named_parameters()
                    if 'projector' not in n
                ],
                'lr': lr_adapter,
                'name': 'adapter',
            },
            {
                'params': self.adapter.projector.parameters(),
                'lr': lr_projector,
                'name': 'projector',
            },
        ]
        return optim.AdamW(param_groups, weight_decay=0.01)
    
    def _setup_scheduler(
        self,
        total_steps: int,
        warmup_steps: int,
    ):
        """Setup cosine scheduler with warmup."""
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def extract_features(
        self,
        pixel_values: torch.Tensor,
        questions: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual and text features using frozen encoders.
        
        Args:
            pixel_values: [B, C, H, W] image tensor
            questions: List of question strings
            
        Returns:
            vis_feats: [B, N, D_v] visual features
            text_feats: [B, D_t] text embeddings
        """
        with torch.no_grad():
            # Visual features
            if self.vision_encoder:
                vis_feats = self.vision_encoder(pixel_values)
                # Handle different encoder output formats
                if hasattr(vis_feats, 'last_hidden_state'):
                    vis_feats = vis_feats.last_hidden_state
                elif isinstance(vis_feats, tuple):
                    vis_feats = vis_feats[0]
            else:
                # Mock features for testing
                B = pixel_values.size(0)
                vis_feats = torch.randn(B, 576, self.adapter.vis_dim, device=self.device)
            
            # Text features
            if self.text_encoder and self.text_tokenizer:
                tokens = self.text_tokenizer(
                    questions,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=32,
                ).to(self.device)
                
                text_outputs = self.text_encoder(tokens.input_ids)
                # Take mean pooling over sequence
                if hasattr(text_outputs, 'last_hidden_state'):
                    text_feats = text_outputs.last_hidden_state.mean(dim=1)
                else:
                    text_feats = text_outputs.mean(dim=1)
            else:
                # Mock text features
                B = len(questions)
                text_feats = torch.randn(B, self.adapter.text_dim, device=self.device)
        
        return vis_feats, text_feats
    
    def interpolate_mask(
        self,
        gt_mask: torch.Tensor,
        target_n: int,
        is_zero_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interpolate GT mask to match feature map size.
        
        Args:
            gt_mask: [B, 1, H, W] or [B, 1, 1] (for zero masks)
            target_n: Number of patches (e.g., 576)
            is_zero_mask: [B] boolean tensor
            
        Returns:
            gt_mask_flat: [B, N, 1]
        """
        B = gt_mask.size(0)
        grid_size = int(math.sqrt(target_n))
        
        result = []
        for i in range(B):
            if is_zero_mask[i]:
                # Zero mask: create all zeros
                result.append(torch.zeros(target_n, 1, device=gt_mask.device))
            else:
                # Normal mask: interpolate
                mask_i = gt_mask[i:i+1]  # [1, 1, H, W]
                if mask_i.size(-1) == 1:
                    # Already tiny, expand to zeros
                    result.append(torch.zeros(target_n, 1, device=gt_mask.device))
                else:
                    mask_resized = F.interpolate(
                        mask_i,
                        size=(grid_size, grid_size),
                        mode='nearest',
                    )
                    mask_flat = mask_resized.flatten(2).transpose(1, 2)  # [1, N, 1]
                    result.append(mask_flat.squeeze(0))
        
        return torch.stack(result)  # [B, N, 1]
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.adapter.train()
        
        total_loss = 0.0
        total_iou = 0.0
        loss_components = {'focal': 0.0, 'dice': 0.0, 'l1': 0.0}
        num_batches = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.max_epochs}",
            leave=True,
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move images to device
            pixel_values = batch['pixel_values'].to(self.device)
            gt_mask = batch['gt_mask'].to(self.device)
            questions = batch.get('question', [''] * pixel_values.size(0))
            sample_weights = batch.get('weight', None)
            is_zero_mask = batch.get('is_zero_mask', torch.zeros(pixel_values.size(0), dtype=torch.bool))
            
            if sample_weights is not None:
                sample_weights = sample_weights.to(self.device)
            
            # Extract features (frozen encoders)
            vis_feats, text_feats = self.extract_features(pixel_values, questions)
            
            N = vis_feats.size(1)
            
            # Interpolate GT mask to feature map size
            gt_mask_flat = self.interpolate_mask(gt_mask, N, is_zero_mask)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                pred_mask, _ = self.adapter(vis_feats, text_feats, training=True)
                
                loss, loss_dict = calc_sib_loss(
                    pred_mask, gt_mask_flat,
                    sample_weights=sample_weights,
                )
                loss = loss / self.grad_accumulation
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accumulation == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()
                self.global_step += 1
            
            # Compute IoU (skip zero masks for meaningful metric)
            with torch.no_grad():
                # Only compute IoU for non-zero masks
                valid_mask = ~is_zero_mask
                if valid_mask.any():
                    iou = compute_iou(
                        pred_mask[valid_mask],
                        gt_mask_flat[valid_mask],
                    )
                else:
                    iou = torch.tensor(0.0)
            
            # Accumulate metrics
            total_loss += loss_dict['total'].item()
            total_iou += iou.item()
            for k in loss_components:
                loss_components[k] += loss_dict.get(k, torch.tensor(0.0)).item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
                'IoU': f"{total_iou/num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}" if self.scheduler else "N/A",
            })
        
        # Average metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_iou = total_iou / max(num_batches, 1)
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_iou'].append(avg_iou)
        
        return {
            'loss': avg_loss,
            'iou': avg_iou,
            **{k: v / max(num_batches, 1) for k, v in loss_components.items()},
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        if self.val_loader is None:
            return {}
        
        self.adapter.eval()
        
        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            pixel_values = batch['pixel_values'].to(self.device)
            gt_mask = batch['gt_mask'].to(self.device)
            questions = batch.get('question', [''] * pixel_values.size(0))
            is_zero_mask = batch.get('is_zero_mask', torch.zeros(pixel_values.size(0), dtype=torch.bool))
            
            vis_feats, text_feats = self.extract_features(pixel_values, questions)
            N = vis_feats.size(1)
            gt_mask_flat = self.interpolate_mask(gt_mask, N, is_zero_mask)
            
            pred_mask, _ = self.adapter(vis_feats, text_feats, training=False)
            
            loss, loss_dict = calc_sib_loss(pred_mask, gt_mask_flat)
            
            valid_mask = ~is_zero_mask
            if valid_mask.any():
                iou = compute_iou(pred_mask[valid_mask], gt_mask_flat[valid_mask])
            else:
                iou = torch.tensor(0.0)
            
            total_loss += loss_dict['total'].item()
            total_iou += iou.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_iou = total_iou / max(num_batches, 1)
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_iou'].append(avg_iou)
        
        return {'val_loss': avg_loss, 'val_iou': avg_iou}
    
    def train(
        self,
        save_dir: Optional[Path] = None,
        save_every: int = 1,
    ) -> Dict[str, Any]:
        """Full training loop."""
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Device: {self.device}, AMP: {self.use_amp}")
        
        for epoch in range(self.max_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            log_msg = f"Epoch {epoch+1}: "
            log_msg += f"train_loss={train_metrics['loss']:.4f}, "
            log_msg += f"train_IoU={train_metrics['iou']:.4f}"
            if val_metrics:
                log_msg += f", val_IoU={val_metrics['val_iou']:.4f}"
            logger.info(log_msg)
            
            current_iou = val_metrics.get('val_iou', train_metrics['iou'])
            if current_iou > 0.4:
                logger.info(f"✓ IoU > 0.4! Model is learning to focus on key regions.")
            
            if current_iou > self.best_iou:
                self.best_iou = current_iou
                if save_dir:
                    self.save_checkpoint(save_dir / 'best_model.pt')
            
            if save_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch{epoch+1}.pt')
        
        logger.info(f"Training complete. Best IoU: {self.best_iou:.4f}")
        return self.history
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'adapter_state': self.adapter.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'best_iou': self.best_iou,
            'history': self.history,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.adapter.load_state_dict(checkpoint['adapter_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.scheduler and checkpoint.get('scheduler_state'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.global_step = checkpoint['global_step']
        self.best_iou = checkpoint['best_iou']
        self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from {path}")
