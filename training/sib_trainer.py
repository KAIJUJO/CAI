"""
S-IB Adapter Trainer

Handles the training loop for the Spatial Information Bottleneck Adapter,
including mixed dataset loading, loss computation, and metrics tracking.
"""

import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from models.sib_adapter import SIB_Adapter, interpolate_gt_mask
from models.losses import calc_sib_loss, compute_iou

logger = logging.getLogger(__name__)


class SIBTrainer:
    """
    Trainer for S-IB Adapter.
    
    Handles:
    - Mixed dataset loading (GQA + IV-VQA)
    - Dynamic GT mask interpolation
    - Training loop with AMP
    - Metrics tracking (IoU, loss components)
    
    Args:
        adapter: SIB_Adapter instance
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
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda',
        lr_adapter: float = 1e-4,
        lr_projector: float = 1e-5,
        warmup_ratio: float = 0.05,
        max_epochs: int = 5,
        grad_accumulation: int = 1,
        use_amp: bool = True,
    ):
        self.adapter = adapter.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.grad_accumulation = grad_accumulation
        self.use_amp = use_amp
        
        # Setup optimizer with different LRs
        self.optimizer = self._setup_optimizer(lr_adapter, lr_projector)
        
        # Setup scheduler
        total_steps = len(train_loader) * max_epochs // grad_accumulation
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = self._setup_scheduler(total_steps, warmup_steps)
        
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
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dict with epoch metrics
        """
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
            # Move to device
            vis_feats = batch['vis_feats'].to(self.device)  # [B, N, D]
            text_feats = batch['text_feats'].to(self.device)  # [B, D_t]
            gt_mask = batch['mask'].to(self.device)  # [B, 1, H, W]
            sample_weights = batch.get('weight', None)
            
            if sample_weights is not None:
                sample_weights = sample_weights.to(self.device)
            
            # Get target patch count from vis_feats
            N = vis_feats.size(1)
            
            # Interpolate GT mask to match patch grid
            gt_mask_flat = interpolate_gt_mask(gt_mask, N)  # [B, N, 1]
            
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
                self.scheduler.step()
                self.global_step += 1
            
            # Compute IoU
            with torch.no_grad():
                iou = compute_iou(pred_mask, gt_mask_flat)
            
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
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}",
            })
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_iou'].append(avg_iou)
        
        return {
            'loss': avg_loss,
            'iou': avg_iou,
            **{k: v / num_batches for k, v in loss_components.items()},
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
            vis_feats = batch['vis_feats'].to(self.device)
            text_feats = batch['text_feats'].to(self.device)
            gt_mask = batch['mask'].to(self.device)
            
            N = vis_feats.size(1)
            gt_mask_flat = interpolate_gt_mask(gt_mask, N)
            
            pred_mask, _ = self.adapter(vis_feats, text_feats, training=False)
            
            loss, loss_dict = calc_sib_loss(pred_mask, gt_mask_flat)
            iou = compute_iou(pred_mask, gt_mask_flat)
            
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
        """
        Full training loop.
        
        Args:
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Device: {self.device}, AMP: {self.use_amp}")
        
        for epoch in range(self.max_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log
            log_msg = f"Epoch {epoch+1}: "
            log_msg += f"train_loss={train_metrics['loss']:.4f}, "
            log_msg += f"train_IoU={train_metrics['iou']:.4f}"
            if val_metrics:
                log_msg += f", val_IoU={val_metrics['val_iou']:.4f}"
            logger.info(log_msg)
            
            # Check convergence
            current_iou = val_metrics.get('val_iou', train_metrics['iou'])
            if current_iou > 0.4:
                logger.info(f"✓ IoU > 0.4! Model is learning to focus on key regions.")
            
            # Save best model
            if current_iou > self.best_iou:
                self.best_iou = current_iou
                if save_dir:
                    self.save_checkpoint(save_dir / 'best_model.pt')
            
            # Save periodic checkpoint
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
            'scheduler_state': self.scheduler.state_dict(),
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
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.global_step = checkpoint['global_step']
        self.best_iou = checkpoint['best_iou']
        self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from {path}")
