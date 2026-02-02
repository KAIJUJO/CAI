"""
Training entry script for S-IB Adapter.

Usage:
    python scripts/train_sib.py --config configs/sib_training.yaml
    python scripts/train_sib.py --epochs 5 --batch-size 64 --lr 1e-4
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SIB_Adapter
from training.sib_trainer import SIBTrainer
from data_pipeline.gqa.sib_dataset import SIBDataset
from data_pipeline.config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train S-IB Adapter')
    
    # Data
    parser.add_argument(
        '--meta-file',
        type=str,
        default='data/processed/gqa_sib_150k.json',
        help='Path to preprocessed metadata file',
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/raw/gqa/images',
        help='Path to image directory',
    )
    
    # Model
    parser.add_argument('--vis-dim', type=int, default=1152, help='Vision dim')
    parser.add_argument('--text-dim', type=int, default=4096, help='Text dim')
    parser.add_argument('--hidden-dim', type=int, default=4096, help='Hidden dim')
    
    # Training
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Adapter learning rate')
    parser.add_argument('--lr-proj', type=float, default=1e-5, help='Projector LR')
    parser.add_argument('--warmup', type=float, default=0.05, help='Warmup ratio')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP')
    
    # System
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--output-dir', type=str, default='outputs/sib', help='Output dir')
    
    # Mock mode for testing
    parser.add_argument('--mock', action='store_true', help='Use mock data for testing')
    
    return parser.parse_args()


def create_mock_dataloader(batch_size: int, num_batches: int = 10):
    """Create mock dataloader for testing without real data."""
    
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, size: int):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Simulate 576 patches (24x24) from 384x384 image with patch_size=16
            N = 576
            return {
                'vis_feats': torch.randn(N, 1152),  # [N, D_v]
                'text_feats': torch.randn(4096),    # [D_t]
                'mask': torch.randint(0, 2, (1, 384, 384)).float(),  # [1, H, W]
                'weight': torch.tensor(1.0),
            }
    
    dataset = MockDataset(batch_size * num_batches)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )


def main():
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("S-IB Adapter Training")
    logger.info("="*60)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    
    # Create model
    adapter = SIB_Adapter(
        vis_dim=args.vis_dim,
        text_dim=args.text_dim,
        hidden_dim=args.hidden_dim,
    )
    
    param_count = sum(p.numel() for p in adapter.parameters())
    logger.info(f"Model parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # Create dataloader
    if args.mock:
        logger.info("Using MOCK data for testing")
        train_loader = create_mock_dataloader(args.batch_size)
    else:
        # Check if data exists
        meta_path = Path(args.meta_file)
        if not meta_path.exists():
            logger.error(f"Metadata file not found: {meta_path}")
            logger.info("Run: python scripts/preprocess_gqa.py first")
            return 1
        
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            logger.error(f"Image directory not found: {image_dir}")
            logger.info("Please download GQA images first")
            return 1
        
        # Create real dataset
        # Note: SIBDataset needs to be modified to return vis_feats
        # For now, this is a placeholder
        logger.info(f"Loading dataset from {meta_path}")
        # train_dataset = SIBDataset(meta_path, image_dir)
        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     num_workers=args.workers,
        #     pin_memory=True,
        # )
        
        # Fallback to mock for now
        logger.warning("Full dataset loading not implemented, using mock data")
        train_loader = create_mock_dataloader(args.batch_size)
    
    # Create trainer
    trainer = SIBTrainer(
        adapter=adapter,
        train_loader=train_loader,
        val_loader=None,
        device=args.device,
        lr_adapter=args.lr,
        lr_projector=args.lr_proj,
        warmup_ratio=args.warmup,
        max_epochs=args.epochs,
        grad_accumulation=args.grad_accum,
        use_amp=not args.no_amp,
    )
    
    # Train
    logger.info(f"Starting training for {args.epochs} epochs")
    history = trainer.train(save_dir=output_dir, save_every=1)
    
    # Summary
    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info(f"Best IoU: {trainer.best_iou:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
