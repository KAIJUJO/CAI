"""
Unified SIB Dataset for S-IB Adapter Training

This dataset handles both GQA and IV-VQA data with proper tier classification.

Key Architecture Decision:
- Dataset returns RAW images (not features)
- Feature extraction (Vision Encoder) happens in Trainer on GPU
- This avoids CPU->GPU tensor transfer bottleneck

Tier Processing:
- GQA: Load pre-generated masks from meta file
- Tier 1 (COCO): Load masks rendered from COCO annotations
- Tier 2 (DINO): Load pre-generated masks from Grounding DINO
- Tier 3 (Zero): Return zero mask placeholder
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class UnifiedSIBDataset(Dataset):
    """
    Unified dataset for S-IB training with GQA and IV-VQA data.
    
    Args:
        meta_file: Path to unified metadata JSON
        image_transform: Transform for images (SigLIP preprocessing)
        mask_transform: Transform for masks (Resize, ToTensor)
        tokenizer: Text tokenizer (optional, for text input_ids)
        max_text_length: Maximum text token length
    """
    
    def __init__(
        self,
        meta_file: Union[str, Path],
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        max_text_length: int = 32,
    ):
        self.meta_file = Path(meta_file)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # Load metadata
        logger.info(f"Loading metadata from {self.meta_file}")
        with open(self.meta_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'samples' in data:
            self.samples = data['samples']
            self.metadata = data.get('metadata', {})
        else:
            self.samples = data
            self.metadata = {}
        
        logger.info(f"Loaded {len(self.samples)} samples")
        
        # Infer base directories from first sample
        if self.samples:
            sample_path = Path(self.samples[0].get('image_path', ''))
            # Handle relative vs absolute paths
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample.
        
        Returns:
            pixel_values: Image tensor [C, H, W] (after transform)
            question: Raw question text (for tokenizer in Trainer)
            input_ids: Token IDs [L] (if tokenizer provided)
            gt_mask: Ground truth mask [1, H, W] or [1, 1] for zero mask
            tier: Sample tier (gqa, t1, t2, t3)
            is_zero_mask: Boolean flag for zero mask samples
            weight: Sample weight (IV-VQA = 3, GQA = 1)
        """
        item = self.samples[idx]
        
        # 1. Load image
        image_path = Path(item['image_path'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Return placeholder
            image = Image.new('RGB', (384, 384), color='gray')
        
        orig_size = image.size  # (W, H)
        
        if self.image_transform:
            pixel_values = self.image_transform(image)
        else:
            pixel_values = torch.from_numpy(
                np.array(image).transpose(2, 0, 1)
            ).float() / 255.0
        
        # 2. Get question text
        question = item.get('question', '')
        
        # Tokenize if tokenizer provided
        if self.tokenizer:
            tokens = self.tokenizer(
                question,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_text_length,
            )
            input_ids = tokens.input_ids.squeeze(0)
        else:
            input_ids = torch.zeros(self.max_text_length, dtype=torch.long)
        
        # 3. Load GT Mask based on tier
        tier = item.get('tier', 'gqa')
        is_zero_mask = False
        
        if tier == 't3':
            # Tier 3: Zero mask (existence negation questions)
            # ==========================================
            # DESIGN DECISION: Do NOT store T3 masks on disk!
            # Generate zero mask in memory to save disk I/O and storage.
            # The [1,1,1] shape is a placeholder; Trainer will expand it
            # to match feature map size during training.
            # ==========================================
            gt_mask = torch.zeros((1, 1, 1), dtype=torch.float32)
            is_zero_mask = True
            
        elif tier in ['t2', 't1']:
            # Tier 1/2: Load pre-generated mask
            mask_path = item.get('mask_path')
            if mask_path and Path(mask_path).exists():
                try:
                    mask_img = Image.open(mask_path).convert('L')
                    # Resize to match image size
                    mask_img = mask_img.resize(orig_size, Image.NEAREST)
                    gt_mask = torch.from_numpy(
                        np.array(mask_img)
                    ).float() / 255.0
                    gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                except Exception as e:
                    logger.warning(f"Failed to load mask {mask_path}: {e}")
                    gt_mask = torch.zeros((1, 1, 1), dtype=torch.float32)
                    is_zero_mask = True
            else:
                # Mask not found, treat as zero
                gt_mask = torch.zeros((1, 1, 1), dtype=torch.float32)
                is_zero_mask = True
                
        else:
            # GQA: Generate mask from bboxes
            gt_mask = self._generate_mask_from_bboxes(
                item, orig_size
            )
        
        # Apply mask transform if provided
        if self.mask_transform and not is_zero_mask:
            gt_mask = self.mask_transform(gt_mask)
        
        # 4. Sample weight (IV-VQA gets higher weight)
        if tier.startswith('t'):
            weight = 3.0
        else:
            weight = 1.0
        
        return {
            'pixel_values': pixel_values,
            'question': question,
            'input_ids': input_ids,
            'gt_mask': gt_mask.squeeze(0) if gt_mask.dim() == 4 else gt_mask,  # [1, H, W]
            'tier': tier,
            'is_zero_mask': is_zero_mask,
            'weight': torch.tensor(weight, dtype=torch.float32),
        }
    
    def _generate_mask_from_bboxes(
        self,
        item: Dict,
        orig_size: tuple,
    ) -> torch.Tensor:
        """Generate mask from GQA bounding boxes."""
        W, H = orig_size
        mask = np.zeros((H, W), dtype=np.float32)
        
        # Try normalized bboxes first
        bboxes_norm = item.get('causal_bboxes_normalized', [])
        if bboxes_norm:
            for bbox in bboxes_norm:
                x, y, w, h = bbox
                x1 = int(x * W)
                y1 = int(y * H)
                x2 = int((x + w) * W)
                y2 = int((y + h) * H)
                mask[y1:y2, x1:x2] = 1.0
        else:
            # Fall back to absolute bboxes
            bboxes = item.get('causal_bboxes', [])
            for bbox in bboxes:
                x, y, w, h = bbox[:4]
                mask[int(y):int(y+h), int(x):int(x+w)] = 1.0
        
        return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


class GQASIBDataset(Dataset):
    """
    GQA-specific dataset for S-IB training.
    
    Simplified version that only handles GQA data format.
    """
    
    def __init__(
        self,
        meta_file: Union[str, Path],
        image_dir: Union[str, Path],
        image_transform: Optional[Callable] = None,
        target_size: int = 384,
    ):
        self.meta_file = Path(meta_file)
        self.image_dir = Path(image_dir)
        self.image_transform = image_transform
        self.target_size = target_size
        
        # Load metadata
        with open(self.meta_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'samples' in data:
            self.samples = data['samples']
        else:
            self.samples = data
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        
        # Build image path
        image_id = item.get('image_id', '')
        image_path = self.image_dir / f"{image_id}.jpg"
        
        # Load and transform image
        try:
            image = Image.open(image_path).convert('RGB')
            orig_w, orig_h = image.size
            
            if self.image_transform:
                pixel_values = self.image_transform(image)
            else:
                image = image.resize((self.target_size, self.target_size))
                pixel_values = torch.from_numpy(
                    np.array(image).transpose(2, 0, 1)
                ).float() / 255.0
                
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            pixel_values = torch.zeros((3, self.target_size, self.target_size))
            orig_w, orig_h = self.target_size, self.target_size
        
        # Generate mask from bboxes
        mask = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        
        bboxes_norm = item.get('causal_bboxes_normalized', [])
        for bbox in bboxes_norm:
            x, y, w, h = bbox
            x1 = int(x * self.target_size)
            y1 = int(y * self.target_size)
            x2 = int((x + w) * self.target_size)
            y2 = int((y + h) * self.target_size)
            mask[y1:y2, x1:x2] = 1.0
        
        gt_mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        
        return {
            'pixel_values': pixel_values,
            'question': item.get('question', ''),
            'gt_mask': gt_mask,
            'weight': torch.tensor(1.0),
            'is_zero_mask': False,
        }
