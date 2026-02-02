"""
S-IB Training Dataset for GQA.

Provides efficient data loading for S-IB Adapter training,
reading from pre-processed metadata file.

Output format:
    image: Tensor [3, H, W] - normalized image
    mask: Tensor [1, H, W] - binary foreground mask
    question: str - question text
    answer: str - answer text
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
import numpy as np

import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from .mask_generator import MaskGenerator

logger = logging.getLogger(__name__)


class SIBDataset(Dataset):
    """
    S-IB Adapter training dataset.
    
    Reads pre-processed metadata (from preprocess_gqa.py) containing
    pre-computed causal object bounding boxes. At training time,
    only performs:
    1. Image loading and transformation
    2. Mask rendering from cached bboxes
    
    This design ensures fast data loading without complex parsing
    during training.
    
    Attributes:
        samples: List of sample metadata dicts
        image_dir: Directory containing images
        image_size: Target image/mask size (H, W)
        transform: Optional image transform
    """
    
    def __init__(
        self,
        meta_file: Path,
        image_dir: Path,
        image_size: Tuple[int, int] = (384, 384),
        transform: Optional[Callable] = None,
        normalize: bool = True,
        return_meta: bool = False,
    ):
        """
        Initialize S-IB dataset.
        
        Args:
            meta_file: Path to preprocessed metadata JSON
            image_dir: Directory containing GQA images
            image_size: Target image size (H, W)
            transform: Optional custom transform (applied before normalize)
            normalize: Apply ImageNet normalization
            return_meta: Include question/answer in output
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.custom_transform = transform
        self.normalize = normalize
        self.return_meta = return_meta
        
        # Load metadata
        logger.info(f"Loading metadata from {meta_file}")
        with open(meta_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = data.get('samples', data)  # Support both formats
        self.metadata = data.get('metadata', {})
        
        logger.info(f"Loaded {len(self.samples)} samples")
        
        # Initialize mask generator
        self.mask_generator = MaskGenerator(
            use_lemmatization=False,  # Not needed - using cached bboxes
            use_synonyms=False,
            default_output_size=image_size,
        )
        
        # Build default transform
        self._build_transforms()
    
    def _build_transforms(self):
        """Build image transforms."""
        if not HAS_TORCHVISION:
            logger.warning("torchvision not available, using basic transforms")
            self.transform = None
            return
        
        transforms = [
            T.Resize(self.image_size),
            T.ToTensor(),
        ]
        
        if self.normalize:
            # ImageNet normalization
            transforms.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        
        self.transform = T.Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict with:
                image: Tensor [3, H, W]
                mask: Tensor [1, H, W]
                (optional) question: str
                (optional) answer: str
        """
        item = self.samples[idx]
        
        # Load image
        image = self._load_image(item)
        
        # Render mask from cached bboxes
        mask = self._render_mask(item)
        
        # Apply custom transform if provided
        if self.custom_transform:
            image = self.custom_transform(image)
        
        # Build output
        output = {
            'image': image,
            'mask': mask,
        }
        
        if self.return_meta:
            output['question'] = item.get('question', '')
            output['answer'] = item.get('answer', '')
            output['image_id'] = item.get('image_id', '')
        
        return output
    
    def _load_image(self, item: Dict) -> torch.Tensor:
        """
        Load and transform image.
        
        Args:
            item: Sample metadata dict
            
        Returns:
            Transformed image tensor [3, H, W]
        """
        # Construct image path
        image_path = item.get('image_path', '')
        if not image_path:
            image_path = f"{item['image_id']}.jpg"
        
        # Handle relative path
        if not Path(image_path).is_absolute():
            full_path = self.image_dir / image_path
        else:
            full_path = Path(image_path)
        
        # Load image
        if not full_path.exists():
            # Fallback: try without 'images/' prefix
            fallback_path = self.image_dir / f"{item['image_id']}.jpg"
            if fallback_path.exists():
                full_path = fallback_path
            else:
                logger.warning(f"Image not found: {full_path}")
                # Return zero tensor
                return torch.zeros(3, *self.image_size)
        
        image = Image.open(full_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Basic fallback
            image = image.resize((self.image_size[1], self.image_size[0]))
            image = torch.from_numpy(
                np.array(image).transpose(2, 0, 1)
            ).float() / 255.0
        
        return image
    
    def _render_mask(self, item: Dict) -> torch.Tensor:
        """
        Render mask from cached bounding boxes.
        
        Prefers normalized coordinates for flexibility with different
        input sizes. Falls back to absolute coordinates if normalized
        are not available.
        
        Args:
            item: Sample metadata dict with 'causal_bboxes' or 'causal_bboxes_normalized'
            
        Returns:
            Binary mask tensor [1, H, W]
        """
        out_h, out_w = self.image_size
        
        # Prefer normalized bboxes (more flexible)
        if 'causal_bboxes_normalized' in item:
            norm_bboxes = item['causal_bboxes_normalized']
            # Convert normalized coords to output size
            bbox_dicts = []
            for bbox in norm_bboxes:
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    bbox_dicts.append({
                        'x': bbox[0] * out_w,
                        'y': bbox[1] * out_h,
                        'w': bbox[2] * out_w,
                        'h': bbox[3] * out_h,
                    })
            
            # Render directly at output size (no rescaling needed)
            mask = self.mask_generator.render_mask(
                bbox_dicts,
                image_size=(out_w, out_h),
                output_size=self.image_size,
            )
        else:
            # Fallback: use absolute coordinates
            bboxes = item.get('causal_bboxes', [])
            image_size = item.get('image_size', [640, 480])  # [W, H]
            
            bbox_dicts = []
            for bbox in bboxes:
                if isinstance(bbox, dict):
                    bbox_dicts.append(bbox)
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    bbox_dicts.append({
                        'x': bbox[0],
                        'y': bbox[1],
                        'w': bbox[2],
                        'h': bbox[3],
                    })
            
            mask = self.mask_generator.render_mask(
                bbox_dicts,
                image_size=(image_size[0], image_size[1]),
                output_size=self.image_size,
            )
        
        # Convert to tensor [1, H, W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return mask_tensor
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for a sample (for debugging).
        
        Args:
            idx: Sample index
            
        Returns:
            Sample metadata dict
        """
        return self.samples[idx]
    
    def compute_stats(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.
        
        Returns:
            Statistics dict
        """
        coverage_ratios = [
            s.get('coverage_ratio', 0) for s in self.samples
        ]
        
        return {
            'total_samples': len(self.samples),
            'image_size': self.image_size,
            'coverage_min': min(coverage_ratios) if coverage_ratios else 0,
            'coverage_max': max(coverage_ratios) if coverage_ratios else 0,
            'coverage_mean': sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else 0,
        }


def create_sib_dataloader(
    meta_file: Path,
    image_dir: Path,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (384, 384),
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for S-IB training.
    
    Args:
        meta_file: Path to preprocessed metadata
        image_dir: Directory containing images
        batch_size: Batch size
        image_size: Target image size
        shuffle: Shuffle data
        num_workers: Number of data loading workers
        **kwargs: Additional Dataset arguments
        
    Returns:
        DataLoader instance
    """
    dataset = SIBDataset(
        meta_file=meta_file,
        image_dir=image_dir,
        image_size=image_size,
        **kwargs,
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader
