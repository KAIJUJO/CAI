"""
Visualization Tools for S-IB Mask Validation.

Provides utilities to visualize generated foreground masks
overlaid on original images for manual quality checking.

Usage:
    python scripts/visualize_masks.py --n-samples 20
    python scripts/visualize_masks.py --meta data/processed/gqa_sib_meta.json
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_pipeline.config import DataConfig
from data_pipeline.gqa.mask_generator import MaskGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def create_overlay_image(
    image: np.ndarray,
    mask: np.ndarray,
    bboxes: List[List[int]],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create overlay visualization.
    
    - Background regions shown in grayscale
    - Foreground regions shown in color
    - Bounding boxes drawn with red borders
    
    Args:
        image: Original RGB image [H, W, 3]
        mask: Binary mask [H, W]
        bboxes: List of [x, y, w, h] bounding boxes
        alpha: Blending alpha for foreground
        
    Returns:
        Overlay image [H, W, 3]
    """
    h, w = image.shape[:2]
    
    # Resize mask to match image if needed
    if mask.shape[:2] != (h, w):
        if HAS_CV2:
            mask = cv2.resize(mask.astype(np.float32), (w, h))
        else:
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((w, h), PILImage.BILINEAR)
            mask = np.array(mask_pil).astype(np.float32) / 255.0
    
    # Binarize
    mask = (mask > 0.5).astype(np.float32)
    
    # Create grayscale version
    gray = np.mean(image, axis=2, keepdims=True).repeat(3, axis=2)
    
    # Blend: foreground in color, background in grayscale
    mask_3d = mask[:, :, np.newaxis]
    overlay = image * mask_3d + gray * (1 - mask_3d)
    overlay = overlay.astype(np.uint8)
    
    # Draw bounding boxes
    if HAS_PIL:
        overlay_pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay_pil)
        for bbox in bboxes:
            x, y, bw, bh = bbox[:4]
            draw.rectangle(
                [x, y, x + bw, y + bh],
                outline='red',
                width=2,
            )
        overlay = np.array(overlay_pil)
    
    return overlay


def visualize_single_sample(
    sample: Dict[str, Any],
    image_dir: Path,
    output_path: Optional[Path] = None,
    mask_size: Tuple[int, int] = (384, 384),
    show: bool = True,
) -> Optional[np.ndarray]:
    """
    Visualize a single sample.
    
    Args:
        sample: Sample metadata dict
        image_dir: Directory containing images
        output_path: Optional path to save visualization
        mask_size: Size to render mask
        show: Display using matplotlib
        
    Returns:
        Overlay image or None if failed
    """
    if not HAS_MPL:
        logger.error("matplotlib required for visualization")
        return None
    
    # Load image
    image_id = sample.get('image_id', '')
    image_path = image_dir / f"{image_id}.jpg"
    
    if not image_path.exists():
        # Try with images/ subdirectory
        image_path = image_dir / "images" / f"{image_id}.jpg"
    
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return None
    
    image = np.array(Image.open(image_path).convert('RGB'))
    img_h, img_w = image.shape[:2]
    
    # Get bboxes
    bboxes = sample.get('causal_bboxes', [])
    
    # Convert bbox format if needed
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
    
    # Render mask
    mask_gen = MaskGenerator(default_output_size=mask_size)
    mask = mask_gen.render_mask(
        bbox_dicts,
        image_size=(img_w, img_h),
        output_size=(img_h, img_w),  # Keep original aspect
    )
    
    # Create overlay
    bbox_list = [[b['x'], b['y'], b['w'], b['h']] for b in bbox_dicts]
    overlay = create_overlay_image(image, mask, bbox_list)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'M_fg Mask\n(coverage: {np.mean(mask):.1%})')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Foreground = Color)')
    axes[2].axis('off')
    
    # Add text info
    question = sample.get('question', '')[:60]
    answer = sample.get('answer', '')
    objects = sample.get('causal_objects', [])
    
    fig.suptitle(
        f"Q: {question}...\nA: {answer}  |  Objects: {objects}",
        fontsize=10,
        y=0.02,
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return overlay


def batch_visualize(
    meta_file: Path,
    image_dir: Path,
    output_dir: Path,
    n_samples: int = 20,
    seed: int = 42,
):
    """
    Batch visualize random samples.
    
    Args:
        meta_file: Path to preprocessed metadata
        image_dir: Directory containing images
        output_dir: Directory to save visualizations
        n_samples: Number of samples to visualize
        seed: Random seed for sampling
    """
    logger.info(f"Loading metadata from {meta_file}")
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('samples', data)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Random sample
    random.seed(seed)
    selected = random.sample(samples, min(n_samples, len(samples)))
    
    logger.info(f"Visualizing {len(selected)} samples...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(selected):
        output_path = output_dir / f"sample_{i:03d}_{sample['image_id']}.png"
        
        try:
            visualize_single_sample(
                sample,
                image_dir,
                output_path=output_path,
                show=False,
            )
        except Exception as e:
            logger.error(f"Failed to visualize sample {i}: {e}")
    
    logger.info(f"Saved {len(selected)} visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize S-IB masks for quality checking",
    )
    
    parser.add_argument(
        '--meta',
        type=str,
        default='data/processed/gqa_sib_meta.json',
        help='Path to preprocessed metadata',
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help='Directory containing images',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/visualizations',
        help='Output directory for visualizations',
    )
    parser.add_argument(
        '--n-samples', '-n',
        type=int,
        default=20,
        help='Number of samples to visualize',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    
    args = parser.parse_args()
    
    # Load config for paths
    config = DataConfig()
    
    # Resolve paths
    if args.meta.startswith('/') or ':' in args.meta:
        meta_file = Path(args.meta)
    else:
        meta_file = config.data_dir / args.meta
    
    if args.image_dir:
        image_dir = Path(args.image_dir)
    else:
        image_dir = config.gqa_raw_dir
    
    if args.output_dir.startswith('/') or ':' in args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.data_dir / args.output_dir
    
    # Run visualization
    batch_visualize(
        meta_file=meta_file,
        image_dir=image_dir,
        output_dir=output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
