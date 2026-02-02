"""
Offline Mask Generation using Grounded-SAM (Grounding DINO + SAM)

This script processes Tier 2 IV-VQA samples and generates
high-quality CONTOUR masks (not just bounding boxes).

Pipeline:
1. NLP extracts object names from questions
2. Grounding DINO detects bounding boxes
3. SAM refines boxes into precise segmentation masks

Usage:
    python scripts/build_ivvqa_masks.py --limit 100  # Test
    python scripts/build_ivvqa_masks.py              # Full run (~42k samples)
    python scripts/build_ivvqa_masks.py --no-sam     # Box only (faster)
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch


def check_dependencies(use_sam: bool = True):
    """Check if required packages are installed."""
    missing = []
    
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        missing.append("transformers")
    
    if use_sam:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError:
            missing.append("segment-anything")
    
    if missing:
        print("="*60)
        print("ERROR: Required packages not installed!")
        print("Please run:")
        for pkg in missing:
            if pkg == "segment-anything":
                print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
            else:
                print(f"  pip install {pkg}")
        print("="*60)
        return False
    return True


def load_sam_model(checkpoint: str, model_type: str, device: str):
    """Load SAM model."""
    from segment_anything import SamPredictor, sam_model_registry
    
    print(f"Loading SAM ({model_type}) from {checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def generate_sam_mask(
    sam_predictor,
    image_np: np.ndarray,
    boxes: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    Generate precise contour mask using SAM.
    
    Args:
        sam_predictor: SAM predictor
        image_np: Image as numpy array [H, W, 3]
        boxes: Detection boxes [N, 4] (x1, y1, x2, y2)
        device: Device string
        
    Returns:
        Binary mask [H, W] as uint8 (0 or 255)
    """
    if len(boxes) == 0:
        return np.zeros(image_np.shape[:2], dtype=np.uint8)
    
    # Set image for SAM
    sam_predictor.set_image(image_np)
    
    # Transform boxes for SAM
    boxes_tensor = torch.from_numpy(boxes).to(device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_tensor, image_np.shape[:2]
    )
    
    # SAM inference
    masks, scores, logits = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    # masks shape: [N_boxes, 1, H, W] (bool Tensor)
    
    # Merge masks (logical OR)
    final_mask = torch.any(masks.squeeze(1), dim=0)  # [H, W] bool
    
    return (final_mask.cpu().numpy().astype(np.uint8) * 255)


def generate_box_mask(
    image_shape: tuple,
    boxes: np.ndarray,
) -> np.ndarray:
    """
    Generate simple box-filled mask (fallback when SAM not available).
    
    Args:
        image_shape: (H, W) tuple
        boxes: Detection boxes [N, 4] (x1, y1, x2, y2)
        
    Returns:
        Binary mask [H, W] as uint8 (0 or 255)
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = 255
    
    return mask


def generate_masks_grounded_sam(
    tier2_file: Path,
    image_base_dir: Path,
    output_dir: Path,
    dino_model_name: str = "IDEA-Research/grounding-dino-base",
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_h",
    box_threshold: float = 0.35,
    device: str = "cuda",
    limit: Optional[int] = None,
    save_visualization: bool = False,
    use_sam: bool = True,
):
    """
    Generate contour masks using Grounded-SAM pipeline.
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations' if save_visualization else None
    if vis_dir:
        vis_dir.mkdir(exist_ok=True)
    
    # Load Grounding DINO
    print(f"Loading Grounding DINO: {dino_model_name}")
    processor = AutoProcessor.from_pretrained(dino_model_name)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_name).to(device)
    dino_model.eval()
    print(f"  DINO loaded on {device}")
    
    # Load SAM (optional)
    sam_predictor = None
    if use_sam and sam_checkpoint:
        sam_predictor = load_sam_model(sam_checkpoint, sam_model_type, device)
        print(f"  SAM loaded on {device}")
    elif use_sam:
        print("  WARNING: SAM checkpoint not provided, falling back to box masks")
        use_sam = False
    
    # Load Tier 2 data
    print(f"Loading Tier 2 data from {tier2_file}")
    with open(tier2_file, 'r') as f:
        tier2_items = json.load(f)
    
    if limit:
        tier2_items = tier2_items[:limit]
        print(f"  Limited to {limit} samples")
    else:
        print(f"  Total samples: {len(tier2_items)}")
    
    # Stats
    stats = {
        'total': len(tier2_items),
        'success': 0,
        'no_detection': 0,
        'error': 0,
        'mode': 'grounded-sam' if use_sam else 'box-only',
    }
    
    # Process samples
    for item in tqdm(tier2_items, desc="Generating masks"):
        image_id = item['image_id']
        prompt = item.get('dino_prompt', item.get('prompt', 'object'))
        
        img_filename = f"COCO_train2014_{image_id:012d}.jpg"
        img_path = image_base_dir / img_filename
        
        mask_path = output_dir / f"{image_id}.png"
        
        # Skip if exists
        if mask_path.exists():
            stats['success'] += 1
            continue
        
        try:
            # Load image
            if not img_path.exists():
                alt_path = image_base_dir / f"{image_id}.jpg"
                if alt_path.exists():
                    img_path = alt_path
                else:
                    stats['error'] += 1
                    continue
            
            image_pil = Image.open(img_path).convert("RGB")
            image_np = np.array(image_pil)
            h, w = image_np.shape[:2]
            
            # Format prompt for DINO
            objects = [o.strip() for o in prompt.split(',')]
            text_prompt = '. '.join([f"a {obj}" for obj in objects if obj]) + '.'
            
            # DINO detection
            inputs = processor(
                images=image_pil,
                text=text_prompt,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = dino_model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([[h, w]])
            results = processor.image_processor.post_process_object_detection(
                outputs,
                threshold=box_threshold,
                target_sizes=target_sizes,
            )[0]
            
            boxes = results["boxes"].cpu().numpy()
            
            # Generate mask
            if len(boxes) > 0:
                if use_sam and sam_predictor:
                    mask = generate_sam_mask(sam_predictor, image_np, boxes, device)
                else:
                    mask = generate_box_mask((h, w), boxes)
                stats['success'] += 1
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
                stats['no_detection'] += 1
            
            # Save mask
            Image.fromarray(mask).save(mask_path)
            
            # Save visualization
            if vis_dir and len(boxes) > 0:
                vis_img = image_np.copy()
                # Overlay mask in green
                green_overlay = np.zeros_like(vis_img)
                green_overlay[:, :, 1] = mask
                vis_img = (vis_img * 0.7 + green_overlay * 0.3).astype(np.uint8)
                Image.fromarray(vis_img).save(vis_dir / f"{image_id}_vis.jpg")
                
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            stats['error'] += 1
    
    # Save metadata
    with open(output_dir / 'generation_meta.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate IV-VQA Tier 2 masks with Grounded-SAM')
    parser.add_argument(
        '--tier2-file', type=str,
        default='data/processed/ivvqa_tiers/tier2_with_prompts.json',
        help='Path to tier2 file with dino_prompts'
    )
    parser.add_argument(
        '--image-dir', type=str,
        default='data/raw/ivvqa/train2014',
        help='Path to COCO train2014 images'
    )
    parser.add_argument(
        '--output', type=str,
        default='data/processed/ivvqa_tiers/tier2_masks',
        help='Output directory for masks'
    )
    parser.add_argument(
        '--sam-checkpoint', type=str,
        default=None,
        help='Path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth)'
    )
    parser.add_argument(
        '--sam-type', type=str,
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='SAM model type'
    )
    parser.add_argument(
        '--no-sam', action='store_true',
        help='Use box masks only (no SAM refinement)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.35,
        help='DINO detection threshold'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit samples for testing'
    )
    parser.add_argument(
        '--save-vis', action='store_true',
        help='Save visualizations'
    )
    
    args = parser.parse_args()
    
    use_sam = not args.no_sam
    
    if not check_dependencies(use_sam):
        return 1
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("="*60)
    print("IV-VQA Tier 2 Mask Generation")
    print(f"Mode: {'Grounded-SAM (contour)' if use_sam else 'Box-only'}")
    print("="*60)
    
    stats = generate_masks_grounded_sam(
        tier2_file=Path(args.tier2_file),
        image_base_dir=Path(args.image_dir),
        output_dir=Path(args.output),
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_type,
        box_threshold=args.threshold,
        device=args.device,
        limit=args.limit,
        save_visualization=args.save_vis,
        use_sam=use_sam,
    )
    
    print("\n" + "="*60)
    print("Generation Complete!")
    print(f"  Mode: {stats['mode']}")
    print(f"  Total: {stats['total']}")
    print(f"  Success: {stats['success']} ({stats['success']/max(stats['total'],1)*100:.1f}%)")
    print(f"  No Detection: {stats['no_detection']}")
    print(f"  Errors: {stats['error']}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
