"""
Build COCO Masks for IV-VQA Tier 1 samples.

This script renders masks from COCO annotations for
Tier 1 samples that matched the CausalVQA pickle.

Uses pycocotools to decode RLE masks or render polygons.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

import numpy as np
from PIL import Image

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils
except ImportError:
    print("Please install pycocotools: pip install pycocotools")
    COCO = None


def load_coco_annotations(annotations_file: Path) -> COCO:
    """Load COCO annotations."""
    print(f"Loading COCO annotations from {annotations_file}")
    coco = COCO(str(annotations_file))
    return coco


def render_mask_from_annotation(coco: COCO, ann_id: int, img_info: Dict) -> np.ndarray:
    """
    Render binary mask from COCO annotation.
    
    Args:
        coco: COCO object
        ann_id: Annotation ID
        img_info: Image info dict with height/width
        
    Returns:
        Binary mask [H, W] as uint8 (0 or 255)
    """
    ann = coco.loadAnns([ann_id])[0]
    
    h, w = img_info['height'], img_info['width']
    
    # Handle different segmentation formats
    if 'segmentation' in ann:
        seg = ann['segmentation']
        
        if isinstance(seg, list):
            # Polygon format
            rle = mask_utils.frPyObjects(seg, h, w)
            if isinstance(rle, list):
                rle = mask_utils.merge(rle)
            mask = mask_utils.decode(rle)
        elif isinstance(seg, dict):
            # RLE format
            if isinstance(seg['counts'], list):
                # Uncompressed RLE
                rle = mask_utils.frPyObjects([seg], h, w)[0]
            else:
                # Compressed RLE
                rle = seg
            mask = mask_utils.decode(rle)
        else:
            # Unknown format, fallback to bbox
            bbox = ann.get('bbox', [0, 0, w, h])
            x, y, bw, bh = map(int, bbox)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y:y+bh, x:x+bw] = 1
    else:
        # No segmentation, use bbox
        bbox = ann.get('bbox', [0, 0, w, h])
        x, y, bw, bh = map(int, bbox)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y:y+bh, x:x+bw] = 1
    
    return (mask * 255).astype(np.uint8)


def build_tier1_masks(
    tier1_file: Path,
    pickle_file: Path,
    coco_annotations_file: Path,
    output_dir: Path,
    limit: Optional[int] = None,
):
    """
    Build masks for Tier 1 samples from COCO annotations.
    
    Args:
        tier1_file: tier1_coco_indices.json
        pickle_file: CausalVQA pickle file
        coco_annotations_file: COCO instances_train2014.json
        output_dir: Output directory for masks
    """
    if COCO is None:
        print("pycocotools not installed!")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Tier 1 indices
    print(f"Loading Tier 1 indices from {tier1_file}")
    with open(tier1_file, 'r') as f:
        tier1_items = json.load(f)
    
    if limit:
        tier1_items = tier1_items[:limit]
    
    print(f"Processing {len(tier1_items)} Tier 1 samples")
    
    # Load CausalVQA pickle to get ann_id mapping
    print(f"Loading CausalVQA pickle from {pickle_file}")
    with open(pickle_file, 'rb') as f:
        cv_data = pickle.load(f)
    
    # Build qid -> ann_id mapping
    qid_to_ann = {}
    for item in cv_data:
        qid = item.get('question_id')
        ann_id = item.get('id')  # COCO annotation ID
        if qid and ann_id:
            qid_to_ann[qid] = ann_id
    
    print(f"  Loaded {len(qid_to_ann)} question->annotation mappings")
    
    # Load COCO
    coco = load_coco_annotations(coco_annotations_file)
    
    # Process each sample
    stats = {'total': len(tier1_items), 'success': 0, 'no_ann': 0, 'error': 0}
    
    for item in tqdm(tier1_items, desc="Generating Tier 1 masks"):
        image_id = item['image_id']
        coco_ann_id = item.get('coco_ann_id')
        
        mask_filename = f"{image_id}.png"
        mask_path = output_dir / mask_filename
        
        # Skip if exists
        if mask_path.exists():
            stats['success'] += 1
            continue
        
        if not coco_ann_id:
            stats['no_ann'] += 1
            continue
        
        try:
            # Get image info
            img_info = coco.loadImgs([image_id])[0]
            
            # Render mask
            mask = render_mask_from_annotation(coco, coco_ann_id, img_info)
            
            # Save
            Image.fromarray(mask).save(mask_path)
            stats['success'] += 1
            
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            stats['error'] += 1
    
    # Save metadata
    with open(output_dir / 'generation_meta.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build COCO masks for Tier 1')
    parser.add_argument(
        '--tier1-file', type=str,
        default='data/processed/ivvqa_tiers/tier1_coco_indices.json',
        help='Path to tier1 indices file'
    )
    parser.add_argument(
        '--pickle', type=str,
        default='data/raw/ivvqa/CausalVQA/cv_vqa_generation/train2014coco_counting_id_area_overlap_only_one_considered_at_a_time.pickle',
        help='Path to CausalVQA pickle'
    )
    parser.add_argument(
        '--coco-ann', type=str,
        default='data/raw/ivvqa/annotations/instances_train2014.json',
        help='Path to COCO instances annotations'
    )
    parser.add_argument(
        '--output', type=str,
        default='data/processed/ivvqa_tiers/tier1_masks',
        help='Output directory'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit samples for testing'
    )
    
    args = parser.parse_args()
    
    stats = build_tier1_masks(
        tier1_file=Path(args.tier1_file),
        pickle_file=Path(args.pickle),
        coco_annotations_file=Path(args.coco_ann),
        output_dir=Path(args.output),
        limit=args.limit,
    )
    
    if stats:
        print("\n" + "="*50)
        print("Tier 1 Mask Generation Complete!")
        print(f"  Success: {stats['success']}")
        print(f"  No Annotation: {stats['no_ann']}")
        print(f"  Errors: {stats['error']}")


if __name__ == '__main__':
    main()
