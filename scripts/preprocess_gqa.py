"""
GQA Offline Preprocessing Script for S-IB Training.

This script processes GQA data offline to generate a clean metadata file
containing pre-computed causal object bounding boxes for each sample.

Usage:
    python scripts/preprocess_gqa.py --output data/processed/gqa_sib_meta.json
    python scripts/preprocess_gqa.py --dry-run --limit 1000  # Test run
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_pipeline.config import DataConfig
from data_pipeline.gqa.parser import GQAParser
from data_pipeline.gqa.mask_generator import MaskGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def preprocess_gqa(
    config: DataConfig,
    output_path: Path,
    sample_size: Optional[int] = None,
    split: str = "train_balanced",
    dry_run: bool = False,
    id_match_only: bool = False,
) -> Dict[str, Any]:
    """
    Preprocess GQA data for S-IB training.
    
    Performs offline processing:
    1. Parse questions and scene graphs
    2. Extract causal objects from semantic field
    3. Match objects to scene graph bounding boxes
    4. Filter out samples with empty masks
    5. Save processed metadata to JSON
    
    Args:
        config: DataConfig instance
        output_path: Path to save processed metadata
        sample_size: Number of samples to process (None = all)
        split: Dataset split to process
        dry_run: If True, don't save output
        
    Returns:
        Statistics dictionary
    """
    logger.info("=" * 60)
    logger.info("GQA Preprocessing for S-IB Training")
    logger.info("=" * 60)
    
    # Initialize components
    parser = GQAParser(config)
    mask_gen = MaskGenerator(
        use_lemmatization=True,
        use_synonyms=True,
        default_output_size=(384, 384),
    )
    
    # Parse scene graphs first (needed for matching)
    logger.info("Loading scene graphs...")
    sg_split = split.replace("_balanced", "")
    scene_graphs = parser.parse_scene_graphs(split=sg_split)
    logger.info(f"Loaded {len(scene_graphs)} scene graphs")
    
    # Parse questions
    logger.info("Loading questions...")
    questions = parser.parse_questions(split=split, limit=sample_size)
    logger.info(f"Loaded {len(questions)} questions")
    
    # Process each sample
    logger.info("Processing samples...")
    processed = []
    stats = {
        'total': len(questions),
        'valid': 0,
        'no_semantic': 0,
        'no_scene_graph': 0,
        'no_match': 0,
        'empty_mask': 0,
        'match_types': defaultdict(int),
        'coverage_ratios': [],
    }
    
    for q in tqdm(questions, desc="Processing"):
        image_id = q.get('image_id', '')
        
        # Check scene graph exists
        if image_id not in scene_graphs:
            stats['no_scene_graph'] += 1
            continue
        
        scene_graph = scene_graphs[image_id]
        
        # Check semantic field exists
        semantic = q.get('semantic', [])
        if not semantic:
            stats['no_semantic'] += 1
            continue
        
        # Process sample
        result = mask_gen.process_sample(q, scene_graph)
        
        if result is None:
            stats['no_match'] += 1
            continue
        
        if not result['mask_stats']['is_valid']:
            stats['empty_mask'] += 1
            continue
        
        # Filter: only keep samples with ID match (most reliable)
        if id_match_only:
            match_types = result.get('match_types', {})
            has_id_match = any(mt == 'id' for mt in match_types.values())
            if not has_id_match:
                stats['no_match'] += 1
                continue
        
        # Build output sample
        processed_sample = {
            'question_id': result['question_id'],
            'image_id': result['image_id'],
            'image_path': f"images/{result['image_id']}.jpg",
            'question': result['question'],
            'answer': result['answer'],
            'causal_objects': result['causal_objects'],
            'causal_object_ids': result.get('causal_object_ids', []),
            'causal_bboxes': result['causal_bboxes'],
            'causal_bboxes_normalized': result.get('causal_bboxes_normalized', []),
            'image_size': result['image_size'],
            'coverage_ratio': result['mask_stats']['coverage_ratio'],
        }
        processed.append(processed_sample)
        
        # Update stats
        stats['valid'] += 1
        stats['coverage_ratios'].append(result['mask_stats']['coverage_ratio'])
        for obj, match_type in result['match_types'].items():
            stats['match_types'][match_type] += 1
    
    # Compute final statistics
    if stats['coverage_ratios']:
        stats['coverage_min'] = min(stats['coverage_ratios'])
        stats['coverage_max'] = max(stats['coverage_ratios'])
        stats['coverage_mean'] = sum(stats['coverage_ratios']) / len(stats['coverage_ratios'])
    else:
        stats['coverage_min'] = 0
        stats['coverage_max'] = 0
        stats['coverage_mean'] = 0
    
    stats['valid_ratio'] = stats['valid'] / stats['total'] if stats['total'] > 0 else 0
    stats['match_types'] = dict(stats['match_types'])
    del stats['coverage_ratios']  # Don't save full list
    
    # Log statistics
    logger.info("=" * 60)
    logger.info("Processing Statistics")
    logger.info("=" * 60)
    logger.info(f"Total questions: {stats['total']}")
    logger.info(f"Valid samples: {stats['valid']} ({stats['valid_ratio']:.1%})")
    logger.info(f"Skipped - no semantic: {stats['no_semantic']}")
    logger.info(f"Skipped - no scene graph: {stats['no_scene_graph']}")
    logger.info(f"Skipped - no match: {stats['no_match']}")
    logger.info(f"Skipped - empty mask: {stats['empty_mask']}")
    logger.info(f"Match types: {stats['match_types']}")
    logger.info(f"Coverage ratio: min={stats['coverage_min']:.3f}, "
                f"max={stats['coverage_max']:.3f}, mean={stats['coverage_mean']:.3f}")
    
    # Save output
    if not dry_run and processed:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'metadata': {
                'split': split,
                'total_samples': len(processed),
                'mask_size': [384, 384],
                'stats': stats,
            },
            'samples': processed,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(processed)} samples to {output_path}")
    elif dry_run:
        logger.info("[DRY RUN] Would save to: " + str(output_path))
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GQA data for S-IB training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/processed/gqa_sib_meta.json',
        help='Output metadata file path',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train_balanced',
        help='Dataset split to process',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of samples (for testing)',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Override data directory',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without saving output',
    )
    parser.add_argument(
        '--id-match-only',
        action='store_true',
        help='Only keep samples with direct ID match (most reliable)',
    )
    
    args = parser.parse_args()
    
    # Load config
    config = DataConfig()
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    
    # Determine output path
    if args.output.startswith('/') or ':' in args.output:
        # Absolute path provided
        output_path = Path(args.output)
    elif args.output.startswith('data/'):
        # Relative path starting with data/ - remove prefix to avoid duplication
        rel_path = args.output[5:]  # Remove 'data/' prefix
        output_path = config.data_dir / rel_path
    else:
        # Simple filename - put in processed/
        output_path = config.data_dir / "processed" / args.output
    
    # Run preprocessing
    stats = preprocess_gqa(
        config=config,
        output_path=output_path,
        sample_size=args.limit,
        split=args.split,
        dry_run=args.dry_run,
        id_match_only=getattr(args, 'id_match_only', False),
    )
    
    return 0 if stats['valid'] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
