"""
Main entry script for perception data preparation.

Usage:
    python -m scripts.prepare_perception_data --gqa --ivvqa
    python -m scripts.prepare_perception_data --config configs/data_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_pipeline.config import DataConfig
from data_pipeline.gqa import GQADataset
from data_pipeline.ivvqa import IVVQADataset


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def prepare_gqa(config: DataConfig, sample_size: int, download: bool = True) -> bool:
    """
    Prepare GQA dataset.
    
    Args:
        config: DataConfig instance
        sample_size: Number of samples to prepare
        download: Whether to download if not present
        
    Returns:
        True if successful
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Preparing GQA Dataset")
    logger.info("=" * 50)
    
    try:
        dataset = GQADataset(config)
        
        if download:
            logger.info("Downloading GQA data (questions and scene graphs)...")
            dataset.download(include_images=False)
        
        logger.info(f"Processing and sampling {sample_size} samples...")
        data = dataset.prepare(sample_size=sample_size, strategy="combined")
        
        # Print statistics
        stats = dataset.get_stats()
        logger.info(f"GQA Stats:")
        logger.info(f"  - Total samples: {stats.get('total_questions', 0)}")
        logger.info(f"  - Question types: {len(stats.get('question_types', {}))}")
        logger.info(f"  - Unique answers: {stats.get('unique_answers', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"GQA preparation failed: {e}")
        return False


def prepare_ivvqa(config: DataConfig, sample_size: int, download: bool = True) -> bool:
    """
    Prepare IV-VQA dataset with positive/negative pairs.
    
    Args:
        config: DataConfig instance
        sample_size: Number of pairs to prepare
        download: Whether to download if not present
        
    Returns:
        True if successful
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Preparing IV-VQA Dataset (Positive/Negative Pairs)")
    logger.info("=" * 50)
    
    try:
        dataset = IVVQADataset(config, use_scored_pairs=True)
        
        if download:
            logger.info("Downloading VQA v2.0 data...")
            dataset.download()
        
        logger.info(f"Building positive/negative pairs (target: {sample_size})...")
        pairs = dataset.prepare_pairs(sample_size=sample_size)
        
        # Print statistics
        stats = dataset.get_stats()
        logger.info(f"IV-VQA Pair Stats:")
        logger.info(f"  - Total pairs: {stats.get('total_pairs', 0)}")
        logger.info(f"  - Unique images: {stats.get('unique_images', 0)}")
        logger.info(f"  - Causal types: {stats.get('causal_types', {})}")
        logger.info(f"  - Background types: {stats.get('background_types', {})}")
        
        return True
        
    except Exception as e:
        logger.error(f"IV-VQA preparation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare perception datasets for CAI-System training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare both datasets with default settings
  python prepare_perception_data.py --gqa --ivvqa

  # Prepare only GQA with custom sample size
  python prepare_perception_data.py --gqa --gqa-samples 100000

  # Use custom config file
  python prepare_perception_data.py --config my_config.yaml --gqa --ivvqa

  # Skip download (use existing data)
  python prepare_perception_data.py --gqa --ivvqa --no-download
        """
    )
    
    # Dataset selection
    parser.add_argument(
        '--gqa', action='store_true',
        help='Prepare GQA dataset'
    )
    parser.add_argument(
        '--ivvqa', action='store_true',
        help='Prepare IV-VQA dataset with positive/negative pairs'
    )
    
    # Sample sizes
    parser.add_argument(
        '--gqa-samples', type=int, default=150000,
        help='Number of GQA samples to prepare (default: 150000)'
    )
    parser.add_argument(
        '--ivvqa-samples', type=int, default=60000,
        help='Number of IV-VQA pairs to prepare (default: 60000)'
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Override data directory'
    )
    
    # Behavior
    parser.add_argument(
        '--no-download', action='store_true',
        help='Skip download, use existing data only'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load or create config
    if args.config and Path(args.config).exists():
        config = DataConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = DataConfig()
        logger.info("Using default configuration")
    
    # Override data directory if specified
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    
    # Ensure directories exist
    config.ensure_dirs()
    logger.info(f"Data directory: {config.data_dir}")
    
    # Prepare selected datasets
    results = {}
    download = not args.no_download
    
    if args.gqa:
        results['gqa'] = prepare_gqa(config, args.gqa_samples, download)
    
    if args.ivvqa:
        results['ivvqa'] = prepare_ivvqa(config, args.ivvqa_samples, download)
    
    if not args.gqa and not args.ivvqa:
        logger.warning("No datasets selected. Use --gqa and/or --ivvqa")
        parser.print_help()
        return 1
    
    # Summary
    logger.info("=" * 50)
    logger.info("Preparation Summary")
    logger.info("=" * 50)
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {name.upper()}: {status}")
    
    if all(results.values()):
        logger.info("All datasets prepared successfully!")
        return 0
    else:
        logger.error("Some datasets failed to prepare")
        return 1


if __name__ == "__main__":
    sys.exit(main())
