"""
Filter GQA images to keep only those referenced in the 150k dataset.

This script:
1. Reads the gqa_sib_150k.json to get unique image IDs
2. Moves unused images to a backup folder (or deletes them)
3. Reports space savings
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def filter_gqa_images(
    meta_file: Path,
    image_dir: Path,
    backup_dir: Path = None,
    dry_run: bool = True,
):
    """
    Filter GQA images to keep only those in the metadata.
    
    Args:
        meta_file: Path to gqa_sib_150k.json
        image_dir: Path to GQA images directory
        backup_dir: If set, move unused images here. If None, delete them.
        dry_run: If True, only report what would be done
    """
    # Load metadata
    print(f"Loading metadata from {meta_file}")
    with open(meta_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('samples', data)
    
    # Get unique image IDs
    used_ids = set()
    for sample in samples:
        image_id = sample.get('image_id')
        if image_id:
            used_ids.add(str(image_id))
    
    print(f"Unique image IDs in dataset: {len(used_ids)}")
    
    # List all images in directory
    all_images = list(image_dir.glob("*.jpg"))
    print(f"Total images in directory: {len(all_images)}")
    
    # Categorize
    keep_images = []
    remove_images = []
    
    for img_path in all_images:
        img_id = img_path.stem  # filename without extension
        if img_id in used_ids:
            keep_images.append(img_path)
        else:
            remove_images.append(img_path)
    
    print(f"\nImages to keep: {len(keep_images)}")
    print(f"Images to remove: {len(remove_images)}")
    
    # Calculate space savings
    remove_size = sum(p.stat().st_size for p in remove_images)
    print(f"Space to free: {remove_size / (1024**3):.2f} GB")
    
    if dry_run:
        print("\n[DRY RUN] No files were modified.")
        print("Run with --execute to actually move/delete files.")
        return
    
    # Execute removal/moving
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nMoving unused images to {backup_dir}")
        for img_path in tqdm(remove_images, desc="Moving"):
            shutil.move(str(img_path), backup_dir / img_path.name)
    else:
        print("\nDeleting unused images...")
        for img_path in tqdm(remove_images, desc="Deleting"):
            img_path.unlink()
    
    print(f"\nDone! Freed {remove_size / (1024**3):.2f} GB")
    print(f"Remaining images: {len(keep_images)}")


def main():
    parser = argparse.ArgumentParser(description='Filter GQA images')
    parser.add_argument(
        '--meta', type=str,
        default='data/processed/gqa_sib_150k.json',
        help='Path to GQA metadata file'
    )
    parser.add_argument(
        '--image-dir', type=str,
        default='data/raw/gqa/images',
        help='Path to GQA images directory'
    )
    parser.add_argument(
        '--backup', type=str,
        default=None,
        help='Backup directory for unused images (if not set, delete them)'
    )
    parser.add_argument(
        '--execute', action='store_true',
        help='Actually perform the operation (default is dry-run)'
    )
    
    args = parser.parse_args()
    
    filter_gqa_images(
        meta_file=Path(args.meta),
        image_dir=Path(args.image_dir),
        backup_dir=Path(args.backup) if args.backup else None,
        dry_run=not args.execute,
    )


if __name__ == '__main__':
    main()
