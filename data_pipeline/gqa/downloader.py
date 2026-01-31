"""
GQA Dataset downloader with support for official Stanford sources.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import zipfile

from ..config import DataConfig
from ..utils.download import download_with_resume, verify_download
from ..utils.io import extract_archive

logger = logging.getLogger(__name__)


class GQADownloader:
    """
    Downloader for GQA dataset components.
    
    Downloads from Stanford NLP official sources:
    - Questions (balanced/all)
    - Scene Graphs
    - Images (optional, large)
    
    Supports:
    - Resume interrupted downloads
    - Selective component download
    - Automatic extraction
    """
    
    # Official download URLs
    URLS = {
        'questions': 'https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip',
        'scene_graphs': 'https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip',
        'images': 'https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip',
        # Alternative: Visual Genome images (same images)
        'images_vg': 'https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip',
    }
    
    # Expected file sizes (approximate, for validation)
    EXPECTED_SIZES = {
        'questions': 1_500_000_000,  # ~1.5GB
        'scene_graphs': 200_000_000,  # ~200MB
        'images': 20_000_000_000,     # ~20GB
    }
    
    def __init__(self, config: DataConfig):
        """
        Initialize GQA downloader.
        
        Args:
            config: DataConfig instance with path configuration
        """
        self.config = config
        self.download_dir = config.gqa_raw_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def download_questions(self, force: bool = False) -> bool:
        """
        Download GQA questions dataset.
        
        Args:
            force: Re-download even if file exists
            
        Returns:
            True if download successful
        """
        dest = self.download_dir / "questions1.2.zip"
        
        if dest.exists() and not force:
            logger.info(f"Questions already downloaded: {dest}")
            return True
        
        logger.info("Downloading GQA questions...")
        success = download_with_resume(
            self.URLS['questions'],
            dest,
            show_progress=True,
        )
        
        if success:
            logger.info("Extracting questions...")
            extract_archive(dest, self.download_dir / "questions")
        
        return success
    
    def download_scene_graphs(self, force: bool = False) -> bool:
        """
        Download GQA scene graphs.
        
        Args:
            force: Re-download even if file exists
            
        Returns:
            True if download successful
        """
        dest = self.download_dir / "sceneGraphs.zip"
        
        if dest.exists() and not force:
            logger.info(f"Scene graphs already downloaded: {dest}")
            return True
        
        logger.info("Downloading GQA scene graphs...")
        success = download_with_resume(
            self.URLS['scene_graphs'],
            dest,
            show_progress=True,
        )
        
        if success:
            logger.info("Extracting scene graphs...")
            extract_archive(dest, self.download_dir / "sceneGraphs")
        
        return success
    
    def download_images(self, force: bool = False) -> bool:
        """
        Download GQA images (warning: ~20GB).
        
        Args:
            force: Re-download even if file exists
            
        Returns:
            True if download successful
        """
        dest = self.download_dir / "images.zip"
        
        if dest.exists() and not force:
            logger.info(f"Images already downloaded: {dest}")
            return True
        
        logger.warning("Downloading GQA images (~20GB). This may take a while...")
        success = download_with_resume(
            self.URLS['images'],
            dest,
            show_progress=True,
        )
        
        if success:
            logger.info("Extracting images...")
            extract_archive(dest, self.download_dir / "images")
        
        return success
    
    def download_all(
        self, 
        include_images: bool = False,
        force: bool = False,
    ) -> Dict[str, bool]:
        """
        Download all GQA components.
        
        Args:
            include_images: Whether to download images (large)
            force: Re-download existing files
            
        Returns:
            Dict mapping component name to success status
        """
        results = {}
        
        results['questions'] = self.download_questions(force)
        results['scene_graphs'] = self.download_scene_graphs(force)
        
        if include_images:
            results['images'] = self.download_images(force)
        
        return results
    
    def verify_download(self) -> Dict[str, bool]:
        """
        Verify downloaded files exist and have reasonable sizes.
        
        Returns:
            Dict mapping component name to verification status
        """
        results = {}
        
        # Check questions
        questions_dir = self.download_dir / "questions"
        results['questions'] = (
            questions_dir.exists() and 
            any(questions_dir.glob("*.json"))
        )
        
        # Check scene graphs
        sg_dir = self.download_dir / "sceneGraphs"
        results['scene_graphs'] = (
            sg_dir.exists() and
            any(sg_dir.glob("*.json"))
        )
        
        # Check images
        images_dir = self.download_dir / "images"
        results['images'] = (
            images_dir.exists() and
            len(list(images_dir.glob("*.jpg"))) > 1000
        )
        
        return results
    
    def get_questions_path(self, split: str = "train_balanced") -> Path:
        """
        Get path to questions file for a specific split.
        
        Args:
            split: Dataset split (train_balanced, val_balanced, etc.)
            
        Returns:
            Path to questions JSON file
        """
        return self.download_dir / "questions" / f"{split}_questions.json"
    
    def get_scene_graphs_path(self, split: str = "train") -> Path:
        """
        Get path to scene graphs file for a specific split.
        
        Args:
            split: Dataset split (train, val)
            
        Returns:
            Path to scene graphs JSON file
        """
        return self.download_dir / "sceneGraphs" / f"{split}_sceneGraphs.json"
    
    def get_image_path(self, image_id: str) -> Path:
        """
        Get path to a specific image.
        
        Args:
            image_id: GQA image ID
            
        Returns:
            Path to image file
        """
        return self.download_dir / "images" / f"{image_id}.jpg"
