"""
IV-VQA Dataset downloader.

IV-VQA is part of the Causal VQA project and can be obtained from:
- GitHub: https://github.com/AgarwalVedika/CausalVQA
- The dataset is based on VQA v2.0 with edited images for counterfactual testing
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional
import shutil

from ..config import DataConfig
from ..utils.download import download_file, download_with_resume
from ..utils.io import extract_archive

logger = logging.getLogger(__name__)


class IVVQADownloader:
    """
    Downloader for IV-VQA (Introspective VQA) dataset.
    
    The dataset consists of:
    - Original VQA v2.0 questions/answers
    - Edited images where objects are removed/modified
    - Annotations linking original and edited samples
    
    Sources:
    - GitHub repo: AgarwalVedika/CausalVQA
    - VQA v2.0 base: https://visualqa.org/download.html
    """
    
    # VQA v2.0 URLs (base dataset)
    VQA_URLS = {
        'questions_train': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip',
        'questions_val': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
        'annotations_train': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip',
        'annotations_val': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',
    }
    
    # COCO images URLs
    COCO_URLS = {
        'train2014': 'http://images.cocodataset.org/zips/train2014.zip',
        'val2014': 'http://images.cocodataset.org/zips/val2014.zip',
    }
    
    # CausalVQA GitHub repo
    CAUSAL_VQA_REPO = 'https://github.com/AgarwalVedika/CausalVQA.git'
    
    def __init__(self, config: DataConfig):
        """
        Initialize IV-VQA downloader.
        
        Args:
            config: DataConfig instance
        """
        self.config = config
        self.download_dir = config.ivvqa_raw_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def download_vqa_questions(self, split: str = "train", force: bool = False) -> bool:
        """
        Download VQA v2.0 questions.
        
        Args:
            split: 'train' or 'val'
            force: Re-download if exists
            
        Returns:
            True if successful
        """
        key = f'questions_{split}'
        if key not in self.VQA_URLS:
            logger.error(f"Invalid split: {split}")
            return False
        
        dest = self.download_dir / f"v2_Questions_{split.capitalize()}_mscoco.zip"
        
        if dest.exists() and not force:
            logger.info(f"VQA questions ({split}) already downloaded")
            return True
        
        logger.info(f"Downloading VQA v2.0 questions ({split})...")
        success = download_with_resume(self.VQA_URLS[key], dest)
        
        if success:
            extract_archive(dest, self.download_dir / "questions")
        
        return success
    
    def download_vqa_annotations(self, split: str = "train", force: bool = False) -> bool:
        """
        Download VQA v2.0 annotations.
        
        Args:
            split: 'train' or 'val'
            force: Re-download if exists
            
        Returns:
            True if successful
        """
        key = f'annotations_{split}'
        if key not in self.VQA_URLS:
            logger.error(f"Invalid split: {split}")
            return False
        
        dest = self.download_dir / f"v2_Annotations_{split.capitalize()}_mscoco.zip"
        
        if dest.exists() and not force:
            logger.info(f"VQA annotations ({split}) already downloaded")
            return True
        
        logger.info(f"Downloading VQA v2.0 annotations ({split})...")
        success = download_with_resume(self.VQA_URLS[key], dest)
        
        if success:
            extract_archive(dest, self.download_dir / "annotations")
        
        return success
    
    def clone_causal_vqa_repo(self, force: bool = False) -> bool:
        """
        Clone CausalVQA GitHub repository for IV-VQA generation scripts.
        
        Args:
            force: Re-clone if exists
            
        Returns:
            True if successful
        """
        repo_dir = self.download_dir / "CausalVQA"
        
        if repo_dir.exists():
            if force:
                shutil.rmtree(repo_dir)
            else:
                logger.info("CausalVQA repo already cloned")
                return True
        
        logger.info("Cloning CausalVQA repository...")
        try:
            result = subprocess.run(
                ['git', 'clone', self.CAUSAL_VQA_REPO, str(repo_dir)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if result.returncode == 0:
                logger.info("CausalVQA repo cloned successfully")
                return True
            else:
                logger.error(f"Git clone failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Git clone timed out")
            return False
        except FileNotFoundError:
            logger.error("Git not found. Please install git.")
            return False
    
    def download_all(
        self,
        include_images: bool = False,
        force: bool = False,
    ) -> Dict[str, bool]:
        """
        Download all IV-VQA components.
        
        Args:
            include_images: Whether to download COCO images (large)
            force: Re-download existing files
            
        Returns:
            Dict of component -> success status
        """
        results = {}
        
        results['questions_train'] = self.download_vqa_questions('train', force)
        results['questions_val'] = self.download_vqa_questions('val', force)
        results['annotations_train'] = self.download_vqa_annotations('train', force)
        results['annotations_val'] = self.download_vqa_annotations('val', force)
        results['causal_vqa_repo'] = self.clone_causal_vqa_repo(force)
        
        if include_images:
            logger.warning("COCO image download not implemented yet (very large)")
            # Would download ~18GB of images
        
        return results
    
    def verify_download(self) -> Dict[str, bool]:
        """
        Verify downloaded files.
        
        Returns:
            Dict of component -> verification status
        """
        results = {}
        
        # Check questions
        q_dir = self.download_dir / "questions"
        results['questions'] = q_dir.exists() and any(q_dir.glob("*.json"))
        
        # Check annotations
        a_dir = self.download_dir / "annotations"
        results['annotations'] = a_dir.exists() and any(a_dir.glob("*.json"))
        
        # Check CausalVQA repo
        repo_dir = self.download_dir / "CausalVQA"
        results['causal_vqa_repo'] = (
            repo_dir.exists() and 
            (repo_dir / "README.md").exists()
        )
        
        return results
    
    def get_questions_path(self, split: str = "train") -> Path:
        """Get path to questions JSON file."""
        return self.download_dir / "questions" / f"v2_OpenEnded_mscoco_{split}2014_questions.json"
    
    def get_annotations_path(self, split: str = "train") -> Path:
        """Get path to annotations JSON file."""
        return self.download_dir / "annotations" / f"v2_mscoco_{split}2014_annotations.json"
