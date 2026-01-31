"""
GQA Dataset class for PyTorch integration.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import random

from ..base import BaseDataset
from ..config import DataConfig
from .downloader import GQADownloader
from .parser import GQAParser
from .sampler import GQASampler

logger = logging.getLogger(__name__)


class GQADataset(BaseDataset):
    """
    Complete GQA dataset handler for CAI-System.
    
    Integrates downloading, parsing, sampling, and data loading.
    
    Usage:
        config = DataConfig()
        dataset = GQADataset(config)
        
        # Download and prepare
        dataset.download()
        dataset.prepare(sample_size=150000)
        
        # Use data
        for sample in dataset:
            process(sample)
    """
    
    def __init__(
        self,
        config: DataConfig,
        split: str = "train_balanced",
        transform: Optional[Callable] = None,
    ):
        """
        Initialize GQA dataset.
        
        Args:
            config: DataConfig instance
            split: Dataset split to use
            transform: Optional transform function for samples
        """
        super().__init__(config)
        self.split = split
        self._transform = transform
        
        self.downloader = GQADownloader(config)
        self.parser = GQAParser(config)
        self.sampler = GQASampler(config)
        
        self._prepared = False
    
    @property
    def name(self) -> str:
        """Dataset name identifier."""
        return f"gqa_{self.split}"
    
    @property
    def raw_dir(self) -> Path:
        """Directory containing raw dataset files."""
        return self.config.gqa_raw_dir
    
    def download(self, force: bool = False, include_images: bool = False) -> bool:
        """
        Download GQA dataset files.
        
        Args:
            force: Re-download even if files exist
            include_images: Whether to download images (~20GB)
            
        Returns:
            True if all downloads successful
        """
        logger.info("Downloading GQA dataset...")
        results = self.downloader.download_all(
            include_images=include_images,
            force=force,
        )
        
        success = all(results.values())
        if success:
            logger.info("GQA download complete")
        else:
            failed = [k for k, v in results.items() if not v]
            logger.error(f"GQA download failed for: {failed}")
        
        return success
    
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse raw GQA data files.
        
        Returns:
            List of parsed question dictionaries
        """
        # Parse scene graphs first
        sg_split = self.split.replace("_balanced", "")
        self.parser.parse_scene_graphs(split=sg_split)
        
        # Parse questions
        questions = self.parser.parse_questions(split=self.split)
        
        # Merge with scene graphs
        questions = self.parser.merge_with_scene_graphs(questions)
        
        # Filter by object count for quality
        questions = self.parser.filter_by_object_count(
            questions,
            min_objects=self.config.gqa.min_objects,
            max_objects=self.config.gqa.max_objects,
        )
        
        return questions
    
    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        strategy: str = "combined",
    ) -> List[Dict[str, Any]]:
        """
        Sample n items from parsed data.
        
        Args:
            n: Number of samples
            seed: Random seed
            strategy: Sampling strategy ('random', 'stratified', 'balanced', 'diverse', 'combined')
            
        Returns:
            List of sampled items
        """
        if not self._is_loaded:
            self.load()
        
        if seed is not None:
            self.sampler._rng = random.Random(seed)
        
        if strategy == "random":
            return self.sampler.sample_random(self._data, n)
        elif strategy == "stratified":
            return self.sampler.sample_stratified_by_type(self._data, n)
        elif strategy == "balanced":
            return self.sampler.sample_balanced_answers(self._data, n)
        elif strategy == "diverse":
            return self.sampler.sample_diverse_scenes(self._data, n)
        else:  # combined
            return self.sampler.sample_combined(self._data, n)
    
    def prepare(
        self,
        sample_size: Optional[int] = None,
        strategy: str = "combined",
        save: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Full preparation pipeline: parse, sample, and optionally save.
        
        Args:
            sample_size: Number of samples (default from config)
            strategy: Sampling strategy
            save: Whether to save processed data
            
        Returns:
            Prepared and sampled data
        """
        if sample_size is None:
            sample_size = self.config.gqa.sample_size
        
        # Load and parse
        self.load()
        
        # Sample
        sampled = self.sample(sample_size, strategy=strategy)
        
        # Apply transforms
        if self._transform:
            sampled = [self.transform(item) for item in sampled]
        
        # Save if requested
        if save:
            self.save_processed(sampled)
        
        self._data = sampled
        self._prepared = True
        
        logger.info(f"GQA preparation complete: {len(sampled)} samples")
        return sampled
    
    def transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transform to a single item.
        
        Args:
            item: Data item
            
        Returns:
            Transformed item
        """
        if self._transform:
            return self._transform(item)
        return item
    
    def get_image_path(self, image_id: str) -> Path:
        """Get path to image file for an image ID."""
        return self.downloader.get_image_path(image_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for current data."""
        if not self._is_loaded:
            return {}
        return self.parser.get_stats(self._data)
    
    def validate(self) -> bool:
        """Validate loaded data."""
        if not super().validate():
            return False
        
        # Additional GQA-specific validation
        required_fields = {'question_id', 'image_id', 'question', 'answer'}
        for item in self._data[:100]:  # Check first 100
            if not required_fields.issubset(item.keys()):
                logger.error(f"Missing required fields in GQA data")
                return False
        
        return True
