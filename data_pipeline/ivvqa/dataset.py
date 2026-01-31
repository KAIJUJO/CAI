"""
IV-VQA Dataset class for PyTorch integration.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import random

from ..base import BaseDataset
from ..config import DataConfig
from .downloader import IVVQADownloader
from .parser import IVVQAParser
from .pair_builder import PairBuilder, CausalPairBuilder

logger = logging.getLogger(__name__)


class IVVQADataset(BaseDataset):
    """
    Complete IV-VQA dataset handler for CAI-System.
    
    Integrates downloading, parsing, pair building, and data loading.
    
    Main purpose: Prepare positive/negative pairs for VJP-S-IB training.
    
    Usage:
        config = DataConfig()
        dataset = IVVQADataset(config)
        
        # Download and prepare
        dataset.download()
        pairs = dataset.prepare_pairs(sample_size=60000)
        
        # Save for training
        dataset.save_processed(pairs)
    """
    
    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        use_scored_pairs: bool = True,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize IV-VQA dataset.
        
        Args:
            config: DataConfig instance
            split: Dataset split to use
            use_scored_pairs: Whether to use CausalPairBuilder for quality scoring
            transform: Optional transform function for samples
        """
        super().__init__(config)
        self.split = split
        self.use_scored_pairs = use_scored_pairs
        self._transform = transform
        
        self.downloader = IVVQADownloader(config)
        self.parser = IVVQAParser(config)
        
        if use_scored_pairs:
            self.pair_builder = CausalPairBuilder(config)
        else:
            self.pair_builder = PairBuilder(config)
        
        self._pairs: List[Dict[str, Any]] = []
        self._prepared = False
    
    @property
    def name(self) -> str:
        """Dataset name identifier."""
        return f"ivvqa_{self.split}"
    
    @property
    def raw_dir(self) -> Path:
        """Directory containing raw dataset files."""
        return self.config.ivvqa_raw_dir
    
    @property
    def pairs_path(self) -> Path:
        """Path to saved pairs file."""
        return self.config.processed_dir / f"ivvqa_{self.split}_pairs.jsonl"
    
    def download(self, force: bool = False) -> bool:
        """
        Download IV-VQA dataset files.
        
        Args:
            force: Re-download even if files exist
            
        Returns:
            True if all downloads successful
        """
        logger.info("Downloading IV-VQA dataset components...")
        results = self.downloader.download_all(force=force)
        
        success = all(results.values())
        if success:
            logger.info("IV-VQA download complete")
        else:
            failed = [k for k, v in results.items() if not v]
            logger.warning(f"IV-VQA download incomplete: {failed}")
        
        return success
    
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse raw IV-VQA data files.
        
        Returns:
            List of parsed QA pairs
        """
        # Parse questions and annotations
        self.parser.parse_questions(split=self.split)
        self.parser.parse_annotations(split=self.split)
        
        # Merge into QA pairs
        qa_pairs = self.parser.merge_questions_annotations()
        
        return qa_pairs
    
    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sample n pairs from built pairs.
        
        Args:
            n: Number of pairs to sample
            seed: Random seed
            
        Returns:
            List of sampled pairs
        """
        if not self._pairs:
            logger.warning("No pairs built yet. Call prepare_pairs first.")
            return []
        
        rng = random.Random(seed if seed is not None else self.config.random_seed)
        
        if n >= len(self._pairs):
            return self._pairs.copy()
        
        return rng.sample(self._pairs, n)
    
    def build_pairs(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build positive/negative pairs from QA data.
        
        Args:
            data: QA data to build pairs from (uses loaded data if None)
            
        Returns:
            List of pair dictionaries
        """
        if data is None:
            if not self._is_loaded:
                self.load()
            data = self._data
        
        if self.use_scored_pairs and isinstance(self.pair_builder, CausalPairBuilder):
            pairs = self.pair_builder.build_scored_pairs(
                data,
                min_positive_score=0.5,
                min_negative_score=0.4,
            )
        else:
            pairs = self.pair_builder.build_pairs(data)
        
        self._pairs = pairs
        return pairs
    
    def prepare_pairs(
        self,
        sample_size: Optional[int] = None,
        save: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Full preparation pipeline: parse, build pairs, sample, and save.
        
        Args:
            sample_size: Number of pairs (default from config)
            save: Whether to save processed pairs
            
        Returns:
            Prepared pairs
        """
        if sample_size is None:
            sample_size = self.config.ivvqa.sample_size
        
        # Load and parse
        self.load()
        
        # Build pairs
        pairs = self.build_pairs()
        
        # Sample if needed
        if sample_size < len(pairs):
            pairs = self.sample(sample_size)
            self._pairs = pairs
        
        # Validate pairs
        valid_pairs = []
        for pair in pairs:
            if self.pair_builder.validate_pair(pair):
                valid_pairs.append(pair)
        
        if len(valid_pairs) < len(pairs):
            logger.warning(f"Filtered {len(pairs) - len(valid_pairs)} invalid pairs")
        
        self._pairs = valid_pairs
        
        # Save if requested
        if save:
            self.save_pairs(valid_pairs)
        
        self._prepared = True
        logger.info(f"IV-VQA preparation complete: {len(valid_pairs)} pairs")
        
        return valid_pairs
    
    def save_pairs(self, pairs: List[Dict[str, Any]]) -> Path:
        """
        Save pairs to JSONL file.
        
        Args:
            pairs: Pairs to save
            
        Returns:
            Path to saved file
        """
        import json
        
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.pairs_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                json.dump(pair, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(pairs)} pairs to {self.pairs_path}")
        return self.pairs_path
    
    def load_pairs(self) -> List[Dict[str, Any]]:
        """
        Load previously saved pairs.
        
        Returns:
            List of pairs
        """
        import json
        
        if not self.pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {self.pairs_path}")
        
        pairs = []
        with open(self.pairs_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        
        self._pairs = pairs
        return pairs
    
    def transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transform to a single item."""
        if self._transform:
            return self._transform(item)
        return item
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for current pairs."""
        if not self._pairs:
            return {'pairs': 0}
        return self.pair_builder.get_stats(self._pairs)
    
    def validate(self) -> bool:
        """Validate loaded pairs."""
        if not self._pairs:
            logger.warning("No pairs loaded")
            return False
        
        # Check sample of pairs
        sample_size = min(100, len(self._pairs))
        valid_count = sum(
            1 for pair in self._pairs[:sample_size]
            if self.pair_builder.validate_pair(pair)
        )
        
        if valid_count < sample_size * 0.9:
            logger.error(f"Too many invalid pairs: {sample_size - valid_count}/{sample_size}")
            return False
        
        return True
    
    def __len__(self) -> int:
        """Return number of pairs."""
        return len(self._pairs)
    
    def __iter__(self):
        """Iterate over pairs."""
        return iter(self._pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get pair by index."""
        return self._pairs[idx]
