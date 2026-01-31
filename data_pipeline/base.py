"""
Base classes for CAI-System data pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
import json
import logging

from .config import DataConfig


logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """
    Abstract base class for all dataset handlers in CAI-System.
    
    Provides unified interface for:
    - Downloading raw data
    - Parsing data formats
    - Sampling subsets
    - Transforming to target format
    
    Subclasses must implement all abstract methods.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize dataset handler.
        
        Args:
            config: DataConfig instance with all configuration parameters
        """
        self.config = config
        self._data: List[Dict[str, Any]] = []
        self._is_loaded = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name identifier."""
        pass
    
    @property
    @abstractmethod
    def raw_dir(self) -> Path:
        """Directory containing raw dataset files."""
        pass
    
    @property
    def processed_path(self) -> Path:
        """Path to processed dataset file."""
        return self.config.processed_dir / f"{self.name}_processed.jsonl"
    
    @abstractmethod
    def download(self, force: bool = False) -> bool:
        """
        Download raw dataset files.
        
        Args:
            force: If True, re-download even if files exist
            
        Returns:
            True if download successful, False otherwise
        """
        pass
    
    @abstractmethod
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse raw data files into structured format.
        
        Returns:
            List of parsed data samples as dictionaries
        """
        pass
    
    @abstractmethod
    def sample(self, n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Sample n items from parsed data.
        
        Args:
            n: Number of samples to select
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled data items
        """
        pass
    
    def transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single item to target format.
        
        Default implementation returns item unchanged.
        Override in subclasses for custom transformations.
        
        Args:
            item: Single data item dictionary
            
        Returns:
            Transformed data item
        """
        return item
    
    def load(self) -> None:
        """Load and parse the dataset into memory."""
        if self._is_loaded:
            logger.info(f"{self.name}: Data already loaded, skipping")
            return
        
        logger.info(f"{self.name}: Loading and parsing data...")
        self._data = self.parse()
        self._is_loaded = True
        logger.info(f"{self.name}: Loaded {len(self._data)} samples")
    
    def save_processed(self, data: Optional[List[Dict[str, Any]]] = None) -> Path:
        """
        Save processed data to JSONL file.
        
        Args:
            data: Data to save. If None, saves self._data
            
        Returns:
            Path to saved file
        """
        data_to_save = data if data is not None else self._data
        
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.processed_path, 'w', encoding='utf-8') as f:
            for item in data_to_save:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"{self.name}: Saved {len(data_to_save)} samples to {self.processed_path}")
        return self.processed_path
    
    def load_processed(self) -> List[Dict[str, Any]]:
        """
        Load previously processed data from JSONL file.
        
        Returns:
            List of data items
        """
        if not self.processed_path.exists():
            raise FileNotFoundError(f"Processed file not found: {self.processed_path}")
        
        data = []
        with open(self.processed_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        self._data = data
        self._is_loaded = True
        return data
    
    def __len__(self) -> int:
        """Return number of loaded samples."""
        return len(self._data)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over loaded samples."""
        return iter(self._data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index."""
        return self._data[idx]
    
    def validate(self) -> bool:
        """
        Validate loaded data integrity.
        
        Returns:
            True if validation passes, False otherwise
        """
        if not self._is_loaded:
            logger.warning(f"{self.name}: Data not loaded, cannot validate")
            return False
        
        if len(self._data) == 0:
            logger.error(f"{self.name}: No data loaded")
            return False
        
        return True


class BasePairBuilder(ABC):
    """
    Abstract base class for building sample pairs (e.g., positive/negative pairs).
    
    Used for constructing training pairs for contrastive learning or VJP training.
    """
    
    @abstractmethod
    def build_pairs(
        self, 
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build pairs from input data.
        
        Args:
            data: List of data samples
            
        Returns:
            List of pair dictionaries
        """
        pass
    
    @abstractmethod
    def validate_pair(self, pair: Dict[str, Any]) -> bool:
        """
        Validate a single pair.
        
        Args:
            pair: Pair dictionary to validate
            
        Returns:
            True if pair is valid
        """
        pass
