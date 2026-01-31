"""
Configuration management for CAI-System data pipeline.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class GQAConfig:
    """GQA dataset configuration."""
    # Download URLs
    questions_url: str = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
    images_url: str = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
    scene_graphs_url: str = "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip"
    
    # Sampling parameters
    sample_size: int = 150000
    balanced_only: bool = True  # Use balanced split for higher quality
    
    # Scene graph filtering
    min_objects: int = 2  # Minimum objects in scene for diversity
    max_objects: int = 50  # Maximum to avoid overly complex scenes


@dataclass
class IVVQAConfig:
    """IV-VQA dataset configuration."""
    # Download source (GitHub + Google Drive)
    github_repo: str = "AgarwalVedika/CausalVQA"
    data_drive_id: str = ""  # Google Drive ID if available
    
    # Sampling parameters  
    sample_size: int = 60000
    
    # Pair building parameters
    require_counterfactual: bool = True  # Only samples with counterfactual pairs
    min_pair_similarity: float = 0.3  # Minimum image similarity for valid pairs


@dataclass
class DataConfig:
    """
    Main configuration for CAI-System data pipeline.
    
    Attributes:
        project_root: Root directory of the CAI project
        data_dir: Directory for all data (raw and processed)
        gqa: GQA dataset specific configuration
        ivvqa: IV-VQA dataset specific configuration
        random_seed: Random seed for reproducibility
        num_workers: Number of parallel workers for data loading
    """
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Optional[Path] = None
    gqa: GQAConfig = field(default_factory=GQAConfig)
    ivvqa: IVVQAConfig = field(default_factory=IVVQAConfig)
    random_seed: int = 42
    num_workers: int = 4
    
    def __post_init__(self):
        """Initialize derived paths after dataclass initialization."""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        
        # Convert string paths to Path objects if needed
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
    
    @property
    def raw_dir(self) -> Path:
        """Directory for raw downloaded data."""
        return self.data_dir / "raw"
    
    @property
    def processed_dir(self) -> Path:
        """Directory for processed/cleaned data."""
        return self.data_dir / "processed"
    
    @property
    def gqa_raw_dir(self) -> Path:
        """Directory for raw GQA data."""
        return self.raw_dir / "gqa"
    
    @property
    def ivvqa_raw_dir(self) -> Path:
        """Directory for raw IV-VQA data."""
        return self.raw_dir / "ivvqa"
    
    def ensure_dirs(self) -> None:
        """Create all necessary directories if they don't exist."""
        dirs = [
            self.raw_dir,
            self.processed_dir,
            self.gqa_raw_dir,
            self.ivvqa_raw_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DataConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse nested configs
        gqa_config = GQAConfig(**config_dict.pop('gqa', {}))
        ivvqa_config = IVVQAConfig(**config_dict.pop('ivvqa', {}))
        
        return cls(gqa=gqa_config, ivvqa=ivvqa_config, **config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'project_root': str(self.project_root),
            'data_dir': str(self.data_dir),
            'random_seed': self.random_seed,
            'num_workers': self.num_workers,
            'gqa': {
                'questions_url': self.gqa.questions_url,
                'images_url': self.gqa.images_url,
                'scene_graphs_url': self.gqa.scene_graphs_url,
                'sample_size': self.gqa.sample_size,
                'balanced_only': self.gqa.balanced_only,
                'min_objects': self.gqa.min_objects,
                'max_objects': self.gqa.max_objects,
            },
            'ivvqa': {
                'github_repo': self.ivvqa.github_repo,
                'data_drive_id': self.ivvqa.data_drive_id,
                'sample_size': self.ivvqa.sample_size,
                'require_counterfactual': self.ivvqa.require_counterfactual,
                'min_pair_similarity': self.ivvqa.min_pair_similarity,
            },
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def get_default_config() -> DataConfig:
    """Get default configuration instance."""
    return DataConfig()
