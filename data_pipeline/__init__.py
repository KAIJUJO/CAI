"""
CAI-System Data Pipeline
=========================

工程级模块化数据清洗管道，用于 CAI-System 感知层训练。

主要组件:
- GQA Dataset Handler: 处理 GQA 视觉推理数据集
- IV-VQA Dataset Handler: 处理 IV-VQA 反事实数据集，构建正负样本对

Usage:
    from data_pipeline import GQADataset, IVVQADataset, DataConfig
    from data_pipeline.ivvqa import PairBuilder, CausalPairBuilder
    
    # Initialize
    config = DataConfig()
    
    # GQA Dataset
    gqa = GQADataset(config)
    gqa.download()
    gqa.prepare(sample_size=150000)
    
    # IV-VQA with positive/negative pairs
    ivvqa = IVVQADataset(config)
    ivvqa.download()
    pairs = ivvqa.prepare_pairs(sample_size=60000)
"""

from .config import DataConfig, GQAConfig, IVVQAConfig, get_default_config
from .base import BaseDataset, BasePairBuilder
from .gqa import GQADataset, GQADownloader, GQAParser, GQASampler
from .ivvqa import IVVQADataset, IVVQADownloader, IVVQAParser, PairBuilder, CausalPairBuilder

__version__ = "0.1.0"

__all__ = [
    # Config
    "DataConfig",
    "GQAConfig", 
    "IVVQAConfig",
    "get_default_config",
    # Base
    "BaseDataset",
    "BasePairBuilder",
    # GQA
    "GQADataset",
    "GQADownloader",
    "GQAParser",
    "GQASampler",
    # IV-VQA
    "IVVQADataset",
    "IVVQADownloader",
    "IVVQAParser",
    "PairBuilder",
    "CausalPairBuilder",
]
