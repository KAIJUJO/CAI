"""
IV-VQA Dataset module for CAI-System.

This module handles the Introspective VQA (IV-VQA) dataset,
which is part of the Causal VQA project for building
positive/negative pairs for VJP training.
"""

from .downloader import IVVQADownloader
from .parser import IVVQAParser
from .pair_builder import PairBuilder, CausalPairBuilder
from .dataset import IVVQADataset

__all__ = [
    "IVVQADownloader",
    "IVVQAParser",
    "PairBuilder",
    "CausalPairBuilder",
    "IVVQADataset",
]
