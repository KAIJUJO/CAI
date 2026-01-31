"""
GQA Dataset sampler with stratified sampling for diversity.
"""

import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict

from ..config import DataConfig

logger = logging.getLogger(__name__)


class GQASampler:
    """
    Intelligent sampler for GQA dataset.
    
    Supports:
    - Stratified sampling by question type
    - Balanced sampling by answer distribution
    - Scene graph diversity sampling
    - Reproducible sampling with seeds
    """
    
    def __init__(self, config: DataConfig, seed: Optional[int] = None):
        """
        Initialize sampler.
        
        Args:
            config: DataConfig instance
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed if seed is not None else config.random_seed
        self._rng = random.Random(self.seed)
    
    def sample_random(
        self,
        data: List[Dict[str, Any]],
        n: int,
    ) -> List[Dict[str, Any]]:
        """
        Simple random sampling.
        
        Args:
            data: Data to sample from
            n: Number of samples
            
        Returns:
            List of n randomly selected items
        """
        if n >= len(data):
            logger.warning(f"Requested {n} samples but only {len(data)} available")
            return data.copy()
        
        return self._rng.sample(data, n)
    
    def sample_stratified_by_type(
        self,
        data: List[Dict[str, Any]],
        n: int,
        type_field: str = 'groups',
        type_key: str = 'global',
    ) -> List[Dict[str, Any]]:
        """
        Stratified sampling to maintain question type distribution.
        
        Args:
            data: Data to sample from
            n: Total number of samples
            type_field: Field containing type info
            type_key: Key within type_field for type value
            
        Returns:
            Stratified sample
        """
        # Group by type
        type_groups: Dict[str, List] = defaultdict(list)
        for item in data:
            type_info = item.get(type_field, {})
            if isinstance(type_info, dict):
                item_type = type_info.get(type_key, 'unknown')
            else:
                item_type = 'unknown'
            type_groups[item_type].append(item)
        
        # Calculate samples per type (proportional)
        total = len(data)
        samples_per_type = {}
        remaining = n
        
        for qtype, items in sorted(type_groups.items()):
            proportion = len(items) / total
            type_samples = int(n * proportion)
            samples_per_type[qtype] = min(type_samples, len(items))
            remaining -= samples_per_type[qtype]
        
        # Distribute remaining samples to largest groups
        sorted_types = sorted(
            type_groups.keys(),
            key=lambda t: len(type_groups[t]),
            reverse=True
        )
        for qtype in sorted_types:
            if remaining <= 0:
                break
            available = len(type_groups[qtype]) - samples_per_type[qtype]
            to_add = min(remaining, available)
            samples_per_type[qtype] += to_add
            remaining -= to_add
        
        # Sample from each type
        result = []
        for qtype, count in samples_per_type.items():
            if count > 0:
                type_sample = self._rng.sample(type_groups[qtype], count)
                result.extend(type_sample)
        
        self._rng.shuffle(result)
        
        logger.info(f"Stratified sampling: {len(result)} samples across {len(samples_per_type)} types")
        for qtype, count in sorted(samples_per_type.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"  {qtype}: {count}")
        
        return result
    
    def sample_balanced_answers(
        self,
        data: List[Dict[str, Any]],
        n: int,
        answer_field: str = 'answer',
        max_per_answer: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sample to balance answer distribution.
        
        Prevents overrepresentation of common answers like 'yes', 'no'.
        
        Args:
            data: Data to sample from
            n: Total number of samples
            answer_field: Field containing answer
            max_per_answer: Maximum samples per unique answer
            
        Returns:
            Answer-balanced sample
        """
        # Group by answer
        answer_groups: Dict[str, List] = defaultdict(list)
        for item in data:
            answer = str(item.get(answer_field, '')).lower()
            answer_groups[answer].append(item)
        
        # Calculate cap per answer
        if max_per_answer is None:
            # Default: at most 5% of total per answer
            max_per_answer = max(n // 20, 100)
        
        # Sample with cap
        result = []
        answers = list(answer_groups.keys())
        self._rng.shuffle(answers)
        
        for answer in answers:
            items = answer_groups[answer]
            cap = min(max_per_answer, len(items))
            sampled = self._rng.sample(items, cap)
            result.extend(sampled)
            
            if len(result) >= n:
                break
        
        # If not enough, sample more from underrepresented answers
        if len(result) < n:
            remaining = n - len(result)
            used_ids = {id(item) for item in result}
            unused = [item for item in data if id(item) not in used_ids]
            
            if unused:
                additional = self._rng.sample(unused, min(remaining, len(unused)))
                result.extend(additional)
        
        result = result[:n]
        self._rng.shuffle(result)
        
        logger.info(f"Answer-balanced sampling: {len(result)} samples")
        return result
    
    def sample_diverse_scenes(
        self,
        data: List[Dict[str, Any]],
        n: int,
        image_id_field: str = 'image_id',
        max_per_image: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Sample for scene diversity (limit questions per image).
        
        Args:
            data: Data to sample from
            n: Total number of samples
            image_id_field: Field containing image ID
            max_per_image: Maximum questions per image
            
        Returns:
            Scene-diverse sample
        """
        # Group by image
        image_groups: Dict[str, List] = defaultdict(list)
        for item in data:
            image_id = item.get(image_id_field, '')
            image_groups[image_id].append(item)
        
        # Sample from each image
        result = []
        images = list(image_groups.keys())
        self._rng.shuffle(images)
        
        for image_id in images:
            items = image_groups[image_id]
            cap = min(max_per_image, len(items))
            sampled = self._rng.sample(items, cap)
            result.extend(sampled)
            
            if len(result) >= n:
                break
        
        result = result[:n]
        self._rng.shuffle(result)
        
        logger.info(
            f"Scene-diverse sampling: {len(result)} samples "
            f"from {len(set(item.get(image_id_field) for item in result))} images"
        )
        return result
    
    def sample_combined(
        self,
        data: List[Dict[str, Any]],
        n: int,
        strategies: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Combined sampling using multiple strategies.
        
        Args:
            data: Data to sample from
            n: Total number of samples
            strategies: Dict of strategy name to weight
                e.g., {'stratified': 0.5, 'balanced': 0.3, 'diverse': 0.2}
            
        Returns:
            Combined sample
        """
        if strategies is None:
            strategies = {
                'stratified': 0.4,
                'balanced': 0.3,
                'diverse': 0.3,
            }
        
        result_set: Set[str] = set()
        result = []
        
        for strategy_name, weight in strategies.items():
            strategy_n = int(n * weight)
            
            if strategy_name == 'stratified':
                sampled = self.sample_stratified_by_type(data, strategy_n)
            elif strategy_name == 'balanced':
                sampled = self.sample_balanced_answers(data, strategy_n)
            elif strategy_name == 'diverse':
                sampled = self.sample_diverse_scenes(data, strategy_n)
            else:
                sampled = self.sample_random(data, strategy_n)
            
            # Add unique samples
            for item in sampled:
                item_id = item.get('question_id', str(id(item)))
                if item_id not in result_set:
                    result_set.add(item_id)
                    result.append(item)
        
        # Fill remaining with random
        if len(result) < n:
            remaining = n - len(result)
            unused = [item for item in data 
                     if item.get('question_id', str(id(item))) not in result_set]
            if unused:
                additional = self._rng.sample(unused, min(remaining, len(unused)))
                result.extend(additional)
        
        result = result[:n]
        self._rng.shuffle(result)
        
        logger.info(f"Combined sampling: {len(result)} total samples")
        return result
