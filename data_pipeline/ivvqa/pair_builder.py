"""
Positive/Negative Pair Builder for VJP Training.

This module implements the core logic for constructing sample pairs
for training the VJP-S-IB (Spatial Information Bottleneck) encoder.

Paper Reference:
- Causal VQA: aims to disentangle foreground (causal) features from background.
- VJP training requires:
  - Positive samples: Questions whose answers depend on foreground object attributes
  - Negative samples: Questions whose answers can be inferred from background/bias

The VJP training objective:
  max |VJP(positive) - VJP(negative)| on foreground regions
  min |VJP(positive) - VJP(negative)| on background regions
"""

import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
import re

from ..base import BasePairBuilder
from ..config import DataConfig

logger = logging.getLogger(__name__)


# Question type categorization for causal analysis
CAUSAL_QUESTION_PATTERNS = {
    # Object attribute questions - answers depend on specific visual features
    'color': [r'\bwhat color\b', r'\bwhat is the color\b', r'\bcolor of\b'],
    'shape': [r'\bwhat shape\b', r'\bshape of\b'],
    'material': [r'\bwhat material\b', r'\bmade of\b', r'\bwhat is .* made of\b'],
    'size': [r'\bhow big\b', r'\bhow large\b', r'\bwhat size\b', r'\bhow small\b'],
    'type': [r'\bwhat type\b', r'\bwhat kind\b', r'\bkind of\b'],
    'count_specific': [r'\bhow many .* are\b', r'\bhow many .* on\b'],
    'object_identity': [r'\bwhat is the\b', r'\bwhat are the\b', r'\bwhat animal\b'],
}

BACKGROUND_QUESTION_PATTERNS = {
    # Background/scene questions - answers can be inferred from context
    'existence': [r'\bis there\b', r'\bare there\b', r'\bdo you see\b'],
    'location_general': [r'\bwhere is this\b', r'\bwhat room\b', r'\bwhat place\b'],
    'weather': [r'\bwhat weather\b', r'\bis it sunny\b', r'\bis it raining\b'],
    'time': [r'\bwhat time\b', r'\bday or night\b', r'\btime of day\b'],
    'season': [r'\bwhat season\b'],
    'indoor_outdoor': [r'\bindoors\b', r'\boutdoors\b', r'\binside\b', r'\boutside\b'],
    'yes_no_scene': [r'\bis this a\b', r'\bis this an\b', r'\bare these\b'],
}


class PairBuilder(BasePairBuilder):
    """
    Base class for building positive/negative pairs.
    
    Pairs are constructed for VJP training where:
    - Positive: Question about causal object features (color, shape, etc.)
    - Negative: Question about background or scene context
    
    Output Format:
    {
        "pair_id": "unique_id",
        "image_id": "coco_image_id",
        "positive": {
            "question_id": "...",
            "question": "What color is the dog?",
            "answer": "brown",
            "causal_type": "color",
            "target_objects": ["dog"]
        },
        "negative": {
            "question_id": "...",
            "question": "Is this indoors?",
            "answer": "yes",
            "background_type": "indoor_outdoor"
        }
    }
    """
    
    def __init__(self, config: DataConfig, seed: Optional[int] = None):
        """
        Initialize pair builder.
        
        Args:
            config: DataConfig instance
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed if seed is not None else config.random_seed
        self._rng = random.Random(self.seed)
        
        # Compiled regex patterns
        self._causal_patterns = self._compile_patterns(CAUSAL_QUESTION_PATTERNS)
        self._background_patterns = self._compile_patterns(BACKGROUND_QUESTION_PATTERNS)
    
    def _compile_patterns(
        self, 
        pattern_dict: Dict[str, List[str]]
    ) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for efficiency."""
        compiled = {}
        for category, patterns in pattern_dict.items():
            compiled[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        return compiled
    
    def classify_question(
        self, 
        question: str
    ) -> Tuple[str, Optional[str]]:
        """
        Classify a question as causal, background, or unknown.
        
        Args:
            question: Question text
            
        Returns:
            Tuple of (classification, subtype)
            classification: 'causal', 'background', or 'unknown'
            subtype: Specific category within the classification
        """
        question_lower = question.lower()
        
        # Check causal patterns
        for category, patterns in self._causal_patterns.items():
            for pattern in patterns:
                if pattern.search(question_lower):
                    return ('causal', category)
        
        # Check background patterns
        for category, patterns in self._background_patterns.items():
            for pattern in patterns:
                if pattern.search(question_lower):
                    return ('background', category)
        
        return ('unknown', None)
    
    def extract_target_objects(self, question: str) -> List[str]:
        """
        Extract target object mentions from a question.
        
        Args:
            question: Question text
            
        Returns:
            List of object names mentioned
        """
        # Common pattern: "What color is the [OBJECT]?"
        patterns = [
            r'what (?:color|shape|size|type|kind) (?:is|are) (?:the |a |an )?(\w+)',
            r'how many (\w+)',
            r'is (?:the |a |an )?(\w+) (\w+)',  # "Is the dog brown?"
            r'(?:color|shape|size) of (?:the |a |an )?(\w+)',
        ]
        
        objects = []
        for pattern in patterns:
            matches = re.findall(pattern, question.lower())
            for match in matches:
                if isinstance(match, tuple):
                    objects.extend(match)
                else:
                    objects.append(match)
        
        # Filter common non-object words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'this', 'that', 'these', 'those'}
        objects = [obj for obj in objects if obj not in stopwords and len(obj) > 2]
        
        return objects
    
    def build_pairs(
        self,
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Build positive/negative pairs from QA data.
        
        Args:
            data: List of QA items with question, answer, image_id
            
        Returns:
            List of pair dictionaries
        """
        # Group questions by image
        image_questions: Dict[Any, List[Dict]] = defaultdict(list)
        for item in data:
            image_id = item.get('image_id')
            classification, subtype = self.classify_question(item.get('question', ''))
            
            enriched_item = {
                **item,
                'classification': classification,
                'subtype': subtype,
                'target_objects': self.extract_target_objects(item.get('question', '')),
            }
            image_questions[image_id].append(enriched_item)
        
        # Build pairs for each image
        pairs = []
        for image_id, questions in image_questions.items():
            image_pairs = self._build_pairs_for_image(image_id, questions)
            pairs.extend(image_pairs)
        
        logger.info(f"Built {len(pairs)} pairs from {len(data)} questions")
        return pairs
    
    def _build_pairs_for_image(
        self,
        image_id: Any,
        questions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Build pairs for a single image.
        
        Args:
            image_id: Image identifier
            questions: Questions for this image
            
        Returns:
            List of pairs for this image
        """
        # Separate by classification
        causal_qs = [q for q in questions if q['classification'] == 'causal']
        background_qs = [q for q in questions if q['classification'] == 'background']
        
        if not causal_qs or not background_qs:
            return []
        
        pairs = []
        
        # Create pairs: each causal question paired with a random background question
        for causal_q in causal_qs:
            background_q = self._rng.choice(background_qs)
            
            pair = {
                'pair_id': f"{image_id}_{causal_q.get('question_id', '')}_{background_q.get('question_id', '')}",
                'image_id': image_id,
                'positive': {
                    'question_id': causal_q.get('question_id'),
                    'question': causal_q.get('question'),
                    'answer': causal_q.get('answer'),
                    'causal_type': causal_q.get('subtype'),
                    'target_objects': causal_q.get('target_objects', []),
                },
                'negative': {
                    'question_id': background_q.get('question_id'),
                    'question': background_q.get('question'),
                    'answer': background_q.get('answer'),
                    'background_type': background_q.get('subtype'),
                },
            }
            pairs.append(pair)
        
        return pairs
    
    def validate_pair(self, pair: Dict[str, Any]) -> bool:
        """
        Validate a single pair.
        
        Args:
            pair: Pair dictionary
            
        Returns:
            True if pair is valid
        """
        required_keys = {'pair_id', 'image_id', 'positive', 'negative'}
        if not required_keys.issubset(pair.keys()):
            return False
        
        # Check positive sample
        positive = pair.get('positive', {})
        if not positive.get('question') or not positive.get('answer'):
            return False
        
        # Check negative sample
        negative = pair.get('negative', {})
        if not negative.get('question') or not negative.get('answer'):
            return False
        
        return True
    
    def get_stats(self, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics for built pairs.
        
        Args:
            pairs: List of pairs
            
        Returns:
            Statistics dictionary
        """
        causal_types = defaultdict(int)
        background_types = defaultdict(int)
        images = set()
        
        for pair in pairs:
            images.add(pair.get('image_id'))
            
            pos = pair.get('positive', {})
            causal_types[pos.get('causal_type', 'unknown')] += 1
            
            neg = pair.get('negative', {})
            background_types[neg.get('background_type', 'unknown')] += 1
        
        return {
            'total_pairs': len(pairs),
            'unique_images': len(images),
            'causal_types': dict(causal_types),
            'background_types': dict(background_types),
        }


class CausalPairBuilder(PairBuilder):
    """
    Enhanced pair builder with causal strength scoring.
    
    Extends base PairBuilder with:
    - Causal relevance scoring for positive samples
    - Background correlation scoring for negative samples
    - Quality filtering based on scores
    """
    
    def __init__(
        self, 
        config: DataConfig,
        min_causal_score: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Initialize causal pair builder.
        
        Args:
            config: DataConfig instance
            min_causal_score: Minimum score for causal questions
            seed: Random seed
        """
        super().__init__(config, seed)
        self.min_causal_score = min_causal_score
        
        # Strong causal indicators
        self._strong_causal_words = {
            'color', 'shape', 'material', 'texture', 'pattern',
            'wearing', 'holding', 'eating', 'reading', 'riding',
        }
        
        # Strong background indicators
        self._strong_background_words = {
            'background', 'scene', 'room', 'place', 'weather',
            'indoor', 'outdoor', 'inside', 'outside', 'time',
        }
    
    def score_causal_relevance(self, question: str, answer: str) -> float:
        """
        Score how strongly a question relates to causal foreground features.
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        question_lower = question.lower()
        
        # Check for strong causal indicators
        for word in self._strong_causal_words:
            if word in question_lower:
                score += 0.3
        
        # Object attribute questions score higher
        classification, subtype = self.classify_question(question)
        if classification == 'causal':
            score += 0.4
            if subtype in ('color', 'shape', 'material'):
                score += 0.2
        
        # Specific object mentions increase score
        objects = self.extract_target_objects(question)
        if objects:
            score += 0.1 * min(len(objects), 3)
        
        return min(score, 1.0)
    
    def score_background_relevance(self, question: str, answer: str) -> float:
        """
        Score how strongly a question relates to background/scene.
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        question_lower = question.lower()
        
        # Check for strong background indicators
        for word in self._strong_background_words:
            if word in question_lower:
                score += 0.3
        
        # Scene context questions score higher
        classification, subtype = self.classify_question(question)
        if classification == 'background':
            score += 0.4
            if subtype in ('weather', 'indoor_outdoor', 'time'):
                score += 0.2
        
        # Yes/no questions about existence boost score
        if answer.lower() in ('yes', 'no') and 'is there' in question_lower:
            score += 0.2
        
        return min(score, 1.0)
    
    def build_scored_pairs(
        self,
        data: List[Dict[str, Any]],
        min_positive_score: float = 0.5,
        min_negative_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Build pairs with quality scoring and filtering.
        
        Args:
            data: List of QA items
            min_positive_score: Minimum score for causal (positive) samples
            min_negative_score: Minimum score for background (negative) samples
            
        Returns:
            List of high-quality pairs
        """
        # Score all questions
        scored_items = []
        for item in data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            causal_score = self.score_causal_relevance(question, answer)
            background_score = self.score_background_relevance(question, answer)
            
            scored_items.append({
                **item,
                'causal_score': causal_score,
                'background_score': background_score,
            })
        
        # Group by image
        image_questions: Dict[Any, Dict[str, List]] = defaultdict(
            lambda: {'causal': [], 'background': []}
        )
        
        for item in scored_items:
            image_id = item.get('image_id')
            
            if item['causal_score'] >= min_positive_score:
                image_questions[image_id]['causal'].append(item)
            if item['background_score'] >= min_negative_score:
                image_questions[image_id]['background'].append(item)
        
        # Build pairs
        pairs = []
        for image_id, qs in image_questions.items():
            causal_qs = qs['causal']
            background_qs = qs['background']
            
            if not causal_qs or not background_qs:
                continue
            
            for causal_q in causal_qs:
                background_q = self._rng.choice(background_qs)
                
                pair = {
                    'pair_id': f"{image_id}_{causal_q.get('question_id', 'c')}_{background_q.get('question_id', 'b')}",
                    'image_id': image_id,
                    'positive': {
                        'question_id': causal_q.get('question_id'),
                        'question': causal_q.get('question'),
                        'answer': causal_q.get('answer'),
                        'causal_score': causal_q.get('causal_score'),
                        'target_objects': self.extract_target_objects(causal_q.get('question', '')),
                    },
                    'negative': {
                        'question_id': background_q.get('question_id'),
                        'question': background_q.get('question'),
                        'answer': background_q.get('answer'),
                        'background_score': background_q.get('background_score'),
                    },
                    'pair_quality': (causal_q['causal_score'] + background_q['background_score']) / 2,
                }
                pairs.append(pair)
        
        # Sort by quality
        pairs.sort(key=lambda x: x.get('pair_quality', 0), reverse=True)
        
        logger.info(f"Built {len(pairs)} scored pairs from {len(data)} questions")
        return pairs
