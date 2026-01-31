"""
IV-VQA Dataset parser.

Parses VQA v2.0 format questions and annotations,
and prepares data for counterfactual pair construction.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from ..config import DataConfig
from ..utils.io import load_json

logger = logging.getLogger(__name__)


class IVVQAParser:
    """
    Parser for IV-VQA / VQA v2.0 format data.
    
    VQA v2.0 Question Format:
    {
        "questions": [
            {
                "question_id": 123456,
                "image_id": 123456,
                "question": "What color is the dog?"
            }
        ]
    }
    
    VQA v2.0 Annotation Format:
    {
        "annotations": [
            {
                "question_id": 123456,
                "image_id": 123456,
                "question_type": "what color",
                "answer_type": "other",
                "answers": [
                    {"answer": "brown", "answer_confidence": "yes", "answer_id": 1}
                ],
                "multiple_choice_answer": "brown"
            }
        ]
    }
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize parser.
        
        Args:
            config: DataConfig instance
        """
        self.config = config
        self.raw_dir = config.ivvqa_raw_dir
        self._questions: Dict[int, Dict] = {}
        self._annotations: Dict[int, Dict] = {}
        self._image_questions: Dict[int, List[int]] = defaultdict(list)
    
    def parse_questions(self, split: str = "train") -> Dict[int, Dict[str, Any]]:
        """
        Parse VQA questions file.
        
        Args:
            split: Dataset split
            
        Returns:
            Dict mapping question_id to question data
        """
        q_path = self.raw_dir / "questions" / f"v2_OpenEnded_mscoco_{split}2014_questions.json"
        
        if not q_path.exists():
            logger.error(f"Questions file not found: {q_path}")
            return {}
        
        logger.info(f"Parsing VQA questions from {q_path}")
        data = load_json(q_path)
        
        questions = {}
        for q in data.get('questions', []):
            qid = q['question_id']
            image_id = q['image_id']
            
            questions[qid] = {
                'question_id': qid,
                'image_id': image_id,
                'question': q['question'],
            }
            
            # Index by image
            self._image_questions[image_id].append(qid)
        
        self._questions = questions
        logger.info(f"Parsed {len(questions)} questions")
        return questions
    
    def parse_annotations(self, split: str = "train") -> Dict[int, Dict[str, Any]]:
        """
        Parse VQA annotations file.
        
        Args:
            split: Dataset split
            
        Returns:
            Dict mapping question_id to annotation data
        """
        a_path = self.raw_dir / "annotations" / f"v2_mscoco_{split}2014_annotations.json"
        
        if not a_path.exists():
            logger.error(f"Annotations file not found: {a_path}")
            return {}
        
        logger.info(f"Parsing VQA annotations from {a_path}")
        data = load_json(a_path)
        
        annotations = {}
        for ann in data.get('annotations', []):
            qid = ann['question_id']
            
            # Get majority answer
            answer = ann.get('multiple_choice_answer', '')
            
            # Get all answer variants
            answer_list = [a['answer'] for a in ann.get('answers', [])]
            
            annotations[qid] = {
                'question_id': qid,
                'image_id': ann['image_id'],
                'question_type': ann.get('question_type', ''),
                'answer_type': ann.get('answer_type', ''),
                'answer': answer,
                'answer_list': answer_list,
            }
        
        self._annotations = annotations
        logger.info(f"Parsed {len(annotations)} annotations")
        return annotations
    
    def merge_questions_annotations(self) -> List[Dict[str, Any]]:
        """
        Merge questions with their annotations.
        
        Returns:
            List of merged QA pairs
        """
        if not self._questions or not self._annotations:
            logger.error("Questions or annotations not loaded")
            return []
        
        merged = []
        for qid, question in self._questions.items():
            annotation = self._annotations.get(qid, {})
            
            merged_item = {
                **question,
                'question_type': annotation.get('question_type', ''),
                'answer_type': annotation.get('answer_type', ''),
                'answer': annotation.get('answer', ''),
                'answer_list': annotation.get('answer_list', []),
            }
            merged.append(merged_item)
        
        logger.info(f"Merged {len(merged)} QA pairs")
        return merged
    
    def get_questions_for_image(self, image_id: int) -> List[Dict[str, Any]]:
        """
        Get all questions for a specific image.
        
        Args:
            image_id: COCO image ID
            
        Returns:
            List of question dictionaries
        """
        question_ids = self._image_questions.get(image_id, [])
        
        questions = []
        for qid in question_ids:
            q = self._questions.get(qid, {})
            ann = self._annotations.get(qid, {})
            if q:
                questions.append({
                    **q,
                    'answer': ann.get('answer', ''),
                    'question_type': ann.get('question_type', ''),
                })
        
        return questions
    
    def categorize_by_type(
        self,
        data: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize QA pairs by question type.
        
        Args:
            data: List of QA pairs
            
        Returns:
            Dict mapping question_type to list of items
        """
        categories = defaultdict(list)
        for item in data:
            qtype = item.get('question_type', 'unknown')
            categories[qtype].append(item)
        
        return dict(categories)
    
    def identify_attribute_questions(
        self,
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Identify questions about object attributes.
        
        These are ideal for causal/foreground questions.
        
        Args:
            data: List of QA pairs
            
        Returns:
            Filtered list of attribute questions
        """
        attribute_types = {
            'what color', 'what kind', 'what type', 'what shape',
            'what is the color', 'what material', 'what size',
        }
        
        return [
            item for item in data
            if any(t in item.get('question_type', '').lower() for t in attribute_types)
        ]
    
    def identify_background_questions(
        self,
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Identify questions about background/scene.
        
        These are ideal for negative samples.
        
        Args:
            data: List of QA pairs
            
        Returns:
            Filtered list of background/scene questions
        """
        background_types = {
            'is there', 'are there', 'where is', 'what room',
            'what is in the background', 'what weather',
            'is it', 'what time', 'what season',
        }
        
        return [
            item for item in data
            if any(t in item.get('question_type', '').lower() for t in background_types)
            or any(t in item.get('question', '').lower() for t in background_types)
        ]
    
    def get_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics for parsed data.
        
        Args:
            data: List of QA pairs
            
        Returns:
            Statistics dictionary
        """
        type_counts = defaultdict(int)
        answer_type_counts = defaultdict(int)
        images = set()
        
        for item in data:
            type_counts[item.get('question_type', 'unknown')] += 1
            answer_type_counts[item.get('answer_type', 'unknown')] += 1
            images.add(item.get('image_id'))
        
        return {
            'total_questions': len(data),
            'unique_images': len(images),
            'question_types': dict(sorted(type_counts.items(), key=lambda x: -x[1])[:20]),
            'answer_types': dict(answer_type_counts),
        }
