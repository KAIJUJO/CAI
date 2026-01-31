"""
GQA Dataset parser for questions and scene graphs.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
from collections import defaultdict

from ..config import DataConfig
from ..utils.io import load_json, iter_jsonl

logger = logging.getLogger(__name__)


class GQAParser:
    """
    Parser for GQA dataset files.
    
    Handles:
    - Question files (JSON format)
    - Scene graph files (JSON format)
    - Merging questions with scene graph information
    
    GQA Question Structure:
    {
        "question_id": {
            "imageId": "2375429",
            "question": "Is the horse to the left of the fence?",
            "answer": "yes",
            "fullAnswer": "Yes, the horse is to the left of the fence.",
            "isBalanced": true,
            "groups": {"global": "relation", "local": "2obj-relS"},
            "semantic": [...],
            "semanticStr": "select: horse -> verify rel: left (fence)"
        }
    }
    
    GQA Scene Graph Structure:
    {
        "image_id": {
            "width": 640,
            "height": 480,
            "objects": {
                "object_id": {
                    "name": "horse",
                    "x": 100, "y": 50, "w": 200, "h": 300,
                    "attributes": ["brown", "large"],
                    "relations": [{"name": "to the left of", "object": "other_id"}]
                }
            }
        }
    }
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize GQA parser.
        
        Args:
            config: DataConfig instance
        """
        self.config = config
        self.raw_dir = config.gqa_raw_dir
        self._scene_graphs: Dict[str, Dict] = {}
        self._questions: Dict[str, Dict] = {}
    
    def parse_questions(
        self,
        split: str = "train_balanced",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse GQA questions file.
        
        Args:
            split: Dataset split name
            limit: Maximum number of questions to parse
            
        Returns:
            List of question dictionaries with standardized format
        """
        questions_path = self.raw_dir / "questions" / f"{split}_questions.json"
        
        if not questions_path.exists():
            logger.error(f"Questions file not found: {questions_path}")
            return []
        
        logger.info(f"Parsing questions from {questions_path}")
        raw_questions = load_json(questions_path)
        
        parsed = []
        for qid, q_data in raw_questions.items():
            if limit and len(parsed) >= limit:
                break
            
            parsed_q = {
                'question_id': qid,
                'image_id': q_data.get('imageId', ''),
                'question': q_data.get('question', ''),
                'answer': q_data.get('answer', ''),
                'full_answer': q_data.get('fullAnswer', ''),
                'is_balanced': q_data.get('isBalanced', False),
                'semantic_str': q_data.get('semanticStr', ''),
                'groups': q_data.get('groups', {}),
                'semantic': q_data.get('semantic', []),
            }
            parsed.append(parsed_q)
        
        logger.info(f"Parsed {len(parsed)} questions")
        return parsed
    
    def parse_scene_graphs(
        self,
        split: str = "train",
        limit: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parse GQA scene graphs file.
        
        Args:
            split: Dataset split name
            limit: Maximum number of scene graphs to parse
            
        Returns:
            Dict mapping image_id to scene graph data
        """
        sg_path = self.raw_dir / "sceneGraphs" / f"{split}_sceneGraphs.json"
        
        if not sg_path.exists():
            logger.error(f"Scene graphs file not found: {sg_path}")
            return {}
        
        logger.info(f"Parsing scene graphs from {sg_path}")
        raw_sg = load_json(sg_path)
        
        parsed = {}
        count = 0
        for image_id, sg_data in raw_sg.items():
            if limit and count >= limit:
                break
            
            # Extract and normalize objects
            objects = {}
            for obj_id, obj_data in sg_data.get('objects', {}).items():
                objects[obj_id] = {
                    'id': obj_id,
                    'name': obj_data.get('name', ''),
                    'bbox': {
                        'x': obj_data.get('x', 0),
                        'y': obj_data.get('y', 0),
                        'w': obj_data.get('w', 0),
                        'h': obj_data.get('h', 0),
                    },
                    'attributes': obj_data.get('attributes', []),
                    'relations': obj_data.get('relations', []),
                }
            
            parsed[image_id] = {
                'image_id': image_id,
                'width': sg_data.get('width', 0),
                'height': sg_data.get('height', 0),
                'objects': objects,
                'num_objects': len(objects),
            }
            count += 1
        
        self._scene_graphs = parsed
        logger.info(f"Parsed {len(parsed)} scene graphs")
        return parsed
    
    def merge_with_scene_graphs(
        self,
        questions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge question data with corresponding scene graph information.
        
        Args:
            questions: List of parsed questions
            
        Returns:
            Questions enriched with scene graph data
        """
        if not self._scene_graphs:
            logger.warning("Scene graphs not loaded, cannot merge")
            return questions
        
        merged = []
        missing_sg = 0
        
        for q in questions:
            image_id = q.get('image_id', '')
            sg = self._scene_graphs.get(image_id)
            
            if sg:
                q_merged = {
                    **q,
                    'scene_graph': sg,
                    'num_objects': sg.get('num_objects', 0),
                }
                merged.append(q_merged)
            else:
                missing_sg += 1
                # Include question without scene graph
                q_merged = {**q, 'scene_graph': None, 'num_objects': 0}
                merged.append(q_merged)
        
        if missing_sg > 0:
            logger.warning(f"{missing_sg} questions have no matching scene graph")
        
        return merged
    
    def extract_objects_from_semantic(
        self,
        semantic: List[Dict],
    ) -> List[str]:
        """
        Extract object names from semantic parse.
        
        Args:
            semantic: Semantic parse list from GQA
            
        Returns:
            List of object names mentioned
        """
        objects = []
        for step in semantic:
            if 'argument' in step:
                arg = step['argument']
                if isinstance(arg, str):
                    objects.append(arg)
                elif isinstance(arg, dict) and 'name' in arg:
                    objects.append(arg['name'])
        return list(set(objects))
    
    def get_question_type(self, question: Dict[str, Any]) -> str:
        """
        Determine question type from groups field.
        
        Args:
            question: Parsed question dictionary
            
        Returns:
            Question type string
        """
        groups = question.get('groups', {})
        return groups.get('global', 'unknown')
    
    def filter_by_object_count(
        self,
        questions: List[Dict[str, Any]],
        min_objects: int = 2,
        max_objects: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Filter questions by number of objects in scene.
        
        Ensures diversity and manageable complexity.
        
        Args:
            questions: Questions to filter
            min_objects: Minimum object count
            max_objects: Maximum object count
            
        Returns:
            Filtered questions
        """
        filtered = [
            q for q in questions
            if min_objects <= q.get('num_objects', 0) <= max_objects
        ]
        
        logger.info(
            f"Filtered by object count [{min_objects}, {max_objects}]: "
            f"{len(questions)} -> {len(filtered)}"
        )
        return filtered
    
    def get_stats(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics for parsed questions.
        
        Args:
            questions: List of questions
            
        Returns:
            Statistics dictionary
        """
        type_counts = defaultdict(int)
        answer_counts = defaultdict(int)
        object_counts = []
        
        for q in questions:
            type_counts[self.get_question_type(q)] += 1
            answer_counts[q.get('answer', '')] += 1
            object_counts.append(q.get('num_objects', 0))
        
        return {
            'total_questions': len(questions),
            'question_types': dict(type_counts),
            'unique_answers': len(answer_counts),
            'top_answers': dict(sorted(
                answer_counts.items(), 
                key=lambda x: -x[1]
            )[:20]),
            'avg_objects': sum(object_counts) / len(object_counts) if object_counts else 0,
            'min_objects': min(object_counts) if object_counts else 0,
            'max_objects': max(object_counts) if object_counts else 0,
        }
