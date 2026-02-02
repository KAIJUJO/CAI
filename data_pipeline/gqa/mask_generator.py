"""
Mask Generator for S-IB Training.

Generates Ground Truth foreground masks (M_fg) from GQA semantic parsing
and scene graph bounding boxes.

Core Logic:
1. Extract causal object names from semantic field
2. Match objects to scene graph using hierarchical strategy
3. Render binary mask from matched bounding boxes
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
from collections import defaultdict

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from nltk.stem import WordNetLemmatizer
    import nltk
    # Download wordnet data if not present
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

logger = logging.getLogger(__name__)


# GQA common synonyms mapping (subset for demo, extend as needed)
GQA_SYNONYMS = {
    'man': ['person', 'guy', 'male', 'human'],
    'woman': ['person', 'lady', 'female', 'human'],
    'boy': ['child', 'kid', 'person'],
    'girl': ['child', 'kid', 'person'],
    'car': ['vehicle', 'automobile'],
    'truck': ['vehicle'],
    'bus': ['vehicle'],
    'couch': ['sofa'],
    'sofa': ['couch'],
    'tv': ['television', 'monitor', 'screen'],
    'cellphone': ['phone', 'mobile'],
    'laptop': ['computer', 'notebook'],
    'cup': ['mug', 'glass'],
    'plate': ['dish'],
    'dog': ['animal', 'pet'],
    'cat': ['animal', 'pet'],
    'horse': ['animal'],
    'cow': ['animal'],
    'sheep': ['animal'],
    'bird': ['animal'],
}

# Build reverse mapping for efficient lookup
SYNONYM_REVERSE = defaultdict(set)
for key, synonyms in GQA_SYNONYMS.items():
    SYNONYM_REVERSE[key].add(key)
    for syn in synonyms:
        SYNONYM_REVERSE[syn].add(key)
        SYNONYM_REVERSE[key].add(syn)


class MaskGenerator:
    """
    Generator for S-IB training Ground Truth foreground masks.
    
    Uses hierarchical matching strategy:
    1. Exact match (case-insensitive)
    2. Lemmatized match (dogs -> dog)
    3. Synonym match (man -> person)
    
    Attributes:
        use_lemmatization: Whether to use nltk lemmatization
        use_synonyms: Whether to use GQA synonym mapping
        default_output_size: Default output mask size
    """
    
    def __init__(
        self,
        use_lemmatization: bool = True,
        use_synonyms: bool = True,
        default_output_size: Tuple[int, int] = (384, 384),
    ):
        """
        Initialize mask generator.
        
        Args:
            use_lemmatization: Enable nltk lemmatization for matching
            use_synonyms: Enable GQA synonym mapping
            default_output_size: Default (H, W) for output masks
        """
        self.use_lemmatization = use_lemmatization and HAS_NLTK
        self.use_synonyms = use_synonyms
        self.default_output_size = default_output_size
        
        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None
            if use_lemmatization:
                logger.warning("nltk not available, lemmatization disabled")
    
    def normalize_word(self, word: str) -> str:
        """
        Normalize word for matching.
        
        Args:
            word: Input word
            
        Returns:
            Normalized word (lowercase, lemmatized if enabled)
        """
        word = word.lower().strip()
        
        if self.use_lemmatization and self.lemmatizer:
            # Try noun lemmatization first
            lemma = self.lemmatizer.lemmatize(word, pos='n')
            if lemma != word:
                return lemma
            # Try verb lemmatization
            lemma = self.lemmatizer.lemmatize(word, pos='v')
            return lemma
        
        return word
    
    def _parse_gqa_argument(self, arg: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse GQA semantic argument format.
        
        GQA uses format: "object_name (object_id)" e.g. "shirt (4653737)"
        
        Args:
            arg: Argument string from semantic field
            
        Returns:
            Tuple of (object_name, object_id) or (None, None) if invalid
        """
        import re
        
        if not isinstance(arg, str) or not arg:
            return None, None
        
        # Match pattern: "name (id)" or just "name"
        match = re.match(r'^(.+?)\s*\((\d+)\)$', arg.strip())
        if match:
            name = match.group(1).strip()
            obj_id = match.group(2)
            return name, obj_id
        
        # No ID, just return the name (could be attribute like "dark", "red")
        return arg.strip(), None
    
    def extract_causal_objects(
        self,
        semantic: List[Dict],
    ) -> Tuple[List[str], List[str]]:
        """
        Extract causal object names and IDs from GQA semantic field.
        
        Parses the semantic operation list and extracts all object
        references that are causally relevant to answering the question.
        
        GQA semantic format example:
        [
            {'operation': 'select', 'argument': 'shirt (4653737)', 'dependencies': []},
            {'operation': 'query', 'argument': 'name', 'dependencies': [0]}
        ]
        
        Args:
            semantic: GQA semantic field (list of operations)
            
        Returns:
            Tuple of:
            - List of unique object names (normalized)
            - List of object IDs (for direct scene graph lookup)
        """
        object_names = set()
        object_ids = set()
        
        # Operations that typically reference objects (not attributes/properties)
        object_operations = {'select', 'filter', 'relate', 'same'}
        
        for step in semantic:
            operation = step.get('operation', '')
            
            # Extract from 'argument' field
            if 'argument' in step:
                arg = step['argument']
                
                if isinstance(arg, str) and arg:
                    name, obj_id = self._parse_gqa_argument(arg)
                    
                    if name:
                        # Only add as object if it's from an object-referencing operation
                        # or if it has an ID (confirming it's an object reference)
                        if obj_id or operation in object_operations:
                            normalized = self.normalize_word(name)
                            # Filter out common non-object arguments
                            if normalized not in {'name', 'color', 'material', 'shape', 
                                                  'yes', 'no', 'true', 'false'}:
                                object_names.add(normalized)
                        
                        if obj_id:
                            object_ids.add(obj_id)
                            
                elif isinstance(arg, dict):
                    # Dict argument with 'name' field
                    if 'name' in arg:
                        object_names.add(self.normalize_word(arg['name']))
        
        return list(object_names), list(object_ids)
    
    def match_objects_hierarchical(
        self,
        target_names: List[str],
        scene_graph: Dict,
        target_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Match target object names to scene graph objects using hierarchical strategy.
        
        Matching hierarchy:
        1. Direct ID match (highest priority, if IDs provided)
        2. Exact name match
        3. Lemmatized match
        4. Synonym match (lowest priority)
        
        Args:
            target_names: List of target object names to find
            scene_graph: Scene graph dict with 'objects' field
            target_ids: Optional list of object IDs for direct lookup
            
        Returns:
            Tuple of:
            - List of matched bounding boxes [{x, y, w, h}, ...]
            - Dict mapping matched target -> match_type ('id'/'exact'/'lemma'/'synonym')
        """
        matched_bboxes = []
        match_types = {}
        matched_obj_ids = set()  # Avoid duplicate bboxes
        
        objects = scene_graph.get('objects', {})
        
        # Pass 0: Direct ID match (most reliable)
        if target_ids:
            for obj_id in target_ids:
                if obj_id in objects and obj_id not in matched_obj_ids:
                    obj_data = objects[obj_id]
                    bbox = self._extract_bbox(obj_data)
                    if bbox:
                        matched_bboxes.append(bbox)
                        matched_obj_ids.add(obj_id)
                        obj_name = obj_data.get('name', 'unknown')
                        match_types[obj_name] = 'id'
        
        # Normalize targets
        targets = {self.normalize_word(t): t for t in target_names}
        target_set = set(targets.keys())
        
        # Pass 1: Exact name match
        for obj_id, obj_data in objects.items():
            if obj_id in matched_obj_ids:
                continue
                
            obj_name = self.normalize_word(obj_data.get('name', ''))
            
            if obj_name in target_set:
                bbox = self._extract_bbox(obj_data)
                if bbox:
                    matched_bboxes.append(bbox)
                    matched_obj_ids.add(obj_id)
                    if targets[obj_name] not in match_types:
                        match_types[targets[obj_name]] = 'exact'
        
        # Pass 2: Lemmatized match (if targets remain unmatched)
        if self.use_lemmatization:
            matched_names = {self.normalize_word(t) for t in match_types.keys()}
            unmatched_targets = target_set - matched_names
            
            if unmatched_targets:
                for obj_id, obj_data in objects.items():
                    if obj_id in matched_obj_ids:
                        continue
                    
                    obj_name = self.normalize_word(obj_data.get('name', ''))
                    
                    # Check if lemmatized form matches
                    for target in list(unmatched_targets):
                        if obj_name == target:
                            bbox = self._extract_bbox(obj_data)
                            if bbox:
                                matched_bboxes.append(bbox)
                                matched_obj_ids.add(obj_id)
                                match_types[targets[target]] = 'lemma'
                                unmatched_targets.discard(target)
                                break
        
        # Pass 3: Synonym match
        if self.use_synonyms:
            matched_names = {self.normalize_word(t) for t in match_types.keys()}
            unmatched_targets = target_set - matched_names
            
            if unmatched_targets:
                for obj_id, obj_data in objects.items():
                    if obj_id in matched_obj_ids:
                        continue
                    
                    obj_name = self.normalize_word(obj_data.get('name', ''))
                    obj_synonyms = SYNONYM_REVERSE.get(obj_name, set())
                    
                    # Check if any unmatched target is a synonym
                    for target in list(unmatched_targets):
                        if target in obj_synonyms or obj_name in SYNONYM_REVERSE.get(target, set()):
                            bbox = self._extract_bbox(obj_data)
                            if bbox:
                                matched_bboxes.append(bbox)
                                matched_obj_ids.add(obj_id)
                                match_types[targets[target]] = 'synonym'
                                unmatched_targets.discard(target)
                                break
        
        return matched_bboxes, match_types
    
    def _extract_bbox(self, obj_data: Dict) -> Optional[Dict]:
        """
        Extract bounding box from object data.
        
        Args:
            obj_data: Object dict with bbox fields
            
        Returns:
            Bbox dict {x, y, w, h} or None if invalid
        """
        # Handle both nested and flat bbox formats
        if 'bbox' in obj_data:
            bbox = obj_data['bbox']
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            w = bbox.get('w', 0)
            h = bbox.get('h', 0)
        else:
            x = obj_data.get('x', 0)
            y = obj_data.get('y', 0)
            w = obj_data.get('w', 0)
            h = obj_data.get('h', 0)
        
        # Validate bbox
        if w > 0 and h > 0:
            return {'x': x, 'y': y, 'w': w, 'h': h}
        return None
    
    def render_mask(
        self,
        bboxes: List[Dict],
        image_size: Tuple[int, int],
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Render binary foreground mask from bounding boxes.
        
        Creates a binary mask where:
        - 1 = foreground (inside any bbox)
        - 0 = background
        
        Args:
            bboxes: List of bbox dicts [{x, y, w, h}, ...]
            image_size: Original image size (W, H)
            output_size: Target output size (H, W), default 384x384
            
        Returns:
            Binary mask as numpy array, shape (H, W), dtype float32
        """
        if output_size is None:
            output_size = self.default_output_size
        
        img_w, img_h = image_size
        out_h, out_w = output_size
        
        # Create mask at original resolution
        mask = np.zeros((img_h, img_w), dtype=np.float32)
        
        for bbox in bboxes:
            x = int(bbox['x'])
            y = int(bbox['y'])
            w = int(bbox['w'])
            h = int(bbox['h'])
            
            # Clamp to image bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1.0
        
        # Resize to output size
        if (img_h, img_w) != output_size:
            if HAS_CV2:
                mask = cv2.resize(
                    mask, 
                    (out_w, out_h), 
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                # Fallback to numpy resize (less smooth)
                from PIL import Image
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((out_w, out_h), Image.BILINEAR)
                mask = np.array(mask_pil).astype(np.float32) / 255.0
        
        # Binarize after resize (threshold 0.5)
        mask = (mask > 0.5).astype(np.float32)
        
        return mask
    
    def compute_mask_stats(
        self,
        mask: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute statistics for a generated mask.
        
        Args:
            mask: Binary mask array
            
        Returns:
            Dict with coverage ratio and other stats
        """
        total_pixels = mask.size
        fg_pixels = np.sum(mask > 0.5)
        
        return {
            'coverage_ratio': float(fg_pixels / total_pixels),
            'fg_pixels': int(fg_pixels),
            'total_pixels': total_pixels,
            'is_valid': fg_pixels > 0,
        }
    
    def process_sample(
        self,
        question_data: Dict,
        scene_graph: Dict,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict]:
        """
        Process a single sample to generate mask.
        
        Complete pipeline:
        1. Extract causal objects from semantic
        2. Match to scene graph
        3. Render mask
        4. Return processed sample or None if invalid
        
        Args:
            question_data: Parsed question dict with 'semantic' field
            scene_graph: Scene graph for the image
            output_size: Target mask size
            
        Returns:
            Processed sample dict or None if no valid mask
        """
        # Extract causal objects (names and IDs)
        semantic = question_data.get('semantic', [])
        causal_names, causal_ids = self.extract_causal_objects(semantic)
        
        if not causal_names and not causal_ids:
            return None
        
        # Match to scene graph (using both names and IDs)
        bboxes, match_types = self.match_objects_hierarchical(
            causal_names,
            scene_graph,
            target_ids=causal_ids,
        )
        
        if not bboxes:
            return None
        
        # Get image size
        img_w = scene_graph.get('width', 640)
        img_h = scene_graph.get('height', 480)
        
        # Render mask
        mask = self.render_mask(
            bboxes,
            (img_w, img_h),
            output_size=output_size,
        )
        
        # Validate mask
        stats = self.compute_mask_stats(mask)
        if not stats['is_valid']:
            return None
        
        return {
            'question_id': question_data.get('question_id', ''),
            'image_id': question_data.get('image_id', ''),
            'question': question_data.get('question', ''),
            'answer': question_data.get('answer', ''),
            'causal_objects': causal_names,
            'causal_object_ids': causal_ids,
            # Absolute coordinates (original image space)
            'causal_bboxes': [
                [b['x'], b['y'], b['w'], b['h']] for b in bboxes
            ],
            # Normalized coordinates (0.0 ~ 1.0) for flexible model input sizes
            'causal_bboxes_normalized': [
                [
                    b['x'] / img_w,
                    b['y'] / img_h,
                    b['w'] / img_w,
                    b['h'] / img_h,
                ] for b in bboxes
            ],
            'match_types': match_types,
            'image_size': [img_w, img_h],
            'mask_stats': stats,
        }
