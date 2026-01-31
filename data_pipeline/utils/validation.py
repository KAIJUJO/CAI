"""
Data validation utilities.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from PIL import Image

logger = logging.getLogger(__name__)


# Required fields for different data types
REQUIRED_FIELDS_GQA = {'question_id', 'image_id', 'question', 'answer'}
REQUIRED_FIELDS_IVVQA = {'image_id', 'question', 'answer'}
REQUIRED_FIELDS_PAIR = {'image_id', 'positive', 'negative'}


def validate_sample(
    sample: Dict[str, Any],
    required_fields: Set[str],
    sample_type: str = "sample",
) -> bool:
    """
    Validate that a sample contains all required fields.
    
    Args:
        sample: Sample dictionary to validate
        required_fields: Set of required field names
        sample_type: Type identifier for logging
        
    Returns:
        True if all required fields present and non-empty
    """
    missing = required_fields - set(sample.keys())
    if missing:
        logger.warning(f"Invalid {sample_type}: missing fields {missing}")
        return False
    
    # Check for empty values
    for field in required_fields:
        value = sample.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            logger.warning(f"Invalid {sample_type}: empty field '{field}'")
            return False
    
    return True


def validate_gqa_sample(sample: Dict[str, Any]) -> bool:
    """Validate a GQA dataset sample."""
    return validate_sample(sample, REQUIRED_FIELDS_GQA, "GQA sample")


def validate_ivvqa_sample(sample: Dict[str, Any]) -> bool:
    """Validate an IV-VQA dataset sample."""
    return validate_sample(sample, REQUIRED_FIELDS_IVVQA, "IV-VQA sample")


def validate_pair(pair: Dict[str, Any]) -> bool:
    """
    Validate a positive/negative pair for VJP training.
    
    Args:
        pair: Pair dictionary with 'positive' and 'negative' entries
        
    Returns:
        True if pair structure is valid
    """
    if not validate_sample(pair, REQUIRED_FIELDS_PAIR, "pair"):
        return False
    
    # Validate positive sample structure
    positive = pair.get('positive', {})
    if not isinstance(positive, dict):
        logger.warning("Invalid pair: 'positive' is not a dictionary")
        return False
    if 'question' not in positive or 'answer' not in positive:
        logger.warning("Invalid pair: 'positive' missing question or answer")
        return False
    
    # Validate negative sample structure
    negative = pair.get('negative', {})
    if not isinstance(negative, dict):
        logger.warning("Invalid pair: 'negative' is not a dictionary")
        return False
    if 'question' not in negative or 'answer' not in negative:
        logger.warning("Invalid pair: 'negative' missing question or answer")
        return False
    
    return True


def validate_image(
    image_path: Path,
    min_size: int = 32,
    max_size: int = 10000,
    allowed_formats: Optional[Set[str]] = None,
) -> bool:
    """
    Validate image file integrity and properties.
    
    Args:
        image_path: Path to image file
        min_size: Minimum dimension (width or height)
        max_size: Maximum dimension
        allowed_formats: Set of allowed format strings (e.g., {'JPEG', 'PNG'})
        
    Returns:
        True if image is valid
    """
    if allowed_formats is None:
        allowed_formats = {'JPEG', 'PNG', 'GIF', 'WEBP'}
    
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return False
    
    try:
        with Image.open(image_path) as img:
            # Check format
            if img.format not in allowed_formats:
                logger.warning(f"Invalid image format: {img.format} not in {allowed_formats}")
                return False
            
            # Check dimensions
            width, height = img.size
            if width < min_size or height < min_size:
                logger.warning(f"Image too small: {width}x{height}")
                return False
            if width > max_size or height > max_size:
                logger.warning(f"Image too large: {width}x{height}")
                return False
            
            # Verify image can be loaded
            img.verify()
        
        return True
        
    except Exception as e:
        logger.warning(f"Invalid image {image_path}: {e}")
        return False


def validate_batch(
    samples: List[Dict[str, Any]],
    validator_fn,
) -> tuple:
    """
    Validate a batch of samples and return valid/invalid counts.
    
    Args:
        samples: List of samples to validate
        validator_fn: Validation function to apply to each sample
        
    Returns:
        Tuple of (valid_samples, invalid_count)
    """
    valid_samples = []
    invalid_count = 0
    
    for sample in samples:
        if validator_fn(sample):
            valid_samples.append(sample)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        logger.info(f"Validation: {len(valid_samples)} valid, {invalid_count} invalid")
    
    return valid_samples, invalid_count


class DataValidator:
    """
    Comprehensive data validator for CAI-System datasets.
    """
    
    def __init__(
        self,
        image_dir: Optional[Path] = None,
        validate_images: bool = False,
    ):
        """
        Initialize validator.
        
        Args:
            image_dir: Directory containing images (for image validation)
            validate_images: Whether to validate image files
        """
        self.image_dir = image_dir
        self.validate_images = validate_images
        self.stats = {
            'total': 0,
            'valid': 0,
            'invalid_structure': 0,
            'invalid_image': 0,
        }
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.stats = {k: 0 for k in self.stats}
    
    def validate(
        self,
        sample: Dict[str, Any],
        required_fields: Set[str],
    ) -> bool:
        """
        Validate a single sample.
        
        Args:
            sample: Sample to validate
            required_fields: Required field names
            
        Returns:
            True if sample is valid
        """
        self.stats['total'] += 1
        
        # Structure validation
        if not validate_sample(sample, required_fields, "sample"):
            self.stats['invalid_structure'] += 1
            return False
        
        # Optional image validation
        if self.validate_images and self.image_dir:
            image_id = sample.get('image_id', '')
            image_path = self.image_dir / f"{image_id}.jpg"
            if not validate_image(image_path):
                self.stats['invalid_image'] += 1
                return False
        
        self.stats['valid'] += 1
        return True
    
    def get_stats(self) -> Dict[str, int]:
        """Return validation statistics."""
        return dict(self.stats)
