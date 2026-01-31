"""
Unit tests for CAI-System data pipeline.

Run with: pytest tests/test_data_pipeline.py -v
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.config import DataConfig, GQAConfig, IVVQAConfig
from data_pipeline.base import BaseDataset
from data_pipeline.utils.io import load_json, save_json, load_jsonl, save_jsonl
from data_pipeline.utils.validation import validate_sample, validate_pair, validate_gqa_sample
from data_pipeline.gqa.parser import GQAParser
from data_pipeline.gqa.sampler import GQASampler
from data_pipeline.ivvqa.pair_builder import PairBuilder, CausalPairBuilder


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.random_seed == 42
        assert config.gqa.sample_size == 150000
        assert config.ivvqa.sample_size == 60000
    
    def test_config_paths(self):
        """Test derived path properties."""
        config = DataConfig()
        assert config.raw_dir == config.data_dir / "raw"
        assert config.processed_dir == config.data_dir / "processed"
        assert config.gqa_raw_dir == config.raw_dir / "gqa"
    
    def test_config_from_yaml(self, tmp_path):
        """Test loading config from YAML."""
        yaml_content = """
random_seed: 123
gqa:
  sample_size: 50000
ivvqa:
  sample_size: 20000
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)
        
        config = DataConfig.from_yaml(str(yaml_path))
        assert config.random_seed == 123
        assert config.gqa.sample_size == 50000


class TestIOUtils:
    """Tests for I/O utilities."""
    
    def test_json_roundtrip(self, tmp_path):
        """Test JSON save and load."""
        data = {"key": "value", "nested": {"a": 1}}
        path = tmp_path / "test.json"
        
        save_json(data, path)
        loaded = load_json(path)
        
        assert loaded == data
    
    def test_jsonl_roundtrip(self, tmp_path):
        """Test JSONL save and load."""
        data = [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"},
        ]
        path = tmp_path / "test.jsonl"
        
        save_jsonl(data, path)
        loaded = load_jsonl(path)
        
        assert loaded == data


class TestValidation:
    """Tests for validation utilities."""
    
    def test_validate_sample_success(self):
        """Test successful sample validation."""
        sample = {
            "question_id": "123",
            "image_id": "456",
            "question": "What color?",
            "answer": "blue",
        }
        required = {"question_id", "image_id", "question", "answer"}
        assert validate_sample(sample, required) is True
    
    def test_validate_sample_missing_field(self):
        """Test validation with missing field."""
        sample = {
            "question_id": "123",
            "question": "What color?",
        }
        required = {"question_id", "image_id", "question", "answer"}
        assert validate_sample(sample, required) is False
    
    def test_validate_pair_success(self):
        """Test successful pair validation."""
        pair = {
            "pair_id": "test_pair",
            "image_id": "123",
            "positive": {
                "question": "What color is the dog?",
                "answer": "brown",
            },
            "negative": {
                "question": "Is this indoors?",
                "answer": "yes",
            },
        }
        assert validate_pair(pair) is True
    
    def test_validate_pair_missing_negative(self):
        """Test pair validation with missing negative."""
        pair = {
            "pair_id": "test_pair",
            "image_id": "123",
            "positive": {
                "question": "What color?",
                "answer": "blue",
            },
            "negative": {},  # Missing question/answer
        }
        assert validate_pair(pair) is False


class TestGQASampler:
    """Tests for GQA sampler."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample GQA data for testing."""
        return [
            {
                "question_id": f"q{i}",
                "image_id": f"img{i % 10}",
                "question": f"Question {i}?",
                "answer": ["yes", "no", "blue", "red"][i % 4],
                "groups": {"global": ["relation", "attribute", "object"][i % 3]},
            }
            for i in range(100)
        ]
    
    def test_random_sampling(self, sample_data):
        """Test random sampling."""
        config = DataConfig()
        sampler = GQASampler(config, seed=42)
        
        result = sampler.sample_random(sample_data, 20)
        
        assert len(result) == 20
        assert all(item in sample_data for item in result)
    
    def test_stratified_sampling(self, sample_data):
        """Test stratified sampling maintains type distribution."""
        config = DataConfig()
        sampler = GQASampler(config, seed=42)
        
        result = sampler.sample_stratified_by_type(sample_data, 30)
        
        assert len(result) == 30
        # Should have samples from different types
        types = set()
        for item in result:
            groups = item.get("groups", {})
            if isinstance(groups, dict):
                types.add(groups.get("global"))
        assert len(types) > 1


class TestPairBuilder:
    """Tests for positive/negative pair builder."""
    
    @pytest.fixture
    def config(self):
        return DataConfig()
    
    def test_classify_causal_question(self, config):
        """Test causal question classification."""
        builder = PairBuilder(config)
        
        classification, subtype = builder.classify_question("What color is the dog?")
        assert classification == "causal"
        assert subtype == "color"
    
    def test_classify_background_question(self, config):
        """Test background question classification."""
        builder = PairBuilder(config)
        
        classification, subtype = builder.classify_question("Is there a tree in the background?")
        assert classification == "background"
        assert subtype == "existence"
    
    def test_classify_unknown_question(self, config):
        """Test unknown question classification."""
        builder = PairBuilder(config)
        
        classification, subtype = builder.classify_question("Random text here")
        assert classification == "unknown"
    
    def test_extract_objects(self, config):
        """Test object extraction from questions."""
        builder = PairBuilder(config)
        
        objects = builder.extract_target_objects("What color is the dog?")
        assert "dog" in objects
    
    def test_build_pairs(self, config):
        """Test pair building."""
        builder = PairBuilder(config, seed=42)
        
        data = [
            {"question_id": 1, "image_id": "img1", "question": "What color is the cat?", "answer": "brown"},
            {"question_id": 2, "image_id": "img1", "question": "Is there a tree?", "answer": "yes"},
            {"question_id": 3, "image_id": "img1", "question": "What shape is the ball?", "answer": "round"},
        ]
        
        pairs = builder.build_pairs(data)
        
        # Should create pairs (causal questions paired with background)
        assert len(pairs) > 0
        for pair in pairs:
            assert "positive" in pair
            assert "negative" in pair
            assert pair["image_id"] == "img1"


class TestCausalPairBuilder:
    """Tests for causal pair builder with scoring."""
    
    def test_score_causal_relevance(self):
        """Test causal relevance scoring."""
        config = DataConfig()
        builder = CausalPairBuilder(config)
        
        # High causal score
        score1 = builder.score_causal_relevance("What color is the dog wearing?", "blue")
        assert score1 > 0.5
        
        # Lower causal score
        score2 = builder.score_causal_relevance("Is it daytime?", "yes")
        assert score2 < score1
    
    def test_score_background_relevance(self):
        """Test background relevance scoring."""
        config = DataConfig()
        builder = CausalPairBuilder(config)
        
        # High background score
        score1 = builder.score_background_relevance("Is this indoors or outdoors?", "outdoors")
        assert score1 > 0.5
        
        # Lower background score
        score2 = builder.score_background_relevance("What color is the shirt?", "red")
        assert score2 < score1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
