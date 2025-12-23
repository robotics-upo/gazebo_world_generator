import pytest
from unittest.mock import MagicMock, patch
from gazebo_world_generator.src.models.resolver import SmartModelResolver

def test_model_resolver_fallback():
    resolver = SmartModelResolver()
    # Test fallback for unknown object
    fallback = resolver._get_fallback_model("completely_unknown_type_xyz")
    assert fallback is None
    
    # Test fallback for known object
    fallback = resolver._get_fallback_model("desk")
    assert "desk" in fallback or "table" in fallback

def test_find_best_model_keyword_match():
    resolver = SmartModelResolver()
    # Mock local_models to have some candidates
    resolver._local_models = {
        "Office Chair": "model://office_chair",
        "Wooden Table": "model://wooden_table"
    }
    
    # Mock LLM to return nothing to force keyword match
    resolver.llm = MagicMock()
    resolver.llm.query.return_value = ""
    
    model = resolver.find_best_model("chair")
    assert "chair" in model.lower()

def test_model_name_from_config(tmp_path):
    resolver = SmartModelResolver()
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    config_file = model_dir / "model.config"
    config_file.write_text("""<?xml version="1.0"?>
<model>
  <name>Test Model Name</name>
</model>""")
    
    name = resolver._get_model_name_from_config(model_dir)
    assert name == "Test Model Name"
