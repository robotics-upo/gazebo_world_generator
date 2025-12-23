import pytest
import json
from unittest.mock import MagicMock, patch
from gazebo_world_generator.src.placement.semantic_grouping import SemanticGroupingEngine, SemanticGroup

@pytest.fixture
def mock_llm():
    return MagicMock()

@pytest.fixture
def engine(mock_llm):
    return SemanticGroupingEngine(llm_interface=mock_llm)

def test_extract_json_from_response(engine):
    # Test valid JSON
    response = '{"key": "value"}'
    assert engine._extract_json_from_response(response) == {"key": "value"}
    
    # Test markdown block
    response = "```json\n[{\"group_id\": \"1\"}]\n```"
    assert engine._extract_json_from_response(response) == [{"group_id": "1"}]
    
    # Test garbage before/after
    response = "Here is the response: [{\"a\": 1}] hope it helps"
    assert engine._extract_json_from_response(response) == [{"a": 1}]

def test_identify_object_groups(engine, mock_llm):
    # Mock LLM response
    mock_llm.query.return_value = json.dumps([
        {
            "group_id": "desk_cluster",
            "primary_object": "desk",
            "related_objects": ["chair", "lamp"],
            "relationship_type": "functional",
            "proximity": "adjacent",
            "spatial_hint": "in_front"
        }
    ])
    
    objects = [
        {"type": "desk", "count": 1},
        {"type": "chair", "count": 1},
        {"type": "lamp", "count": 1}
    ]
    
    groups = engine.identify_object_groups(objects, "office", "main_room")
    
    assert len(groups) == 1
    assert isinstance(groups[0], SemanticGroup)
    assert groups[0].group_id == "desk_cluster"
    assert groups[0].primary_object == "desk"

def test_expand_groups_to_instances(engine):
    groups = [
        SemanticGroup(
            group_id="test_group",
            primary_object="desk",
            related_objects=["chair"],
            relationship_type="pairing",
            proximity="adjacent",
            spatial_hint="in_front",
            spatial_arrangement={"chair": "in_front"}
        )
    ]
    
    objects_to_place = [
        {"type": "desk", "count": 2},
        {"type": "chair", "count": 2}
    ]
    
    expanded = engine.expand_groups_to_instances(groups, objects_to_place)
    
    # Should create 2 groups of desk+chair
    assert len(expanded) == 2
    assert expanded[0]['primary_object'] == 'desk'
    assert 'chair' in expanded[0]['objects']
