import pytest
from unittest.mock import MagicMock
from gazebo_world_generator.src.llm.interface import OpenAICompatibleInterface

def test_extract_dimension_hints():
    llm = OpenAICompatibleInterface(server_url="localhost:8000")
    
    # Test pattern 1: "12m x 10m office"
    description = "Create a 12m x 10m office with a desk."
    hints = llm._extract_dimension_hints(description)
    assert len(hints) == 1
    assert hints[0]['room_type'] == 'office'
    assert hints[0]['width'] == 12.0
    assert hints[0]['length'] == 10.0

    # Test pattern 2: "Living room (5x6m)"
    description = "A living room (5x6.5m) with a sofa."
    hints = llm._extract_dimension_hints(description)
    assert len(hints) == 1
    # Current regex only catches the last word before parenthesis if it's alphanumeric
    assert hints[0]['room_type'] == 'room'
    assert hints[0]['width'] == 5.0
    assert hints[0]['length'] == 6.5


@pytest.mark.parametrize("description, room_type, width, length", [
    ("An office of size 8x6 with a desk", "office", 8.0, 6.0),
    ("A warehouse of 20 meters by 15 meters", "warehouse", 20.0, 15.0),
    ("A lab of 7m by 5.5m", "lab", 7.0, 5.5),
    ("An office of 8 x 6 m with 2 chairs", "office", 8.0, 6.0),
    ("An 8 x 6 m office", "office", 8.0, 6.0),
    ("A 5mx4m room", "room", 5.0, 4.0),
])
def test_extract_dimension_hints_of_pattern(description, room_type, width, length):
    llm = OpenAICompatibleInterface(server_url="localhost:8000")
    hints = llm._extract_dimension_hints(description)
    assert [(h['room_type'], h['width'], h['length']) for h in hints] == [(room_type, width, length)]

def test_create_fallback_room():
    llm = OpenAICompatibleInterface(server_url="localhost:8000")
    description = "Something about a warehouse"
    fallback = llm._create_fallback_room(description)
    assert fallback['rooms'][0]['type'] == 'warehouse'
    
    description = "A simple office"
    fallback = llm._create_fallback_room(description)
    assert fallback['rooms'][0]['type'] == 'office'

def test_balance_object_distribution():
    llm = OpenAICompatibleInterface(server_url="localhost:8000")
    parsed_data = {
        "rooms": [
            {
                "name": "Office 1",
                "type": "office",
                "objects": [{"type": "desk", "count": 10}]
            },
            {
                "name": "Office 2",
                "type": "office",
                "objects": []
            }
        ]
    }
    llm._balance_object_distribution(parsed_data)
    # Total 10 desks should be distributed
    count1 = sum(obj['count'] for obj in parsed_data['rooms'][0]['objects'])
    count2 = sum(obj['count'] for obj in parsed_data['rooms'][1]['objects'])
    assert count1 + count2 == 10
    assert count1 == 5
    assert count2 == 5
