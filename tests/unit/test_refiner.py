import pytest
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch
from pathlib import Path
from gazebo_world_generator.src.world.refiner import WorldRefiner

@pytest.fixture
def mock_refiner():
    llm = MagicMock()
    generator = MagicMock()
    return WorldRefiner(llm, generator)

def test_extract_object_type(mock_refiner):
    assert mock_refiner._extract_object_type("Office_Desk_3") == "desk"
    assert mock_refiner._extract_object_type("OfficeChairBlack_1") == "chair"
    assert mock_refiner._extract_object_type("bookshelf_0") == "bookshelf"
    assert mock_refiner._extract_object_type("Simple_Table_Large") == "table"
    assert mock_refiner._extract_object_type("UnknownModel") == "unknownmodel"

def test_calculate_room_bounds(mock_refiner):
    # Mock SDF with walls
    sdf_content = """
    <sdf version='1.6'>
      <world name='default'>
        <model name='room1_wall_north'>
          <pose>0 5 1.5 0 0 0</pose>
        </model>
        <model name='room1_wall_south'>
          <pose>0 -5 1.5 0 0 0</pose>
        </model>
        <model name='room1_wall_east'>
          <pose>5 0 1.5 0 0 0</pose>
        </model>
        <model name='room1_wall_west'>
          <pose -5 0 1.5 0 0 0</pose>
        </model>
      </world>
    </sdf>
    """
    # Fix the typo in the fake SDF for parsing if needed, but WorldRefiner might just regex it or find elements.
    # Actually WorldRefiner uses ElementTree.
    root = ET.fromstring("""
    <world>
        <model name='room1_wall_north'><pose>0 5 1.5 0 0 0</pose></model>
        <model name='room1_wall_south'><pose>0 -5 1.5 0 0 0</pose></model>
        <model name='room1_wall_east'><pose>5 0 1.5 0 0 0</pose></model>
        <model name='room1_wall_west'><pose>-5 0 1.5 0 0 0</pose></model>
    </world>
    """)
    
    wall_names = ['room1_wall_north', 'room1_wall_south', 'room1_wall_east', 'room1_wall_west']
    bounds = mock_refiner._calculate_room_bounds(wall_names, root)
    
    assert 'room1' in bounds
    assert bounds['room1']['min_x'] == -4.85 # -5.0 + 0.15
    assert bounds['room1']['max_x'] == 4.85  # 5.0 - 0.15
    assert bounds['room1']['min_y'] == -4.85
    assert bounds['room1']['max_y'] == 4.85

def test_extract_door_positions(mock_refiner):
    root = ET.fromstring("""
    <world>
        <model name='door_1'>
            <pose>2 0 0 0 0 0</pose>
        </model>
    </world>
    """)
    door_names = ['door_1']
    doors = mock_refiner._extract_door_positions(door_names, root)
    
    assert len(doors) == 1
    assert doors[0]['x'] == 2.0
    assert doors[0]['y'] == 0.0

def test_apply_remove_objects(mock_refiner):
    # Setup mock world with a model to remove
    root = ET.fromstring("""
    <sdf version='1.6'>
      <world name='default'>
        <model name='desk_0'>
            <pose>0 0 0 0 0 0</pose>
        </model>
      </world>
    </sdf>
    """)
    mock_refiner.current_world = ET.ElementTree(root)
    
    data = {
        "operation": "remove",
        "objects": [{"target": "desk_0"}]
    }
    
    with patch('pathlib.Path.write_text'): # Prevent actual file write
        success = mock_refiner._apply_remove_objects(data, Path("test.sdf"))
        
    assert success is True
    # Verify it was removed from ET
    assert root.find('.//model[@name="desk_0"]') is None

def test_apply_batch_operations(mock_refiner):
    # Mock individual operation handlers
    mock_refiner._apply_add_objects = MagicMock(return_value=True)
    mock_refiner._apply_remove_objects = MagicMock(return_value=True)
    
    data = {
        "operation": "batch",
        "operations": [
            {"operation": "add", "objects": []},
            {"operation": "remove", "objects": []}
        ]
    }
    
    success = mock_refiner.apply_refinement(data, Path("test.sdf"))
    
    assert success is True
    assert mock_refiner._apply_add_objects.call_count == 1
    assert mock_refiner._apply_remove_objects.call_count == 1

def test_apply_add_objects_simple(mock_refiner):
    root = ET.fromstring("""
    <sdf version='1.6'>
      <world name='default'></world>
    </sdf>
    """)
    mock_refiner.current_world = ET.ElementTree(root)
    mock_refiner.generator = MagicMock()
    mock_refiner.generator.model_db = MagicMock()
    mock_refiner.generator.placement_engine = MagicMock()
    
    # Mocking resolve_and_place_new_objects or similar depends on the implementation
    # Let's mock _add_single_object as it's likely a helper
    mock_refiner._add_single_object = MagicMock(return_value=True)
    
    data = {
        "operation": "add",
        "objects": [{"type": "chair", "count": 1}]
    }
    
    with patch.object(mock_refiner, '_add_single_object', return_value=True):
         success = mock_refiner._apply_add_objects(data, Path("test.sdf"))
         
    assert success is True
