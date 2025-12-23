import pytest
import os
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch
from gazebo_world_generator.src.world.generator import WorldGenerator

def test_generate_world_integration(mock_llm_interface, tmp_path):
    # Mock LLMInterface
    llm = mock_llm_interface
    
    # We need to mock PromptManager to avoid issues with missing templates
    with patch('gazebo_world_generator.src.llm.interface.PromptManager'):
        # Mock SmartModelResolver to avoid real model database lookups if possible
        # Or just let it use the online_db mock
        
        generator = WorldGenerator(llm_interface=llm)
        
        output_file = str(tmp_path / "test_world.sdf")
        description = "A small empty room"
        
        # Mock parse_room_description specifically for this test
        llm.parse_room_description.return_value = {
            "rooms": [
                {
                    "name": "Small Room",
                    "type": "office",
                    "dimensions": {"width": 4.0, "length": 4.0, "height": 3.0},
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "objects": []
                }
            ]
        }
        
        # Run generation
        generator.generate_world(description, output_path=output_file)
        
        # Assertions
        assert os.path.exists(output_file)
        
        # Parse the generated SDF to check for expected elements
        tree = ET.parse(output_file)
        root = tree.getroot()
        
        # Check for ground plane
        ground_plane = root.find(".//include[uri='model://ground_plane']")
        if ground_plane is None:
            # Maybe it's a <model> or differently structured
            ground_plane = root.find(".//model[@name='ground_plane']")
        assert ground_plane is not None
        
        # Check for walls
        # The generator creates walls with names like {room_name}_wall_{side}
        walls = [m for m in root.findall(".//model") if "Small_Room_wall" in m.get("name", "")]
        assert len(walls) > 0

def test_calculate_room_dimensions(mock_llm_interface):
    generator = WorldGenerator(llm_interface=mock_llm_interface)
    
    # Test specific dimensions
    room_data = {"name": "test", "type": "office", "dimensions": {"width": 8.0, "length": 6.0}}
    dims = generator._calculate_room_dimensions(room_data)
    assert dims["width"] == 8.0
    assert dims["length"] == 6.0
    
    # Test corridor default width
    room_data = {"name": "corr", "type": "corridor", "dimensions": {"length": 10.0}}
    dims = generator._calculate_room_dimensions(room_data)
    assert dims["width"] == generator.placement_config.corridor_width
    assert dims["length"] == 10.0
