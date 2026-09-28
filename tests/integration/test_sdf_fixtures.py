"""Headless SDF fixtures for simulator validation."""

import xml.etree.ElementTree as ET

import pytest

from gazebo_world_generator.src.config.validated_settings import ValidatedConfig
from gazebo_world_generator.src.core.data_models import GazeboModel
from gazebo_world_generator.src.world.generator import WorldGenerator


@pytest.mark.parametrize("simulator", ["classic", "harmonic"])
@pytest.mark.parametrize("layout", ["empty", "connected", "warehouse"])
def test_generated_sdf_layouts(mock_llm_interface, tmp_path, simulator, layout):
    rooms = [{"name": "Office", "type": "office",
              "dimensions": {"width": 6, "length": 6}, "objects": []}]
    if layout == "connected":
        rooms.append({"name": "Storage", "type": "storage",
                      "dimensions": {"width": 5, "length": 5}, "objects": [],
                      "connections": {"Office": "west"}})
    elif layout == "warehouse":
        rooms = [{"name": "Warehouse", "type": "warehouse",
                  "dimensions": {"width": 12, "length": 10},
                  "objects": [{"type": "shelf", "count": 1}]}]
    mock_llm_interface.parse_room_description.return_value = {"rooms": rooms}
    generator = WorldGenerator(mock_llm_interface, config=ValidatedConfig(simulator=simulator))
    if layout == "warehouse":
        generator.placement_engine.place_all_objects = lambda *_: [GazeboModel(
            name="shelf_0", model_path="model://test_crate", category="furniture",
            pose={"x": 1, "y": 1, "z": 0.9, "roll": 0, "pitch": 0, "yaw": 0},
            size=[0.8, 0.4, 1.8], static=True)]
    output = tmp_path / f"{simulator}_{layout}.sdf"
    generator.generate_world(layout, str(output))
    world = ET.parse(output).getroot().find("world")
    assert world is not None
    assert world.find("physics") is not None
    assert world.findtext("gravity") == "0 0 -9.8"
    assert world.find("physics/gravity") is None
    assert world.findall("model")
    if simulator == "harmonic":
        assert world.find("model[@name='ground_plane']") is not None
        assert world.find("include[uri='model://ground_plane']") is None
