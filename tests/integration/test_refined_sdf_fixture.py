"""A generated world remains valid after a refinement operation."""

import xml.etree.ElementTree as ET

import pytest

from gazebo_world_generator.src.core.data_models import GazeboModel
from gazebo_world_generator.src.world.generator import WorldGenerator
from gazebo_world_generator.src.world.refiner import WorldRefiner
from gazebo_world_generator.src.world.refinement import OperationExecutor
from gazebo_world_generator.src.config.validated_settings import ValidatedConfig


@pytest.mark.parametrize("simulator", ["classic", "harmonic"])
def test_refined_world_sdf(mock_llm_interface, tmp_path, simulator):
    mock_llm_interface.parse_room_description.return_value = {"rooms": [{
        "name": "Warehouse", "type": "warehouse",
        "dimensions": {"width": 10, "length": 10},
        "objects": [{"type": "shelf", "count": 1}],
    }]}
    generator = WorldGenerator(mock_llm_interface,
                               config=ValidatedConfig(simulator=simulator))
    generator.placement_engine.place_all_objects = lambda *_: [GazeboModel(
        name="shelf_0", model_path="model://test_crate", category="furniture",
        pose={"x": 1, "y": 1, "z": 0.9, "roll": 0, "pitch": 0, "yaw": 0},
        size=[0.8, 0.4, 1.8], static=True)]
    output = tmp_path / "refined.sdf"
    generator.generate_world("warehouse", str(output))
    refiner = WorldRefiner(mock_llm_interface, generator)
    refiner.current_world = ET.parse(output)
    refiner.executor = OperationExecutor(refiner)
    assert refiner.executor._apply_remove_objects({
        "operation": "remove", "objects": [{"target": "shelf_0"}]}, output)
    world = ET.parse(output).getroot().find("world")
    assert world.find("include[name='shelf_0']") is None
    assert world.find("model[@name='ground_plane']") is not None
