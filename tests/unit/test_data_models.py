import pytest
from gazebo_world_generator.src.core.data_models import Room, GazeboModel

def test_room_initialization():
    room = Room(
        name="Test Room",
        type="office",
        dimensions={"width": 10.0, "length": 10.0, "height": 3.0},
        position={"x": 0.0, "y": 0.0, "z": 0.0}
    )
    assert room.name == "Test Room"
    assert room.type == "office"
    assert room.dimensions["width"] == 10.0
    assert len(room.objects) == 0

def test_gazebo_model_initialization():
    model = GazeboModel(
        name="test_model",
        model_path="model://test",
        category="furniture",
        pose={"x": 1.0, "y": 2.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 1.57}
    )
    assert model.name == "test_model"
    assert model.pose["x"] == 1.0
    assert model.pose["yaw"] == 1.57
    assert model.static is False
