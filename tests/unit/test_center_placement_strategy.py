"""
Unit tests for CenterPlacementStrategy

Tests center-based placement logic.
"""

import pytest
from unittest.mock import MagicMock
from gazebo_world_generator.src.placement.strategies.center_placement import CenterPlacementStrategy
from gazebo_world_generator.src.core.data_models import Room


@pytest.fixture
def test_room():
    """Create a test room."""
    return Room(
        name='test_room',
        type='office',
        dimensions={'width': 10.0, 'length': 10.0, 'height': 3.0},
        position={'x': 0.0, 'y': 0.0, 'z': 0.0}
    )


@pytest.fixture
def center_strategy():
    """Create CenterPlacementStrategy instance."""
    return CenterPlacementStrategy()


def test_single_object_center(center_strategy, test_room):
    """Test placing single object at center."""
    objects = [
        {'name': 'table', 'type': 'table', 'dimensions': (1.5, 0.8, 0.75)}
    ]
    
    placements = center_strategy.place(objects, test_room)
    
    assert len(placements) == 1
    # Should be at or near center
    x, y = placements[0]['pose']['x'], placements[0]['pose']['y']
    assert abs(x) < 1.0
    assert abs(y) < 1.0


def test_place_with_offset(center_strategy, test_room):
    """Test placement with custom offset from center."""
    objects = [
        {'name': 'desk', 'type': 'desk', 'dimensions': (1.2, 0.6, 0.75)}
    ]
    
    offset_x, offset_y = 2.0, 1.5
    placements = center_strategy.place_with_offset(objects, test_room, offset_x, offset_y)
    
    # Should be at offset position
    x, y = placements[0]['pose']['x'], placements[0]['pose']['y']
    assert abs(x - offset_x) < 0.5
    assert abs(y - offset_y) < 0.5


def test_collision_validation(center_strategy, test_room):
    """Test that collision detection is used."""
    objects = [{'name': 'obj', 'type': 'desk', 'dimensions': (1.0, 0.5, 0.75)}]
    
    existing = [
        {
            'name': 'existing',
            'position': (0.0, 0.0, 0.0),
            'dimensions': (1.0, 1.0, 1.0)
        }
    ]
    
    placements = center_strategy.place(objects, test_room, existing)
    
    # Should still place (may adjust position for collision)
    assert len(placements) == 1


def test_empty_objects_list(center_strategy, test_room):
    """Test handling of empty objects list."""
    placements = center_strategy.place([], test_room)
    assert len(placements) == 0


def test_multiple_objects_center(center_strategy, test_room):
    """Test placing multiple objects near center."""
    objects = [
        {'name': f'obj_{i}', 'type': 'chair', 'dimensions': (0.6, 0.6, 0.9)}
        for i in range(3)
    ]
    
    placements = center_strategy.place(objects, test_room)
    
    # All should be placed
    assert len(placements) == 3
    # All should be relatively close to center
    for p in placements:
        x, y = p['pose']['x'], p['pose']['y']
        distance = (x**2 + y**2)**0.5
        assert distance <= 3.5  # Within 3.5m of center


def test_uses_collision_detector(center_strategy, test_room):
    """Test that collision detector is initialized."""
    objects = [{'name': 'obj', 'type': 'desk', 'dimensions': (1.0, 0.5, 0.75)}]
    
    center_strategy.place(objects, test_room)
    
    assert center_strategy.collision_detector is not None
