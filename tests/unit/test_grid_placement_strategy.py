"""
Unit tests for GridPlacementStrategy

Tests grid-based placement logic including wall alignment and collision avoidance.
"""

import pytest
import math
from unittest.mock import MagicMock, patch
from gazebo_world_generator.src.placement.strategies.grid_placement import GridPlacementStrategy
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
def grid_strategy():
    """Create GridPlacementStrategy instance."""
    return GridPlacementStrategy()


def test_basic_grid_placement(grid_strategy, test_room):
    """Test basic grid placement of objects."""
    objects = [
        {'name': 'desk_1', 'type': 'desk', 'dimensions': (1.2, 0.6, 0.75)},
        {'name': 'desk_2', 'type': 'desk', 'dimensions': (1.2, 0.6, 0.75)}
    ]
    
    placements = grid_strategy.place(objects, test_room)
    
    # Should return 2 placements
    assert len(placements) == 2
    # Each should have required keys
    for p in placements:
        assert 'name' in p
        assert 'pose' in p
        assert 'x' in p['pose']
        assert 'y' in p['pose']
        assert 'yaw' in p['pose']


def test_grid_positions_in_room_bounds(grid_strategy, test_room):
    """Test that grid positions stay within room bounds."""
    objects = [
        {'name': f'obj_{i}', 'type': 'desk', 'dimensions': (1.0, 0.5, 0.75)}
        for i in range(4)
    ]
    
    placements = grid_strategy.place(objects, test_room)
    
    # All objects should be within room bounds
    half_w = test_room.dimensions['width'] / 2
    half_l = test_room.dimensions['length'] / 2
    margin = 0.3
    
    for p in placements:
        x, y = p['pose']['x'], p['pose']['y']
        assert -half_w + margin < x < half_w - margin
        assert -half_l + margin < y < half_l - margin


def test_grid_with_wall_alignment(grid_strategy, test_room):
    """Test grid placement with wall alignment hint."""
    objects = [
        {'name': f'shelf_{i}', 'type': 'shelf', 'dimensions': (0.8, 0.3, 1.8)}
        for i in range(3)
    ]
    
    placements = grid_strategy.place_with_hint(objects, test_room, "along north wall")
    
    # Should have 3 placements
    assert len(placements) == 3
    
    # Objects should be near north wall (high Y values)
    for p in placements:
        y = p['pose']['y']
        assert y > 3.0  # Near north wall
    
    # Orientation should face into room (yaw ≈ π for north wall)
    for p in placements:
        yaw = p['pose']['yaw']
        assert abs(yaw - math.pi) < 0.5


def test_grid_collision_avoidance(grid_strategy, test_room):
    """Test that grid placement avoids existing objects."""
    objects = [
        {'name': 'new_obj', 'type': 'desk', 'dimensions': (1.0, 0.5, 0.75)}
    ]
    
    # Existing object at center
    existing = [
        {
            'name': 'existing',
            'position': (0.0, 0.0, 0.0),
            'dimensions': (2.0, 2.0, 1.0)
        }
    ]
    
    placements = grid_strategy.place(objects, test_room, existing)
    
    # New object should be placed away from center
    x, y = placements[0]['pose']['x'], placements[0]['pose']['y']
    distance = math.sqrt(x**2 + y**2)
    assert distance >= 1.4  # Should be away from center (allowing for floating point precision)


def test_place_with_hint_center(grid_strategy, test_room):
    """Test center placement hint."""
    objects = [
        {'name': 'centerpiece', 'type': 'table', 'dimensions': (1.5, 0.8, 0.75)}
    ]
    
    placements = grid_strategy.place_with_hint(objects, test_room, "center")
    
    # Should be near center
    x, y = placements[0]['pose']['x'], placements[0]['pose']['y']
    assert abs(x) < 2.0
    assert abs(y) < 2.0


def test_grid_builds_room_bounds(grid_strategy, test_room):
    """Test that room bounds are properly constructed."""
    objects = [{'name': 'obj', 'type': 'desk', 'dimensions': (1.0, 0.5, 0.75)}]
    
    # This should work without errors
    placements = grid_strategy.place(objects, test_room)
    
    assert len(placements) == 1


def test_empty_objects_list(grid_strategy, test_room):
    """Test handling of empty objects list."""
    placements = grid_strategy.place([], test_room)
    
    assert len(placements) == 0


def test_grid_uses_collision_detector(grid_strategy, test_room):
    """Test that grid strategy uses CollisionDetector."""
    objects = [{'name': 'obj', 'type': 'desk', 'dimensions': (1.0, 0.5, 0.75)}]
    
    # After first placement, collision detector should be initialized
    placements = grid_strategy.place(objects, test_room)
    
    # Verify collision detector was created
    assert grid_strategy.collision_detector is not None


def test_multiple_wall_hints(grid_strategy, test_room):
    """Test different wall alignment hints."""
    objects = [{'name': 'obj', 'type': 'shelf', 'dimensions': (0.8, 0.3, 1.8)}]
    
    # Test each wall
    walls = {
        'north': (lambda y: y > 3.0, math.pi),
        'south': (lambda y: y < -3.0, 0.0),
        'east': (lambda x: x > 3.0, -math.pi/2),
        'west': (lambda x: x < -3.0, math.pi/2)
    }
    
    for wall_name, (check_fn, expected_yaw) in walls.items():
        placements = grid_strategy.place_with_hint(objects, test_room, f"along {wall_name} wall")
        
        if wall_name in ['north', 'south']:
            assert check_fn(placements[0]['pose']['y'])
        else:
            assert check_fn(placements[0]['pose']['x'])
        
        # Check orientation
        yaw = placements[0]['pose']['yaw']
        assert abs(yaw - expected_yaw) < 0.5
