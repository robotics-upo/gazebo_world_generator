import pytest
import math
from gazebo_world_generator.src.utils.collision_detection import CollisionDetector

def test_collision_detection_overlap():
    detector = CollisionDetector()
    # Object 1 at (0, 0, 0) dims (1, 1, 1)
    # Object 2 at (0.5, 0.5, 0) dims (1, 1, 1)
    # Margin is 0.15. Half dims are 0.5.
    # Required dist = 0.5 + 0.5 + 0.15 = 1.15
    # Actual dx = 0.5, dy = 0.5. Collision expected.
    
    existing_objects = [
        {
            'name': 'obj1',
            'position': (0.0, 0.0, 0.0),
            'dimensions': (1.0, 1.0, 1.0)
        }
    ]
    
    position = (0.5, 0.5, 0.0)
    dimensions = (1.0, 1.0, 1.0)
    
    assert detector._check_collisions(position, dimensions, existing_objects) is False

def test_collision_detection_no_overlap():
    detector = CollisionDetector()
    existing_objects = [
        {
            'name': 'obj1',
            'position': (0.0, 0.0, 0.0),
            'dimensions': (1.0, 1.0, 1.0)
        }
    ]
    
    # Position far away
    position = (3.0, 3.0, 0.0)
    dimensions = (1.0, 1.0, 1.0)
    
    assert detector._check_collisions(position, dimensions, existing_objects) is True

def test_room_bounds_check():
    detector = CollisionDetector()
    room_bounds = {'min_x': -5, 'max_x': 5, 'min_y': -5, 'max_y': 5}
    
    # Center (0,0), dims (2,2). Edges at +/-1. Margin 0.3.
    # Fits within [-5+0.3, 5-0.3] = [-4.7, 4.7]
    assert detector._check_room_bounds_with_extent(0, 0, (2, 2, 2), room_bounds) is True
    
    # Position at edge
    assert detector._check_room_bounds_with_extent(4.5, 0, (2, 2, 2), room_bounds) is False

def test_doorway_clearance():
    world_metadata = {
        'door_positions': [
            {'name': 'door1', 'x': 2.0, 'y': 0.0, 'clearance_radius': 1.0}
        ]
    }
    detector = CollisionDetector(world_metadata=world_metadata)
    
    # Object near door
    assert detector._check_doorway_clearance(2.5, 0.0, (1, 1, 1)) is False
    
    # Object far from door
    assert detector._check_doorway_clearance(0.0, 0.0, (1, 1, 1)) is True

def test_calculate_grid_positions():
    detector = CollisionDetector()
    room_bounds = {
        'center_x': 0.0, 'center_y': 0.0,
        'width': 10.0, 'length': 10.0,
        'min_x': -5.0, 'max_x': 5.0,
        'min_y': -5.0, 'max_y': 5.0
    }
    objects = [
        {'name': 'shelf1', 'dimensions': (1.0, 0.5, 2.0)},
        {'name': 'shelf2', 'dimensions': (1.0, 0.5, 2.0)}
    ]
    
    # Test "center" positioning
    positions = detector.calculate_grid_positions(objects, "center", room_bounds)
    assert len(positions) == 2
    # First object should be near center
    assert abs(positions[0][0]) < 2.0
    assert abs(positions[0][1]) < 2.0

def test_wall_start_position():
    detector = CollisionDetector()
    # Mock parameters
    center_x, center_y = 0, 0
    min_x, max_x, min_y, max_y = -5, 5, -5, 5
    cols, rows = 1, 1
    spacing_x, spacing_y = 1, 1
    max_length = 1.0
    
    # North wall
    start_x, start_y, yaw = detector._calculate_wall_start_position(
        "north wall", center_x, center_y, min_x, max_x, min_y, max_y,
        cols, rows, spacing_x, spacing_y, max_length
    )
    assert start_y > 4.0
    assert yaw == math.pi
    
    # South wall
    start_x, start_y, yaw = detector._calculate_wall_start_position(
        "south wall", center_x, center_y, min_x, max_x, min_y, max_y,
        cols, rows, spacing_x, spacing_y, max_length
    )
    assert start_y < -4.0
    assert yaw == 0.0
