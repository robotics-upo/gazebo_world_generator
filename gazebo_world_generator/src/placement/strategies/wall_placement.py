"""
Wall Placement Strategy

Implements wall-snapping placement for furniture that should be against walls.
Handles proper orientation to face into the room.
"""

import logging
import math
from typing import List, Dict, Optional, Tuple
from gazebo_world_generator.src.core.data_models import Room
from gazebo_world_generator.src.placement.strategies import PlacementStrategy
from gazebo_world_generator.src.placement.strategies.grid_placement import GridPlacementStrategy
from gazebo_world_generator.src.utils.collision_detection import CollisionDetector

logger = logging.getLogger(__name__)


class WallPlacementStrategy(PlacementStrategy):
    """
    Wall-snapping placement strategy for furniture.
    
    Use cases:
    - Bookshelves along walls
    - Wardrobes against walls
    - Cabinets and storage units
    
    Features:
    - Automatic wall detection
    - Proper orientation (facing into room)
    - Grid positioning for multiple items on same wall
    - Collision avoidance
    """
    
    # Object types that can be placed against walls
    WALL_FURNITURE_TYPES = {
        'bookshelf', 'shelf', 'cabinet', 'shelving_unit', 
        'wardrobe', 'dresser', 'tv_stand'
    }
    
    def __init__(self, collision_detector: Optional[CollisionDetector] = None,
                 grid_strategy: Optional[GridPlacementStrategy] = None,
                 snap_threshold: float = 2.0,
                 wall_thickness: float = 0.15):
        """
        Initialize wall placement strategy.
        
        Args:
            collision_detector: Optional collision detector
            grid_strategy: Optional grid strategy for multiple items
            snap_threshold: Maximum distance to snap to wall (meters)
            wall_thickness: Wall thickness (meters)
        """
        self.collision_detector = collision_detector
        self.grid_strategy = grid_strategy
        self.snap_threshold = snap_threshold
        self.wall_thickness = wall_thickness
    
    def place(self, objects: List[Dict], room: Room, 
              existing_objects: List[Dict] = None) -> List[Dict]:
        """
        Place objects against walls.
        
        Args:
            objects: Objects to place (each with 'name', 'type', 'dimensions')
            room: Room to place objects in
            existing_objects: Already placed objects to avoid
            
        Returns:
            List of placements with pose information
        """
        if not objects:
            return []
        
        if existing_objects is None:
            existing_objects = []
        
        # Initialize strategies if needed
        if not self.collision_detector:
            self._initialize_collision_detector(room)
        if not self.grid_strategy:
            self.grid_strategy = GridPlacementStrategy(self.collision_detector)
        
        # Calculate room bounds
        wall_thickness = self.wall_thickness
        inner_half_w = (room.dimensions['width'] / 2) - (wall_thickness / 2)
        inner_half_l = (room.dimensions['length'] / 2) - (wall_thickness / 2)
        
        # Group items by their target wall
        wall_groups = {'west': [], 'east': [], 'north': [], 'south': []}
        non_wall_items = []
        
        # Analyze each object to determine wall assignment
        for obj in objects:
            obj_type = obj.get('type', '').lower()
            
            # Check if this object type can be against walls
            if not any(wf_type in obj_type for wf_type in self.WALL_FURNITURE_TYPES):
                non_wall_items.append(obj)
                continue
            
            # Determine target wall based on initial position or default
            pose = obj.get('pose', {})
            x = pose.get('x', 0.0)
            y = pose.get('y', 0.0)
            
            # Calculate distances to each wall
            dist_to_west = abs(x - (-inner_half_w))
            dist_to_east = abs(x - inner_half_w)
            dist_to_south = abs(y - (-inner_half_l))
            dist_to_north = abs(y - inner_half_l)
            
            min_dist = min(dist_to_west, dist_to_east, dist_to_south, dist_to_north)
            
            # Only snap if furniture is already close to a wall
            if min_dist > self.snap_threshold:
                non_wall_items.append(obj)
                continue
            
            # Determine target wall and add to group
            if min_dist == dist_to_west:
                wall_groups['west'].append(obj)
            elif min_dist == dist_to_east:
                wall_groups['east'].append(obj)
            elif min_dist == dist_to_south:
                wall_groups['south'].append(obj)
            else:  # north wall
                wall_groups['north'].append(obj)
        
        # Build room bounds
        room_bounds = self._build_room_bounds(room, inner_half_w, inner_half_l)
        
        # Process each wall group
        placements = []
        for wall_name, items in wall_groups.items():
            if not items:
                continue
            
            # If multiple items on same wall, use grid positioning
            if len(items) >= 3:
                logger.info(f"Using grid positioning for {len(items)} items on {wall_name} wall")
                wall_placements = self.grid_strategy.place_with_hint(
                    objects=items,
                    room=room,
                    position_hint=f"along {wall_name} wall",
                    existing_objects=existing_objects
                )
                placements.extend(wall_placements)
            else:
                # Single or few items, use simple snapping
                for item in items:
                    placement = self._snap_to_wall(item, wall_name, inner_half_w, inner_half_l)
                    placements.append(placement)
        
        logger.info(f"Wall placement: positioned {len(placements)} objects against walls, "
                   f"{len(non_wall_items)} items not wall-snapped")
        
        return placements
    
    def _snap_to_wall(self, obj: Dict, wall_name: str, 
                     inner_half_w: float, inner_half_l: float) -> Dict:
        """
        Snap a single object to a wall.
        
        Args:
            obj: Object to snap
            wall_name: Wall name (west/east/north/south)
            inner_half_w: Half width of room interior
            inner_half_l: Half length of room interior
            
        Returns:
            Placement dictionary with updated pose
        """
        pose = obj.get('pose', {})
        x = pose.get('x', 0.0)
        y = pose.get('y', 0.0)
        
        # Get object dimensions
        dims = obj.get('dimensions', (1.0, 1.0, 1.0))
        half_length = dims[1] / 2  # Object depth
        
        # Snap to wall and set orientation
        if wall_name == 'west':
            x = -inner_half_w + half_length
            yaw = math.pi / 2  # Face into room (east)
        elif wall_name == 'east':
            x = inner_half_w - half_length
            yaw = -math.pi / 2  # Face into room (west)
        elif wall_name == 'south':
            y = -inner_half_l + half_length
            yaw = 0.0  # Face into room (north)
        else:  # north
            y = inner_half_l - half_length
            yaw = math.pi  # Face into room (south)
        
        return {
            'name': obj['name'],
            'type': obj.get('type', 'unknown'),
            'pose': {
                'x': x,
                'y': y,
                'z': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': yaw
            },
            '_wall_snapped': True
        }
    
    def _initialize_collision_detector(self, room: Room):
        """Initialize collision detector with room metadata."""
        world_metadata = {
            'door_positions': getattr(room, 'doorways', []),
            'room_bounds': {
                room.name: self._build_room_bounds(room)
            }
        }
        self.collision_detector = CollisionDetector(world_metadata)
    
    def _build_room_bounds(self, room: Room, 
                          inner_half_w: Optional[float] = None,
                          inner_half_l: Optional[float] = None) -> Dict:
        """Build room bounds dictionary."""
        width = room.dimensions['width']
        length = room.dimensions['length']
        
        if inner_half_w is None:
            inner_half_w = width / 2 - self.wall_thickness / 2
        if inner_half_l is None:
            inner_half_l = length / 2 - self.wall_thickness / 2
        
        return {
            'min_x': -inner_half_w,
            'max_x': inner_half_w,
            'min_y': -inner_half_l,
            'max_y': inner_half_l,
            'center_x': 0.0,
            'center_y': 0.0,
            'width': width,
            'length': length
        }
