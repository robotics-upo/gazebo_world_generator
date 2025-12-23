"""
Grid Placement Strategy

Implements systematic grid-based placement for multiple objects.
Handles row/column layouts with wall alignment and collision avoidance.
"""

import logging
import math
from typing import List, Dict, Optional, Tuple
from gazebo_world_generator.src.core.data_models import Room
from gazebo_world_generator.src.placement.strategies import PlacementStrategy
from gazebo_world_generator.src.utils.collision_detection import CollisionDetector

logger = logging.getLogger(__name__)


class GridPlacementStrategy(PlacementStrategy):
    """
    Grid-based placement strategy for systematic object layouts.
    
    Use cases:
    - Warehouse storage racks in rows
    - Office desks in grid pattern
    - Multiple similar objects
    
    Features:
    - Configurable row/column layout
    - Wall alignment (north/south/east/west)
    - Collision avoidance
    - Automatic spacing calculation
    """
    
    def __init__(self, collision_detector: Optional[CollisionDetector] = None,
                 room_margin: float = 0.3, collision_margin: float = 0.15):
        """
        Initialize grid placement strategy.
        
        Args:
            collision_detector: Optional collision detector (creates default if None)
            room_margin: Clearance from walls (meters)
            collision_margin: Minimum clearance between objects (meters)
        """
        self.collision_detector = collision_detector
        self.room_margin = room_margin
        self.collision_margin = collision_margin
    
    def place(self, objects: List[Dict], room: Room, 
              existing_objects: List[Dict] = None) -> List[Dict]:
        """
        Place objects in a grid pattern.
        
        Args:
            objects: Objects to place (each with 'name' and 'dimensions')
            room: Room to place objects in
            existing_objects: Already placed objects to avoid
            
        Returns:
            List of placements with pose information
        """
        if not objects:
            return []
        
        if existing_objects is None:
            existing_objects = []
        
        # Initialize collision detector if needed
        if not self.collision_detector:
            self._initialize_collision_detector(room)
        
        # Build room bounds
        room_bounds = self._build_room_bounds(room)
        
        # Calculate grid positions using CollisionDetector
        positions = self.collision_detector.calculate_grid_positions(
            objects=objects,
            position_desc="in rows",  # Default to row layout
            room_bounds=room_bounds,
            existing_objects=existing_objects
        )
        
        # Convert positions to placement dictionaries
        placements = []
        for obj, position in zip(objects, positions):
            x, y, yaw = position
            placements.append({
                'name': obj['name'],
                'type': obj.get('type', 'unknown'),
                'pose': {
                    'x': x,
                    'y': y,
                    'z': 0.0,
                    'roll': 0.0,
                    'pitch': 0.0,
                    'yaw': yaw
                }
            })
        
        logger.info(f"Grid placement: positioned {len(placements)} objects")
        return placements
    
    def place_with_hint(self, objects: List[Dict], room: Room, 
                       position_hint: str,
                       existing_objects: List[Dict] = None) -> List[Dict]:
        """
        Place objects in a grid pattern with a position hint.
        
        Args:
            objects: Objects to place
            room: Room to place objects in
            position_hint: Hint for positioning (e.g., "against north wall", "in center")
            existing_objects: Already placed objects to avoid
            
        Returns:
            List of placements with pose information
        """
        if not objects:
            return []
        
        if existing_objects is None:
            existing_objects = []
        
        # Initialize collision detector if needed
        if not self.collision_detector:
            self._initialize_collision_detector(room)
        
        # Build room bounds
        room_bounds = self._build_room_bounds(room)
        
        # Calculate grid positions with hint
        positions = self.collision_detector.calculate_grid_positions(
            objects=objects,
            position_desc=position_hint,
            room_bounds=room_bounds,
            existing_objects=existing_objects
        )
        
        # Convert positions to placement dictionaries
        placements = []
        for obj, position in zip(objects, positions):
            x, y, yaw = position
            placements.append({
                'name': obj['name'],
                'type': obj.get('type', 'unknown'),
                'pose': {
                    'x': x,
                    'y': y,
                    'z': 0.0,
                    'roll': 0.0,
                    'pitch': 0.0,
                    'yaw': yaw
                }
            })
        
        logger.info(f"Grid placement with hint '{position_hint}': positioned {len(placements)} objects")
        return placements
    
    def _initialize_collision_detector(self, room: Room):
        """Initialize collision detector with room metadata."""
        world_metadata = {
            'door_positions': getattr(room, 'doorways', []),
            'room_bounds': {
                room.name: self._build_room_bounds(room)
            }
        }
        self.collision_detector = CollisionDetector(world_metadata)
        self.collision_detector.room_margin = self.room_margin
        self.collision_detector.collision_margin = self.collision_margin
    
    def _build_room_bounds(self, room: Room) -> Dict:
        """
        Build room bounds dictionary.
        
        Args:
            room: Room to build bounds for
            
        Returns:
            Dictionary with room boundary information
        """
        width = room.dimensions['width']
        length = room.dimensions['length']
        
        return {
            'min_x': -width / 2,
            'max_x': width / 2,
            'min_y': -length / 2,
            'max_y': length / 2,
            'center_x': 0.0,
            'center_y': 0.0,
            'width': width,
            'length': length
        }
