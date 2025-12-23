"""
Center Placement Strategy

Implements center-based placement for focal objects.
Places objects in the center of the room or a specified area.
"""

import logging
from typing import List, Dict, Optional
from gazebo_world_generator.src.core.data_models import Room
from gazebo_world_generator.src.placement.strategies import PlacementStrategy
from gazebo_world_generator.src.utils.collision_detection import CollisionDetector

logger = logging.getLogger(__name__)


class CenterPlacementStrategy(PlacementStrategy):
    """
    Center-based placement strategy for focal objects.
    
    Use cases:
    - Conference tables in center of room
    - Single desks in home offices
    - Featured furniture pieces
    
    Features:
    - Centered positioning
    - Collision validation
    - Supports offset from center
    """
    
    def __init__(self, collision_detector: Optional[CollisionDetector] = None):
        """
        Initialize center placement strategy.
        
        Args:
            collision_detector: Optional collision detector for validation
        """
        self.collision_detector = collision_detector
    
    def place(self, objects: List[Dict], room: Room, 
              existing_objects: List[Dict] = None) -> List[Dict]:
        """
        Place objects in the center of the room.
        
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
        
        placements = []
        center_x = 0.0
        center_y = 0.0
        
        # Place objects in center with slight offsets for multiple objects
        for idx, obj in enumerate(objects):
            # Slight offset for multiple objects to avoid overlap
            offset_x = 0.0
            offset_y = 0.0
            
            if len(objects) > 1:
                # Space objects out slightly
                offset_x = (idx - len(objects) / 2) * 2.0
            
            x = center_x + offset_x
            y = center_y + offset_y
            yaw = 0.0  # Default orientation
            
            # Validate position
            room_bounds = self._build_room_bounds(room)
            position = (x, y, yaw)
            dimensions = obj.get('dimensions', (1.0, 1.0, 1.0))
            
            is_valid = self.collision_detector.validate_position(
                position, dimensions, existing_objects, room_bounds
            )
            
            if not is_valid:
                logger.warning(f"Center position for {obj['name']} has collision or out of bounds, "
                             f"using position anyway")
            
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
            
            # Add to existing objects for next iteration
            existing_objects.append({
                'name': obj['name'],
                'position': position,
                'dimensions': dimensions
            })
        
        logger.info(f"Center placement: positioned {len(placements)} objects")
        return placements
    
    def place_with_offset(self, objects: List[Dict], room: Room,
                         offset_x: float = 0.0, offset_y: float = 0.0,
                         existing_objects: List[Dict] = None) -> List[Dict]:
        """
        Place objects with offset from center.
        
        Args:
            objects: Objects to place
            room: Room to place objects in
            offset_x: X offset from center (meters)
            offset_y: Y offset from center (meters)
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
        
        placements = []
        center_x = 0.0 + offset_x
        center_y = 0.0 + offset_y
        
        for obj in objects:
            x = center_x
            y = center_y
            yaw = 0.0
            
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
        
        logger.info(f"Center placement with offset ({offset_x}, {offset_y}): "
                   f"positioned {len(placements)} objects")
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
    
    def _build_room_bounds(self, room: Room) -> Dict:
        """Build room bounds dictionary."""
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
