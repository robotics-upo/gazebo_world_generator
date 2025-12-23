"""
Placement Strategies

Provides different strategies for object placement in rooms.
Each strategy implements a specific placement pattern (grid, wall, center, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Dict
from gazebo_world_generator.src.core.data_models import Room


class PlacementStrategy(ABC):
    """
    Base interface for object placement strategies.
    
    Strategies determine how objects are positioned in a room based on
    their types, counts, and the room layout.
    """
    
    @abstractmethod
    def place(self, objects: List[Dict], room: Room, 
              existing_objects: List[Dict] = None) -> List[Dict]:
        """
        Generate placement positions for objects.
        
        Args:
            objects: Objects to place with their types and metadata
                     Each dict should have 'name' and 'dimensions' keys
            room: Room to place objects in
            existing_objects: Already placed objects to avoid (optional)
            
        Returns:
            List of placement dictionaries with 'pose' information:
            [{'name': str, 'pose': {'x': float, 'y': float, 'yaw': float}}]
        """
        pass


# Export strategy interface
__all__ = ['PlacementStrategy']
