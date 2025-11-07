"""
Core data structures for the Gazebo World Generator.

Defines Room, GazeboModel, and PlacementConstraint classes to represent
the world layout, objects, and placement rules.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Room:
    """Represents a room or space in the world"""
    name: str
    type: str  # office, living_room, warehouse, outdoor, etc.
    dimensions: Dict[str, float]  # width, length, height
    position: Dict[str, float]  # x, y, z
    walls: bool = True
    objects: List['GazeboModel'] = field(default_factory=list)
    object_requests: List[Dict] = field(default_factory=list)
    shape: str = "rectangle"
    rotation: float = 0.0  

    connections: Dict[str, str] = field(default_factory=dict)
    doorways: List[Dict[str, float]] = field(default_factory=list)  # List of {side, x, y, width}


@dataclass
class GazeboModel:
    """Represents a Gazebo model with its properties"""
    name: str
    model_path: str  # Full path or URI
    category: str
    pose: Dict[str, float]  # x, y, z, roll, pitch, yaw
    room: Optional[str] = None
    scale: Optional[Dict[str, float]] = None
    static: bool = False
    natural_height: float = 0.0  # Height at which object naturally sits
    size: Optional[List[float]] = None  # For inline geometry like walls [width, height, depth]


@dataclass
class PlacementConstraint:
    """Constraints for placing objects naturally"""
    near_wall: bool = False
    center: bool = False
    corner: bool = False
    on_surface: Optional[str] = None  # e.g., "table", "desk"
    min_distance_from_others: float = 0.5
    orientation_to_room: bool = True  # Face into room
