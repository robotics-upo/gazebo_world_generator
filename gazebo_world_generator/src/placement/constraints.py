"""
Placement Constraint Validation

Validates placement constraints for objects based on PlacementConstraint rules.
Ensures objects satisfy spatial requirements like near_wall, center, corner, etc.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
from gazebo_world_generator.src.core.data_models import PlacementConstraint, Room

logger = logging.getLogger(__name__)


class ConstraintValidator:
    """
    Validator for placement constraints.
    
    Checks that object placements satisfy constraint rules such as:
    - near_wall: Object should be close to a wall
    - center: Object should be in the center of the room
    - corner: Object should be in a corner
    - on_surface: Object should be on another object
    - min_distance_from_others: Minimum spacing from other objects
    - orientation_to_room: Object should face into the room
    """
    
    def __init__(self, wall_threshold: float = 1.0, 
                 corner_threshold: float = 1.5,
                 center_threshold: float = 2.0):
        """
        Initialize constraint validator.
        
        Args:
            wall_threshold: Maximum distance from wall to be "near wall" (meters)
            corner_threshold: Maximum distance from corner to be "in corner" (meters)
            center_threshold: Maximum distance from center to be "centered" (meters)
        """
        self.wall_threshold = wall_threshold
        self.corner_threshold = corner_threshold
        self.center_threshold = center_threshold
    
    def validate_constraint(self, pose: Dict[str, float], 
                           constraint: PlacementConstraint,
                           room: Room,
                           existing_objects: Optional[List[Dict]] = None) -> Tuple[bool, str]:
        """
        Validate a single placement against a constraint.
        
        Args:
            pose: Object pose {'x': float, 'y': float, 'yaw': float, ...}
            constraint: Placement constraint to validate
            room: Room the object is in
            existing_objects: Other objects in the room (for distance checks)
            
        Returns:
            Tuple of (is_valid, reason) where reason explains validation failure
        """
        x = pose['x']
        y = pose['y']
        yaw = pose.get('yaw', 0.0)
        
        # Get room bounds
        half_w = room.dimensions['width'] / 2
        half_l = room.dimensions['length'] / 2
        
        # Check near_wall constraint
        if constraint.near_wall:
            is_near_wall, reason = self._check_near_wall(x, y, half_w, half_l)
            if not is_near_wall:
                return False, reason
        
        # Check center constraint
        if constraint.center:
            is_centered, reason = self._check_center(x, y, half_w, half_l)
            if not is_centered:
                return False, reason
        
        # Check corner constraint
        if constraint.corner:
            is_in_corner, reason = self._check_corner(x, y, half_w, half_l)
            if not is_in_corner:
                return False, reason
        
        # Check minimum distance from others
        if existing_objects and constraint.min_distance_from_others > 0:
            is_spaced, reason = self._check_min_distance(
                x, y, constraint.min_distance_from_others, existing_objects
            )
            if not is_spaced:
                return False, reason
        
        # Check orientation to room
        if constraint.orientation_to_room:
            is_oriented, reason = self._check_orientation_to_room(
                x, y, yaw, half_w, half_l
            )
            if not is_oriented:
                return False, reason
        
        return True, "All constraints satisfied"
    
    def _check_near_wall(self, x: float, y: float, 
                        half_w: float, half_l: float) -> Tuple[bool, str]:
        """Check if position is near a wall."""
        dist_to_west = abs(x - (-half_w))
        dist_to_east = abs(x - half_w)
        dist_to_south = abs(y - (-half_l))
        dist_to_north = abs(y - half_l)
        
        min_dist = min(dist_to_west, dist_to_east, dist_to_south, dist_to_north)
        
        if min_dist <= self.wall_threshold:
            return True, ""
        else:
            return False, f"Not near wall (closest wall is {min_dist:.2f}m away, threshold={self.wall_threshold}m)"
    
    def _check_center(self, x: float, y: float,
                     half_w: float, half_l: float) -> Tuple[bool, str]:
        """Check if position is in center of room."""
        distance_from_center = math.sqrt(x**2 + y**2)
        
        if distance_from_center <= self.center_threshold:
            return True, ""
        else:
            return False, f"Not centered (distance from center is {distance_from_center:.2f}m, threshold={self.center_threshold}m)"
    
    def _check_corner(self, x: float, y: float,
                     half_w: float, half_l: float) -> Tuple[bool, str]:
        """Check if position is in a corner."""
        # Define corner positions
        corners = [
            (-half_w, -half_l),  # Southwest
            (half_w, -half_l),   # Southeast
            (-half_w, half_l),   # Northwest
            (half_w, half_l)     # Northeast
        ]
        
        # Find distance to nearest corner
        min_corner_dist = min(
            math.sqrt((x - cx)**2 + (y - cy)**2)
            for cx, cy in corners
        )
        
        if min_corner_dist <= self.corner_threshold:
            return True, ""
        else:
            return False, f"Not in corner (closest corner is {min_corner_dist:.2f}m away, threshold={self.corner_threshold}m)"
    
    def _check_min_distance(self, x: float, y: float,
                           min_distance: float,
                           existing_objects: List[Dict]) -> Tuple[bool, str]:
        """Check if position maintains minimum distance from other objects."""
        for obj in existing_objects:
            obj_pos = obj.get('position', obj.get('pose', {}))
            obj_x = obj_pos.get('x', 0.0)
            obj_y = obj_pos.get('y', 0.0)
            
            distance = math.sqrt((x - obj_x)**2 + (y - obj_y)**2)
            
            if distance < min_distance:
                obj_name = obj.get('name', 'unknown')
                return False, f"Too close to {obj_name} (distance={distance:.2f}m, min={min_distance}m)"
        
        return True, ""
    
    def _check_orientation_to_room(self, x: float, y: float, yaw: float,
                                   half_w: float, half_l: float) -> Tuple[bool, str]:
        """
        Check if orientation faces into the room.
        
        For objects near walls, they should face away from the wall.
        """
        # Determine which wall is closest
        dist_to_west = abs(x - (-half_w))
        dist_to_east = abs(x - half_w)
        dist_to_south = abs(y - (-half_l))
        dist_to_north = abs(y - half_l)
        
        min_dist = min(dist_to_west, dist_to_east, dist_to_south, dist_to_north)
        
        # Only check orientation if near a wall
        if min_dist > self.wall_threshold:
            return True, ""  # Not near wall, orientation doesn't matter
        
        # Determine expected orientation based on nearest wall
        expected_yaw = None
        wall_name = ""
        
        if min_dist == dist_to_west:
            expected_yaw = math.pi / 2  # Face east (into room)
            wall_name = "west"
        elif min_dist == dist_to_east:
            expected_yaw = -math.pi / 2  # Face west (into room)
            wall_name = "east"
        elif min_dist == dist_to_south:
            expected_yaw = 0.0  # Face north (into room)
            wall_name = "south"
        else:  # north
            expected_yaw = math.pi  # Face south (into room)
            wall_name = "north"
        
        # Normalize yaw to [-pi, pi]
        yaw_normalized = math.atan2(math.sin(yaw), math.cos(yaw))
        expected_yaw_normalized = math.atan2(math.sin(expected_yaw), math.cos(expected_yaw))
        
        # Check if orientation is approximately correct (within 45 degrees)
        angle_diff = abs(yaw_normalized - expected_yaw_normalized)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        tolerance = math.pi / 4  # 45 degrees
        
        if angle_diff <= tolerance:
            return True, ""
        else:
            return False, f"Facing away from room (near {wall_name} wall, yaw={yaw:.2f}, expected≈{expected_yaw:.2f})"
    
    def validate_plan(self, plan: List[Dict], room: Room,
                     constraints: Optional[Dict[str, PlacementConstraint]] = None) -> List[Dict]:
        """
        Validate an entire placement plan against constraints.
        
        Args:
            plan: List of placement dictionaries with 'type', 'pose', etc.
            room: Room the objects are in
            constraints: Optional dict mapping object types to constraints
            
        Returns:
            List of validation results with 'name', 'is_valid', 'reason' keys
        """
        if constraints is None:
            constraints = {}
        
        results = []
        
        # Build list of existing objects for distance checks
        existing_objects = [
            {
                'name': item.get('name', item.get('type', 'unknown')),
                'position': item['pose']
            }
            for item in plan
        ]
        
        for item in plan:
            obj_type = item.get('type', 'unknown')
            pose = item.get('pose', {})
            
            # Get constraint for this object type
            constraint = constraints.get(obj_type)
            
            if constraint:
                # Validate against constraint
                is_valid, reason = self.validate_constraint(
                    pose, constraint, room, existing_objects
                )
                
                results.append({
                    'name': item.get('name', obj_type),
                    'type': obj_type,
                    'is_valid': is_valid,
                    'reason': reason
                })
                
                if not is_valid:
                    logger.warning(f"Constraint validation failed for {obj_type}: {reason}")
            else:
                # No constraint for this type
                results.append({
                    'name': item.get('name', obj_type),
                    'type': obj_type,
                    'is_valid': True,
                    'reason': "No constraints defined"
                })
        
        # Log summary
        valid_count = sum(1 for r in results if r['is_valid'])
        logger.info(f"Constraint validation: {valid_count}/{len(results)} objects satisfy constraints")
        
        return results
