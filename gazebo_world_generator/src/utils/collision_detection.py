#!/usr/bin/env python3
"""
Unified Collision Detection Module

Provides shared collision detection, validation, and grid positioning logic
used by both the World Generator (initial placement) and World Refiner (modifications).

This module ensures consistent behavior across the entire system for:
- Collision detection between objects
- Room boundary validation
- Doorway clearance checking
- Grid-based collision-free positioning
- Wall alignment with proper orientation
"""

import logging
import math
from typing import List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class CollisionDetector:
    """
    Unified collision detection and validation for object placement.
    
    Handles:
    - Collision detection between objects
    - Room boundary validation
    - Doorway clearance
    - Grid positioning with collision avoidance
    - Wall alignment
    """
    
    def __init__(self, world_metadata: Dict = None):
        """
        Initialize collision detector.
        
        Args:
            world_metadata: Metadata dict with door_positions, room_bounds, etc.
        """
        self.world_metadata = world_metadata or {}
        self.collision_margin = 0.15  # Minimum clearance between objects (meters)
        self.room_margin = 0.3  # Clearance from walls (meters)
    
    def validate_position(self, position: Tuple[float, float, float], 
                         dimensions: Tuple[float, float, float],
                         existing_objects: List[Dict],
                         room_bounds: Optional[Dict] = None) -> bool:
        """
        Validate that a position is collision-free, within bounds, and clear of doorways.
        
        Args:
            position: (x, y, yaw) position to validate
            dimensions: (length, width, height) of object
            existing_objects: List of dicts with 'position', 'dimensions', 'name'
            room_bounds: Optional dict with min_x, max_x, min_y, max_y
            
        Returns:
            bool: True if position is valid
        """
        x, y, yaw = position
        obj_length, obj_width, obj_height = dimensions
        
        # Check room bounds (must include object extent)
        if room_bounds:
            if not self._check_room_bounds_with_extent(x, y, dimensions, room_bounds):
                logger.debug(f"Position ({x:.2f}, {y:.2f}) with dims ({obj_length:.2f}x{obj_width:.2f}) outside room bounds")
                return False
        
        # Check collision with existing objects
        if not self._check_collisions(position, dimensions, existing_objects):
            logger.debug(f"Position ({x:.2f}, {y:.2f}) has collision")
            return False
        
        # Check doorway clearance
        if not self._check_doorway_clearance(x, y, dimensions):
            logger.debug(f"Position ({x:.2f}, {y:.2f}) blocks doorway")
            return False
        
        return True
    
    def validate_position_basic(self, position: Tuple[float, float, float],
                               dimensions: Tuple[float, float, float],
                               room_bounds: Optional[Dict] = None) -> bool:
        """
        Basic validation without collision checking (for grouped objects).
        
        Used when objects should be close together (e.g., desk-chair pairs).
        Only checks room bounds and doorway clearance.
        
        Args:
            position: (x, y, yaw) position to validate
            dimensions: (length, width, height) of object
            room_bounds: Optional room bounds
            
        Returns:
            bool: True if position is valid
        """
        x, y, yaw = position
        
        # Check room bounds (must include object extent)
        if room_bounds:
            if not self._check_room_bounds_with_extent(x, y, dimensions, room_bounds):
                logger.warning(f"Basic validation failed: position ({x:.2f}, {y:.2f}) with dims {dimensions[:2]} outside room bounds")
                return False
        
        # Check doorway clearance
        if not self._check_doorway_clearance(x, y, dimensions):
            logger.warning(f"Basic validation failed: position ({x:.2f}, {y:.2f}) blocks doorway")
            return False
        
        return True
    
    def _check_room_bounds_with_extent(self, x: float, y: float, 
                                      dimensions: Tuple[float, float, float],
                                      room_bounds: Dict) -> bool:
        """
        Check if object's full extent (bounding box) is within room boundaries with margin.
        
        Args:
            x, y: Center position of object
            dimensions: (length, width, height) of object
            room_bounds: Dict with min_x, max_x, min_y, max_y
            
        Returns:
            bool: True if object fits completely within room bounds
        """
        margin = self.room_margin
        obj_length, obj_width, _ = dimensions
        
        # Calculate object's bounding box edges
        obj_min_x = x - obj_length / 2
        obj_max_x = x + obj_length / 2
        obj_min_y = y - obj_width / 2
        obj_max_y = y + obj_width / 2
        
        # Check if bounding box is within room bounds with margin
        fits_x = (room_bounds['min_x'] + margin <= obj_min_x and 
                 obj_max_x <= room_bounds['max_x'] - margin)
        fits_y = (room_bounds['min_y'] + margin <= obj_min_y and 
                 obj_max_y <= room_bounds['max_y'] - margin)
        
        if not (fits_x and fits_y):
            logger.debug(f"Object extent check: x=[{obj_min_x:.2f}, {obj_max_x:.2f}] vs room=[{room_bounds['min_x']+margin:.2f}, {room_bounds['max_x']-margin:.2f}], "
                        f"y=[{obj_min_y:.2f}, {obj_max_y:.2f}] vs room=[{room_bounds['min_y']+margin:.2f}, {room_bounds['max_y']-margin:.2f}]")
        
        return fits_x and fits_y
    
    def _check_room_bounds(self, x: float, y: float, room_bounds: Dict) -> bool:
        """
        DEPRECATED: Use _check_room_bounds_with_extent instead.
        Check if position is within room boundaries with margin.
        """
        margin = self.room_margin
        return (room_bounds['min_x'] + margin <= x <= room_bounds['max_x'] - margin and
                room_bounds['min_y'] + margin <= y <= room_bounds['max_y'] - margin)
    
    def _check_collisions(self, position: Tuple[float, float, float],
                         dimensions: Tuple[float, float, float],
                         existing_objects: List[Dict]) -> bool:
        """
        Check for collisions with existing objects.
        
        Args:
            position: (x, y, yaw) position to check
            dimensions: (length, width, height) of object
            existing_objects: List of existing objects
            
        Returns:
            bool: True if no collision
        """
        x, y, yaw = position
        obj_length, obj_width, obj_height = dimensions
        
        obj_half_length = obj_length / 2
        obj_half_width = obj_width / 2
        
        for existing in existing_objects:
            ex_pos = existing['position']
            ex_dims = existing['dimensions']
            
            ex_half_length = ex_dims[0] / 2
            ex_half_width = ex_dims[1] / 2
            
            dx = abs(x - ex_pos[0])
            dy = abs(y - ex_pos[1])
            
            required_dist_x = obj_half_length + ex_half_length + self.collision_margin
            required_dist_y = obj_half_width + ex_half_width + self.collision_margin
            
            if dx < required_dist_x and dy < required_dist_y:
                logger.debug(f"Collision detected with {existing['name']}")
                return False
        
        return True
    
    def _check_doorway_clearance(self, x: float, y: float, 
                                dimensions: Tuple[float, float, float]) -> bool:
        """
        Check if position maintains clearance from doorways.
        
        Args:
            x, y: Position coordinates
            dimensions: Object dimensions
            
        Returns:
            bool: True if doorway clearance is maintained
        """
        door_positions = self.world_metadata.get('door_positions', [])
        obj_length, obj_width, _ = dimensions
        obj_extent = max(obj_length, obj_width) / 2
        
        for door in door_positions:
            door_x, door_y = door['x'], door['y']
            clearance_radius = door['clearance_radius']
            
            distance = math.sqrt((x - door_x)**2 + (y - door_y)**2)
            
            if distance < (clearance_radius + obj_extent):
                logger.debug(f"Doorway clearance violation near {door['name']}")
                return False
        
        return True
    
    def calculate_grid_positions(self, objects: List[Dict], position_desc: str,
                                room_bounds: Dict,
                                existing_objects: List[Dict] = None) -> List[Tuple[float, float, float]]:
        """
        Calculate collision-free positions for multiple objects in a grid/row layout.
        
        This method handles:
        - Grid-based layouts with configurable spacing
        - Wall alignment (north/south/east/west) with proper orientation
        - Collision avoidance through position validation
        - Automatic fallback for crowded scenarios
        
        Args:
            objects: List of dicts with 'name', 'dimensions' keys
            position_desc: Description (e.g., "center", "against north wall", "in rows")
            room_bounds: Room boundary dict (min_x, max_x, min_y, max_y, center_x, center_y, width, length)
            existing_objects: List of existing objects to avoid (optional, defaults to empty)
            
        Returns:
            List of (x, y, yaw) positions
        """
        if existing_objects is None:
            existing_objects = []
        
        # Make a copy to track placements
        all_objects = existing_objects.copy()
        
        logger.info(f"Grid positioning: {len(objects)} objects to place, "
                   f"{len(existing_objects)} existing objects to avoid")
        
        center_x = room_bounds.get('center_x', 0.0)
        center_y = room_bounds.get('center_y', 0.0)
        width = room_bounds.get('width', 10.0)
        length = room_bounds.get('length', 10.0)
        
        # Calculate available space
        available_width = width - (2 * self.room_margin)
        available_length = length - (2 * self.room_margin)
        
        positions = []
        hint_lower = position_desc.lower()
        
        # Calculate grid dimensions
        num_objects = len(objects)
        cols = math.ceil(math.sqrt(num_objects))
        rows = math.ceil(num_objects / cols)
        
        # Calculate spacing based on largest object
        max_width = max(obj['dimensions'][0] for obj in objects)
        max_length = max(obj['dimensions'][1] for obj in objects)
        
        # Determine layout pattern and spacing
        if 'row' in hint_lower or 'line' in hint_lower:
            spacing_x = max_width + 1.5  # Generous spacing for rows
            spacing_y = max_length + 1.5
        else:
            spacing_x = max_width + 1.0  # Standard spacing
            spacing_y = max_length + 1.0
        
        # Calculate grid total size
        grid_width = (cols - 1) * spacing_x + max_width
        grid_length = (rows - 1) * spacing_y + max_length
        
        # Check if grid fits in available space
        if grid_width > available_width or grid_length > available_length:
            logger.warning(f"Grid too large ({grid_width:.1f}x{grid_length:.1f}m) "
                         f"for room ({available_width:.1f}x{available_length:.1f}m), reducing spacing")
            # Reduce spacing to fit
            spacing_x = min(spacing_x, (available_width - max_width) / max(1, cols - 1))
            spacing_y = min(spacing_y, (available_length - max_length) / max(1, rows - 1))
            spacing_x = max(spacing_x, max_width + 0.3)  # Minimum spacing
            spacing_y = max(spacing_y, max_length + 0.3)
        
        # Calculate starting position and orientation based on description
        min_x = room_bounds.get('min_x', center_x - width/2)
        max_x = room_bounds.get('max_x', center_x + width/2)
        min_y = room_bounds.get('min_y', center_y - length/2)
        max_y = room_bounds.get('max_y', center_y + length/2)
        
        start_x, start_y, default_yaw = self._calculate_wall_start_position(
            hint_lower, center_x, center_y, min_x, max_x, min_y, max_y,
            cols, rows, spacing_x, spacing_y, max_length
        )
        
        # Generate and validate positions
        max_attempts_per_object = 20
        collision_count = 0
        
        for idx, obj in enumerate(objects):
            col = idx % cols
            row = idx // cols
            obj_dims = obj['dimensions']
            
            # Try initial grid position
            base_x = start_x + col * spacing_x
            base_y = start_y + row * spacing_y
            yaw = default_yaw
            
            position = (base_x, base_y, yaw)
            
            # Validate position (check collisions and bounds)
            if self.validate_position(position, obj_dims, all_objects, room_bounds):
                positions.append(position)
                # Add to tracking so subsequent objects avoid this one
                all_objects.append({
                    'name': obj['name'],
                    'position': position,
                    'dimensions': obj_dims
                })
            else:
                # Grid position has collision, try nearby positions
                collision_count += 1
                found_valid = False
                
                for attempt in range(max_attempts_per_object):
                    # Try positions in a spiral pattern around the grid position
                    offset_radius = 0.3 * (attempt + 1)
                    angle = (2 * math.pi * attempt) / max_attempts_per_object
                    
                    test_x = base_x + offset_radius * math.cos(angle)
                    test_y = base_y + offset_radius * math.sin(angle)
                    test_pos = (test_x, test_y, yaw)
                    
                    if self.validate_position(test_pos, obj_dims, all_objects, room_bounds):
                        positions.append(test_pos)
                        all_objects.append({
                            'name': obj['name'],
                            'position': test_pos,
                            'dimensions': obj_dims
                        })
                        found_valid = True
                        logger.debug(f"Found valid position for {obj['name']} at offset "
                                   f"{offset_radius:.1f}m after {attempt+1} attempts")
                        break
                
                if not found_valid:
                    # Last resort: use the grid position anyway with warning
                    logger.warning(f"Could not find collision-free position for {obj['name']}, "
                                 f"using grid position ({base_x:.2f}, {base_y:.2f})")
                    positions.append(position)
                    all_objects.append({
                        'name': obj['name'],
                        'position': position,
                        'dimensions': obj_dims
                    })
        
        if collision_count > 0:
            logger.info(f"Grid positioning: {collision_count}/{num_objects} positions "
                       f"had initial collisions, resolved with offsets")
        
        return positions
    
    def _calculate_wall_start_position(self, hint_lower: str,
                                       center_x: float, center_y: float,
                                       min_x: float, max_x: float,
                                       min_y: float, max_y: float,
                                       cols: int, rows: int,
                                       spacing_x: float, spacing_y: float,
                                       max_length: float) -> Tuple[float, float, float]:
        """
        Calculate starting position and orientation for grid based on wall/position hint.
        
        Args:
            hint_lower: Lowercase position description
            center_x, center_y: Room center
            min_x, max_x, min_y, max_y: Room boundaries
            cols, rows: Grid dimensions
            spacing_x, spacing_y: Grid spacing
            max_length: Maximum object length
            
        Returns:
            Tuple of (start_x, start_y, default_yaw)
        """
        default_yaw = 0.0
        
        if 'wall' in hint_lower or 'edge' in hint_lower:
            # Position along a wall
            if 'north' in hint_lower or 'top' in hint_lower or 'back' in hint_lower:
                # Along north wall (max_y)
                start_x = center_x - (cols - 1) * spacing_x / 2
                start_y = max_y - self.room_margin - max_length / 2
                default_yaw = math.pi  # Face into room
            elif 'south' in hint_lower or 'bottom' in hint_lower or 'front' in hint_lower:
                # Along south wall (min_y)
                start_x = center_x - (cols - 1) * spacing_x / 2
                start_y = min_y + self.room_margin + max_length / 2
                default_yaw = 0.0  # Face into room
            elif 'east' in hint_lower or 'right' in hint_lower:
                # Along east wall (max_x)
                start_x = max_x - self.room_margin - max_length / 2
                start_y = center_y - (cols - 1) * spacing_y / 2
                default_yaw = -math.pi / 2  # Face into room
            elif 'west' in hint_lower or 'left' in hint_lower:
                # Along west wall (min_x)
                start_x = min_x + self.room_margin + max_length / 2
                start_y = center_y - (cols - 1) * spacing_y / 2
                default_yaw = math.pi / 2  # Face into room
            else:
                # Default: along south wall (most common for shelving)
                start_x = center_x - (cols - 1) * spacing_x / 2
                start_y = min_y + self.room_margin + max_length / 2
                default_yaw = 0.0
                logger.info("Wall positioning: no specific wall mentioned, using south wall")
        elif 'center' in hint_lower or 'middle' in hint_lower:
            # Center the grid in the room
            start_x = center_x - (cols - 1) * spacing_x / 2
            start_y = center_y - (rows - 1) * spacing_y / 2
        else:
            # Default: center the grid
            start_x = center_x - (cols - 1) * spacing_x / 2
            start_y = center_y - (rows - 1) * spacing_y / 2
        
        return start_x, start_y, default_yaw
    
    def get_existing_object_positions(self, world_elem: ET.Element,
                                     room_bounds: Optional[Dict] = None) -> List[Dict]:
        """
        Extract positions and dimensions of all existing objects for collision detection.
        
        Args:
            world_elem: World XML element
            room_bounds: Optional room bounds to filter objects (only return objects in this room)
            
        Returns:
            List of dicts with 'name', 'position', 'dimensions' keys
        """
        from ..utils.sdf_parser import SDFDimensionExtractor
        
        existing = []
        sdf_extractor = SDFDimensionExtractor()
        
        try:
            # Get all model and include elements
            for model in world_elem.findall('.//model') + world_elem.findall('.//include'):
                name_elem = model.find('name')
                if name_elem is None:
                    continue
                
                name_text = name_elem.text
                
                # Skip walls, doors, ground, sun, and materials/scripts
                name_lower = name_text.lower()
                if any(skip in name_lower for skip in ['wall', 'door', 'ground', 'sun', 'gazebo/', 'script:']):
                    continue
                
                # Skip if name contains slashes (materials/scripts)
                if '/' in name_text:
                    continue
                
                # Get pose
                pose = model.find('pose')
                if pose is not None and pose.text:
                    coords = list(map(float, pose.text.split()))
                    position = (coords[0], coords[1], coords[5] if len(coords) > 5 else 0.0)
                else:
                    continue
                
                # If room_bounds specified, filter objects to only those in this room
                if room_bounds:
                    x, y = position[0], position[1]
                    # Add margin to room bounds for objects near edges
                    margin = 1.5
                    if not (room_bounds['min_x'] - margin <= x <= room_bounds['max_x'] + margin and
                            room_bounds['min_y'] - margin <= y <= room_bounds['max_y'] + margin):
                        continue
                
                # Extract model URI
                model_uri = ""
                uri_elem = model.find('uri')
                if uri_elem is not None and uri_elem.text:
                    model_uri = uri_elem.text
                
                # Extract dimensions (use default if extraction fails)
                try:
                    if model_uri:
                        dimensions = sdf_extractor.extract_model_dimensions(model_uri)
                        if dimensions is None:
                            # SDF file not found or dimensions could not be extracted
                            dimensions = (1.0, 1.0, 1.5)  # Default
                    else:
                        dimensions = (1.0, 1.0, 1.5)  # Default
                except Exception as e:
                    logger.debug(f"Could not extract dimensions for {name_text}: {e}")
                    dimensions = (1.0, 1.0, 1.5)  # Default
                
                existing.append({
                    'name': name_text,
                    'position': position,
                    'dimensions': dimensions
                })
        
        except Exception as e:
            logger.warning(f"Error extracting existing objects: {e}")
        
        return existing
