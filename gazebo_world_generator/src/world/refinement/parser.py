#!/usr/bin/env python3
"""
World State Parser
Extracts and parses world state metadata from SDF/XML files.
"""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

logger = logging.getLogger(__name__)


class WorldStateParser:
    """
    Parses SDF/XML world files and extracts comprehensive metadata.
    
    This class is responsible for reading the world state and returning
    structured information about rooms, objects, walls, doors, etc.
    """
    
    def __init__(self, world_tree: ET.ElementTree):
        """
        Initialize the parser with a world XML tree.
        
        Args:
            world_tree: Parsed ElementTree of the world SDF file
        """
        self.world_tree = world_tree
    
    def extract_metadata(self) -> Dict:
        """
        Extract comprehensive metadata from the current world.
        
        Returns:
            dict: Metadata including room info, object counts, types, positions, etc.
        """
        if not self.world_tree:
            return {}
        
        root = self.world_tree.getroot()
        world = root.find('.//world')
        
        if world is None:
            return {}
        
        # Categorize models (both <model> tags and <include> tags)
        models = world.findall('.//model')
        includes = world.findall('.//include')
        
        walls = []
        doors = []
        furniture = []
        room_identifiers = set()
        object_types = {}
        
        # Process explicit <model> tags
        for model in models:
            name = model.get('name', '')
            name_lower = name.lower()
            
            # Categorize by type
            if 'wall' in name_lower:
                walls.append(name)
                room_name = name.split('_wall')[0] if '_wall' in name_lower else name
                room_identifiers.add(room_name)
            elif 'door' in name_lower:
                doors.append(name)
            else:
                furniture.append(name)
                # Count object types
                obj_type = self.extract_object_type(name)
                object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        # Process <include> tags (furniture models)
        for include in includes:
            name_elem = include.find('name')
            uri_elem = include.find('uri')
            
            if name_elem is not None and name_elem.text:
                name = name_elem.text
                name_lower = name.lower()
                
                # Skip ground plane and sun
                if 'ground' in name_lower or 'sun' in name_lower:
                    continue
                
                # Skip walls and doors if they appear as includes
                if 'wall' in name_lower:
                    walls.append(name)
                    room_name = name.split('_wall')[0] if '_wall' in name_lower else name
                    room_identifiers.add(room_name)
                elif 'door' in name_lower:
                    doors.append(name)
                else:
                    furniture.append(name)
                    # Count object types
                    obj_type = self.extract_object_type(name)
                    object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        # Calculate room bounds from walls
        room_bounds = self.calculate_room_bounds(walls, world)
        
        # Extract door positions for doorway clearance
        door_positions = self.extract_door_positions(doors, world)
        
        logger.info(f"Extracted {len(door_positions)} door(s) for doorway clearance checking")
        for door in door_positions:
            logger.info(f"  Door '{door['name']}' at ({door['x']:.2f}, {door['y']:.2f}), clearance radius: {door['clearance_radius']}m")
        
        return {
            'rooms': len(room_identifiers),
            'room_names': sorted(list(room_identifiers)),
            'objects': len(furniture),
            'object_names': furniture,
            'object_types': object_types,
            'walls': len(walls),
            'doors': len(doors),
            'door_positions': door_positions,
            'total_models': len(models) + len(includes),
            'room_bounds': room_bounds
        }
    
    def extract_object_type(self, model_name: str) -> str:
        """
        Extract the object type from a model name.
        
        Examples:
            "Office_Desk_3" -> "desk"
            "OfficeChairBlack_1" -> "chair"
            "bookshelf_0" -> "bookshelf"
        
        Args:
            model_name: Name of the model
            
        Returns:
            str: Extracted object type
        """
        # Remove trailing numbers
        name = re.sub(r'_\d+$', '', model_name)
        
        # Extract type keywords
        name_lower = name.lower()
        
        if 'desk' in name_lower:
            return 'desk'
        elif 'chair' in name_lower:
            return 'chair'
        elif 'bookshelf' in name_lower:
            return 'bookshelf'
        elif 'shelf' in name_lower:
            return 'shelf'
        elif 'table' in name_lower:
            return 'table'
        elif 'wardrobe' in name_lower or 'cabinet' in name_lower:
            return 'storage'
        else:
            # Return the base name
            return name.split('_')[0].lower()
    
    def calculate_room_bounds(self, wall_names: List[str], world_element: ET.Element) -> Dict:
        """
        Calculate room boundaries from wall positions.
        
        Args:
            wall_names: List of wall model names
            world_element: The world XML element
            
        Returns:
            dict: Room bounds {room_name: {'min_x': float, 'max_x': float, ...}}
        """
        room_bounds = {}
        
        for wall_name in wall_names:
            # Find the wall model
            wall_model = world_element.find(f".//model[@name='{wall_name}']")
            if wall_model is None:
                continue
            
            # Extract room name
            room_name = wall_name.split('_wall')[0] if '_wall' in wall_name.lower() else 'main'
            
            # Get wall pose
            pose_elem = wall_model.find('.//pose')
            if pose_elem is not None and pose_elem.text:
                try:
                    pose = [float(x) for x in pose_elem.text.split()]
                    x, y = pose[0], pose[1]
                    
                    # Update room bounds
                    if room_name not in room_bounds:
                        room_bounds[room_name] = {
                            'min_x': x, 'max_x': x,
                            'min_y': y, 'max_y': y
                        }
                    else:
                        bounds = room_bounds[room_name]
                        bounds['min_x'] = min(bounds['min_x'], x)
                        bounds['max_x'] = max(bounds['max_x'], x)
                        bounds['min_y'] = min(bounds['min_y'], y)
                        bounds['max_y'] = max(bounds['max_y'], y)
                except (ValueError, IndexError):
                    continue
        
        # Calculate dimensions for each room
        for room_name, bounds in room_bounds.items():
            bounds['width'] = abs(bounds['max_x'] - bounds['min_x'])
            bounds['length'] = abs(bounds['max_y'] - bounds['min_y'])
            bounds['center_x'] = (bounds['min_x'] + bounds['max_x']) / 2
            bounds['center_y'] = (bounds['min_y'] + bounds['max_y']) / 2
            
            # Expand bounds inward to get actual interior space
            wall_thickness = 0.15
            bounds['min_x'] += wall_thickness
            bounds['max_x'] -= wall_thickness
            bounds['min_y'] += wall_thickness
            bounds['max_y'] -= wall_thickness
            
            logger.debug(f"Room '{room_name}' bounds: ({bounds['min_x']:.2f}, {bounds['min_y']:.2f}) to ({bounds['max_x']:.2f}, {bounds['max_y']:.2f}), size: {bounds['width']:.2f}m x {bounds['length']:.2f}m")
        
        return room_bounds
    
    def extract_door_positions(self, door_names: List[str], world_element: ET.Element) -> List[Dict]:
        """
        Extract door positions and dimensions for doorway clearance checking.
        
        Args:
            door_names: List of door model names
            world_element: The world XML element
            
        Returns:
            list: List of dicts with door position and clearance zone info
        """
        door_positions = []
        
        for door_name in door_names:
            # Find the door model
            door_model = world_element.find(f".//model[@name='{door_name}']")
            if door_model is None:
                # Try as include
                for include in world_element.findall('.//include'):
                    name_elem = include.find('name')
                    if name_elem is not None and name_elem.text == door_name:
                        door_model = include
                        break
            
            if door_model is None:
                continue
            
            # Get door pose
            pose_elem = door_model.find('.//pose')
            if pose_elem is not None and pose_elem.text:
                try:
                    pose = [float(x) for x in pose_elem.text.split()]
                    x, y = pose[0], pose[1]
                    yaw = pose[5] if len(pose) > 5 else 0.0
                    
                    # Doorway clearance zone: 1.0m radius around door (reduced from 1.5m)
                    door_positions.append({
                        'name': door_name,
                        'x': x,
                        'y': y,
                        'yaw': yaw,
                        'clearance_radius': 1.0  # meters
                    })
                    logger.debug(f"Door '{door_name}' at ({x:.2f}, {y:.2f})")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse door pose for {door_name}: {e}")
                    continue
        
        return door_positions
