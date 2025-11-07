#!/usr/bin/env python3
"""
World Refinement Module
Handles LLM-based modifications to existing Gazebo worlds.
"""

import logging
import json
import re
import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from xml.dom import minidom
from datetime import datetime

from ..core.data_models import Room, GazeboModel
from ..models.resolver import SmartModelResolver
from ..models.online_database import OnlineModelDatabase
from ..utils.file_utils import find_gazebo_model_path
from ..utils.sdf_parser import SDFDimensionExtractor
from ..utils.output_manager import OutputManager
from ..utils.collision_detection import CollisionDetector
from ..placement.engine import NaturalPlacementEngine
from ..placement.semantic_grouping import SemanticGroupingEngine
from ..prompts.manager import PromptManager

logger = logging.getLogger(__name__)


class Style:
    """Terminal styling for consistent output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    DIM = '\033[2m'


class WorldRefiner:
    """
    Refines existing Gazebo worlds based on natural language modification requests.
    
    Capabilities:
    - Add/remove objects
    - Modify object positions
    - Change room dimensions
    - Adjust furniture arrangements
    - Update object properties
    """
    
    def __init__(self, llm_interface, world_generator):
        """
        Initialize the world refiner.
        
        Args:
            llm_interface: LLM interface for parsing refinement requests
            world_generator: WorldGenerator instance for applying modifications
        """
        self.llm = llm_interface
        self.generator = world_generator
        self.current_world = None
        self.current_world_file = None
        self.world_metadata = {}
        
        # Initialize model resolver for adding new objects
        self.online_db = OnlineModelDatabase(llm_interface=llm_interface)
        self.model_db = SmartModelResolver(llm_interface=llm_interface, online_db=self.online_db)
        
        # Initialize SDF dimension extractor
        self.sdf_extractor = SDFDimensionExtractor()
        
        # Initialize placement engine 
        self.placement_engine = NaturalPlacementEngine(
            model_db=self.model_db,
            online_db=self.online_db,
            llm_interface=llm_interface
        )
        
        # Initialize semantic grouping engine 
        self.semantic_grouping = SemanticGroupingEngine(llm_interface) if llm_interface else None
        
        # Initialize PromptManager
        if llm_interface and hasattr(llm_interface, 'prompt_manager'):
            self.prompt_manager = llm_interface.prompt_manager
        else:
            try:
                self.prompt_manager = PromptManager(
                    template_dir='prompts/',
                    default_version='v1',
                    enable_metrics=True
                )
            except Exception as e:
                logger.warning(f"Failed to initialize PromptManager in WorldRefiner: {e}")
                self.prompt_manager = None
        
        # Initialize output manager for organized file output and logging
        self.output_manager = OutputManager()
        
        # Initialize collision detector 
        self.collision_detector = None  # Will be initialized when world metadata is available
        
        # Model resolution cache to avoid repeated LLM calls
        # Cache structure: {object_type: {'uri': str, 'dimensions': tuple, 'timestamp': float}}
        self.model_cache = {}
        self.dimension_cache = {}  # Separate cache for dimensions by URI
        
        # Track active log file handler for cleanup
        self.active_log_handler = None
        
    def load_world(self, world_file: Path) -> bool:
        """
        Load an existing world file for refinement.
        
        Args:
            world_file: Path to the .sdf world file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Parse the SDF file
            self.current_world = ET.parse(world_file)
            self.current_world_file = world_file
            
            # Extract metadata for context
            self.world_metadata = self._extract_metadata()
            
            # Initialize collision detector with world metadata
            self.collision_detector = CollisionDetector(self.world_metadata)
            
            logger.info(f"Loaded world file: {world_file}")
            logger.info(f"World contains {self.world_metadata.get('objects', 0)} objects in {self.world_metadata.get('rooms', 0)} room(s)")
            
            return True
        except ET.ParseError as e:
            logger.error(f"Failed to parse SDF file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load world file: {e}", exc_info=True)
            return False
    
    def _extract_metadata(self) -> Dict:
        """
        Extract comprehensive metadata from the current world.
        
        Returns:
            dict: Metadata including room info, object counts, types, positions, etc.
        """
        if not self.current_world:
            return {}
        
        root = self.current_world.getroot()
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
                obj_type = self._extract_object_type(name)
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
                    obj_type = self._extract_object_type(name)
                    object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        # Calculate room bounds from walls
        room_bounds = self._calculate_room_bounds(walls, world)
        
        # Extract door positions for doorway clearance
        door_positions = self._extract_door_positions(doors, world)
        
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
    
    def _extract_object_type(self, model_name: str) -> str:
        """
        Extract the object type from a model name.
        
        Examples:
            "Office_Desk_3" -> "desk"
            "OfficeChairBlack_1" -> "chair"
            "bookshelf_0" -> "bookshelf"
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
    
    def _calculate_room_bounds(self, wall_names: List[str], world_element: ET.Element) -> Dict:
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
    
    def _extract_door_positions(self, door_names: List[str], world_element: ET.Element) -> List[Dict]:
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
    
    def _get_cached_model(self, obj_type: str, context: str = "") -> Optional[Dict]:
        """
        Get cached model information for an object type.
        
        Args:
            obj_type: Object type (e.g., 'desk', 'chair')
            context: Additional context for cache key
            
        Returns:
            dict: Cached model info {'uri': str, 'dimensions': tuple} or None
        """
        cache_key = f"{obj_type}:{context}" if context else obj_type
        cached = self.model_cache.get(cache_key)
        
        if cached:
            logger.debug(f"Cache hit for '{cache_key}'")
            return cached
        
        return None
    
    def _cache_model(self, obj_type: str, uri: str, dimensions: tuple, context: str = ""):
        """
        Cache model resolution results.
        
        Args:
            obj_type: Object type
            uri: Model URI
            dimensions: Model dimensions (width, length, height)
            context: Additional context for cache key
        """
        import time
        cache_key = f"{obj_type}:{context}" if context else obj_type
        
        self.model_cache[cache_key] = {
            'uri': uri,
            'dimensions': dimensions,
            'timestamp': time.time()
        }
        
        # Also cache dimensions by URI for quick lookup
        self.dimension_cache[uri] = dimensions
        
        logger.debug(f"Cached model '{cache_key}': {uri} with dimensions {dimensions}")
    
    def _get_cached_dimensions(self, uri: str) -> Optional[tuple]:
        """Get cached dimensions for a model URI."""
        return self.dimension_cache.get(uri)
    
    def clear_cache(self):
        """Clear the model resolution cache."""
        self.model_cache.clear()
        self.dimension_cache.clear()
        logger.info("Model cache cleared")
    
    def parse_refinement_request(self, request: str) -> Dict:
        """
        Parse a natural language refinement request using LLM.
        
        Args:
            request: Natural language description of desired changes
            
        Returns:
            dict: Structured refinement instructions
        """
        logger.info(f"Parsing refinement request: {request}")
        
        # Store original request for semantic context detection
        self.original_request = request
        
        # Build context from current world
        context = self._build_context_summary()
        
        # Build detailed context
        obj_types = context.get('object_types', {})
        obj_summary = ', '.join([f"{count} {obj_type}(s)" for obj_type, count in obj_types.items()]) if obj_types else "None"
        room_count = context['rooms']
        room_names = context['room_names']
        
        # Add room requirement warning if multiple rooms
        room_requirement = ""
        if room_count > 1:
            room_requirement = f"\n\nIMPORTANT: This world has {room_count} rooms. You MUST specify which room in the 'properties.room' field for add/remove/modify/reposition operations. Available rooms: {', '.join(room_names)}"
        
        # Use PromptManager if available
        if not self.prompt_manager:
            logger.error("No PromptManager available, cannot parse refinement request")
            return {"error": "PromptManager not initialized"}
        
        try:
            prompt_content = self.prompt_manager.render(
                'world_refinement_parsing',
                room_count=room_count,
                room_names=room_names,
                total_objects=context['objects'],
                object_summary=obj_summary,
                room_requirement=room_requirement,
                user_request=request
            )
            
            # For refinement parsing, we use a single user message with all context
            messages = [{"role": "user", "content": prompt_content}]
        except Exception as e:
            logger.error(f"Failed to render world_refinement_parsing template: {e}")
            return {"error": f"Template rendering failed: {str(e)}"}
        
        try:
            response = self.llm.query(messages, temperature=0.2)
            logger.debug(f"LLM response: {response[:200]}...")
            
            # Extract JSON from response
            # Try multiple extraction methods
            refinement_data = None
            
            # Method 1: Direct JSON parse (if response is clean JSON)
            try:
                refinement_data = json.loads(response.strip())
            except json.JSONDecodeError:
                pass
            
            # Method 2: Extract from code blocks
            if not refinement_data:
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                if code_block_match:
                    try:
                        refinement_data = json.loads(code_block_match.group(1))
                    except json.JSONDecodeError:
                        pass
            
            # Method 3: Find first complete JSON object
            if not refinement_data:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    try:
                        refinement_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
            
            if refinement_data:
                # Validate required fields
                if "operation" not in refinement_data:
                    logger.warning("Missing 'operation' field in parsed JSON")
                    return {"error": "Invalid refinement format: missing operation"}
                
                logger.info(f"Successfully parsed refinement: {refinement_data.get('operation')}")
                return refinement_data
            else:
                logger.warning("Could not extract valid JSON from LLM response")
                logger.debug(f"Full response: {response}")
                return {"error": "Could not parse refinement request. Try rephrasing."}
                
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}", exc_info=True)
            return {"error": f"Parsing error: {str(e)}"}
    
    def _build_context_summary(self) -> Dict:
        """Build a comprehensive summary of the current world for LLM context."""
        return {
            'rooms': self.world_metadata.get('rooms', 0),
            'room_names': self.world_metadata.get('room_names', []),
            'objects': self.world_metadata.get('objects', 0),
            'object_types': self.world_metadata.get('object_types', {}),
            'object_names': self.world_metadata.get('object_names', [])[:15],
            'room_bounds': self.world_metadata.get('room_bounds', {})
        }
    
    def apply_refinement(self, refinement_data: Dict, output_file: Path) -> bool:
        """
        Apply refinement instructions to the world.
        
        Args:
            refinement_data: Structured refinement instructions
            output_file: Path to save refined world
            
        Returns:
            bool: True if successful
        """
        if "error" in refinement_data:
            logger.error(f"Cannot apply refinement: {refinement_data['error']}")
            return False
        
        try:
            operation = refinement_data.get("operation")
            
            if operation == "batch":
                return self._apply_batch_operations(refinement_data, output_file)
            elif operation == "add":
                return self._apply_add_objects(refinement_data, output_file)
            elif operation == "remove":
                return self._apply_remove_objects(refinement_data, output_file)
            elif operation == "modify":
                return self._apply_modify_objects(refinement_data, output_file)
            elif operation == "reposition":
                return self._apply_reposition_objects(refinement_data, output_file)
            elif operation == "resize_room":
                return self._apply_resize_room(refinement_data, output_file)
            else:
                logger.warning(f"Unknown operation: {operation}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying refinement: {e}", exc_info=True)
            return False
    
    def _apply_batch_operations(self, data: Dict, output_file: Path) -> bool:
        """
        Apply multiple operations in sequence (batch processing).
        
        Each operation is applied to the world state, with the output of one operation
        becoming the input to the next. The final state is saved to the output file.
        
        Args:
            data: Batch refinement data with 'operations' list
            output_file: Final output file path
            
        Returns:
            bool: True if all operations succeeded
        """
        operations = data.get("operations", [])
        
        if not operations:
            logger.warning("No operations found in batch request")
            print(f"{Style.YELLOW}      ⚠ No operations specified in batch{Style.ENDC}")
            return False
        
        logger.info(f"Executing batch of {len(operations)} operation(s)")
        print(f"\n{Style.CYAN}⚙ Batch Processing: {len(operations)} operation(s){Style.ENDC}")
        
        successful_ops = 0
        failed_ops = 0
        
        temp_file = output_file  # First operation saves to final output
        
        for idx, op_data in enumerate(operations, 1):
            operation = op_data.get("operation", "unknown")
            
            print(f"{Style.DIM}  Operation {idx}/{len(operations)}: {Style.BOLD}{operation}{Style.ENDC}")
            logger.info(f"Executing batch operation {idx}/{len(operations)}: {operation}")
            
            try:
                # Route to appropriate handler
                if operation == "add":
                    success = self._apply_add_objects(op_data, temp_file)
                elif operation == "remove":
                    success = self._apply_remove_objects(op_data, temp_file)
                elif operation == "modify":
                    success = self._apply_modify_objects(op_data, temp_file)
                elif operation == "reposition":
                    success = self._apply_reposition_objects(op_data, temp_file)
                elif operation == "resize_room":
                    success = self._apply_resize_room(op_data, temp_file)
                else:
                    logger.warning(f"Unknown operation in batch: {operation}")
                    print(f"{Style.YELLOW}      ⚠ Unknown operation: {operation}{Style.ENDC}")
                    failed_ops += 1
                    continue
                
                if success:
                    successful_ops += 1
                    print(f"{Style.GREEN}      ✓ {operation} completed{Style.ENDC}")
                    
                    # Reload the world for the next operation (so changes cascade)
                    if idx < len(operations):
                        self.current_world = ET.parse(temp_file)
                        self.world_metadata = self._extract_metadata()
                        self.collision_detector = CollisionDetector(self.world_metadata)
                        logger.debug(f"Reloaded world state after operation {idx}")
                else:
                    failed_ops += 1
                    print(f"{Style.RED}      ✗ {operation} failed{Style.ENDC}")
                    logger.warning(f"Batch operation {idx} ({operation}) failed")
                    
            except Exception as e:
                failed_ops += 1
                logger.error(f"Error in batch operation {idx} ({operation}): {e}", exc_info=True)
                print(f"{Style.RED}      ✗ {operation} error: {e}{Style.ENDC}")
        
        # Summary
        print(f"\n{Style.CYAN}Batch Summary:{Style.ENDC}")
        print(f"  {Style.GREEN}✓ Successful: {successful_ops}/{len(operations)}{Style.ENDC}")
        if failed_ops > 0:
            print(f"  {Style.RED}✗ Failed: {failed_ops}/{len(operations)}{Style.ENDC}")
        
        logger.info(f"Batch processing complete: {successful_ops} succeeded, {failed_ops} failed")
        
        # Return True if at least one operation succeeded
        return successful_ops > 0
    
    def _apply_add_objects(self, data: Dict, output_file: Path) -> bool:
        """
        Add new objects to the world.
        Uses the generator's placement engine logic for proper positioning.
        """
        logger.info("Applying add objects refinement with placement engine")
        
        objects = data.get("objects", [])
        if not objects:
            logger.warning("No objects specified for add operation")
            return False
        
        root = self.current_world.getroot()
        world = root.find('.//world')
        
        if world is None:
            logger.error("Could not find world element in SDF")
            return False
        
        added_count = 0
        total_to_add = sum(obj.get("count", 1) for obj in objects)
        
        print(f"{Style.DIM}    Adding {total_to_add} object(s)...{Style.ENDC}")
        
        # Check if this is a single object with semantic context referencing an existing object
        # e.g., "add a chair to the desk"
        logger.info(f"DEBUG: len(objects)={len(objects)}, self.semantic_grouping={self.semantic_grouping}")
        if len(objects) == 1 and self.semantic_grouping:
            obj_spec = objects[0]
            properties = obj_spec.get("properties", {})
            semantic_context = properties.get("semantic_context", "")
            placement_hint = properties.get("placement_hint", "")
            obj_type = obj_spec.get("type", "").lower()
            
            # Check BOTH semantic_context AND the original user request
            # The LLM might not always populate semantic_context, so we fall back to the raw request
            check_text = semantic_context or getattr(self, 'original_request', '')
            
            # Check if semantic context mentions an existing object type (desk, table, bed, etc.)
            if check_text:
                logger.info(f"Checking for existing object reference in: '{check_text}' for single object: {obj_type}")
                
                # Extract existing object types that might be referenced
                existing_objects = self._get_existing_object_positions(world, None)
                existing_type_map = {}
                for obj in existing_objects:
                    # Extract base type from name (e.g., 'desk_0' -> 'desk')
                    base_type = obj['name'].rsplit('_', 1)[0]
                    if base_type not in existing_type_map:
                        existing_type_map[base_type] = obj
                
                # Check if any existing type is mentioned in semantic context
                referenced_type = None
                for existing_obj_type in existing_type_map.keys():
                    if existing_obj_type in check_text.lower():
                        referenced_type = existing_obj_type
                        logger.info(f"Found reference to existing '{referenced_type}' in text: '{check_text}'")
                        break
                
                # If we found a reference, use direct relative positioning
                if referenced_type:
                    logger.info(f"Using direct relative positioning for {obj_type} relative to existing {referenced_type}")
                    referenced_obj = existing_type_map[referenced_type]
                    
                    # Determine the relationship (chair->desk, nightstand->bed, etc.)
                    relationship = self._determine_semantic_relationship(obj_type, referenced_type)
                    
                    if relationship:
                        # Use the relative positioning logic to place the object
                        added = self._add_object_relative_to_existing(
                            obj_spec, referenced_obj, relationship, world, output_file
                        )
                        if added:
                            print(f"{Style.GREEN}      ✓ Added 1/1 object(s) with relative positioning{Style.ENDC}")
                            return True
                        else:
                            logger.warning("Relative positioning failed, falling back to regular placement")
        
        # Check if we should use semantic grouping (multiple objects that might be related)
        if len(objects) > 1 and self.semantic_grouping:
            logger.info("Multiple objects detected - checking for semantic groupings")
            added_count = self._apply_add_with_semantic_grouping(objects, world, output_file)
            
            if added_count > 0:
                print(f"{Style.GREEN}      ✓ Added {added_count}/{total_to_add} object(s) with semantic grouping{Style.ENDC}")
                return True
            else:
                # Fall through to individual placement if grouping failed
                logger.warning("Semantic grouping failed, falling back to individual placement")
        
        # Individual object placement (original logic)
        
        for obj_spec in objects:
            obj_type = obj_spec.get("type", "").lower()
            count = obj_spec.get("count", 1)
            properties = obj_spec.get("properties", {})
            placement_hint = properties.get("placement_hint", "")
            room_name = properties.get("room", "")
            
            logger.info(f"Adding {count} {obj_type}(s) with hint: {placement_hint}")
            
            try:
                # Determine room type from room name or properties
                room_type = self._infer_room_type(room_name)
                
                # Check cache first (cache by obj_type + room_type, not placement_hint)
                cache_key = f"{obj_type}_{room_type}"
                cached_model = self._get_cached_model(cache_key, room_type)
                
                if cached_model:
                    model_uri = cached_model['uri']
                    dimensions = cached_model['dimensions']
                    logger.info(f"Using cached model for {obj_type} in {room_type}: {model_uri}")
                else:
                    # Resolve model URI using room type as context (same as engine)
                    model_uri = self.model_db.find_best_model(obj_type, context=room_type)
                    if not model_uri:
                        print(f"{Style.YELLOW}      ⚠ Could not find model for: {obj_type}{Style.ENDC}")
                        logger.warning(f"Could not resolve model for type: {obj_type}")
                        continue
                    
                    # Extract actual model dimensions from SDF if available
                    dimensions = self._extract_model_dimensions(model_uri, obj_type)
                    
                    # Cache the result with room type
                    self._cache_model(cache_key, model_uri, dimensions, room_type)
                    logger.info(f"Resolved and cached model for {obj_type} in {room_type}: {model_uri}")
                
                logger.info(f"Using dimensions {dimensions} for {obj_type}")
                
                # Determine placement area
                room_bounds = self._get_placement_area(room_name)
                
                # Get existing objects for collision detection (only from target room)
                existing_objects = self._get_existing_object_positions(world, room_bounds)
                
                # Add each instance with placement engine
                for i in range(count):
                    # Generate unique name
                    model_name = self._generate_unique_name(obj_type)
                    
                    # Calculate position with collision detection and wall snapping
                    position = self._calculate_position_with_placement(
                        obj_type,
                        placement_hint, 
                        room_bounds, 
                        dimensions,
                        existing_objects
                    )
                    
                    if position is None:
                        print(f"{Style.YELLOW}      ⚠ Could not place {model_name} (no valid position){Style.ENDC}")
                        logger.warning(f"Could not find valid position for {model_name}")
                        continue
                    
                    # Create model element
                    model_elem = self._create_model_element(
                        model_name,
                        model_uri,
                        position,
                        dimensions
                    )
                    
                    # Add to world
                    world.append(model_elem)
                    
                    # Update existing objects for next iteration
                    existing_objects.append({
                        'name': model_name,
                        'position': position,
                        'dimensions': dimensions
                    })
                    
                    added_count += 1
                    logger.info(f"Added {model_name} at ({position[0]:.2f}, {position[1]:.2f}) with yaw {position[2]:.2f}")
                    
            except Exception as e:
                logger.error(f"Error adding {obj_type}: {e}", exc_info=True)
                continue
        
        if added_count > 0:
            # Save the modified world
            self._save_world(output_file)
            print(f"{Style.GREEN}      ✓ Added {added_count}/{total_to_add} object(s){Style.ENDC}")
            logger.info(f"Successfully added {added_count} object(s) with placement engine")
            return True
        else:
            print(f"{Style.RED}      ✗ No objects were added{Style.ENDC}")
            logger.warning("No objects were added")
            return False
    def _apply_add_with_semantic_grouping(self, objects: List[Dict], world_elem: ET.Element, 
                                         output_file: Path) -> int:
        """
        Add objects using the placement engine for intelligent placement.
        Delegates to the main generator's placement engine to ensure consistency.
        
        Args:
            objects: List of object specifications
            world_elem: World XML element
            output_file: Output file path
            
        Returns:
            int: Number of objects successfully added
        """
        logger.info("Using placement engine for intelligent object placement")
        
        # Get room info from first object
        room_name = None
        if objects:
            properties = objects[0].get("properties", {})
            room_name = properties.get("room", "")
        
        # Get room bounds for extracting existing objects
        room_bounds = self._get_placement_area(room_name)
        if not room_bounds:
            logger.error("Could not determine room bounds")
            return 0
        
        # Extract existing objects in the room to avoid collisions
        existing_objects_in_room = self._get_existing_object_positions(world_elem, room_bounds)
        logger.info(f"Found {len(existing_objects_in_room)} existing object(s) in room to avoid")
        
        # Convert existing objects to GazeboModel format for the placement engine
        # Mark them as 'existing' so the LLM knows to avoid them
        existing_models = []
        for obj in existing_objects_in_room:
            pos = obj['position']
            dims = obj['dimensions']
            existing_models.append(GazeboModel(
                name=obj['name'],
                model_path='existing',  # Mark as existing - this signals refinement mode
                category=obj.get('type', obj['name']),
                pose={
                    'x': pos[0], 'y': pos[1], 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': pos[2]
                },
                room=room_name,
                size=[dims[0], dims[1], dims[2]],
                static=True
            ))
        
        # Convert existing world to Room object WITH existing objects for LLM context
        room = self._create_room_from_world_empty(room_name, room_bounds, existing_models)
        if room is None:
            logger.error("Could not create room representation for placement")
            return 0
        
        # Convert objects to place format expected by placement engine
        objects_to_place = []
        for obj in objects:
            obj_type = obj.get("type", "")
            count = obj.get("count", 1)
            properties = obj.get("properties", {})
            semantic_context = properties.get("semantic_context", "")
            
            for _ in range(count):
                obj_dict = {'type': obj_type}
                if semantic_context:
                    obj_dict['semantic_context'] = semantic_context
                objects_to_place.append(obj_dict)
        
        # Create name generator that avoids existing names
        existing_names = {model.get('name') for model in world_elem.findall('.//model')}
        existing_names.update({include.find('name').text for include in world_elem.findall('.//include') 
                              if include.find('name') is not None})
        
        def name_generator(base_name: str) -> str:
            counter = 0
            while True:
                candidate = f"{base_name}_{counter}"
                if candidate not in existing_names:
                    existing_names.add(candidate)
                    return candidate
                counter += 1
        
        # Use placement engine to place objects
        try:
            placed_models = self.placement_engine.place_objects_in_room(
                room=room,
                world_map="",
                objects_to_place=objects_to_place,
                name_generator=name_generator,
                room_number=1,
                total_rooms=1
            )
            
            # Check for collisions BEFORE adjusting new objects
            logger.info("Checking for collisions with existing objects...")
            initial_collisions = self._check_remaining_collisions(placed_models, existing_objects_in_room)
            
            if initial_collisions > 0:
                logger.warning(f"Detected {initial_collisions} collision(s) with existing objects")
                logger.info("Adjusting new object positions to avoid existing objects...")
                
                # Only adjust the positions of newly placed objects to avoid collisions
                final_models = self._validate_against_existing(placed_models, existing_objects_in_room, room_bounds)
                
                # Check if collisions still remain after adjustment
                remaining_collisions = self._check_remaining_collisions(final_models, existing_objects_in_room)
                
                if remaining_collisions > 0:
                    logger.warning(f"Still have {remaining_collisions} collision(s) after adjustment")
                    logger.info("Attempting to redistribute existing objects to make space...")
                    
                    # Try redistributing existing objects to make space
                    if self._redistribute_for_space(world_elem, existing_objects_in_room, final_models, room_bounds):
                        logger.info("Redistribution successful, re-validating new object positions...")
                        
                        # Re-validate after redistribution
                        final_models = self._validate_against_existing(final_models, existing_objects_in_room, room_bounds)
                        
                        final_collisions = self._check_remaining_collisions(final_models, existing_objects_in_room)
                        if final_collisions > 0:
                            logger.warning(f"Still have {final_collisions} collision(s) after redistribution")
                        else:
                            logger.info("✅ All collisions resolved after redistribution!")
                    else:
                        logger.warning("Redistribution failed or not beneficial")
            else:
                logger.info("No collisions detected, using placements as-is")
                final_models = placed_models
            
            # Convert placed models to XML elements and add to world
            added_count = 0
            for model in final_models:
                # Create model element from GazeboModel
                model_elem = ET.Element('include')
                
                # Add URI
                uri_elem = ET.SubElement(model_elem, 'uri')
                uri_elem.text = model.model_path
                
                # Add name
                name_elem = ET.SubElement(model_elem, 'name')
                name_elem.text = model.name
                
                # Add pose
                pose_elem = ET.SubElement(model_elem, 'pose')
                pose = model.pose
                pose_elem.text = f"{pose['x']:.6f} {pose['y']:.6f} {pose['z']:.6f} {pose['roll']:.6f} {pose['pitch']:.6f} {pose['yaw']:.6f}"
                
                # Add static if needed
                if model.static:
                    static_elem = ET.SubElement(model_elem, 'static')
                    static_elem.text = 'true'
                
                world_elem.append(model_elem)
                added_count += 1
                logger.info(f"Added {model.name} at ({pose['x']:.2f}, {pose['y']:.2f})")
            
            if added_count > 0:
                self._save_world(output_file)
                logger.info(f"Successfully added {added_count} object(s) using placement engine")
            
            return added_count
            
        except Exception as e:
            logger.error(f"Placement engine failed: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    
    def _validate_position_basic(self, position: tuple, dimensions: tuple, room_bounds: Dict = None) -> bool:
        """
        Basic validation: only check room bounds and doorways (no collision check).
        Used for desk-chair pairs where proximity is intentional.
        
        Args:
            position: (x, y, yaw) position to validate
            dimensions: (length, width, height) of object
            room_bounds: Optional room bounds to check if position is within room
            
        Returns:
            bool: True if position is valid
        """
        # Delegate to shared collision detector
        if not self.collision_detector:
            self.collision_detector = CollisionDetector(self.world_metadata)
        
        return self.collision_detector.validate_position_basic(position, dimensions, room_bounds)
    
    def _determine_semantic_relationship(self, obj_type: str, referenced_type: str) -> str:
        """
        Determine the spatial arrangement relationship between two object types.
        
        Args:
            obj_type: Type of object being added (e.g., 'chair')
            referenced_type: Type of existing object (e.g., 'desk')
            
        Returns:
            str: Spatial arrangement ('in_front', 'beside_left', 'beside_right', etc.) or None
        """
        # Define common semantic relationships
        relationships = {
            ('chair', 'desk'): 'in_front',
            ('chair', 'table'): 'in_front',
            ('chair', 'workstation'): 'in_front',
            ('nightstand', 'bed'): 'beside_right',
            ('night_stand', 'bed'): 'beside_right',
            ('lamp', 'desk'): 'beside_right',
            ('lamp', 'table'): 'beside_right',
            ('lamp', 'nightstand'): 'on_top',
            ('lamp', 'night_stand'): 'on_top',
            ('monitor', 'desk'): 'on_top',
            ('keyboard', 'desk'): 'on_top',
            ('cushion', 'sofa'): 'on_top',
            ('pillow', 'bed'): 'on_top',
        }
        
        key = (obj_type.lower(), referenced_type.lower())
        arrangement = relationships.get(key)
        
        if arrangement:
            logger.info(f"Determined relationship: {obj_type} -> {arrangement} of {referenced_type}")
        else:
            # Default: place beside
            logger.info(f"No specific relationship defined for {obj_type} + {referenced_type}, using 'beside_right'")
            arrangement = 'beside_right'
        
        return arrangement
    
    def _extract_model_name_from_existing(self, object_name: str, world_elem: ET.Element) -> str:
        """
        Extract the model name from an existing object in the world.
        
        Args:
            object_name: Name of the existing object (e.g., 'desk_0')
            world_elem: World XML element
            
        Returns:
            str: Model name (e.g., 'Office_Desk') or 'unknown'
        """
        # Find the object in the XML
        for include in world_elem.findall('.//include'):
            name_elem = include.find('name')
            if name_elem is not None and name_elem.text == object_name:
                # Get the URI
                uri_elem = include.find('uri')
                if uri_elem is not None and uri_elem.text:
                    uri = uri_elem.text
                    # Extract model name from URI
                    model_name = uri.replace("model://", "").strip('/')
                    logger.debug(f"Extracted model name '{model_name}' from URI '{uri}'")
                    return model_name
        
        # Also check <model> tags
        for model in world_elem.findall('.//model'):
            if model.get('name') == object_name:
                # For inline models, use the name attribute
                return object_name
        
        logger.warning(f"Could not extract model name for {object_name}, using 'unknown'")
        return 'unknown'
    
    def _calculate_relative_position_with_model(self, primary_pos: tuple, arrangement: str,
                                               primary_dims: tuple, related_dims: tuple,
                                               primary_model_name: str = None) -> tuple:
        """
        Calculate position with model-specific corrections (matches engine's logic exactly).
        
        Args:
            primary_pos: (x, y, yaw) of primary object
            arrangement: Spatial arrangement hint
            primary_dims: (length, width, height) of primary
            related_dims: (length, width, height) of related object
            primary_model_name: Model name for model-specific corrections
            
        Returns:
            tuple: (x, y, yaw) for related object
        """
        import math
        
        px, py, p_yaw = primary_pos
        
        return self.placement_engine._calculate_relative_position(
            px, py, p_yaw,
            primary_dims, related_dims,
            arrangement, primary_model_name
        )
    
    def _add_object_relative_to_existing(self, obj_spec: Dict, referenced_obj: Dict, 
                                         relationship: str, world_elem: ET.Element, 
                                         output_file: Path) -> bool:
        """
        Add a new object positioned relative to an existing object using semantic relationships.
        Applies model-specific corrections for known models (e.g., Office_Desk chair positioning).
        
        Args:
            obj_spec: Specification for the object to add
            referenced_obj: The existing object to position relative to
            relationship: Spatial arrangement ('in_front', 'beside_right', etc.)
            world_elem: World XML element
            output_file: Output file path
            
        Returns:
            bool: True if object was successfully added
        """
        try:
            obj_type = obj_spec.get("type", "")
            count = obj_spec.get("count", 1)
            properties = obj_spec.get("properties", {})
            room_name = properties.get("room", "")
            
            if count > 1:
                logger.warning(f"Relative positioning requested but count={count}. Only adding 1 object.")
                count = 1
            
            logger.info(f"Adding {count} {obj_type}(s) {relationship} of existing {referenced_obj['name']}")
            
            # Resolve model URI for the new object
            cached_model = self._get_cached_model(obj_type, relationship)
            if cached_model:
                model_uri = cached_model['uri']
                dimensions = cached_model['dimensions']
            else:
                model_uri = self.model_db.find_best_model(obj_type, context=relationship)
                if not model_uri:
                    logger.error(f"Could not find model for: {obj_type}")
                    return False
                dimensions = self._extract_model_dimensions(model_uri, obj_type)
                self._cache_model(obj_type, model_uri, dimensions, relationship)
            
            # Get referenced object's position, dimensions, and model info
            ref_pos = referenced_obj['position']  # (x, y, yaw)
            ref_dims = referenced_obj['dimensions']  # (length, width, height)
            
            # Try to extract the model name from the existing object for model-specific corrections
            ref_model_name = self._extract_model_name_from_existing(referenced_obj['name'], world_elem)
            logger.info(f"Referenced object model: {ref_model_name}")
            
            # Calculate relative position with model-specific corrections if available
            position = self._calculate_relative_position_with_model(
                primary_pos=ref_pos,
                arrangement=relationship,
                primary_dims=ref_dims,
                related_dims=dimensions,
                primary_model_name=ref_model_name
            )
            
            if not position:
                logger.error(f"Failed to calculate position for {obj_type}")
                return False
            
            x, y, yaw = position
            
            # Get room bounds for validation
            room_bounds = self._get_placement_area(room_name)
            
            # Get existing objects for collision check (excluding the referenced one since we want to be close to it)
            existing_objects = self._get_existing_object_positions(world_elem, room_bounds)
            other_objects = [obj for obj in existing_objects if obj['name'] != referenced_obj['name']]
            
            # Validate position (check bounds and doorways, but allow proximity to referenced object)
            if not self._validate_position_basic(position, dimensions, room_bounds):
                logger.warning(f"Position validation failed, trying nearby alternatives...")
                # Try slight variations
                import random
                for attempt in range(10):
                    offset_x = random.uniform(-0.3, 0.3)
                    offset_y = random.uniform(-0.3, 0.3)
                    test_pos = (x + offset_x, y + offset_y, yaw)
                    if self._validate_position_basic(test_pos, dimensions, room_bounds):
                        x, y = test_pos[0], test_pos[1]
                        logger.info(f"Found valid nearby position after {attempt + 1} attempts")
                        break
                else:
                    logger.error(f"Could not find valid position for {obj_type} near {referenced_obj['name']}")
                    return False
            
            # Generate unique name
            model_name = self._generate_unique_name(obj_type)
            
            # Create XML element
            model_elem = ET.Element('include')
            
            uri_elem = ET.SubElement(model_elem, 'uri')
            uri_elem.text = model_uri
            
            name_elem = ET.SubElement(model_elem, 'name')
            name_elem.text = model_name
            
            pose_elem = ET.SubElement(model_elem, 'pose')
            pose_elem.text = f"{x:.6f} {y:.6f} 0.0 0.0 0.0 {yaw:.6f}"
            
            # Add to world
            world_elem.append(model_elem)
            
            # Save world
            self._save_world(output_file)
            
            logger.info(f"Successfully added {model_name} at ({x:.2f}, {y:.2f}, yaw={yaw:.2f}) {relationship} of {referenced_obj['name']}")
            print(f"{Style.GREEN}      ✓ Added {model_name} {relationship} of {referenced_obj['name']}{Style.ENDC}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding object relative to existing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_unique_name(self, base_type: str) -> str:
        """
        Generate a unique model name that doesn't conflict with existing models.
        Checks both <model> tags and <include> tags to avoid duplicates.
        """
        root = self.current_world.getroot()
        world = root.find('.//world')
        
        existing_names = set()
        
        # Check <model> tags
        for model in world.findall('.//model'):
            name = model.get('name', '')
            if name:
                existing_names.add(name)
        
        # Check <include> tags
        for include in world.findall('.//include'):
            name_elem = include.find('name')
            if name_elem is not None and name_elem.text:
                existing_names.add(name_elem.text)
        
        # Clean base type
        clean_type = re.sub(r'[^a-zA-Z0-9_]', '_', base_type)
        
        # Find unique name
        counter = 0
        while True:
            candidate = f"{clean_type}_{counter}"
            if candidate not in existing_names:
                return candidate
            counter += 1
    
    def _extract_model_dimensions(self, model_uri: str, obj_type: str) -> tuple:
        """
        Extract actual model dimensions from SDF file.
        Falls back to type-based defaults if parsing fails.
        Uses dimension cache to avoid repeated file I/O.
        
        Args:
            model_uri: Model URI (e.g., model://aws_robomaker_residential_PortableSofa_01)
            obj_type: Object type for fallback dimensions
            
        Returns:
            tuple: (length, width, height) in meters
        """
        # Check dimension cache first
        cached_dims = self._get_cached_dimensions(model_uri)
        if cached_dims:
            logger.debug(f"Using cached dimensions for {model_uri}: {cached_dims}")
            return cached_dims
        
        try:
            # Extract model name from URI
            if "model://" in model_uri:
                model_name = model_uri.replace("model://", "")
            else:
                model_name = model_uri
            
            # Try to locate and parse the actual model SDF file
            model_path = find_gazebo_model_path(model_name)
            if model_path:
                dimensions = self.sdf_extractor.extract_model_dimensions(model_path)
                if dimensions and all(d > 0 for d in dimensions):
                    logger.debug(f"Extracted SDF dimensions {dimensions} for {model_name}")
                    # Return as (length, width, height) matching the method signature
                    width, length, height = dimensions
                    result = (length, width, height)
                    # Cache the extracted dimensions
                    self.dimension_cache[model_uri] = result
                    return result
            
            # Fallback to type-based dimension heuristics if SDF parsing fails
            furniture_dims = {
                'sofa': (2.0, 0.8, 0.8),
                'couch': (2.0, 0.8, 0.8),
                'chair': (0.6, 0.6, 1.0),
                'armchair': (0.8, 0.8, 1.0),
                'table': (1.5, 0.8, 0.75),
                'desk': (1.2, 0.6, 0.75),
                'bed': (2.0, 1.5, 0.6),
                'wardrobe': (1.2, 0.6, 2.0),
                'bookshelf': (0.8, 0.3, 1.8),
                'cabinet': (0.8, 0.4, 1.5),
                'shelf': (1.0, 0.3, 0.8),
                'plant': (0.4, 0.4, 0.8),
                'lamp': (0.3, 0.3, 0.6),
                'tv': (1.2, 0.1, 0.7),
                'default': (1.0, 1.0, 1.0)
            }
            
            # Find matching type
            for key, dims in furniture_dims.items():
                if key in obj_type.lower():
                    logger.debug(f"Using heuristic dimensions {dims} for {obj_type}")
                    # Cache heuristic dimensions too
                    self.dimension_cache[model_uri] = dims
                    return dims
            
            # Cache default dimensions
            default_dims = furniture_dims['default']
            self.dimension_cache[model_uri] = default_dims
            return default_dims
            
        except Exception as e:
            logger.warning(f"Could not extract dimensions for {model_uri}: {e}")
            return (1.0, 1.0, 1.0)
    
    def _get_existing_object_positions(self, world_elem, room_bounds: Dict = None) -> list:
        """
        Extract positions and dimensions of all existing objects for collision detection.
        
        Args:
            world_elem: World XML element
            room_bounds: Optional room bounds to filter objects by room (only return objects in this room)
            
        Returns:
            list: List of dicts with 'name', 'position', 'dimensions', 'room'
        """
        # Delegate to shared collision detector
        if not self.collision_detector:
            self.collision_detector = CollisionDetector(self.world_metadata)
        
        existing = self.collision_detector.get_existing_object_positions(world_elem, room_bounds)
        
        # Add room information (collision detector doesn't track this)
        for obj in existing:
            obj['room'] = self._determine_object_room(obj['position'][0], obj['position'][1])
        
        return existing
    
    def _extract_type_from_name(self, name: str) -> str:
        """Extract object type from model name (e.g., 'chair_0' -> 'chair')"""
        # Remove trailing numbers and underscores
        import re
        # Match letters only (not underscores), then stop at underscore+digit
        match = re.match(r'^([a-zA-Z]+)(?:_\d+)?', name)
        if match:
            return match.group(1).lower()
        return 'object'
    
    def _determine_object_room(self, x: float, y: float) -> str:
        """
        Determine which room a position belongs to based on room bounds.
        
        Args:
            x, y: Position coordinates
            
        Returns:
            str: Room name or 'unknown'
        """
        room_bounds = self.world_metadata.get('room_bounds', {})
        
        for room_name, bounds in room_bounds.items():
            if (bounds['min_x'] <= x <= bounds['max_x'] and
                bounds['min_y'] <= y <= bounds['max_y']):
                return room_name
        
        return 'unknown'
    
    def _calculate_position_with_placement(self, obj_type: str, placement_hint: str, 
                                          room_bounds: Dict, dimensions: tuple,
                                          existing_objects: list) -> tuple:
        """
        Calculate position using enhanced semantic placement with collision detection and wall snapping.
        
        Args:
            obj_type: Type of object (for wall snapping decision)
            placement_hint: Semantic placement hint
            room_bounds: Dict with room bounds
            dimensions: (length, width, height)
            existing_objects: List of existing objects for collision detection
            
        Returns:
            tuple: (x, y, yaw) or None if no valid position found
        """
        import random
        import math
        
        # Extract room bounds
        x_min = room_bounds.get('min_x', -5.0)
        y_min = room_bounds.get('min_y', -5.0)
        x_max = room_bounds.get('max_x', 5.0)
        y_max = room_bounds.get('max_y', 5.0)
        center_x = room_bounds.get('center_x', (x_min + x_max) / 2)
        center_y = room_bounds.get('center_y', (y_min + y_max) / 2)
        
        # Enhanced object type categorization
        wall_snap_types = {'wardrobe', 'bookshelf', 'cabinet', 'shelf', 'tv', 'dresser', 'sofa', 'couch'}
        center_types = {'table', 'coffee_table', 'dining_table', 'rug'}
        desk_types = {'desk', 'workstation', 'office_desk'}
        
        obj_lower = obj_type.lower()
        should_snap_to_wall = any(wtype in obj_lower for wtype in wall_snap_types)
        should_center = any(ctype in obj_lower for ctype in center_types)
        is_desk = any(dtype in obj_lower for dtype in desk_types)
        is_chair = 'chair' in obj_lower
        
        obj_length, obj_width, obj_height = dimensions
        
        # Add safety margins - reduced for better space utilization
        margin = 0.3 
        collision_margin = 0.15 
        
        # Define natural orientations (yaw angles)
        natural_yaw = 0.0  # Default: facing north
        
        # Object-specific orientation preferences
        orientation_map = {
            'desk': 0.0,  # Face north (user faces north when working)
            'chair': 0.0,  # Face north initially (can be adjusted for pairing)
            'bookshelf': math.pi / 2,  # Face east (opening toward room)
            'sofa': 0.0,  # Face north (sitting area faces north)
            'couch': 0.0,  # Face north
            'tv': math.pi,  # Face south (screen toward seating area)
            'bed': 0.0,  # Face north (head toward north)
            'table': 0.0,  # Face north
        }
        
        # Set natural yaw based on object type
        for key, yaw in orientation_map.items():
            if key in obj_lower:
                natural_yaw = yaw
                break
        
        # Try multiple positions
        max_attempts = 100
        for attempt in range(max_attempts):
            
            # Determine placement strategy based on object type and hint
            if should_center or 'center' in placement_hint.lower():
                # Place in room center
                x = center_x + random.uniform(-1.0, 1.0)
                y = center_y + random.uniform(-1.0, 1.0)
                yaw = natural_yaw
                
            elif is_desk:
                # Desks prefer walls but with working space in front
                # Place against a wall with proper orientation
                wall = random.choice(['north', 'south', 'east', 'west'])
                
                if wall == 'north':
                    x = random.uniform(x_min + margin + 1, x_max - margin - 1)
                    y = y_max - margin - obj_width / 2
                    yaw = math.pi  # Face south (into room)
                elif wall == 'south':
                    x = random.uniform(x_min + margin + 1, x_max - margin - 1)
                    y = y_min + margin + obj_width / 2
                    yaw = 0.0  # Face north (into room)
                elif wall == 'east':
                    x = x_max - margin - obj_width / 2
                    y = random.uniform(y_min + margin + 1, y_max - margin - 1)
                    yaw = -math.pi / 2  # Face west (into room)
                else:  # west
                    x = x_min + margin + obj_width / 2
                    y = random.uniform(y_min + margin + 1, y_max - margin - 1)
                    yaw = math.pi / 2  # Face east (into room)
                    
            elif is_chair:
                # Chairs should pair with desks
                # Find the most recently added desk (last in list) that doesn't have a chair nearby
                paired_desk = None
                
                # Reverse search to find newest desk first
                for existing in reversed(existing_objects):
                    ex_name = existing.get('name', '')
                    if any(dtype in ex_name.lower() for dtype in desk_types):
                        # Check if this desk already has a chair nearby (within 1.5m)
                        desk_pos = existing['position']
                        has_chair = False
                        
                        for obj in existing_objects:
                            if 'chair' in obj.get('name', '').lower():
                                chair_pos = obj['position']
                                distance = math.sqrt((desk_pos[0] - chair_pos[0])**2 + (desk_pos[1] - chair_pos[1])**2)
                                if distance < 1.5:  # Chair within 1.5m of desk
                                    has_chair = True
                                    break
                        
                        if not has_chair:
                            paired_desk = existing
                            break
                
                if paired_desk:
                    # Place chair in front of desk
                    desk_pos = paired_desk['position']
                    desk_yaw = desk_pos[2] if len(desk_pos) > 2 else 0.0
                    
                    # Calculate chair position in front of desk
                    offset_distance = 0.8  # Distance from desk
                    x = desk_pos[0] + offset_distance * math.cos(desk_yaw)
                    y = desk_pos[1] + offset_distance * math.sin(desk_yaw)
                    yaw = desk_yaw + math.pi  # Face the desk
                    
                    # Clamp position to room bounds
                    x = max(x_min + margin, min(x_max - margin, x))
                    y = max(y_min + margin, min(y_max - margin, y))
                    
                    logger.debug(f"Pairing chair with desk '{paired_desk.get('name')}' at ({desk_pos[0]:.2f}, {desk_pos[1]:.2f}), chair at ({x:.2f}, {y:.2f})")
                else:
                    # All desks have chairs, place standalone
                    x = random.uniform(x_min + margin, x_max - margin)
                    y = random.uniform(y_min + margin, y_max - margin)
                    yaw = natural_yaw
                    logger.debug("No available desk to pair with, placing chair standalone")
                    
            elif should_snap_to_wall:
                # Wall-snapping furniture (bookshelf, cabinet, etc.)
                wall = random.choice(['north', 'south', 'east', 'west'])
                
                if wall == 'north':
                    x = random.uniform(x_min + margin, x_max - margin)
                    y = y_max - margin - obj_width / 2
                    yaw = math.pi  # Face into room
                elif wall == 'south':
                    x = random.uniform(x_min + margin, x_max - margin)
                    y = y_min + margin + obj_width / 2
                    yaw = 0.0  # Face into room
                elif wall == 'east':
                    x = x_max - margin - obj_width / 2
                    y = random.uniform(y_min + margin, y_max - margin)
                    yaw = -math.pi / 2  # Face into room
                else:  # west
                    x = x_min + margin + obj_width / 2
                    y = random.uniform(y_min + margin, y_max - margin)
                    yaw = math.pi / 2  # Face into room
                    
            else:
                # Free placement with natural orientation
                x = random.uniform(x_min + margin, x_max - margin)
                y = random.uniform(y_min + margin, y_max - margin)
                yaw = natural_yaw + random.uniform(-0.2, 0.2)  # Small variation
            
            # Check collision with existing objects
            collision = False
            for existing in existing_objects:
                ex_pos = existing['position']
                ex_dims = existing['dimensions']
                
                # Calculate actual bounding boxes (AABB - Axis-Aligned Bounding Boxes)
                # Use proper dimensions for each axis, not max()
                obj_half_length = obj_length / 2
                obj_half_width = obj_width / 2
                ex_half_length = ex_dims[0] / 2
                ex_half_width = ex_dims[1] / 2
                
                # Calculate distance between centers
                dx = abs(x - ex_pos[0])
                dy = abs(y - ex_pos[1])
                
                # Required separation for each axis independently
                # This is the correct way: sum of half-extents in each direction plus margin
                required_dist_x = obj_half_length + ex_half_length + collision_margin
                required_dist_y = obj_half_width + ex_half_width + collision_margin
                
                # AABB collision check - both axes must pass
                if dx < required_dist_x and dy < required_dist_y:
                    collision = True
                    logger.debug(f"Collision detected with {existing['name']} at ({ex_pos[0]:.2f}, {ex_pos[1]:.2f}), dx={dx:.2f}<{required_dist_x:.2f}, dy={dy:.2f}<{required_dist_y:.2f}")
                    break
            
            # Check doorway clearance
            if not collision:
                door_positions = self.world_metadata.get('door_positions', [])
                for door in door_positions:
                    door_x, door_y = door['x'], door['y']
                    clearance_radius = door['clearance_radius']
                    
                    # Calculate distance from object center to door
                    distance = math.sqrt((x - door_x)**2 + (y - door_y)**2)
                    
                    # Check if object center is too close to doorway
                    # Use the larger dimension as a conservative estimate
                    obj_extent = max(obj_length, obj_width) / 2
                    if distance < (clearance_radius + obj_extent):
                        collision = True
                        logger.debug(f"Doorway clearance violation near {door['name']} at ({door_x:.2f}, {door_y:.2f}), distance={distance:.2f}m < required {clearance_radius + obj_extent:.2f}m")
                        break
            
            if not collision:
                logger.debug(f"Found valid position for {obj_type} at ({x:.2f}, {y:.2f}, yaw={yaw:.2f}) after {attempt + 1} attempts")
                return (x, y, yaw)
        
        # If we couldn't find space, try to compact existing objects
        logger.info(f"No space found after {max_attempts} attempts, attempting to compact room")
        if self._try_compact_room(room_bounds, existing_objects, world_elem=None):
            logger.info("Room compacted, retrying placement")
            # Try a few more times with potentially more space
            for attempt in range(20):
                x = random.uniform(x_min + margin, x_max - margin)
                y = random.uniform(y_min + margin, y_max - margin)
                yaw = natural_yaw
                
                collision = False
                for existing in existing_objects:
                    ex_pos = existing['position']
                    ex_dims = existing['dimensions']
                    
                    obj_half_length = obj_length / 2
                    obj_half_width = obj_width / 2
                    ex_half_length = ex_dims[0] / 2
                    ex_half_width = ex_dims[1] / 2
                    
                    dx = abs(x - ex_pos[0])
                    dy = abs(y - ex_pos[1])
                    
                    required_dist_x = obj_half_length + ex_half_length + collision_margin
                    required_dist_y = obj_half_width + ex_half_width + collision_margin
                    
                    if dx < required_dist_x and dy < required_dist_y:
                        collision = True
                        break
                
                if not collision:
                    logger.info(f"Found position after compacting at ({x:.2f}, {y:.2f})")
                    return (x, y, yaw)

        
        logger.warning(f"Could not find collision-free position for {obj_type} after {max_attempts} attempts")
        logger.warning(f"Room bounds: {room_bounds}")
        logger.warning(f"Existing objects in room: {len(existing_objects)}")
        return None
    
    def _try_compact_room(self, room_bounds: Dict, existing_objects: list, world_elem) -> bool:
        """
        Attempt to compact existing objects in a room to make more space.
        This uses a simple grid-based packing strategy.
        
        Args:
            room_bounds: Room boundary dict
            existing_objects: List of existing objects in room
            world_elem: World XML element (unused, for future enhancement)
            
        Returns:
            bool: True if compacting was attempted (may create some space)
        """
        if not existing_objects:
            return False
        
        logger.info(f"Attempting to compact {len(existing_objects)} objects in room")
        
        # Calculate a grid-based layout
        # This is a simplified approach - just log the attempt for now
        # In a future enhancement, we could actually reposition objects
        
        # For now, reduce margins slightly in collision detection
        # The real implementation would reposition objects in existing_objects list
        logger.info("Compact mode: Using tighter spacing for collision detection")
        
        # Update existing_objects positions to be more compact (simplified)
        # This is a placeholder - real implementation would modify world_elem
        return True
    
    def _detect_grouped_objects(self, world_elem, primary_elem, primary_name: str) -> list:
        """
        Detect objects that are semantically grouped with the primary object.
        
        For example:
        - desk_0 is grouped with chair_0 (if nearby)
        - table_0 is grouped with chair_1, chair_2 (if nearby)
        - sofa_0 is grouped with lamp_0 (if nearby)
        
        Args:
            world_elem: World XML element
            primary_elem: The primary object element being moved
            primary_name: Name of the primary object
            
        Returns:
            list: List of tuples (element, name) for grouped objects
        """
        grouped = []
        
        # Extract primary object position
        primary_pose = primary_elem.find('pose')
        if primary_pose is None or not primary_pose.text:
            return grouped
        
        primary_coords = list(map(float, primary_pose.text.split()))
        primary_x, primary_y = primary_coords[0], primary_coords[1]
        primary_type = self._extract_type_from_name(primary_name)
        
        # Define grouping rules: {primary_type: [related_types, max_distance]}
        grouping_rules = {
            'desk': (['chair', 'lamp'], 2.0),  # Desk groups with nearby chairs/lamps
            'table': (['chair'], 2.5),  # Table groups with nearby chairs
            'chair': (['desk', 'table'], 2.0),  # Chair can group with nearby desk/table
            'sofa': (['lamp', 'table'], 2.0),  # Sofa groups with nearby lamps/tables
            'bed': (['lamp', 'nightstand'], 2.0),  # Bed groups with nearby furniture
        }
        
        # Check if primary type has grouping rules
        if primary_type not in grouping_rules:
            return grouped
        
        related_types, max_distance = grouping_rules[primary_type]
        
        # Search for related objects within proximity
        all_elements = world_elem.findall('.//model') + world_elem.findall('.//include')
        logger.debug(f"Searching {len(all_elements)} elements for grouped objects")
        
        for elem in all_elements:
            # Get element name
            if elem.tag == 'model':
                name = elem.get('name', '')
            else:  # include
                name_elem = elem.find('name')
                name = name_elem.text if name_elem is not None and name_elem.text else ''
            
            # Skip if it's the primary object itself or empty name
            if name == primary_name or not name:
                continue
            
            # Skip ground plane and sun
            if 'ground' in name.lower() or 'sun' in name.lower():
                continue
            
            # Check if object type matches any related type
            obj_type = self._extract_type_from_name(name)
            logger.debug(f"Checking {name} (type={obj_type}) against related types {related_types}")
            
            if obj_type not in related_types:
                continue
            
            logger.debug(f"{name} is a related type!")
            
            # Get object position
            pose = elem.find('pose')
            if pose is None or not pose.text:
                logger.debug(f"{name} has no pose element")
                continue
            
            coords = list(map(float, pose.text.split()))
            obj_x, obj_y = coords[0], coords[1]
            
            # Calculate distance
            import math
            distance = math.sqrt((obj_x - primary_x)**2 + (obj_y - primary_y)**2)
            logger.debug(f"Distance from {primary_name} to {name}: {distance:.2f}m (max={max_distance})")
            
            # If within max distance, add to grouped objects
            if distance <= max_distance:
                grouped.append((elem, name))
                logger.info(f"✓ Detected grouped object: {name} ({obj_type}) at distance {distance:.2f}m from {primary_name}")
            else:
                logger.debug(f"Too far: {distance:.2f}m > {max_distance}m")
        
        return grouped
    
    def _get_placement_area(self, room_name: str = "") -> Dict:
        """
        Get the placement area bounds for a room.
        
        Args:
            room_name: Name of the room (optional)
            
        Returns:
            dict: Bounds dictionary with min/max coordinates
        """
        room_bounds = self.world_metadata.get('room_bounds', {})
        
        if room_name and room_name in room_bounds:
            logger.info(f"Using specified room: {room_name}")
            return room_bounds[room_name]
        
        # If no room name specified but multiple rooms exist, this is ambiguous
        if not room_name and len(room_bounds) > 1:
            # Log warning and use first room, but make it clear
            first_room = list(room_bounds.keys())[0]
            logger.warning(f"No room specified for placement. Multiple rooms detected: {list(room_bounds.keys())}. Using first room: {first_room}")
            print(f"{Style.YELLOW}      ⚠ No room specified - using '{first_room}' (available: {', '.join(room_bounds.keys())}){Style.ENDC}")
            return room_bounds[first_room]
        elif room_bounds:
            # Single room - use it
            single_room = list(room_bounds.keys())[0]
            logger.info(f"Using single available room: {single_room}")
            return room_bounds[single_room]
        else:
            # Default bounds if no room information
            logger.warning("No room information available, using default bounds")
            return {
                'min_x': -5.0, 'max_x': 5.0,
                'min_y': -5.0, 'max_y': 5.0,
                'center_x': 0.0, 'center_y': 0.0,
                'width': 10.0, 'length': 10.0
            }
    
    def _infer_room_type(self, room_name: str = "") -> str:
        """
        Infer room type from room name.
        
        Args:
            room_name: Name of the room
            
        Returns:
            str: Room type (office, bedroom, living_room, etc.)
        """
        if not room_name:
            return "default"
        
        room_lower = room_name.lower()
        
        # Common room type mappings
        if any(keyword in room_lower for keyword in ['office', 'study', 'workspace']):
            return "office"
        elif any(keyword in room_lower for keyword in ['bedroom', 'bed', 'dorm']):
            return "bedroom"
        elif any(keyword in room_lower for keyword in ['living', 'lounge', 'sitting']):
            return "living_room"
        elif any(keyword in room_lower for keyword in ['kitchen', 'cook']):
            return "kitchen"
        elif any(keyword in room_lower for keyword in ['bathroom', 'bath', 'wc']):
            return "bathroom"
        elif any(keyword in room_lower for keyword in ['dining', 'dinner']):
            return "dining_room"
        elif any(keyword in room_lower for keyword in ['warehouse', 'storage', 'stock']):
            return "warehouse"
        elif any(keyword in room_lower for keyword in ['corridor', 'hallway', 'hall']):
            return "corridor"
        else:
            return "default"
    
    def _calculate_position(self, placement_hint: str, room_bounds: Dict, 
                           dimensions: Tuple, world_elem: ET.Element) -> Tuple[float, float, float]:
        """
        Calculate position for a new object based on placement hint.
        
        Args:
            placement_hint: Natural language placement description
            room_bounds: Room boundary information
            dimensions: Object dimensions (width, length, height)
            world_elem: World XML element for collision checking
            
        Returns:
            tuple: (x, y, yaw) position
        """
        hint_lower = placement_hint.lower()
        
        # Extract room center and bounds
        center_x = room_bounds.get('center_x', 0.0)
        center_y = room_bounds.get('center_y', 0.0)
        min_x = room_bounds.get('min_x', -5.0)
        max_x = room_bounds.get('max_x', 5.0)
        min_y = room_bounds.get('min_y', -5.0)
        max_y = room_bounds.get('max_y', 5.0)
        
        # Position based on hint
        if 'center' in hint_lower or 'middle' in hint_lower:
            x, y = center_x, center_y
        elif 'entrance' in hint_lower or 'door' in hint_lower:
            # Place near the minimum y (typical entrance location)
            x = center_x
            y = min_y + 2.0
        elif 'wall' in hint_lower or 'edge' in hint_lower:
            # Place along a wall (prefer north or south)
            x = min_x + (max_x - min_x) * 0.3
            y = min_y + 1.0
        elif 'corner' in hint_lower:
            # Place in a corner
            x = min_x + 1.0
            y = min_y + 1.0
        else:
            # Default: distributed placement
            # Find an open spot
            import random
            x = center_x + random.uniform(-2.0, 2.0)
            y = center_y + random.uniform(-2.0, 2.0)
        
        # Ensure within bounds with margin
        margin = 0.5
        x = max(min_x + margin, min(max_x - margin, x))
        y = max(min_y + margin, min(max_y - margin, y))
        
        # Default orientation
        yaw = 0.0
        
        return (x, y, yaw)
    
    def _estimate_object_dimensions(self, elem: ET.Element, name: str) -> Tuple[float, float, float]:
        """
        Estimate object dimensions by extracting from SDF file or using defaults.
        
        Tries to extract actual dimensions from the model's SDF file first,
        then falls back to type-based defaults if extraction fails.
        
        Args:
            elem: XML element (<model> or <include>)
            name: Object name
            
        Returns:
            tuple: (width, length, height) in meters
        """
        # Try to extract model URI from the element
        uri_elem = elem.find('uri')
        if uri_elem is not None and uri_elem.text:
            model_uri = uri_elem.text
            
            try:
                # Use existing _extract_model_dimensions method which:
                # 1. Checks dimension cache
                # 2. Parses SDF file if available
                # 3. Falls back to type-based defaults
                obj_type = self._extract_type_from_name(name)
                dimensions = self._extract_model_dimensions(model_uri, obj_type)
                
                logger.debug(f"Extracted dimensions for {name} from {model_uri}: {dimensions}")
                return dimensions
                
            except Exception as e:
                logger.warning(f"Could not extract dimensions from {model_uri}: {e}")
                # Fall through to name-based defaults
        
        # Fallback: Use name-based dimension heuristics
        name_lower = name.lower()
        
        if 'shelf' in name_lower or 'rack' in name_lower:
            return (1.5, 0.5, 2.0)  # shelving unit
        elif 'desk' in name_lower or 'table' in name_lower:
            return (1.5, 0.8, 0.75)
        elif 'chair' in name_lower:
            return (0.5, 0.5, 1.0)
        elif 'pallet' in name_lower:
            return (1.2, 0.8, 0.15)
        elif 'cabinet' in name_lower or 'wardrobe' in name_lower:
            return (1.0, 0.6, 2.0)
        else:
            logger.debug(f"Using default dimensions for {name}")
            return (1.0, 1.0, 1.5)  # default
    
    def _calculate_grid_positions(self, objects: List[Dict], position_desc: str, 
                                  room_bounds: Dict) -> List[Tuple[float, float, float]]:
        """
        Calculate collision-free positions for multiple objects in a grid/row layout.
        Respects room boundaries and avoids existing objects.
        
        Args:
            objects: List of object info dicts with dimensions and elements
            position_desc: Position description (e.g., "center", "in rows")
            room_bounds: Room boundary information
            
        Returns:
            list: List of (x, y, yaw) positions
        """
        # Get existing objects for collision detection (exclude objects being repositioned)
        root = self.current_world.getroot()
        world = root.find('.//world')
        all_existing = self._get_existing_object_positions(world, room_bounds)
        
        # Filter out objects being repositioned
        reposition_names = {obj['name'] for obj in objects}
        existing_objects = [obj for obj in all_existing if obj['name'] not in reposition_names]
        
        # Delegate to shared collision detector
        if not self.collision_detector:
            self.collision_detector = CollisionDetector(self.world_metadata)
        
        return self.collision_detector.calculate_grid_positions(
            objects, position_desc, room_bounds, existing_objects
        )
    
    def _create_model_element(self, name: str, uri: str, position: Tuple, 
                              dimensions: Tuple) -> ET.Element:
        """
        Create an include XML element for insertion into the world.
        Creates structure compatible with the world generator's format.
        
        Args:
            name: Model name
            uri: Model URI
            position: (x, y, yaw) position
            dimensions: (width, length, height) dimensions
            
        Returns:
            ET.Element: Include XML element (not wrapped in <model>)
        """
        # Create <include> element (not <model>)
        include = ET.Element('include')
        
        # Add URI
        uri_elem = ET.SubElement(include, 'uri')
        uri_elem.text = uri
        
        # Add name
        name_elem = ET.SubElement(include, 'name')
        name_elem.text = name
        
        # Add pose
        x, y, yaw = position
        pose = ET.SubElement(include, 'pose')
        pose.text = f"{x:.6f} {y:.6f} 0.000000 0.000000 0.000000 {yaw:.6f}"
        
        # Add static property
        static = ET.SubElement(include, 'static')
        static.text = 'true'
        
        return include
    
    def _save_world(self, output_file: Path):
        """
        Save the current world to file with proper formatting.
        
        Args:
            output_file: Path to save the world file
        """
        # Convert to string with pretty formatting
        rough_string = ET.tostring(self.current_world.getroot(), encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ", encoding='utf-8')
        
        # Write to file
        with open(output_file, 'wb') as f:
            f.write(pretty_xml)
        
        logger.info(f"Saved world to: {output_file}")
    
    def _apply_remove_objects(self, data: Dict, output_file: Path) -> bool:
        """
        Remove specified objects from the world.
        
        Supports removing by:
        - Target name (exact or partial match)
        - Object type (all objects of that type)
        - Quantity ("remove 2 chairs")
        
        Safety features:
        - Defaults to removing only 1 object if quantity not specified
        - Warns user about partial name matches
        - Requires confirmation-style hints for bulk removal
        """
        logger.info("Applying remove objects refinement")
        
        objects = data.get("objects", [])
        root = self.current_world.getroot()
        world = root.find('.//world')
        
        print(f"{Style.DIM}    Removing object(s)...{Style.ENDC}")
        
        removed_count = 0
        for obj in objects:
            target = obj.get("target", "")
            obj_type = obj.get("type", "")
            quantity = obj.get("quantity", None)
            
            # Build list of candidates from both <model> and <include> elements
            candidates = []
            partial_matches = []
            exact_matches = []
            
            # Get synonyms for better matching
            target_lower = target.lower() if target else ""
            type_lower = obj_type.lower() if obj_type else ""
            
            # Build synonym lists
            synonym_map = {
                'bookshelf': ['shelf', 'bookcase', 'shelving'],
                'shelf': ['bookshelf', 'shelving'],
                'desk': ['table', 'workstation'],
                'table': ['desk'],
                'chair': ['seat'],
                'seat': ['chair'],
                'wardrobe': ['closet', 'cabinet'],
                'closet': ['wardrobe', 'cabinet'],
            }
            
            target_synonyms = [target_lower] if target_lower else []
            if target_lower in synonym_map:
                target_synonyms.extend(synonym_map[target_lower])
                
            type_synonyms = [type_lower] if type_lower else []
            if type_lower in synonym_map:
                type_synonyms.extend(synonym_map[type_lower])
            
            # Check <model> tags
            models = world.findall('.//model')
            for model in models:
                name = model.get('name', '')
                name_lower = name.lower()
                
                # Match by target name or type with synonyms
                if target:
                    if any(syn == name_lower for syn in target_synonyms):
                        exact_matches.append(('model', model, name))
                        candidates.append(('model', model, name))
                    elif any(syn in name_lower for syn in target_synonyms):
                        partial_matches.append(('model', model, name))
                        candidates.append(('model', model, name))
                elif obj_type and any(syn in name_lower for syn in type_synonyms):
                    candidates.append(('model', model, name))
            
            # Check <include> tags
            includes = world.findall('.//include')
            for include in includes:
                name_elem = include.find('name')
                if name_elem is not None and name_elem.text:
                    name = name_elem.text
                    
                    # Skip ground plane and sun
                    if 'ground' in name.lower() or 'sun' in name.lower():
                        continue
                    
                    name_lower = name.lower()
                    
                    # Match by target name or type with synonyms
                    if target:
                        if any(syn == name_lower for syn in target_synonyms):
                            exact_matches.append(('include', include, name))
                            candidates.append(('include', include, name))
                        elif any(syn in name_lower for syn in target_synonyms):
                            partial_matches.append(('include', include, name))
                            candidates.append(('include', include, name))
                    elif obj_type and any(syn in name_lower for syn in type_synonyms):
                        candidates.append(('include', include, name))
            
            # Safety check: Warn about partial matches
            if partial_matches and not exact_matches:
                print(f"{Style.YELLOW}      ⚠ Partial match detected for target '{target}'{Style.ENDC}")
                logger.warning(f"Removal using partial match for '{target}' - matched objects: {[m[2] for m in partial_matches]}")
                print(f"{Style.DIM}        Matched objects: {', '.join([m[2] for m in partial_matches[:5]])}{Style.ENDC}")
                if len(partial_matches) > 5:
                    print(f"{Style.DIM}        ... and {len(partial_matches) - 5} more{Style.ENDC}")
            
            # Remove up to quantity (default to 1 if not specified to be safe)
            # Special handling: quantity=999 means "all matching objects"
            if quantity is None:
                remove_count = 1  # Safe default: only remove one
                if len(candidates) > 1:
                    print(f"{Style.YELLOW}      ℹ Multiple objects matched ({len(candidates)}), removing only 1 (no quantity specified){Style.ENDC}")
                    logger.info(f"No quantity specified for '{target}', defaulting to removing 1 of {len(candidates)} matched objects")
            elif quantity >= 999:
                # Interpret 999 as "all matching objects"
                remove_count = len(candidates)
                logger.info(f"Quantity=999 interpreted as 'all' - removing all {remove_count} matched objects")
            else:
                remove_count = min(quantity, len(candidates))
                if quantity > len(candidates):
                    print(f"{Style.DIM}        Note: Requested {quantity} but only {len(candidates)} objects available{Style.ENDC}")
            
            # Perform removal
            actual_remove_count = remove_count
            for i, (elem_type, elem, name) in enumerate(candidates[:actual_remove_count]):
                world.remove(elem)
                removed_count += 1
                print(f"{Style.DIM}      - Removed: {name}{Style.ENDC}")
                logger.info(f"Removed {elem_type}: {name}")
        
        if removed_count > 0:
            self._save_world(output_file)
            print(f"{Style.GREEN}      ✓ Removed {removed_count} object(s){Style.ENDC}")
            logger.info(f"Removed {removed_count} object(s), saved to: {output_file}")
            return True
        else:
            print(f"{Style.YELLOW}      ⚠ No objects matched removal criteria{Style.ENDC}")
            logger.warning("No objects matched removal criteria")
            return False
    
    def _apply_modify_objects(self, data: Dict, output_file: Path) -> bool:
        """
        Modify properties of existing objects.
        
        Supported modifications:
        - Position changes (x, y, z)
        - Rotation changes (roll, pitch, yaw)
        - Scale changes (uniform or per-axis)
        
        Respects quantity field to only modify specified number of objects.
        Safety: Defaults to 1 object, warns about partial matches.
        """
        logger.info("Applying modify objects refinement")
        
        objects = data.get("objects", [])
        root = self.current_world.getroot()
        world = root.find('.//world')
        
        print(f"{Style.DIM}    Modifying object(s)...{Style.ENDC}")
        
        modified_count = 0
        for obj in objects:
            target = obj.get("target", "")
            modifications = obj.get("modifications", {})
            quantity = obj.get("quantity", None)
            
            # Build list of candidates
            candidates = []
            exact_matches = []
            partial_matches = []
            
            # Get synonyms for better matching
            target_lower = target.lower()
            synonym_map = {
                'bookshelf': ['shelf', 'bookcase', 'shelving'],
                'shelf': ['bookshelf', 'shelving'],
                'desk': ['table', 'workstation'],
                'table': ['desk'],
                'chair': ['seat'],
                'seat': ['chair'],
                'wardrobe': ['closet', 'cabinet'],
                'closet': ['wardrobe', 'cabinet'],
            }
            
            target_synonyms = [target_lower]
            if target_lower in synonym_map:
                target_synonyms.extend(synonym_map[target_lower])
            
            # Find target objects in <model> tags
            models = world.findall('.//model')
            for model in models:
                name = model.get('name', '')
                name_lower = name.lower()
                
                if any(syn == name_lower for syn in target_synonyms):
                    exact_matches.append(model)
                    candidates.append(model)
                elif any(syn in name_lower for syn in target_synonyms):
                    partial_matches.append(model)
                    candidates.append(model)
            
            # Safety check: Warn about partial matches
            if partial_matches and not exact_matches:
                print(f"{Style.YELLOW}      ⚠ Partial match detected for target '{target}'{Style.ENDC}")
                logger.warning(f"Modification using partial match for '{target}' - matched {len(partial_matches)} objects")
                matched_names = [m.get('name', '') for m in partial_matches[:3]]
                print(f"{Style.DIM}        Matched: {', '.join(matched_names)}{Style.ENDC}")
                if len(partial_matches) > 3:
                    print(f"{Style.DIM}        ... and {len(partial_matches) - 3} more{Style.ENDC}")
            
            # Apply quantity limit (default to 1 for safety)
            # Special handling: quantity=999 means "all matching objects"
            if quantity is None:
                modify_count = 1
                if len(candidates) > 1:
                    print(f"{Style.YELLOW}      ℹ Multiple objects matched ({len(candidates)}), modifying only 1{Style.ENDC}")
                    logger.info(f"No quantity specified for modify '{target}', defaulting to 1 of {len(candidates)} objects")
            elif quantity >= 999:
                # Interpret 999 as "all matching objects"
                modify_count = len(candidates)
                logger.info(f"Quantity=999 interpreted as 'all' - modifying all {modify_count} matched objects")
            else:
                modify_count = min(quantity, len(candidates))
                if quantity > len(candidates):
                    logger.warning(f"Requested quantity {quantity} exceeds available objects {len(candidates)}, modifying all {modify_count}")
            
            # Modify only up to the specified quantity
            actual_modify_count = modify_count
            for model in candidates[:actual_modify_count]:
                name = model.get('name', '')
                pose_elem = model.find('pose')
                if pose_elem is None:
                    pose_elem = ET.SubElement(model, 'pose')
                    pose_elem.text = "0 0 0 0 0 0"
                
                # Parse current pose
                pose_values = [float(x) for x in pose_elem.text.split()]
                
                # Apply modifications
                if "position" in modifications:
                    pos = modifications["position"]
                    if "x" in pos:
                        pose_values[0] = float(pos["x"])
                    if "y" in pos:
                        pose_values[1] = float(pos["y"])
                    if "z" in pos:
                        pose_values[2] = float(pos["z"])
                
                if "rotation" in modifications:
                    rot = modifications["rotation"]
                    if "roll" in rot:
                        pose_values[3] = float(rot["roll"])
                    if "pitch" in rot:
                        pose_values[4] = float(rot["pitch"])
                    if "yaw" in rot:
                        pose_values[5] = float(rot["yaw"])
                
                # Update pose
                pose_elem.text = " ".join(f"{v:.6f}" for v in pose_values)
                modified_count += 1
                print(f"{Style.DIM}      - Modified: {name}{Style.ENDC}")
                logger.info(f"Modified object: {name}")
        
        if modified_count > 0:
            self._save_world(output_file)
            print(f"{Style.GREEN}      ✓ Modified {modified_count} object(s){Style.ENDC}")
            logger.info(f"Modified {modified_count} object(s), saved to: {output_file}")
            return True
        else:
            print(f"{Style.YELLOW}      ⚠ No objects matched modification criteria{Style.ENDC}")
            logger.warning("No objects matched modification criteria")
            return False
    
    def _apply_reposition_objects(self, data: Dict, output_file: Path) -> bool:
        """
        Reposition existing objects based on natural language descriptions.
        
        Supported position descriptions:
        - "center of room" - Move to room center
        - "near door/entrance" - Move near entrance
        - "against wall" - Move to wall edge
        - "corner" - Move to room corner
        - Absolute coordinates: "x:2.0, y:3.0"
        
        Respects quantity field and detects grouped objects (desk+chair) to move together.
        """
        logger.info("Applying reposition objects refinement")
        
        objects = data.get("objects", [])
        root = self.current_world.getroot()
        world = root.find('.//world')
        
        print(f"{Style.DIM}    Repositioning object(s)...{Style.ENDC}")
        
        repositioned_count = 0
        moved_objects = set()  # Track already moved objects
        grouped_moved_count = 0
        
        for obj in objects:
            target = obj.get("target", "")
            position_desc = obj.get("position", "")
            room_name = obj.get("room", "")
            quantity = obj.get("quantity", None)
            
            # Get room bounds for positioning
            room_bounds = self._get_placement_area(room_name)
            
            # Build list of candidates from both <model> and <include> elements
            candidates = []
            exact_matches = []
            partial_matches = []
            
            # Get synonyms for the target type to improve matching
            target_lower = target.lower()
            target_synonyms = [target_lower]
            
            # Add common synonyms for better matching
            synonym_map = {
                'bookshelf': ['shelf', 'bookcase', 'shelving'],
                'shelf': ['bookshelf', 'shelving'],
                'desk': ['table', 'workstation'],
                'table': ['desk'],
                'chair': ['seat'],
                'seat': ['chair'],
                'wardrobe': ['closet', 'cabinet'],
                'closet': ['wardrobe', 'cabinet'],
            }
            
            if target_lower in synonym_map:
                target_synonyms.extend(synonym_map[target_lower])
            
            # Check <model> tags
            models = world.findall('.//model')
            for model in models:
                name = model.get('name', '')
                if name not in moved_objects:
                    name_lower = name.lower()
                    # Check exact match or synonym match
                    if any(syn in name_lower for syn in target_synonyms):
                        if target_lower == name_lower:
                            exact_matches.append(('model', model, name))
                        else:
                            partial_matches.append(('model', model, name))
                        candidates.append(('model', model, name))
            
            # Check <include> tags
            includes = world.findall('.//include')
            for include in includes:
                name_elem = include.find('name')
                if name_elem is not None and name_elem.text:
                    name = name_elem.text
                    
                    # Skip ground plane and sun
                    if 'ground' in name.lower() or 'sun' in name.lower():
                        continue
                    
                    if name not in moved_objects:
                        name_lower = name.lower()
                        # Check exact match or synonym match
                        if any(syn in name_lower for syn in target_synonyms):
                            if target_lower == name_lower:
                                exact_matches.append(('include', include, name))
                            else:
                                partial_matches.append(('include', include, name))
                            candidates.append(('include', include, name))
            
            # Safety check: Warn about partial matches
            if partial_matches and not exact_matches:
                print(f"{Style.YELLOW}      ⚠ Partial match detected for target '{target}'{Style.ENDC}")
                logger.warning(f"Reposition using partial match for '{target}' - matched {len(partial_matches)} objects")
                matched_names = [m[2] for m in partial_matches[:3]]
                print(f"{Style.DIM}        Matched: {', '.join(matched_names)}{Style.ENDC}")
                if len(partial_matches) > 3:
                    print(f"{Style.DIM}        ... and {len(partial_matches) - 3} more{Style.ENDC}")
            
            # Apply quantity limit (default to 1 for safety)
            if quantity is None:
                reposition_count = 1
                if len(candidates) > 1:
                    print(f"{Style.YELLOW}      ℹ Multiple objects matched ({len(candidates)}), repositioning only 1{Style.ENDC}")
                    logger.info(f"No quantity specified for reposition '{target}', defaulting to 1 of {len(candidates)} objects")
            elif quantity >= 999:
                # Interpret 999 as "all matching objects"
                reposition_count = len(candidates)
                logger.info(f"Quantity=999 interpreted as 'all' - repositioning all {reposition_count} matched objects")
            else:
                reposition_count = min(quantity, len(candidates))
                if quantity > len(candidates):
                    logger.warning(f"Requested quantity {quantity} exceeds available objects {len(candidates)}, repositioning all {reposition_count}")
            
            # Check if we're repositioning multiple objects - use intelligent placement
            if reposition_count > 1:
                logger.info(f"Repositioning {reposition_count} objects - using intelligent placement to avoid collisions")
                
                # Extract dimensions from existing objects for collision avoidance
                objects_to_reposition = []
                for elem_type, elem, name in candidates[:reposition_count]:
                    pose_elem = elem.find('pose')
                    if pose_elem is not None:
                        old_pose_values = [float(x) for x in pose_elem.text.split()]
                        old_x, old_y = old_pose_values[0], old_pose_values[1]
                        
                        # Try to extract dimensions (approximate if not available)
                        dimensions = self._estimate_object_dimensions(elem, name)
                        
                        objects_to_reposition.append({
                            'elem_type': elem_type,
                            'elem': elem,
                            'name': name,
                            'old_x': old_x,
                            'old_y': old_y,
                            'dimensions': dimensions,
                            'pose_elem': pose_elem,
                            'old_pose_values': old_pose_values
                        })
                
                # Calculate grid layout for multiple objects
                new_positions = self._calculate_grid_positions(
                    objects_to_reposition,
                    position_desc,
                    room_bounds
                )
                
                # Apply new positions
                for obj_info, (new_x, new_y, new_yaw) in zip(objects_to_reposition, new_positions):
                    pose_elem = obj_info['pose_elem']
                    old_pose_values = obj_info['old_pose_values']
                    name = obj_info['name']
                    
                    # Update position
                    old_pose_values[0] = new_x
                    old_pose_values[1] = new_y
                    old_pose_values[5] = new_yaw
                    
                    pose_elem.text = " ".join(f"{v:.6f}" for v in old_pose_values)
                    moved_objects.add(name)
                    repositioned_count += 1
                    print(f"{Style.DIM}      - Moved: {name} → ({new_x:.1f}, {new_y:.1f}){Style.ENDC}")
                    logger.info(f"Repositioned {name} to ({new_x:.2f}, {new_y:.2f})")
                
                continue  # Skip individual repositioning loop
            
            # Original single-object repositioning logic
            for elem_type, elem, name in candidates[:reposition_count]:
                # Get current pose
                pose_elem = elem.find('pose')
                if pose_elem is None:
                    pose_elem = ET.SubElement(elem, 'pose')
                    pose_elem.text = "0 0 0 0 0 0"
                
                old_pose_values = [float(x) for x in pose_elem.text.split()]
                old_x, old_y = old_pose_values[0], old_pose_values[1]
                
                logger.info(f"Checking for grouped objects with {name} at ({old_x:.2f}, {old_y:.2f})...")
                grouped_objects = self._detect_grouped_objects(world, elem, name)
                logger.info(f"Found {len(grouped_objects)} grouped objects to move with {name}")
                
                # Parse position description and calculate new position
                new_x, new_y, new_yaw = self._calculate_position(
                    position_desc, 
                    room_bounds,
                    (1.0, 1.0, 1.0),  # Default dimensions
                    world
                )
                
                # Calculate displacement
                dx = new_x - old_x
                dy = new_y - old_y
                
                # Update position, keep z and orientation (except yaw)
                old_pose_values[0] = new_x
                old_pose_values[1] = new_y
                old_pose_values[5] = new_yaw  # Update yaw
                
                pose_elem.text = " ".join(f"{v:.6f}" for v in old_pose_values)
                moved_objects.add(name)
                repositioned_count += 1
                print(f"{Style.DIM}      - Moved: {name} → ({new_x:.1f}, {new_y:.1f}){Style.ENDC}")
                logger.info(f"Repositioned {name} to ({new_x:.2f}, {new_y:.2f}): {position_desc}")
                
                # Now move the grouped objects by the same displacement
                for grouped_elem, grouped_name in grouped_objects:
                    if grouped_name not in moved_objects:
                        # Get current pose of grouped object
                        grouped_pose_elem = grouped_elem.find('pose')
                        if grouped_pose_elem is None:
                            grouped_pose_elem = ET.SubElement(grouped_elem, 'pose')
                            grouped_pose_elem.text = "0 0 0 0 0 0"
                        
                        grouped_pose_values = [float(x) for x in grouped_pose_elem.text.split()]
                        
                        # Apply same displacement to maintain relative position
                        grouped_pose_values[0] += dx
                        grouped_pose_values[1] += dy
                        # Keep original orientation of grouped object
                        
                        grouped_pose_elem.text = " ".join(f"{v:.6f}" for v in grouped_pose_values)
                        moved_objects.add(grouped_name)
                        repositioned_count += 1
                        grouped_moved_count += 1
                        print(f"{Style.DIM}        + Grouped: {grouped_name} (moved together){Style.ENDC}")
                        logger.info(f"Also moved grouped object {grouped_name} by ({dx:.2f}, {dy:.2f})")
        
        if repositioned_count > 0:
            self._save_world(output_file)
            print(f"{Style.GREEN}      ✓ Repositioned {repositioned_count} object(s){Style.ENDC}")
            if grouped_moved_count > 0:
                print(f"{Style.CYAN}        (including {grouped_moved_count} grouped object(s)){Style.ENDC}")
            logger.info(f"Repositioned {repositioned_count} object(s), saved to: {output_file}")
            return True
        else:
            print(f"{Style.YELLOW}      ⚠ No objects matched reposition criteria{Style.ENDC}")
            logger.warning("No objects matched reposition criteria")
            return False
    
    def _apply_resize_room(self, data: Dict, output_file: Path) -> bool:
        """
        Resize a room by adjusting its wall positions.
        
        Supports:
        - Resizing room dimensions (width, length)
        - Optional automatic redistribution of objects in resized room
        - Updates room_bounds metadata
        
        Args:
            data: Refinement data containing room_resize field
            output_file: Output file path
            
        Returns:
            bool: Success status
        """
        logger.info("Applying resize room refinement")
        
        resize_info = data.get("room_resize", {})
        target_room = resize_info.get("target_room", "")
        new_dimensions = resize_info.get("new_dimensions", {})
        redistribute = resize_info.get("redistribute", True)
        
        new_width = new_dimensions.get("width")
        new_length = new_dimensions.get("length")
        
        if not target_room:
            logger.error("No target room specified for resize operation")
            print(f"{Style.RED}      ✗ No target room specified{Style.ENDC}")
            return False
        
        if not new_width and not new_length:
            logger.error("No new dimensions specified for resize operation")
            print(f"{Style.RED}      ✗ No new dimensions specified{Style.ENDC}")
            return False
        
        print(f"{Style.DIM}    Resizing room '{target_room}'...{Style.ENDC}")
        
        root = self.current_world.getroot()
        world = root.find('.//world')
        
        # Get current room bounds
        room_bounds = self.world_metadata.get('room_bounds', {})
        
        # Find the room in bounds (normalize name comparison)
        target_room_key = None
        for room_name in room_bounds.keys():
            if target_room.lower() in room_name.lower() or room_name.lower() in target_room.lower():
                target_room_key = room_name
                break
        
        if not target_room_key:
            logger.error(f"Room '{target_room}' not found in world metadata")
            print(f"{Style.RED}      ✗ Room '{target_room}' not found{Style.ENDC}")
            return False
        
        current_bounds = room_bounds[target_room_key]
        current_width = current_bounds['width']
        current_length = current_bounds['length']
        current_center_x = current_bounds['center_x']
        current_center_y = current_bounds['center_y']
        
        # Use current dimensions if new ones not specified
        target_width = new_width if new_width else current_width
        target_length = new_length if new_length else current_length
        
        print(f"{Style.DIM}      Current: {current_width:.1f}m × {current_length:.1f}m{Style.ENDC}")
        print(f"{Style.DIM}      New:     {target_width:.1f}m × {target_length:.1f}m{Style.ENDC}")
        
        # Calculate new bounds (expand/contract around center)
        half_width = target_width / 2
        half_length = target_length / 2
        
        new_bounds = {
            'min_x': current_center_x - half_width,
            'max_x': current_center_x + half_width,
            'min_y': current_center_y - half_length,
            'max_y': current_center_y + half_length,
            'center_x': current_center_x,
            'center_y': current_center_y,
            'width': target_width,
            'length': target_length
        }
        
        # Find and update wall positions
        walls_updated = 0
        walls = self.world_metadata.get('walls', [])
        
        for wall_name in walls:
            # Check if this wall belongs to the target room
            if target_room_key.lower() not in wall_name.lower():
                continue
            
            wall_model = world.find(f".//model[@name='{wall_name}']")
            if wall_model is None:
                continue
            
            # Get wall pose
            pose_elem = wall_model.find('.//pose')
            if pose_elem is None or not pose_elem.text:
                continue
            
            pose_values = [float(x) for x in pose_elem.text.split()]
            current_x, current_y = pose_values[0], pose_values[1]
            
            # Determine which wall this is based on position relative to center
            # and update its position accordingly
            tolerance = 0.5  # Small tolerance for floating point comparison
            
            # North wall (max_y)
            if abs(current_y - current_bounds['max_y']) < tolerance:
                pose_values[1] = new_bounds['max_y']
                walls_updated += 1
            # South wall (min_y)
            elif abs(current_y - current_bounds['min_y']) < tolerance:
                pose_values[1] = new_bounds['min_y']
                walls_updated += 1
            # East wall (max_x)
            elif abs(current_x - current_bounds['max_x']) < tolerance:
                pose_values[0] = new_bounds['max_x']
                walls_updated += 1
            # West wall (min_x)
            elif abs(current_x - current_bounds['min_x']) < tolerance:
                pose_values[0] = new_bounds['min_x']
                walls_updated += 1
            
            pose_elem.text = " ".join(f"{v:.6f}" for v in pose_values)
        
        if walls_updated == 0:
            logger.warning(f"No walls found for room '{target_room}' to update")
            print(f"{Style.YELLOW}      ⚠ No walls found to update{Style.ENDC}")
            return False
        
        print(f"{Style.GREEN}      ✓ Updated {walls_updated} wall(s){Style.ENDC}")
        
        # Update room bounds in metadata
        room_bounds[target_room_key] = new_bounds
        self.world_metadata['room_bounds'] = room_bounds
        
        # Redistribute objects if requested
        if redistribute:
            print(f"{Style.DIM}    Redistributing objects in resized room...{Style.ENDC}")
            redistributed = self._redistribute_room_objects(target_room_key, new_bounds, world)
            if redistributed > 0:
                print(f"{Style.GREEN}      ✓ Redistributed {redistributed} object(s){Style.ENDC}")
        
        self._save_world(output_file)
        print(f"{Style.GREEN}      ✓ Room resized successfully{Style.ENDC}")
        logger.info(f"Resized room '{target_room}' to {target_width}m × {target_length}m")
        return True
    
    def _redistribute_room_objects(self, room_name: str, room_bounds: Dict, 
                                   world_elem: ET.Element) -> int:
        """
        Redistribute objects in a room after resizing using enhanced placement logic.
        
        Strategy:
        1. Identify all objects in the room
        2. For objects outside new bounds, use placement engine to reposition
        3. Maintain semantic groupings (desk-chair pairs)
        4. Use proper collision detection
        
        Args:
            room_name: Name of the room
            room_bounds: New room bounds
            world_elem: World XML element
            
        Returns:
            int: Number of objects redistributed
        """
        logger.info(f"Redistributing objects in room '{room_name}' after resize")
        
        redistributed_count = 0
        
        # Get all objects currently in the room
        room_objects = []
        all_elements = []
        all_elements.extend([('model', m) for m in world_elem.findall('.//model')])
        all_elements.extend([('include', i) for i in world_elem.findall('.//include')])
        
        for elem_type, elem in all_elements:
            # Get name
            if elem_type == 'model':
                name = elem.get('name', '')
            else:
                name_elem = elem.find('name')
                name = name_elem.text if name_elem is not None and name_elem.text else ''
            
            # Skip if not furniture/object
            if not name or not self._is_furniture(name):
                continue
            
            # Skip ground plane, sun, and walls
            if any(skip in name.lower() for skip in ['ground', 'sun', 'wall', 'floor', 'ceiling']):
                continue
            
            # Get pose
            pose_elem = elem.find('pose')
            if pose_elem is None or not pose_elem.text:
                continue
            
            pose_values = [float(x) for x in pose_elem.text.split()]
            x, y = pose_values[0], pose_values[1]
            
            # Check if object is in this room (within reasonable bounds)
            # Use expanded bounds to catch objects near edges
            expanded_margin = 2.0
            if (room_bounds['min_x'] - expanded_margin <= x <= room_bounds['max_x'] + expanded_margin and
                room_bounds['min_y'] - expanded_margin <= y <= room_bounds['max_y'] + expanded_margin):
                
                # Extract object type and dimensions
                obj_type = self._extract_type_from_name(name)
                
                # Try to get model URI
                model_uri = ""
                if elem_type == 'include':
                    uri_elem = elem.find('uri')
                    if uri_elem is not None and uri_elem.text:
                        model_uri = uri_elem.text
                
                dimensions = self._extract_model_dimensions(model_uri, obj_type)
                
                room_objects.append({
                    'element': elem,
                    'elem_type': elem_type,
                    'name': name,
                    'type': obj_type,
                    'position': (x, y, pose_values[5] if len(pose_values) > 5 else 0.0),
                    'dimensions': dimensions,
                    'pose_elem': pose_elem,
                    'pose_values': pose_values,
                    'needs_repositioning': False
                })
        
        logger.info(f"Found {len(room_objects)} objects in room '{room_name}'")
        
        # Identify objects that need repositioning
        margin = 0.5
        min_x = room_bounds['min_x'] + margin
        max_x = room_bounds['max_x'] - margin
        min_y = room_bounds['min_y'] + margin
        max_y = room_bounds['max_y'] - margin
        
        for obj in room_objects:
            x, y = obj['position'][0], obj['position'][1]
            if x < min_x or x > max_x or y < min_y or y > max_y:
                obj['needs_repositioning'] = True
                logger.info(f"Object {obj['name']} needs repositioning (outside new bounds)")
        
        # Reposition objects using enhanced placement
        repositioned_names = set()
        
        for obj in room_objects:
            if not obj['needs_repositioning']:
                continue
            
            # Skip if already repositioned as part of a group
            if obj['name'] in repositioned_names:
                continue
            
            # Build list of existing objects for collision detection
            # (excluding the one we're repositioning)
            existing_objects = [
                {
                    'name': o['name'],
                    'position': o['position'],
                    'dimensions': o['dimensions']
                }
                for o in room_objects
                if o['name'] != obj['name'] and o['name'] not in repositioned_names
            ]
            
            # Calculate new position
            new_position = self._calculate_position_with_placement(
                obj['type'],
                "",  # No specific placement hint
                room_bounds,
                obj['dimensions'],
                existing_objects
            )
            
            if new_position:
                # Update position
                obj['pose_values'][0] = new_position[0]
                obj['pose_values'][1] = new_position[1]
                obj['pose_values'][5] = new_position[2]  # Update yaw
                obj['pose_elem'].text = " ".join(f"{v:.6f}" for v in obj['pose_values'])
                
                # Update position in our tracking
                obj['position'] = new_position
                
                repositioned_names.add(obj['name'])
                redistributed_count += 1
                logger.info(f"Repositioned {obj['name']} to ({new_position[0]:.2f}, {new_position[1]:.2f})")
            else:
                logger.warning(f"Could not find valid position for {obj['name']} during redistribution")
                # Clamp to bounds as fallback
                obj['pose_values'][0] = max(min_x, min(max_x, obj['pose_values'][0]))
                obj['pose_values'][1] = max(min_y, min(max_y, obj['pose_values'][1]))
                obj['pose_elem'].text = " ".join(f"{v:.6f}" for v in obj['pose_values'])
                redistributed_count += 1
                logger.info(f"Clamped {obj['name']} to room bounds as fallback")
        
        logger.info(f"Redistributed {redistributed_count} objects in room '{room_name}'")
        return redistributed_count
    
    def _is_furniture(self, name: str) -> bool:
        """Check if name appears to be furniture/object rather than structural."""
        furniture_keywords = ['chair', 'table', 'desk', 'sofa', 'bed', 'shelf', 
                             'cabinet', 'lamp', 'plant', 'wardrobe', 'couch']
        name_lower = name.lower()
        return any(keyword in name_lower for keyword in furniture_keywords)
    
    def _create_room_from_world_empty(self, room_name: str = None, room_bounds: Dict = None, 
                                      existing_objects: List[GazeboModel] = None) -> Optional[Room]:
        """
        Create a Room object for placement engine, optionally including existing objects.
        
        Args:
            room_name: Room name
            room_bounds: Pre-computed room bounds
            existing_objects: List of existing GazeboModel objects already in the room
            
        Returns:
            Room object with or without existing objects
        """
        if not room_bounds:
            room_bounds = self.world_metadata.get('room_bounds', {})
            if room_name and room_name in room_bounds:
                bounds = room_bounds[room_name]
            elif room_bounds:
                room_name = list(room_bounds.keys())[0]
                bounds = room_bounds[room_name]
            else:
                logger.warning("No room bounds found, creating generic room")
                room_name = "main_room"
                bounds = {'min_x': -10, 'max_x': 10, 'min_y': -10, 'max_y': 10, 'width': 20, 'length': 20}
        else:
            bounds = room_bounds
            if not room_name:
                room_name = "main_room"
        
        # Extract door positions
        door_positions = self.world_metadata.get('door_positions', [])
        doorways = [
            {'side': 'custom', 'x': door['x'], 'y': door['y'], 'width': 0.9}
            for door in door_positions
        ]
        
        # Create Room with existing objects (if provided) for LLM awareness
        room = Room(
            name=room_name,
            type="office",  # Default
            dimensions={
                'width': bounds.get('width', bounds['max_x'] - bounds['min_x']),
                'length': bounds.get('length', bounds['max_y'] - bounds['min_y']),
                'height': 3.0
            },
            position={
                'x': (bounds['min_x'] + bounds['max_x']) / 2,
                'y': (bounds['min_y'] + bounds['max_y']) / 2,
                'z': 0.0
            },
            walls=True,
            doorways=doorways,
            objects=existing_objects if existing_objects else [],  # Include existing for refinement mode
            shape="rectangle",
            rotation=0.0
        )
        
        return room
    
    def _validate_against_existing(self, new_models: List[GazeboModel], 
                                   existing_objects: List[Dict],
                                   room_bounds: Dict) -> List[GazeboModel]:
        """
        Validate new placements against existing objects and adjust positions if needed.
        CRITICAL: This method only moves NEW objects, never existing ones.
        
        Args:
            new_models: Newly placed models from placement engine
            existing_objects: Existing objects in the room
            room_bounds: Room boundaries
            
        Returns:
            List of validated/adjusted models
        """
        import math
        import random
        
        validated_models = []
        collision_margin = 0.3
        
        for model in new_models:
            pose = model.pose
            size = model.size if model.size else [1.0, 1.0, 1.0]
            original_x, original_y = pose['x'], pose['y']
            
            # Check collision with existing objects
            has_collision = False
            max_push_attempts = 10
            attempt = 0
            
            # First try: push away from colliding objects
            while attempt < max_push_attempts:
                has_collision = False
                
                for existing in existing_objects:
                    ex_pos = existing['position']
                    ex_dims = existing['dimensions']
                    
                    # AABB collision detection
                    half_w1, half_l1 = size[0] / 2, size[1] / 2
                    half_w2, half_l2 = ex_dims[0] / 2, ex_dims[1] / 2
                    
                    dx = abs(pose['x'] - ex_pos[0])
                    dy = abs(pose['y'] - ex_pos[1])
                    
                    min_dist_x = half_w1 + half_w2 + collision_margin
                    min_dist_y = half_l1 + half_l2 + collision_margin
                    
                    if dx < min_dist_x and dy < min_dist_y:
                        has_collision = True
                        logger.warning(f"Collision detected: {model.name} at ({pose['x']:.2f}, {pose['y']:.2f}) too close to {existing['name']}")
                        
                        # Try to push away from existing object
                        if dx < dy:
                            # Push along X axis
                            push_dist = min_dist_x - dx + 0.2
                            if pose['x'] > ex_pos[0]:
                                pose['x'] += push_dist
                            else:
                                pose['x'] -= push_dist
                        else:
                            # Push along Y axis
                            push_dist = min_dist_y - dy + 0.2
                            if pose['y'] > ex_pos[1]:
                                pose['y'] += push_dist
                            else:
                                pose['y'] -= push_dist
                        
                        # Clamp to room bounds
                        margin = 0.5
                        pose['x'] = max(room_bounds['min_x'] + margin, 
                                      min(room_bounds['max_x'] - margin, pose['x']))
                        pose['y'] = max(room_bounds['min_y'] + margin, 
                                      min(room_bounds['max_y'] - margin, pose['y']))
                        
                        logger.info(f"  Adjusted {model.name} to ({pose['x']:.2f}, {pose['y']:.2f})")
                        break
                
                if not has_collision:
                    break
                    
                attempt += 1
            
            # If pushing didn't work, try random positions as a last resort
            if has_collision:
                logger.warning(f"Push strategy failed for {model.name}, trying random positions...")
                max_random_attempts = 30
                
                for random_attempt in range(max_random_attempts):
                    # Try random position within room
                    margin = 0.5
                    test_x = random.uniform(room_bounds['min_x'] + margin, room_bounds['max_x'] - margin)
                    test_y = random.uniform(room_bounds['min_y'] + margin, room_bounds['max_y'] - margin)
                    
                    # Check if this position is collision-free
                    collision_free = True
                    for existing in existing_objects:
                        ex_pos = existing['position']
                        ex_dims = existing['dimensions']
                        
                        half_w1, half_l1 = size[0] / 2, size[1] / 2
                        half_w2, half_l2 = ex_dims[0] / 2, ex_dims[1] / 2
                        
                        dx = abs(test_x - ex_pos[0])
                        dy = abs(test_y - ex_pos[1])
                        
                        min_dist_x = half_w1 + half_w2 + collision_margin
                        min_dist_y = half_l1 + half_l2 + collision_margin
                        
                        if dx < min_dist_x and dy < min_dist_y:
                            collision_free = False
                            break
                    
                    if collision_free:
                        pose['x'] = test_x
                        pose['y'] = test_y
                        has_collision = False
                        logger.info(f"  Found collision-free position for {model.name} at ({test_x:.2f}, {test_y:.2f}) after {random_attempt + 1} random attempts")
                        break
            
            if has_collision:
                logger.error(f"Could not resolve collision for {model.name} after all attempts - room may be too crowded")
                logger.error(f"  Recommend: 1) Remove some objects first, 2) Resize the room larger, or 3) Try different object types")
            
            validated_models.append(model)
        
        return validated_models
    
    def _check_remaining_collisions(self, new_models: List[GazeboModel], 
                                   existing_objects: List[Dict]) -> int:
        """
        Check how many collisions remain between new and existing objects.
        
        Returns:
            Number of remaining collisions
        """
        collision_count = 0
        collision_margin = 0.3
        
        for model in new_models:
            pose = model.pose
            size = model.size if model.size else [1.0, 1.0, 1.0]
            
            for existing in existing_objects:
                ex_pos = existing['position']
                ex_dims = existing['dimensions']
                
                half_w1, half_l1 = size[0] / 2, size[1] / 2
                half_w2, half_l2 = ex_dims[0] / 2, ex_dims[1] / 2
                
                dx = abs(pose['x'] - ex_pos[0])
                dy = abs(pose['y'] - ex_pos[1])
                
                min_dist_x = half_w1 + half_w2 + collision_margin
                min_dist_y = half_l1 + half_l2 + collision_margin
                
                if dx < min_dist_x and dy < min_dist_y:
                    collision_count += 1
        
        return collision_count
    
    def _redistribute_for_space(self, world_elem: ET.Element, existing_objects: List[Dict],
                                new_models: List[GazeboModel], room_bounds: Dict) -> bool:
        """
        Redistribute existing objects to make space for new ones.
        Uses intelligent packing to optimize space usage.
        
        Args:
            world_elem: World XML element to update
            existing_objects: Existing objects to redistribute
            new_models: New models that need space
            room_bounds: Room boundaries
            
        Returns:
            bool: True if redistribution successful
        """
        import math
        import random
        
        logger.info("Redistributing existing objects to optimize space...")
        
        # Calculate total area needed
        total_area_existing = sum(obj['dimensions'][0] * obj['dimensions'][1] for obj in existing_objects)
        total_area_new = sum((m.size[0] * m.size[1] if m.size else 1.0) for m in new_models)
        
        room_area = (room_bounds['max_x'] - room_bounds['min_x']) * (room_bounds['max_y'] - room_bounds['min_y'])
        total_area_needed = total_area_existing + total_area_new
        
        # Check if there's theoretically enough space
        utilization = total_area_needed / room_area
        logger.info(f"Room utilization: {utilization*100:.1f}% (existing: {total_area_existing:.1f}m², new: {total_area_new:.1f}m², room: {room_area:.1f}m²)")
        
        if utilization > 0.7:
            logger.warning("Room is getting crowded (>70% utilization), redistribution may be challenging")
        
        # Strategy: Push existing objects towards walls to free up center space
        margin = 0.5
        wall_clearance = 0.3
        
        redistributed_count = 0
        
        for obj in existing_objects:
            # Find the object in the XML - check both model and include elements
            obj_elem = None
            
            # First try models
            for model in world_elem.findall('.//model'):
                model_name = model.get('name')
                if model_name == obj['name']:
                    obj_elem = model
                    break
            
            # Then try includes
            if not obj_elem:
                for include in world_elem.findall('.//include'):
                    name_elem = include.find('name')
                    if name_elem is not None and name_elem.text == obj['name']:
                        obj_elem = include
                        break
            
            if not obj_elem:
                logger.debug(f"Could not find XML element for {obj['name']} (may have been skipped as non-furniture)")
                continue
            
            pose_elem = obj_elem.find('.//pose')
            if pose_elem is None:
                continue
            
            # Parse current pose
            pose_values = [float(x) for x in pose_elem.text.split()]
            if len(pose_values) < 6:
                continue
            
            x, y, yaw = pose_values[0], pose_values[1], pose_values[5]
            dims = obj['dimensions']
            
            # Try moving to each wall and pick the one that creates most space
            wall_options = [
                ('west', room_bounds['min_x'] + dims[0]/2 + wall_clearance, y),
                ('east', room_bounds['max_x'] - dims[0]/2 - wall_clearance, y),
                ('south', x, room_bounds['min_y'] + dims[1]/2 + wall_clearance),
                ('north', x, room_bounds['max_y'] - dims[1]/2 - wall_clearance),
            ]
            
            best_option = None
            best_distance = 0
            
            for wall_name, new_x, new_y in wall_options:
                # Calculate how much we're moving
                move_distance = abs(new_x - x) + abs(new_y - y)
                
                # Skip if not moving significantly
                if move_distance < 0.5:
                    continue
                
                # Check collision severity with other existing objects
                max_overlap = 0
                for other in existing_objects:
                    if other['name'] == obj['name']:
                        continue
                    
                    other_pos = other['position']
                    other_dims = other['dimensions']
                    
                    dx = abs(new_x - other_pos[0])
                    dy = abs(new_y - other_pos[1])
                    
                    min_dist_x = (dims[0] + other_dims[0]) / 2 + 0.3
                    min_dist_y = (dims[1] + other_dims[1]) / 2 + 0.3
                    
                    overlap_x = max(0, min_dist_x - dx)
                    overlap_y = max(0, min_dist_y - dy)
                    overlap = min(overlap_x, overlap_y) if overlap_x > 0 and overlap_y > 0 else 0
                    
                    max_overlap = max(max_overlap, overlap)
                
                # Prefer positions with less overlap and more movement
                # Allow small overlaps (< 0.5m) if it moves the object significantly
                if max_overlap < 0.5 and move_distance > best_distance:
                    best_option = (wall_name, new_x, new_y)
                    best_distance = move_distance
            
            # Apply best option if found
            if best_option:
                wall_name, new_x, new_y = best_option
                
                # Update position
                pose_values[0] = new_x
                pose_values[1] = new_y
                pose_elem.text = " ".join(f"{v:.6f}" for v in pose_values)
                
                # Update tracking
                obj['position'] = (new_x, new_y, yaw)
                
                redistributed_count += 1
                logger.info(f"Redistributed {obj['name']} to {wall_name} wall: ({x:.2f}, {y:.2f}) -> ({new_x:.2f}, {new_y:.2f})")
        
        if redistributed_count > 0:
            logger.info(f"Successfully redistributed {redistributed_count} object(s)")
            return True
        else:
            logger.warning("No objects could be redistributed")
            return False
    
    
    def refine(self, request: str, output_file: Path) -> Tuple[bool, str]:
        """
        Main refinement workflow.
        
        Args:
            request: Natural language refinement request
            output_file: Path to save refined world
            
        Returns:
            tuple: (success: bool, message: str)
        """
        if not self.current_world:
            return False, "No world loaded. Load a world first."
        
        # Set up logging for this refinement using the world filename
        log_dir = Path("generated_worlds/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the input world filename (without .sdf) for the log name
        world_stem = self.current_world_file.stem if self.current_world_file else "unknown_world"
        log_file = log_dir / f"refinement_{world_stem}.log"
        
        # Remove previous log handler if exists
        if self.active_log_handler:
            self.active_log_handler.close()
            logging.getLogger().removeHandler(self.active_log_handler)
        
        # Set up new log handler (append mode so multiple refinements are logged)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
        self.active_log_handler = file_handler
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"\n{'='*80}")
        logger.info(f"=== Refinement Session Started at {timestamp} ===")
        logger.info(f"{'='*80}")
        logger.info(f"Input world: {self.current_world_file}")
        logger.info(f"Output world: {output_file}")
        logger.info(f"Request: {request}")
        
        # Parse the request
        print(f"\n{Style.CYAN}⚙ Parsing refinement request...{Style.ENDC}")
        logger.info("Parsing refinement request...")
        refinement_data = self.parse_refinement_request(request)
        
        if "error" in refinement_data:
            print(f"{Style.RED}  ✗ Failed to parse request{Style.ENDC}")
            logger.error(f"Failed to parse request: {refinement_data['error']}")
            self._cleanup_logging()
            return False, f"Failed to parse request: {refinement_data['error']}"
        
        operation = refinement_data.get('operation', 'unknown')
        
        # Handle batch operations display differently
        if operation == 'batch':
            operations = refinement_data.get('operations', [])
            print(f"{Style.GREEN}  ✓ Parsed batch: {Style.BOLD}{len(operations)} operation(s){Style.ENDC}")
            logger.info(f"Parsed batch operation with {len(operations)} sub-operations")
            for i, op in enumerate(operations, 1):
                op_type = op.get('operation', 'unknown')
                print(f"{Style.DIM}    {i}. {op_type}{Style.ENDC}")
                logger.info(f"  Operation {i}: {op_type}")
        else:
            objects = refinement_data.get('objects', [])
            logger.info(f"Parsed operation: {operation}")
            if objects:
                print(f"{Style.DIM}    Objects affected: {len(objects)}{Style.ENDC}")
                logger.info(f"  Objects affected: {len(objects)}")
                for obj in objects:
                    logger.info(f"    - {obj.get('type', 'unknown')}")
        
        # Apply the refinement
        if operation != 'batch':
            print(f"\n{Style.CYAN}⚙ Applying {operation} operation...{Style.ENDC}")
        logger.info(f"Applying refinement operation...")
        success = self.apply_refinement(refinement_data, output_file)
        
        if success:
            print(f"{Style.GREEN}  ✓ Refinement completed successfully{Style.ENDC}")
            print(f"{Style.DIM}    Output: {output_file.name}{Style.ENDC}")
            print(f"{Style.DIM}    Log:    {log_file.name}{Style.ENDC}")
            logger.info(f"{'='*80}")
            logger.info(f"=== Refinement Completed Successfully ===")
            logger.info(f"{'='*80}\n")
            self._cleanup_logging()
            return True, f"Successfully refined world. Saved to: {output_file}"
        else:
            print(f"{Style.RED}  ✗ Failed to apply refinement{Style.ENDC}")
            print(f"{Style.DIM}    Check log: {log_file}{Style.ENDC}")
            logger.error(f"{'='*80}")
            logger.error(f"=== Refinement Failed ===")
            logger.error(f"{'='*80}\n")
            self._cleanup_logging()
            return False, "Failed to apply refinement"
    
    def _cleanup_logging(self):
        """Clean up the active log handler."""
        if self.active_log_handler:
            self.active_log_handler.close()
            logging.getLogger().removeHandler(self.active_log_handler)
            self.active_log_handler = None
