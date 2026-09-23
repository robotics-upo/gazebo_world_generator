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
from .refinement import WorldStateParser, RefinementPlanner, OperationExecutor

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
        
        # Components (initialized when world is loaded)
        self.parser = None
        self.planner = None
        self.executor = None
        
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
            
            # Initialize parser and extract metadata
            self.parser = WorldStateParser(self.current_world)
            self.world_metadata = self.parser.extract_metadata()
            
            # Initialize planner
            self.planner = RefinementPlanner(self.llm, self.prompt_manager)
            
            # Initialize executor (passes self for delegation)
            self.executor = OperationExecutor(self)
            
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
        # Store original request for semantic context detection (used by executor)
        self.original_request = request
        
        # Delegate to planner
        if not self.planner:
            logger.error("Planner not initialized. Load a world first.")
            return {"error": "Planner not initialized"}
        
        return self.planner.parse_request(request, self.world_metadata, request)
    
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
        # Delegate to executor
        if not self.executor:
            logger.error("Executor not initialized. Load a world first.")
            return False
        
        return self.executor.execute_operation(refinement_data, output_file)
    

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
