"""
Natural Placement Engine

Handles intelligent object placement within rooms using LLM-guided positioning.
Includes validation, collision detection, and warehouse-specific optimizations.
"""

import logging
import math
import json
import random
from typing import Dict, List, Optional, Tuple
import tiktoken
from transformers import AutoTokenizer

from gazebo_world_generator.src.core.data_models import Room, GazeboModel
from gazebo_world_generator.src.utils import llm_utils
from gazebo_world_generator.src.utils.file_utils import find_gazebo_model_path
from gazebo_world_generator.src.utils.sdf_utils import extract_model_metadata
from gazebo_world_generator.src.utils.sdf_parser import SDFDimensionExtractor
from gazebo_world_generator.src.utils.collision_detection import CollisionDetector
from gazebo_world_generator.src.placement.semantic_grouping import SemanticGroupingEngine
from gazebo_world_generator.src.config.settings import DEFAULT_MODEL
from gazebo_world_generator.src.config.validated_settings import PlacementConfig, RoomConfig
from gazebo_world_generator.src.utils.json_validator import JSONValidator
from gazebo_world_generator.src.prompts.manager import PromptManager


logger = logging.getLogger(__name__)

CONTEXT_LIMIT = 8192
SAFETY_MARGIN = 256 

PLACEMENT_PLAN_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "pose": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
                    "roll": {"type": "number"}, "pitch": {"type": "number"}, "yaw": {"type": "number"}
                },
                "required": ["x", "y", "z", "roll", "pitch", "yaw"]
            }
        },
        "required": ["type", "pose"]
    }
}


class NaturalPlacementEngine:
    """Engine for natural and intelligent object placement in rooms."""

    def __init__(self, model_db, online_db=None, llm_interface=None,
                 placement_config: Optional[PlacementConfig] = None,
                 room_config: Optional[RoomConfig] = None):
        self.model_db = model_db
        self.llm_interface = llm_interface
        self.sdf_extractor = SDFDimensionExtractor()
        self.semantic_grouping = SemanticGroupingEngine(llm_interface) if llm_interface else None

        # Use provided configs or create defaults
        self.config = placement_config if placement_config else PlacementConfig()
        self.room_config = room_config if room_config else RoomConfig()
        
        # Initialize collision detector (will be properly configured when placing objects)
        self.collision_detector = None

        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
        
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
                logger.warning(f"Failed to initialize PromptManager in NaturalPlacementEngine: {e}")
                self.prompt_manager = None

        # Cache for extracted model dimensions to avoid re-parsing
        self.model_dimensions_cache = {}
        self.model_offsets_cache = {}
        self.object_sizes = {
            'desk': (1.2, 0.6, 0.55), 'chair': (0.6, 0.6, 0.9),
            'table': (1.5, 0.8, 0.75), 'shelf': (0.8, 0.3, 1.8),
            'bookshelf': (0.8, 0.3, 1.8), 'monitor': (0.5, 0.2, 0.4),
            'keyboard': (0.4, 0.15, 0.05), 'plant': (0.4, 0.4, 0.6), 'lamp': (0.4, 0.4, 1.5),
            'storage_rack': (3.6, 0.6, 1.8), 'shelving_unit': (3.6, 0.6, 1.8),
            'pallet': (1.2, 0.8, 0.144),
            'bed': (2.0, 1.5, 0.5), 'nightstand': (0.5, 0.4, 0.5), 'wardrobe': (1.2, 0.6, 2.0),
            'sofa': (2.0, 0.9, 0.8), 'couch': (2.0, 0.9, 0.8), 'tv_stand': (1.2, 0.4, 0.5),
            'coffee_table': (1.0, 0.6, 0.4), 'dining_table': (1.5, 0.9, 0.75),
            'dresser': (1.0, 0.5, 1.0), 'armchair': (0.8, 0.8, 0.9),
            'default': (0.5, 0.5, 0.5)
        }

        # Object orientation guidance - default facing directions and natural orientations
        self.object_orientations = {
            'desk': {
                'default_facing': 'south',  # Most desks face south (user sits facing north)
                'natural_yaw': 0.0,  # Facing north (working direction)
                'description': 'Desk front (where user sits) should face the working area'
            },
            'chair': {
                'default_facing': 'north',  # Chair faces the desk/work surface
                'natural_yaw': 0.0,  # Facing north toward desk
                'description': 'Chair should face toward the desk or work area'
            },
            'bookshelf': {
                'default_facing': 'east',  # Bookshelf face/opening toward room center
                'natural_yaw': 1.57,  # Facing east into room
                'description': 'Bookshelf front should face into the room for access'
            },
            'table': {
                'default_facing': 'north',
                'natural_yaw': 0.0,
                'description': 'Table oriented for optimal access'
            },
            'default': {
                'default_facing': 'north',
                'natural_yaw': 0.0,
                'description': 'Object facing toward room center'
            }
        }

        # Model-specific orientation corrections for desk-chair positioning
        self.model_orientation_corrections = {
            'Desk': {
                'placement_direction': 'right_side',  # To the right of desk (when viewed from behind)
                'chair_offset_x': 0.6,  # Distance along desk's right (local X)
                'chair_offset_y': 0.0,  # Centered with desk front-back
                'chair_yaw_offset': -math.pi / 2  # Face left (perpendicular to desk)
            },
            'Office_Desk': {
                'placement_direction': 'right_side',  # To the right of desk
                'chair_offset_x': 0.834,  # 0.834m to the East (right side) from manual correction
                'chair_offset_y': -0.081,  # Slightly toward south (front of desk)
                'chair_yaw_offset': -math.pi / 2  # Face East (yaw=-π/2), perpendicular to desk
            },
            'default': {
                'placement_direction': 'right_side',
                'chair_offset_x': 0.75,
                'chair_offset_y': 0.0,
                'chair_yaw_offset': -math.pi / 2
            }
        }

    def _get_placement_rules(self, room_type: str) -> str:
        """
        Get placement rules for a given room type from PromptManager.
        
        Args:
            room_type: Type of room (office, warehouse, etc.)
            
        Returns:
            Placement rules string
        """
        if not self.prompt_manager:
            logger.warning("PromptManager not available, using generic rules")
            return "• LOGICAL GROUPING: Arrange related objects together in functional clusters.\n• NAVIGATION SPACE: Maintain 1.5m minimum clearance for human movement.\n• ROOM BALANCE: Distribute objects evenly, avoid clustering in corners.\n• NATURAL FLOW: Orient objects to face room center or primary activity areas."
        
        try:
            # Try to load room-specific placement rules
            template_name = f"placement_rules_{room_type}"
            rules = self.prompt_manager.render(template_name)
            return rules.strip()
        except Exception:
            # Fall back to default rules if room-specific template doesn't exist
            try:
                rules = self.prompt_manager.render("placement_rules_default")
                return rules.strip()
            except Exception as e:
                logger.warning(f"Failed to load placement rules: {e}")
                return "• LOGICAL GROUPING: Arrange related objects together in functional clusters.\n• NAVIGATION SPACE: Maintain 1.5m minimum clearance for human movement.\n• ROOM BALANCE: Distribute objects evenly, avoid clustering in corners.\n• NATURAL FLOW: Orient objects to face room center or primary activity areas."

    def _resolve_models_early(self, objects_to_place: List[Dict], room: Room) -> Dict[str, Dict]:
        """
        Resolve models and extract dimensions before placement.

        Resolves actual 3D models first to obtain real sizes instead of guessing during placement.

        Returns:
            Dict mapping object_type -> {
                'uri': model URI,
                'model_name': clean model name,
                'dimensions': (width, length, height),
                'description': model description,
                'metadata': full metadata
            }
        """
        from gazebo_world_generator.src.utils.file_utils import find_gazebo_model_path
        from gazebo_world_generator.src.utils.sdf_utils import extract_model_metadata

        model_info_cache = {}
        object_types = set(obj['type'] for obj in objects_to_place)

        logger.info(f"🔍 Resolving {len(object_types)} model types early...")

        for obj_type in object_types:
            # Find best model for this object type
            model_uri = self.model_db.find_best_model(obj_type, room.type)

            if not model_uri:
                logger.warning(f"Could not resolve model for '{obj_type}', using fallback")
                model_info_cache[obj_type] = {
                    'uri': f"model://{obj_type}",
                    'model_name': obj_type,
                    'dimensions': self.object_sizes.get(obj_type, self.object_sizes['default']),
                    'description': f"A standard {obj_type}",
                    'metadata': {}
                }
                continue

            model_name = model_uri.replace("model://", "")

            # Extract metadata and dimensions
            model_path = find_gazebo_model_path(model_name)
            metadata = extract_model_metadata(model_path) if model_path else {}
            description = metadata.get("description", f"A standard {obj_type}")

            # Get actual dimensions from SDF
            dimensions = self.get_actual_model_dimensions(obj_type, model_name, room.type)

            model_info_cache[obj_type] = {
                'uri': model_uri,
                'model_name': model_name,
                'dimensions': dimensions,
                'description': description,
                'metadata': metadata
            }

            logger.debug(f"  ✓ {obj_type} -> {model_name} ({dimensions[0]:.2f}×{dimensions[1]:.2f}×{dimensions[2]:.2f}m)")

        logger.info(f"✅ Model resolution complete - all dimensions known")
        return model_info_cache

    def get_actual_model_dimensions(self, object_type: str, model_name: str = None, room_type: str = None) -> Tuple[float, float, float]:
        """
        Get actual model dimensions from SDF files or fallback to hardcoded values.

        Args:
            object_type: The object type (e.g., 'shelving_unit')
            model_name: The specific model name (e.g., 'Shelf with ARUco boxes')
            room_type: The room type for context-aware model resolution

        Returns:
            Tuple of (width, length, height) in meters
        """
        cache_key = f"{object_type}_{model_name or 'default'}_{room_type or 'default'}"

        if cache_key in self.model_dimensions_cache:
            return self.model_dimensions_cache[cache_key]

        dimensions = None
        resolved_model_name = model_name

        if model_name:
            try:
                model_path = find_gazebo_model_path(model_name)
                if model_path:
                    # Try to get complete bounding box with offset
                    bbox_result = self.sdf_extractor.extract_model_bounding_box(model_path)
                    if bbox_result:
                        dimensions, offset = bbox_result
                        # Store with multiple cache keys for easier retrieval
                        self.model_offsets_cache[cache_key] = offset
                        simple_key = f"{object_type}_resolved"
                        self.model_offsets_cache[simple_key] = offset
                        simple_dim_key = f"{object_type}_default_{room_type or 'default'}"
                        self.model_dimensions_cache[simple_dim_key] = dimensions
                        logger.info(f"Extracted dims for '{model_name}': {dimensions}, offset: {offset}")
                    else:
                        dimensions = self.sdf_extractor.extract_model_dimensions(model_path)
                        if dimensions:
                            logger.info(f"Extracted dimensions for '{model_name}': {dimensions}")
            except Exception as e:
                logger.warning(f"Failed to extract dimensions for '{model_name}': {e}")

        # If no specific model, try to find the best available model for this object type
        if not dimensions and not model_name:
            try:
                best_model = self.model_db.find_best_model(object_type, room_type)
                if best_model and best_model.startswith("model://"):
                    resolved_model_name = best_model.replace("model://", "")
                    model_path = find_gazebo_model_path(resolved_model_name)
                    if model_path:
                        # Try to get complete bounding box with offset
                        bbox_result = self.sdf_extractor.extract_model_bounding_box(model_path)
                        if bbox_result:
                            dimensions, offset = bbox_result
                            # Store with multiple cache keys for easier retrieval
                            self.model_offsets_cache[cache_key] = offset
                            simple_key = f"{object_type}_resolved"
                            self.model_offsets_cache[simple_key] = offset
                            simple_dim_key = f"{object_type}_default_{room_type or 'default'}"
                            self.model_dimensions_cache[simple_dim_key] = dimensions
                            logger.info(f"Extracted dims from '{resolved_model_name}' for '{object_type}': {dimensions}, offset: {offset}")
                        else:
                            dimensions = self.sdf_extractor.extract_model_dimensions(model_path)
                            if dimensions:
                                logger.info(f"Extracted dimensions from resolved model '{resolved_model_name}' for '{object_type}': {dimensions}")
            except Exception as e:
                logger.debug(f"Failed to resolve and extract dimensions for '{object_type}': {e}")

        # Fallback to hardcoded values
        if not dimensions:
            dimensions = self.object_sizes.get(object_type, self.object_sizes['default'])
            logger.debug(f"Using fallback dimensions for '{object_type}': {dimensions}")

        # Cache the result
        self.model_dimensions_cache[cache_key] = dimensions
        return dimensions

    def get_effective_object_bounds(self, object_type: str, center_x: float, center_y: float, room_type: str = None, yaw: float = 0.0) -> Tuple[float, float, float, float]:
        """
        Get the effective bounding box of an object accounting for pose offsets and rotation.

        Args:
            object_type: The object type
            center_x, center_y: Where we're placing the model origin
            room_type: The room type for model resolution
            yaw: The yaw rotation in radians

        Returns:
            Tuple of (min_x, max_x, min_y, max_y) - the actual space the object occupies
        """
        import math

        # Get the dimensions
        dimensions = self.get_actual_model_dimensions(object_type, room_type=room_type)
        half_w, half_l = dimensions[0] / 2, dimensions[1] / 2

        # Get the bounding box offset (if available)
        # Try to find the offset in cache - check multiple possible keys
        offset = None

        # Try different cache key formats
        possible_keys = [
            f"{object_type}_resolved",  # Simple key for resolved models
            f"{object_type}_default_{room_type or 'default'}",  # Most common case
            f"{object_type}_None_{room_type or 'default'}",
            f"{object_type}_default_default",
            f"{object_type}_None_default",
        ]

        logger.debug(f"Looking for offset for '{object_type}', room_type='{room_type}'")
        logger.debug(f"Cache contents: {list(self.model_offsets_cache.keys())}")

        for key in possible_keys:
            if key in self.model_offsets_cache:
                offset = self.model_offsets_cache[key]
                logger.info(f"Found offset {offset} for '{object_type}' using cache_key: {key}")
                break

        if offset is None:
            offset = (0, 0, 0)
            logger.debug(f"No offset found for '{object_type}', using default (0, 0, 0)")

        # Rotate the offset by the yaw angle
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotated_offset_x = offset[0] * cos_yaw - offset[1] * sin_yaw
        rotated_offset_y = offset[0] * sin_yaw + offset[1] * cos_yaw

        # Calculate the actual center of the bounding box after rotation
        bbox_center_x = center_x + rotated_offset_x
        bbox_center_y = center_y + rotated_offset_y

        # When rotated, we need to compute the axis-aligned bounding box (AABB)
        # For a rotated rectangle, we need to check all 4 corners
        corners = [
            (-half_w, -half_l),
            (half_w, -half_l),
            (-half_w, half_l),
            (half_w, half_l)
        ]

        # Rotate each corner and find min/max
        rotated_corners = []
        for cx, cy in corners:
            rx = cx * cos_yaw - cy * sin_yaw + bbox_center_x
            ry = cx * sin_yaw + cy * cos_yaw + bbox_center_y
            rotated_corners.append((rx, ry))

        xs = [c[0] for c in rotated_corners]
        ys = [c[1] for c in rotated_corners]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        return (min_x, max_x, min_y, max_y)

    def place_objects_in_room(self, room: Room, world_map: str, objects_to_place: List[Dict], name_generator, room_number: int = 0, total_rooms: int = 0) -> List[GazeboModel]:
        """
        Main method to place all objects using model-aware placement.

        Workflow:
        1. Early model resolution with dimension extraction
        2. Semantic grouping with model awareness
        3. LLM placement planning with real dimensions
        4. LLM self-correction review
        5. Pre-collision wall snapping
        6. Collision resolution
        7. Semantic adjustment
        8. Final verification
        """

        if not objects_to_place:
            logger.debug(f"No objects to place in '{room.name}'. Skipping placement.")
            return []

        # Show progress for multi-room scenarios
        if total_rooms > 1:
            logger.info(f"Placing {len(objects_to_place)} objects in '{room.name}' ({room.type}) [{room_number}/{total_rooms}]")
        else:
            logger.info(f"Placing {len(objects_to_place)} objects in '{room.name}' ({room.type}).")

        # Resolve models early
        logger.debug("Resolving models early...")
        model_info_cache = self._resolve_models_early(objects_to_place, room)

        # Semantic grouping and LLM placement with model info
        placement_plan = self._get_llm_placement_plan(room, objects_to_place, model_info_cache) if self.llm_interface else None

        if not placement_plan:
            logger.warning("⚠️ LLM generation failed. Using intelligent fallback placement.")
            placement_plan = self._create_fallback_plan(room, objects_to_place)
        else:
            logger.debug(f"Initial LLM plan: {len(placement_plan)} objects")
            # LLM self-correction phase: let LLM review and fix its own placement
            if self.llm_interface:
                placement_plan = self._llm_self_correction_phase(placement_plan, room)
                logger.debug(f"After self-correction: {len(placement_plan)} objects")

        # Validation with optimized order
        final_plan = self._validate_and_correct_plan(placement_plan, room, model_info_cache)
        logger.debug(f"Final plan after validation: {len(final_plan)} objects")

        return self._create_models_from_plan(final_plan, room, name_generator)
    
    def place_all_objects(self, rooms: List[Room], name_generator) -> List[GazeboModel]:
        """
        Orchestrates the placement of all objects in all rooms using direct spatial understanding.
        """
        all_placed_models = []
        
        # Count non-corridor rooms with objects
        rooms_to_furnish = [r for r in rooms if r.type != "corridor" and r.objects]
        total_rooms = len(rooms_to_furnish)

        # Iterate through each room and place its objects
        current_room_num = 0
        for room in rooms:
            # Skip corridors - they should remain empty for navigation
            if room.type == "corridor":
                logger.debug(f"Skipping object placement in corridor '{room.name}' - keeping it clear for navigation.")
                continue

            if not room.objects:
                logger.debug(f"No objects to place in '{room.name}'. Skipping.")
                continue

            current_room_num += 1
            
            # Use the optimized placement workflow
            newly_placed_models = self.place_objects_in_room(
                room, "", room.objects, name_generator, current_room_num, total_rooms
            )
            all_placed_models.extend(newly_placed_models)

        return all_placed_models

    def _get_model_visual_info(self, model_name: str, obj_type: str) -> Dict[str, str]:
        """Get visual/geometric information about specific models to help with orientation."""

        # Model-specific visual information based on common Gazebo models
        model_visuals = {
            # Desk models
            'Desk': {
                'layout': 'Rectangular desk with drawers on one side, open leg space on user side',
                'front_description': 'User sits facing the desk surface (south side faces user)'
            },
            'Office Desk': {
                'layout': 'Modern office desk with clean front panel, drawers/storage on back',
                'front_description': 'Clean front panel faces user (south), drawers face away (north)'
            },

            # Chair models
            'OfficeChairBlack': {
                'layout': 'Office chair with high backrest and armrests',
                'front_description': 'Seat faces forward, backrest faces away from user workspace'
            },
            'Chair': {
                'layout': 'Standard chair with backrest',
                'front_description': 'Seat faces forward toward table/desk'
            },
            'VisitorChair': {
                'layout': 'Guest chair with backrest',
                'front_description': 'Seat faces forward toward meeting area'
            },

            # Storage models
            'bookshelf': {
                'layout': 'Tall rectangular bookshelf with open shelves facing forward',
                'front_description': 'Shelves open toward room for book access (east when against west wall)'
            },
            'SquareShelf': {
                'layout': 'Square storage unit with open compartments',
                'front_description': 'Open compartments face toward accessible area'
            }
        }

        # Get specific model info or fall back to generic type info
        if model_name in model_visuals:
            return model_visuals[model_name]

        # Generic fallbacks by object type
        generic_visuals = {
            'desk': {
                'layout': 'Rectangular work surface with user side and storage side',
                'front_description': 'User side faces toward chair/workspace'
            },
            'chair': {
                'layout': 'Seat with backrest for sitting',
                'front_description': 'Seat faces toward desk/table/activity area'
            },
            'bookshelf': {
                'layout': 'Vertical storage with open shelves',
                'front_description': 'Shelves face into room for accessibility'
            },
            'table': {
                'layout': 'Flat surface for activities',
                'front_description': 'Surface oriented for optimal access'
            }
        }

        return generic_visuals.get(obj_type, {
            'layout': 'Standard object with front and back',
            'front_description': 'Front faces toward activity area'
        })

    def _get_llm_placement_plan(self, room: Room, objects_to_place: List[Dict], model_info_cache: Dict[str, Dict] = None) -> List[Dict]:
        """
        Generate placement plan with semantic grouping using real model information.

        Args:
            room: Room to place objects in
            objects_to_place: List of objects to place
            model_info_cache: Pre-resolved model information with dimensions
        """
        object_context_str = ""
        total_object_count = 0

        object_summary = {}
        for obj in objects_to_place:
            obj_type = obj['type']
            count = obj.get('count', 1)
            object_summary[obj_type] = object_summary.get(obj_type, 0) + count
            total_object_count += count

        # Identify semantic groups for office layouts
        semantic_groups = []
        group_instructions = ""
        
        # Warehouse layouts skip semantic grouping for systematic grid placement
        is_warehouse_storage = (room.type in ['warehouse', 'storage', 'staging'] and
                              any('rack' in obj['type'].lower() or 'shelf' in obj['type'].lower() or 'pallet' in obj['type'].lower()
                                  for obj in objects_to_place))
        
        if self.semantic_grouping and not is_warehouse_storage:
            logger.info(f"Analyzing object relationships for {total_object_count} objects...")
            groups = self.semantic_grouping.identify_object_groups(
                objects_to_place, room.type, room.name, model_info_cache
            )
            if groups:
                semantic_groups = self.semantic_grouping.expand_groups_to_instances(groups, objects_to_place)

                # Store semantic groups for post-placement enforcement
                self._semantic_groups = semantic_groups

                logger.info(f"✅ Created {len(semantic_groups)} functional groups")

                # Build group instructions for the LLM
                group_instructions = "\n\nSEMANTIC GROUPS (objects that should be placed together):\n"
                for group in semantic_groups:
                    objects_str = ", ".join(group['objects'])
                    group_instructions += f"• {group['group_id']} ({group['group_type']}): [{objects_str}]\n"
                    group_instructions += f"  Proximity: {group['proximity']} | Hint: {group['spatial_hint']}\n"
                    logger.debug(f"  Group {group['group_id']}: {objects_str} - {group['spatial_hint']}")
            else:
                logger.debug("No semantic groups identified, using independent placement")
                self._semantic_groups = None
        elif is_warehouse_storage:
            logger.info(f"Warehouse storage layout detected - using systematic grid placement instead of semantic grouping")
            self._semantic_groups = None
        else:
            self._semantic_groups = None


        # Extract user's semantic context from objects_to_place
        user_context_map = {}
        for obj in objects_to_place:
            obj_type = obj.get('type')
            context = obj.get('semantic_context') or obj.get('properties', {}).get('semantic_context')
            if obj_type and context:
                user_context_map[obj_type] = context
        
        object_list_str = ""
        for obj_type, count in object_summary.items():
            # Use cached model info if available
            if model_info_cache and obj_type in model_info_cache:
                cached_info = model_info_cache[obj_type]
                model_name = cached_info['model_name']
                description = cached_info['description']
                size = cached_info['dimensions']
                logger.debug(f"Using cached model info for {obj_type}: {model_name}")
            else:
                # Fallback if cache not available
                model_uri = self.model_db.find_best_model(obj_type, room.type)
                model_name = model_uri.replace("model://", "") if model_uri else obj_type

                model_path = find_gazebo_model_path(model_name)
                metadata = extract_model_metadata(model_path) if model_path else {}
                description = metadata.get("description", f"A standard {obj_type}.")
                size = self.get_actual_model_dimensions(obj_type, model_name)

            # Get orientation guidance
            orientation_info = self.object_orientations.get(obj_type, self.object_orientations['default'])

            # Get visual/geometric information for this specific model
            visual_info = self._get_model_visual_info(model_name, obj_type)

            # Add user's placement context if available
            user_context_note = ""
            if obj_type in user_context_map:
                user_context_note = f"\n  - USER REQUEST: \"{user_context_map[obj_type]}\" (RESPECT THIS!)"

            # Build the "Model Metadata Card" with visual and orientation guidance
            object_list_str += f"""• {obj_type} (quantity: {count})
  - Model: '{model_name}'
  - Description: {description}
  - Size: {size[0]:.1f}m × {size[1]:.1f}m × {size[2]:.1f}m (ACTUAL DIMENSIONS)
  - Visual Layout: {visual_info['layout']}
  - Functional Front: {visual_info['front_description']}
  - Recommended yaw: {orientation_info['natural_yaw']:.2f} ({orientation_info['default_facing']}){user_context_note}
  - Orientation: {orientation_info['description']}

"""
        placement_rules = self._get_placement_rules(room.type)
        
        # Add semantic grouping instructions if groups exist
        grouping_principle = ""
        if semantic_groups:
            # Use office/desk-chair grouping principles
            grouping_principle = """
SEMANTIC GROUPING (CRITICAL - HIGHEST PRIORITY):
• Objects have been pre-analyzed and grouped by functional relationships
• **MANDATORY**: Place objects within the same group VERY CLOSE TOGETHER:
  - "adjacent" groups: objects must be within 0.5-1.0m of each other
  - "nearby" groups: objects must be within 1.0-1.5m of each other
  - "surrounding" groups: objects must be within 2.0m of primary object
• Each group should form a tight, cohesive functional zone
• Chair MUST be immediately in front of its paired desk (within 0.5-1.0m)
• Monitor MUST be on top of or directly behind its paired desk (within 0.3m)
• Objects in the same group should be oriented to work together
• VIOLATION OF GROUP PROXIMITY RULES IS UNACCEPTABLE
"""
        elif is_warehouse_storage:
            # Warehouse-specific layout rules (already detected above)
            # Determine which axis is longer for row orientation
            is_x_longer = room.dimensions['width'] > room.dimensions['length']
            row_axis = "X-axis (horizontal, left-right)" if is_x_longer else "Y-axis (vertical, top-bottom)"
            perpendicular_axis = "Y-axis" if is_x_longer else "X-axis"

            # Build warehouse-specific type breakdown
            type_breakdown = ", ".join([f"{count} {obj_type}(s)" for obj_type, count in object_summary.items()])
            
            grouping_principle = f"""
WAREHOUSE LAYOUT (CRITICAL - HIGHEST PRIORITY):
• YOU HAVE EXACTLY {total_object_count} OBJECTS TO PLACE: {type_breakdown}
• DO NOT add extra objects to "complete" a pattern - use ONLY what you are given
• Storage racks must be arranged in PARALLEL ROWS with AISLES between them
• ORIENTATION: Rows should run along the LONGEST dimension ({row_axis})
• Each row should contain multiple racks aligned in a straight line along {row_axis}
• Rows should be separated across the {perpendicular_axis} with wide aisles between them
• Aisles between rows must be 3-4m wide for forklift access
• Distribute ALL {total_object_count} objects evenly - do not skip any
• Create a grid pattern with EXACTLY the objects provided ({type_breakdown})
• VIOLATION: Adding objects beyond {total_object_count} is STRICTLY FORBIDDEN
"""

        system_prompt = f"""You are an expert space planner for robotics simulation. Create a realistic {room.type} layout with {total_object_count} objects.

⚠️  CRITICAL: Output EXACTLY {total_object_count} objects - count them before submitting!
⚠️  CRITICAL: Each object MUST have a UNIQUE position - no two objects at same X,Y coordinates!

DESIGN PRINCIPLES:
• Functional arrangements - think like an architect
• Consider human workflow and movement patterns
• Intentional positioning, not random scattering
• SPATIAL DISTRIBUTION: Spread objects across available space - avoid clustering all at one point{grouping_principle}

TECHNICAL REQUIREMENTS:
1. Output ONE JSON array: [{{"type":"name","pose":{{"x":n,"y":n,"z":0,"roll":0,"pitch":0,"yaw":n}}}}]
2. EXACTLY {total_object_count} objects (COUNT THEM!)
3. Each object at DIFFERENT position - verify no duplicate X,Y coordinates
4. Use object types (desk, chair) NOT model names (Desk, OfficeChairBlack)
5. NO markdown, NO explanations, NO ```json blocks

ROOM-SPECIFIC LAYOUT RULES FOR {room.type.upper()}:
{placement_rules}

COORDINATES:
• Center: (0,0), Floor: z=0
• X: west(-) to east(+), Y: south(-) to north(+)
• Yaw: 0=north, 1.57=east, 3.14=south, -1.57=west

ORIENTATION RULES:
• Use "Visual Layout" and "Functional Front" descriptions for each model
• Bookshelves: Shelves face INTO room (not walls) for access
• Chairs: Seat forward, backrest behind - match actual geometry
• Desks: User side faces workspace where people sit"""

        room_width = room.dimensions['width'] - 0.15
        room_length = room.dimensions['length'] - 0.15

        spatial_context = f"""ROOM LAYOUT: '{room.name}' ({room_width:.1f}m × {room_length:.1f}m)
- Center point: (0, 0)
- Room boundaries: X from {-room_width/2:.1f} to {room_width/2:.1f}m, Y from {-room_length/2:.1f} to {room_length/2:.1f}m
- Wall positions: North={room_length/2:.1f}m, South={-room_length/2:.1f}m, East={room_width/2:.1f}m, West={-room_width/2:.1f}m"""

        # Add existing objects information for refinement mode
        existing_objects_info = ""
        if hasattr(room, 'objects') and room.objects:
            # Filter to only real existing objects (not the ones we're about to place)
            # Check if object is a GazeboModel (has model_path attribute) and is marked as existing
            existing_furniture = [obj for obj in room.objects if hasattr(obj, 'model_path') and obj.model_path == 'existing']
            if existing_furniture:
                existing_objects_info = "\n\n⚠️  EXISTING FURNITURE IN ROOM (AVOID THESE AREAS):"
                existing_objects_info += "\nThe room already contains the following objects - you MUST place new objects in DIFFERENT locations:"
                for obj in existing_furniture:
                    pose = obj.pose
                    size = obj.size if obj.size else [1.0, 1.0, 1.0]
                    # Convert world coordinates to room-relative
                    x_rel = pose['x'] - room.position['x']
                    y_rel = pose['y'] - room.position['y']
                    existing_objects_info += f"\n  • {obj.category} at ({x_rel:.1f}, {y_rel:.1f}), size: {size[0]:.1f}m × {size[1]:.1f}m"
                    # Calculate occupied zone
                    x_min = x_rel - size[0]/2 - 0.5  # Add 0.5m clearance
                    x_max = x_rel + size[0]/2 + 0.5
                    y_min = y_rel - size[1]/2 - 0.5
                    y_max = y_rel + size[1]/2 + 0.5
                    existing_objects_info += f" → AVOID zone: X [{x_min:.1f} to {x_max:.1f}], Y [{y_min:.1f} to {y_max:.1f}]"
                
                existing_objects_info += "\n\n🎯 STRATEGY: Place new objects in EMPTY areas, preferably on opposite side of room from existing furniture."
                logger.info(f"Refinement mode: Informing LLM about {len(existing_furniture)} existing object(s) to avoid")
        
        spatial_context += existing_objects_info

        # Add connection and doorway information if available
        if hasattr(room, 'connections') and room.connections:
            connection_info = []
            for connected_room, side in room.connections.items():
                connection_info.append(f"- {side.capitalize()} wall connects to '{connected_room}'")
            spatial_context += f"\n\nCONNECTIONS:\n" + "\n".join(connection_info)

        # Add doorway clearance zones 
        if hasattr(room, 'doorways') and room.doorways:
            doorway_info = ["\n\nDOORWAY CLEARANCE ZONES (CRITICAL - KEEP COMPLETELY CLEAR):"]
            doorway_info.append("⚠️  DO NOT place ANY objects in these zones - they must remain clear for entry/exit:")
            for doorway in room.doorways:
                # Convert from world coordinates to room-relative coordinates
                dx_rel = doorway['x'] - room.position['x']
                dy_rel = doorway['y'] - room.position['y']
                clearance = doorway['clearance_radius']
                width = doorway['width']
                side = doorway['side']

                # Define the clearance zone boundaries
                if side in ['north', 'south']:
                    # Horizontal doorway - clearance extends in Y direction
                    zone_desc = f"X: {dx_rel - width/2:.1f}m to {dx_rel + width/2:.1f}m"
                    if side == 'north':
                        zone_desc += f", Y: {dy_rel - clearance:.1f}m to {dy_rel:.1f}m (extends {clearance}m into room)"
                    else:
                        zone_desc += f", Y: {dy_rel:.1f}m to {dy_rel + clearance:.1f}m (extends {clearance}m into room)"
                else:
                    # Vertical doorway - clearance extends in X direction
                    zone_desc = f"Y: {dy_rel - width/2:.1f}m to {dy_rel + width/2:.1f}m"
                    if side == 'east':
                        zone_desc += f", X: {dx_rel - clearance:.1f}m to {dx_rel:.1f}m (extends {clearance}m into room)"
                    else:
                        zone_desc += f", X: {dx_rel:.1f}m to {dx_rel + clearance:.1f}m (extends {clearance}m into room)"

                doorway_info.append(f"  • {side.upper()} doorway: {zone_desc}")

            spatial_context += "\n".join(doorway_info)

        prompt = f"""{spatial_context}

OBJECTS ({total_object_count} total):
{object_list_str.strip()}{group_instructions}

Generate JSON array with {total_object_count} objects in '{room.name}' ({room.type} layout).
{"⚠️  Place grouped objects together!" if semantic_groups else ""}"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        

        prompt_string = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        input_tokens = len(self.tokenizer.encode(prompt_string))

        available_output_tokens = CONTEXT_LIMIT - input_tokens - SAFETY_MARGIN

        # Ensure it's not negative
        max_tokens_for_output = max(0, available_output_tokens)
        
        logger.info(f"Requesting LLM to place {total_object_count} objects in '{room.name}'...")
        logger.debug(
            f"Model Context: {CONTEXT_LIMIT}, Input Tokens: {input_tokens}, "
            f"Max Output Tokens set to: {max_tokens_for_output}"
        )
        response_text = self.llm_interface.query(messages, max_tokens=max_tokens_for_output) 
        logger.info(f"✅ LLM placement successful")
        
        logger.debug(f"LLM placement response: {len(response_text)} characters")
        
        plan = llm_utils.extract_json_from_response(response_text)

        if not plan:
            logger.error(f"LLM response could not be parsed into JSON. Response length: {len(response_text)}")
            logger.debug(f"Failed LLM response:\n{response_text}")
            return []

        logger.debug(f"LLM plan parsed: {plan}")

        if not llm_utils.validate_json_with_schema(plan, PLACEMENT_PLAN_SCHEMA):
            logger.error(f"LLM placement plan failed schema validation. Plan: {plan}")
            return []

        # Check for duplicate positions in LLM output
        position_counts = {}
        for item in plan:
            pose = item.get('pose', {})
            pos_key = f"{pose.get('x', 0):.2f},{pose.get('y', 0):.2f}"
            obj_type = item.get('type', 'unknown')
            if pos_key in position_counts:
                position_counts[pos_key].append(obj_type)
            else:
                position_counts[pos_key] = [obj_type]
        
        # Log any duplicates
        duplicates = {k: v for k, v in position_counts.items() if len(v) > 1}
        if duplicates:
            logger.warning(f"⚠️  LLM placed {len(duplicates)} groups of objects in same positions:")
            for pos, types in duplicates.items():
                logger.warning(f"    Position ({pos}): {', '.join(types)}")

        # Basic count validation with tolerance
        if len(plan) == total_object_count:
            logger.debug(f"✅ LLM generated a valid placement plan for {len(plan)} objects.")
            return plan
        elif len(plan) > 0 and abs(len(plan) - total_object_count) <= max(3, total_object_count * 0.2):
            logger.warning(f"⚠️ LLM plan count mismatch (expected {total_object_count}, got {len(plan)}), but close enough. Using plan.")
            return plan
        else:
            logger.error(f"❌ LLM plan item count too far off. Expected {total_object_count}, but got {len(plan)}. Using fallback.")
            return []
        
    def _llm_self_correction_phase(self, plan: List[Dict], room: Room) -> List[Dict]:
        """Ask the LLM to review and correct a placement plan."""
        logger.info(f"LLM self-correction: Reviewing placement of {len(plan)} objects...")
        
        # Add indexed names for object reference
        indexed_plan = []
        for i, item in enumerate(plan):
            indexed_plan.append({
                "type": item['type'],
                "temp_name": f"{item['type']}_{i}",
                "pose": item['pose']
            })
        
        plan_str = json.dumps(indexed_plan, indent=2)
        placement_rules = self._get_placement_rules(room.type)

        system_prompt = """You are a QA engineer for robotics simulation. Review layout and provide corrections.

CHECK:
1. Clipping: Objects into walls? → Correct pose
2. Collisions: Objects overlapping? → Correct pose
3. Logic: Follows room rules? 

OUTPUT:
• Issues found: JSON array [{{"temp_name":"obj_0","issue":"<50 chars>","correction":{{"new_pose":{{"x":n,"y":n,"z":n}}}}}}]
• No issues: []

Example: [{{"temp_name":"bookshelf_0","issue":"Clipping wall","correction":{{"new_pose":{{"x":0,"y":4.85,"z":0}}}}}}]"""
        prompt = f"""Room: {room.dimensions['width']-0.15:.2f}m × {room.dimensions['length']-0.15:.2f}m ({room.type})
Rules: {placement_rules}

Layout:
```json
{plan_str}
```
Corrections JSON array (or []):"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        
        prompt_string = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        input_tokens = len(self.tokenizer.encode(prompt_string))

        available_output_tokens = CONTEXT_LIMIT - input_tokens - SAFETY_MARGIN

        max_tokens_for_output = max(0, available_output_tokens)

        response_text = self.llm_interface.query(messages, max_tokens=max_tokens_for_output)

        if not response_text or not response_text.strip():
            logger.info("  ✓ LLM review: No issues found")
            return plan

        corrections = llm_utils.extract_json_from_response(response_text)

        if corrections is None:
            logger.warning("  ⚠️ Could not parse self-correction response. Skipping.")
            return plan

        if not corrections:
            logger.info("  ✓ LLM review: No issues found")
            return plan

        logger.info(f"  📝 Applying {len(corrections)} LLM-suggested corrections...")
        corrections_applied = 0
        for correction in corrections:
            temp_name = correction.get("temp_name")
            if temp_name and "correction" in correction and "new_pose" in correction["correction"]:
                try:
                    obj_index = int(temp_name.split('_')[-1])
                    if 0 <= obj_index < len(plan):
                        old_pose = plan[obj_index]["pose"].copy()
                        new_pose = correction["correction"]["new_pose"]
                        plan[obj_index]["pose"].update(new_pose)
                        corrections_applied += 1
                        logger.info(f"    - Corrected '{temp_name}': ({old_pose['x']:.2f},{old_pose['y']:.2f}) → ({new_pose['x']:.2f},{new_pose['y']:.2f})")
                    else:
                        logger.warning(f"    ⚠️ LLM tried to correct out-of-bounds index: {temp_name}")
                except (ValueError, IndexError):
                    logger.warning(f"    ⚠️ Could not parse index from temp_name: {temp_name}")

        if corrections_applied > 0:
            logger.info(f"  ✓ Applied {corrections_applied} correction(s)")
        
        return plan

    def _validate_and_correct_plan(self, plan: List[Dict], room: Room, model_info_cache: Dict[str, Dict] = None) -> List[Dict]:
        """
        Validate and correct placement plan in optimized order.

        Steps: Pre-collision wall snapping, main collision resolution,
        minimal semantic adjustment, and final verification.

        Args:
            model_info_cache: Model information including resolved model names for orientation corrections
        """
        if not plan:
            return []

        logger.info(f"Validating placement of {len(plan)} objects (optimized workflow)...")

        wall_thickness = self.room_config.wall_thickness
        inner_half_w = (room.dimensions['width'] / 2) - (wall_thickness / 2)
        inner_half_l = (room.dimensions['length'] / 2) - (wall_thickness / 2)

        # Initial boundary enforcement
        for item in plan:
            pose = item['pose']
            size = self.get_actual_model_dimensions(item.get('type'), room_type=room.type)
            half_w, half_l = size[0] / 2, size[1] / 2

            pose['x'] = max(min(pose['x'], inner_half_w - half_w), -inner_half_w + half_w)
            pose['y'] = max(min(pose['y'], inner_half_l - half_l), -inner_half_l + half_l)
            pose['z'] = max(0.0, pose.get('z', 0.0))

        # Doorway clearance enforcement
        if hasattr(room, 'doorways') and room.doorways:
            logger.info("🚪 Enforcing doorway clearance zones...")
            plan = self._enforce_doorway_clearance(plan, room)

        # Optimize repetitive objects with grid positioning
        logger.info("📊 Optimizing repetitive object layouts...")
        plan = self._optimize_repetitive_objects(plan, room)

        # Pre-collision wall snapping
        logger.info("📍 Pre-collision wall snapping...")
        plan = self._snap_wall_furniture_to_walls(plan, room)

        # Main collision resolution
        logger.info("🔄 Collision resolution...")
        plan = self._collision_resolution_pass(plan, room, inner_half_w, inner_half_l, max_iterations=15)

        # Minimal semantic adjustment
        logger.info("🔗 Minimal semantic adjustments...")
        if hasattr(self, '_semantic_groups') and self._semantic_groups:
            plan = self._enforce_group_cohesion(plan, self._semantic_groups, room, model_info_cache)
        else:
            # Fallback: Apply basic desk-chair pairing even without semantic groups
            logger.info("  ℹ️ No semantic groups available - applying basic desk-chair pairing...")
            plan = self._apply_basic_desk_chair_pairing(plan, room, model_info_cache)

        # Doorway clearance enforcement
        logger.info("🚪 Ensuring doorway clearance...")
        if hasattr(room, 'doorways') and room.doorways:
            plan = self._enforce_doorway_clearance(plan, room)
        
        # Robust desk/table-chair positioning
        logger.info("🪑 Enforcing desk/table-chair relationships...")
        plan = self._enforce_robust_chair_positioning(plan, room, model_info_cache)

        # Final verification
        logger.info("✓ Final collision verification...")
        plan = self._final_collision_check(plan, room, inner_half_w, inner_half_l)

        logger.info(f"✅ Validation complete: {len(plan)} objects positioned")
        
        return plan

    def _collision_resolution_pass(self, plan: List[Dict], room: Room, inner_half_w: float, inner_half_l: float, max_iterations: int = 15) -> List[Dict]:
        """
        Main collision detection and resolution using shared CollisionDetector.
        
        Uses the unified collision detection module for consistency with refiner
        and warehouse placement.
        """
        # Initialize collision detector if needed
        if not self.collision_detector:
            world_metadata = {
                'door_positions': getattr(room, 'doorways', []),
                'room_bounds': {
                    room.name: {
                        'min_x': -room.dimensions['width'] / 2,
                        'max_x': room.dimensions['width'] / 2,
                        'min_y': -room.dimensions['length'] / 2,
                        'max_y': room.dimensions['length'] / 2,
                        'center_x': 0.0,
                        'center_y': 0.0,
                        'width': room.dimensions['width'],
                        'length': room.dimensions['length']
                    }
                }
            }
            self.collision_detector = CollisionDetector(world_metadata)
        
        # Room bounds for validation
        room_bounds = {
            'min_x': -inner_half_w,
            'max_x': inner_half_w,
            'min_y': -inner_half_l,
            'max_y': inner_half_l,
            'center_x': 0.0,
            'center_y': 0.0,
            'width': room.dimensions['width'],
            'length': room.dimensions['length']
        }
        
        for iteration in range(max_iterations):
            adjustments_made = False
            collision_count = 0

            # Check each object for collisions
            for i, item in enumerate(plan):
                pose = item['pose']
                size = self.get_actual_model_dimensions(item['type'], room_type=room.type)
                position = (pose['x'], pose['y'], pose.get('yaw', 0.0))
                
                # Build list of other objects for collision checking
                other_objects = []
                for j, other_item in enumerate(plan):
                    if i == j:
                        continue
                    other_pose = other_item['pose']
                    other_size = self.get_actual_model_dimensions(other_item['type'], room_type=room.type)
                    other_objects.append({
                        'name': other_item.get('type', f'obj_{j}'),
                        'position': (other_pose['x'], other_pose['y'], other_pose.get('yaw', 0.0)),
                        'dimensions': other_size
                    })
                
                # Check for collision using shared collision detector
                if not self.collision_detector.validate_position(position, size, other_objects, room_bounds):
                    collision_count += 1
                    adjustments_made = True
                    
                    # Find direction to move away from nearest collision
                    min_dist = float('inf')
                    push_dir_x, push_dir_y = 0.0, 0.0
                    
                    for other in other_objects:
                        other_pos = other['position']
                        dx = pose['x'] - other_pos[0]
                        dy = pose['y'] - other_pos[1]
                        dist = math.sqrt(dx**2 + dy**2)
                        
                        if dist < min_dist and dist > 0.01:
                            min_dist = dist
                            # Normalize direction
                            push_dir_x = dx / dist if dist > 0.01 else random.uniform(-1, 1)
                            push_dir_y = dy / dist if dist > 0.01 else random.uniform(-1, 1)
                    
                    # If no clear direction (objects coincident), pick random
                    if abs(push_dir_x) < 0.01 and abs(push_dir_y) < 0.01:
                        angle = random.uniform(0, 2 * math.pi)
                        push_dir_x = math.cos(angle)
                        push_dir_y = math.sin(angle)
                    
                    # Apply push with adaptive amount based on how close objects are
                    if min_dist < 0.3:
                        push_amount = 0.5
                    elif min_dist < 0.6:
                        push_amount = 0.3
                    else:
                        push_amount = 0.2

                    pose['x'] += push_dir_x * push_amount
                    pose['y'] += push_dir_y * push_amount
                    
                    # Enforce boundaries
                    half_w, half_l = size[0] / 2, size[1] / 2
                    pose['x'] = max(min(pose['x'], inner_half_w - half_w), -inner_half_w + half_w)
                    pose['y'] = max(min(pose['y'], inner_half_l - half_l), -inner_half_l + half_l)

            if not adjustments_made:
                logger.info(f"  ✓ Collision detection converged after {iteration + 1} iteration(s)")
                return plan
            else:
                logger.debug(f"  Iteration {iteration + 1}: {collision_count} objects adjusted")

        # Final validation
        final_collision_count = 0
        for i, item in enumerate(plan):
            pose = item['pose']
            size = self.get_actual_model_dimensions(item['type'], room_type=room.type)
            position = (pose['x'], pose['y'], pose.get('yaw', 0.0))
            
            other_objects = []
            for j, other_item in enumerate(plan):
                if i == j:
                    continue
                other_pose = other_item['pose']
                other_size = self.get_actual_model_dimensions(other_item['type'], room_type=room.type)
                other_objects.append({
                    'name': other_item.get('type', f'obj_{j}'),
                    'position': (other_pose['x'], other_pose['y'], other_pose.get('yaw', 0.0)),
                    'dimensions': other_size
                })
            
            if not self.collision_detector.validate_position(position, size, other_objects, room_bounds):
                final_collision_count += 1
        
        if final_collision_count > 0:
            logger.warning(f"  ⚠ Did not fully converge after {max_iterations} iterations ({final_collision_count} collisions remain)")
        else:
            logger.info(f"  ✓ Completed after {max_iterations} iterations (collision-free)")

        return plan

    def _apply_basic_desk_chair_pairing(self, plan: List[Dict], room: Room, model_info_cache: Dict[str, Dict] = None) -> List[Dict]:
        """
        Fallback method: Pair desks with chairs using simple proximity matching.
        Apply proper positioning in front of desk with correct orientation.

        Args:
            model_info_cache: Model information for desk-specific corrections
        """
        # Find all desks and chairs
        desks = [(i, obj) for i, obj in enumerate(plan) if 'desk' in obj['type'].lower()]
        chairs = [(i, obj) for i, obj in enumerate(plan) if 'chair' in obj['type'].lower()]

        if not desks or not chairs:
            return plan

        # Track which chairs have been assigned
        assigned_chairs = set()

        for desk_idx, desk in desks:
            desk_pose = desk['pose']
            desk_yaw = desk_pose.get('yaw', 0.0)

            # Find closest unassigned chair
            closest_chair_idx = None
            min_distance = float('inf')

            for chair_idx, chair in chairs:
                if chair_idx in assigned_chairs:
                    continue

                chair_pose = chair['pose']
                distance = math.sqrt(
                    (desk_pose['x'] - chair_pose['x'])**2 +
                    (desk_pose['y'] - chair_pose['y'])**2
                )

                if distance < min_distance:
                    min_distance = distance
                    closest_chair_idx = chair_idx

            if closest_chair_idx is not None:
                assigned_chairs.add(closest_chair_idx)
                chair = plan[closest_chair_idx]

                # Get desk model name for model-specific corrections
                desk_model_name = None
                if model_info_cache and desk['type'] in model_info_cache:
                    desk_model_name = model_info_cache[desk['type']]['model_name']

                # Use model-specific positioning if available
                if desk_model_name and desk_model_name in self.model_orientation_corrections:
                    correction = self.model_orientation_corrections[desk_model_name]
                    offset_x = correction['chair_offset_x']
                    offset_y = correction['chair_offset_y']
                    chair_yaw_offset = correction['chair_yaw_offset']
                    
                    # Transform offsets from desk's local frame to world frame
                    cos_yaw = math.cos(desk_yaw)
                    sin_yaw = math.sin(desk_yaw)
                    
                    # In desk's local frame: +X is right, +Y is forward
                    world_offset_x = offset_x * cos_yaw - offset_y * sin_yaw
                    world_offset_y = offset_x * sin_yaw + offset_y * cos_yaw
                    
                    target_x = desk_pose['x'] + world_offset_x
                    target_y = desk_pose['y'] + world_offset_y
                    target_yaw = desk_yaw + chair_yaw_offset

                    logger.info(f"  🔄 Model-specific positioning for '{desk_model_name}': "
                               f"offset=({offset_x:.3f}, {offset_y:.3f}), world=({world_offset_x:.3f}, {world_offset_y:.3f})")
                else:
                    # Fallback: generic positioning
                    desk_size = self.get_actual_model_dimensions(desk['type'], room_type=room.type)
                    chair_size = self.get_actual_model_dimensions(chair['type'], room_type=room.type)
                    clearance = 0.15
                    offset_distance = (desk_size[1] / 2) + (chair_size[1] / 2) + clearance

                    # Generic perpendicular positioning
                    perpendicular_angle = desk_yaw - math.pi / 2
                    target_x = desk_pose['x'] + offset_distance * math.cos(perpendicular_angle)
                    target_y = desk_pose['y'] + offset_distance * math.sin(perpendicular_angle)
                    target_yaw = desk_yaw + math.pi

                    logger.info(f"  🔄 Generic desk-chair positioning: ({target_x:.2f}, {target_y:.2f}, yaw={target_yaw:.3f})")

                old_x, old_y = chair['pose']['x'], chair['pose']['y']
                chair['pose']['x'] = target_x
                chair['pose']['y'] = target_y
                chair['pose']['yaw'] = target_yaw

                # Enforce room boundaries after repositioning
                w, l = room.dimensions["width"], room.dimensions["length"]
                inner_half_w = (w / 2) * 0.9  # 10% margin from walls
                inner_half_l = (l / 2) * 0.9
                chair_size = self.get_actual_model_dimensions(chair['type'], room_type=room.type)
                half_chair_w, half_chair_l = chair_size[0] / 2, chair_size[1] / 2
                
                # Clamp to boundaries
                chair['pose']['x'] = max(min(chair['pose']['x'], inner_half_w - half_chair_w), -inner_half_w + half_chair_w)
                chair['pose']['y'] = max(min(chair['pose']['y'], inner_half_l - half_chair_l), -inner_half_l + half_chair_l)

                logger.info(f"  📍 Moved chair: ({old_x:.2f}, {old_y:.2f}) → ({chair['pose']['x']:.2f}, {chair['pose']['y']:.2f})")

        return plan

    def _enforce_group_cohesion(self, plan: List[Dict], semantic_groups: List[Dict], room: Room, model_info_cache: Dict[str, Dict] = None) -> List[Dict]:
        """
        Apply semantic adjustments with orientation correction.

        Adjusts orientations (yaw) to ensure proper facing directions rather than positions,
        assuming LLM with actual dimensions placed objects roughly correctly.

        Args:
            model_info_cache: Model information to get resolved model names for desk-specific corrections
        """
        # Create a mapping from object type + index to plan item
        type_counters = {}
        object_map = {}

        for i, item in enumerate(plan):
            obj_type = item['type']
            idx = type_counters.get(obj_type, 0)
            type_counters[obj_type] = idx + 1
            object_map[f"{obj_type}_{idx}"] = i

        # Track which objects have been processed to avoid duplicates
        processed_objects = set()

        # Process each semantic group
        for group in semantic_groups:
            if len(group['objects']) < 2:
                continue  # Skip single-object groups

            primary_type = group['primary_object']
            spatial_arr = group.get('spatial_arrangement', {})

            # Find an unprocessed primary object in the plan
            primary_idx = None
            primary_key = None
            for key, idx in object_map.items():
                if key.startswith(f"{primary_type}_") and key not in processed_objects:
                    primary_idx = idx
                    primary_key = key
                    break

            if primary_idx is None:
                continue

            processed_objects.add(primary_key)
            primary_item = plan[primary_idx]
            primary_pose = primary_item['pose']
            primary_yaw = primary_pose.get('yaw', 0.0)

            # Get primary object dimensions
            primary_size = self.get_actual_model_dimensions(primary_type, room_type=room.type)

            # Get model-specific orientation correction (for desk-chair positioning)
            primary_model_name = None
            if model_info_cache and primary_type in model_info_cache:
                primary_model_name = model_info_cache[primary_type].get('model_name')
                logger.info(f"🔧 Retrieved model name for {primary_type}: '{primary_model_name}' from cache")
            else:
                logger.warning(f"⚠️  No model info in cache for {primary_type}. model_info_cache={'exists' if model_info_cache else 'None'}")

            logger.debug(f"Processing group {group['group_id']}: primary={primary_key} (model: {primary_model_name}) at ({primary_pose['x']:.2f}, {primary_pose['y']:.2f})")

            # Count how many of each related object type are in this group (for distribution)
            type_counts = {}
            for obj_type in group['objects']:
                if obj_type != primary_type:
                    type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

            # Track which index we're at for each type (for "around" distribution)
            type_indices = {obj_type: 0 for obj_type in type_counts.keys()}

            # Process each related object in the group
            for obj_type in group['objects']:
                if obj_type == primary_type:
                    continue

                # Find an unprocessed related object in the plan
                obj_idx = None
                obj_key = None
                for key, idx in object_map.items():
                    if key.startswith(f"{obj_type}_") and key not in processed_objects:
                        obj_idx = idx
                        obj_key = key
                        break

                if obj_idx is None:
                    logger.debug(f"  No unprocessed {obj_type} found for {primary_key}")
                    continue

                processed_objects.add(obj_key)
                obj_item = plan[obj_idx]
                obj_pose = obj_item['pose']
                obj_size = self.get_actual_model_dimensions(obj_type, room_type=room.type)

                # Determine desired spatial arrangement
                arrangement = spatial_arr.get(obj_type, 'around')
                
                # Skip desk-chair semantic positioning - let robust chair positioning handle it
                if arrangement == 'in_front' and 'chair' in obj_type.lower():
                    logger.debug(f"  Skipping semantic positioning for {obj_key} (desk-chair pair, will be handled by robust positioning)")
                    processed_objects.add(obj_key)  # Mark as processed so it doesn't get handled again
                    continue

                # Check current distance from primary
                current_x, current_y = obj_pose['x'], obj_pose['y']
                primary_x, primary_y = primary_pose['x'], primary_pose['y']
                current_distance = math.sqrt((current_x - primary_x)**2 + (current_y - primary_y)**2)

                # Calculate target position and orientation based on arrangement
                # Use indexed version for proper distribution of multiple objects
                obj_index = type_indices[obj_type]
                total_of_type = type_counts[obj_type]
                type_indices[obj_type] += 1  # Increment for next object of this type

                target_x, target_y, target_yaw = self._calculate_relative_position_with_index(
                    primary_x, primary_y, primary_yaw,
                    primary_size, obj_size, arrangement, obj_index, total_of_type, primary_model_name
                )

                if arrangement == 'in_front':
                    # For desk-chair: apply both position and orientation corrections
                    old_x, old_y = obj_pose['x'], obj_pose['y']
                    old_yaw = obj_pose.get('yaw', 0.0)

                    # Apply position correction
                    obj_pose['x'] = target_x
                    obj_pose['y'] = target_y

                    # Apply orientation: use target_yaw from model-specific calculation
                    obj_pose['yaw'] = target_yaw

                    position_distance = math.sqrt((target_x - old_x)**2 + (target_y - old_y)**2)
                    if position_distance > 0.2 or abs(old_yaw - target_yaw) > 0.1:
                        logger.info(f"  📍 Corrected {obj_key} position+orientation: ({old_x:.2f}, {old_y:.2f}, yaw={old_yaw:.2f}) → ({target_x:.2f}, {target_y:.2f}, yaw={obj_pose['yaw']:.2f}) [{arrangement}]")

                elif arrangement == 'around':
                    # For "around" arrangement (e.g., chairs around table), apply position and orientation
                    old_x, old_y = obj_pose['x'], obj_pose['y']
                    old_yaw = obj_pose.get('yaw', 0.0)

                    # Apply position correction
                    obj_pose['x'] = target_x
                    obj_pose['y'] = target_y

                    # Enforce room boundaries after repositioning
                    w, l = room.dimensions["width"], room.dimensions["length"]
                    inner_half_w = (w / 2) * 0.9  # 10% margin from walls
                    inner_half_l = (l / 2) * 0.9
                    obj_size = self.get_actual_model_dimensions(obj_type, room_type=room.type)
                    half_obj_w, half_obj_l = obj_size[0] / 2, obj_size[1] / 2
                    
                    # Clamp to boundaries
                    obj_pose['x'] = max(min(obj_pose['x'], inner_half_w - half_obj_w), -inner_half_w + half_obj_w)
                    obj_pose['y'] = max(min(obj_pose['y'], inner_half_l - half_obj_l), -inner_half_l + half_obj_l)

                    # Apply orientation - target_yaw points toward table center
                    obj_pose['yaw'] = target_yaw + math.pi / 2  # Add offset for chair model

                    position_distance = math.sqrt((obj_pose['x'] - old_x)**2 + (obj_pose['y'] - old_y)**2)
                    if position_distance > 0.3 or abs(old_yaw - obj_pose['yaw']) > 0.3:
                        logger.info(f"  🪑 Positioned {obj_key} around {primary_key} [seat {obj_index + 1}/{total_of_type}]: ({old_x:.2f}, {old_y:.2f}) → ({obj_pose['x']:.2f}, {obj_pose['y']:.2f}), yaw={obj_pose['yaw']:.2f}")

                elif arrangement in ['behind', 'beside_left', 'beside_right']:
                    # For these arrangements, both position AND orientation need to be corrected
                    old_x, old_y = obj_pose['x'], obj_pose['y']
                    old_yaw = obj_pose.get('yaw', 0.0)

                    # Apply position correction
                    obj_pose['x'] = target_x
                    obj_pose['y'] = target_y

                    # Calculate relative orientation from the primary's facing
                    if arrangement == 'behind':
                        target_yaw = primary_yaw  # Face same direction as primary
                    elif arrangement == 'beside_left':
                        target_yaw = primary_yaw + math.pi / 2  # Face 90° left
                    elif arrangement == 'beside_right':
                        target_yaw = primary_yaw - math.pi / 2  # Face 90° right

                    obj_pose['yaw'] = target_yaw

                    position_distance = math.sqrt((target_x - old_x)**2 + (target_y - old_y)**2)
                    yaw_change = abs(old_yaw - target_yaw)
                    if position_distance > 0.2 or yaw_change > 0.3:
                        logger.info(f"  � Corrected {obj_key} position+orientation: ({old_x:.2f}, {old_y:.2f}, yaw={old_yaw:.2f}) → ({target_x:.2f}, {target_y:.2f}, yaw={target_yaw:.2f}) [{arrangement}]")

                # For 'on_top' arrangement, adjust Z to be on the surface of the primary object
                if arrangement == 'on_top':
                    obj_pose['z'] = primary_size[2]

        return plan

    def _final_collision_check(self, plan: List[Dict], room: Room, inner_half_w: float, inner_half_l: float) -> List[Dict]:
        """
        Final verification using shared CollisionDetector.
        
        Checks for major overlaps and warns without moving objects,
        preserving semantic corrections from previous steps.
        """
        logger.debug("  Running final collision check (verification only)...")
        
        # Initialize collision detector if needed
        if not self.collision_detector:
            world_metadata = {
                'door_positions': getattr(room, 'doorways', []),
                'room_bounds': {
                    room.name: {
                        'min_x': -room.dimensions['width'] / 2,
                        'max_x': room.dimensions['width'] / 2,
                        'min_y': -room.dimensions['length'] / 2,
                        'max_y': room.dimensions['length'] / 2,
                        'center_x': 0.0,
                        'center_y': 0.0,
                        'width': room.dimensions['width'],
                        'length': room.dimensions['length']
                    }
                }
            }
            self.collision_detector = CollisionDetector(world_metadata)
        
        room_bounds = {
            'min_x': -inner_half_w,
            'max_x': inner_half_w,
            'min_y': -inner_half_l,
            'max_y': inner_half_l,
            'center_x': 0.0,
            'center_y': 0.0,
            'width': room.dimensions['width'],
            'length': room.dimensions['length']
        }

        collision_count = 0
        overlap_pairs = []

        for i, item in enumerate(plan):
            pose = item['pose']
            size = self.get_actual_model_dimensions(item['type'], room_type=room.type)
            position = (pose['x'], pose['y'], pose.get('yaw', 0.0))
            
            # Build list of other objects
            other_objects = []
            for j, other_item in enumerate(plan):
                if i == j:
                    continue
                other_pose = other_item['pose']
                other_size = self.get_actual_model_dimensions(other_item['type'], room_type=room.type)
                other_objects.append({
                    'name': other_item.get('type', f'obj_{j}'),
                    'position': (other_pose['x'], other_pose['y'], other_pose.get('yaw', 0.0)),
                    'dimensions': other_size
                })
            
            # Validate using shared collision detector
            if not self.collision_detector.validate_position(position, size, other_objects, room_bounds):
                collision_count += 1
                overlap_pairs.append((item['type'], f"collision_{i}"))

        if overlap_pairs:
            logger.info(f"  ℹ️ Final check found {len(overlap_pairs)} minor overlaps - acceptable for semantic correctness")
        else:
            logger.info("  ✓ Final verification passed - no overlaps detected")

        return plan


    def _calculate_relative_position_with_index(self, primary_x: float, primary_y: float, primary_yaw: float,
                                               primary_size: Tuple[float, float, float], obj_size: Tuple[float, float, float],
                                               arrangement: str, obj_index: int = 0, total_objects: int = 1, primary_model_name: str = None) -> Tuple[float, float, float]:
        """
        Calculate relative position with support for distributing multiple objects around a primary (e.g., chairs around table).

        Args:
            primary_x, primary_y: Primary object center position
            primary_yaw: Primary object yaw (rotation)
            primary_size: Primary object (width, length, height)
            obj_size: Related object (width, length, height)
            arrangement: Spatial arrangement type
            obj_index: Index of this object among siblings of the same type (0-based)
            total_objects: Total number of objects of this type to place around primary
            primary_model_name: Specific model name for model-specific orientation corrections
        """
        import math

        # For tables with multiple chairs, distribute them around the table
        if arrangement == 'around' and total_objects > 1:
            # Calculate angular distribution around the table
            angle_step = (2 * math.pi) / total_objects
            angle = obj_index * angle_step + primary_yaw

            # Distance from center (use the larger dimension of the table)
            clearance = 0.4  # 40cm clearance from table edge
            table_radius = max(primary_size[0], primary_size[1]) / 2
            chair_depth = obj_size[1] / 2
            distance = table_radius + chair_depth + clearance

            target_x = primary_x + distance * math.cos(angle)
            target_y = primary_y + distance * math.sin(angle)

            # Chair should face toward the table center
            target_yaw = angle + math.pi

            return (target_x, target_y, target_yaw)

        # Fall back to standard calculation with model-specific corrections
        return self._calculate_relative_position(primary_x, primary_y, primary_yaw, primary_size, obj_size, arrangement, primary_model_name)

    def _calculate_relative_position(self, primary_x: float, primary_y: float, primary_yaw: float,
                                     primary_size: Tuple[float, float, float],
                                     obj_size: Tuple[float, float, float],
                                     arrangement: str, primary_model_name: str = None) -> Tuple[float, float, float]:
        """
        Calculate the target position and orientation for an object relative to a primary object.

        Args:
            primary_x, primary_y: Primary object center position
            primary_yaw: Primary object yaw (rotation)
            primary_size: Primary object (width, length, height)
            obj_size: Related object (width, length, height)
            arrangement: Spatial arrangement type ('in_front', 'behind', 'beside_left', etc.)
            primary_model_name: Specific model name for model-specific orientation corrections

        Returns:
            (target_x, target_y, target_yaw) for the related object
        """
        import math

        # Apply model-specific positioning for desk-chair (ONLY for known desk models)
        if arrangement == 'in_front' and primary_model_name and primary_model_name in self.model_orientation_corrections:
            correction = self.model_orientation_corrections[primary_model_name]
            offset_x = correction['chair_offset_x']
            offset_y = correction['chair_offset_y']
            chair_yaw_offset = correction['chair_yaw_offset']

            # Transform offsets from desk's local frame to world frame
            cos_yaw = math.cos(primary_yaw)
            sin_yaw = math.sin(primary_yaw)
            
            # In desk's local frame: +X is right, +Y is forward
            # Rotate to world frame
            world_offset_x = offset_x * cos_yaw - offset_y * sin_yaw
            world_offset_y = offset_x * sin_yaw + offset_y * cos_yaw
            
            target_x = primary_x + world_offset_x
            target_y = primary_y + world_offset_y
            target_yaw = primary_yaw + chair_yaw_offset
            
            logger.info(f"📐 Desk '{primary_model_name}' at ({primary_x:.2f}, {primary_y:.2f}, yaw={primary_yaw:.3f}): "
                       f"chair at ({target_x:.2f}, {target_y:.2f}, yaw={target_yaw:.3f})")
            return (target_x, target_y, target_yaw)

        # Fallback: generic positioning for unknown desk models or other arrangements
        # Standard calculation for non-desk or unknown models
        # Calculate the "front" direction of the primary object (where yaw points)
        cos_yaw = math.cos(primary_yaw)
        sin_yaw = math.sin(primary_yaw)

        # Standard distances (primary half-size + object half-size + clearance)
        # For a desk-chair setup, we want the chair close but not overlapping
        clearance = 0.15  # Small clearance for natural desk-chair spacing
        front_back_dist = (primary_size[1] / 2) + (obj_size[1] / 2) + clearance
        left_right_dist = (primary_size[0] / 2) + (obj_size[0] / 2) + clearance

        target_yaw = primary_yaw  # Default yaw

        if arrangement == 'in_front':
            # Place in front of the primary object
            # For desks: "in front" means where the user sits (OPPOSITE to where desk faces)
            # If desk faces direction (sin_yaw, cos_yaw), chair should be in opposite direction
            # So we use MINUS to place in opposite direction
            target_x = primary_x - front_back_dist * sin_yaw
            target_y = primary_y - front_back_dist * cos_yaw
            # Chair should face TOWARD the desk (same direction as desk faces)
            target_yaw = primary_yaw  # Face same direction as desk  

        elif arrangement == 'behind':
            # Place behind (opposite yaw direction)
            target_x = primary_x - front_back_dist * sin_yaw
            target_y = primary_y - front_back_dist * cos_yaw
            target_yaw = primary_yaw  # Face away from primary

        elif arrangement == 'beside_left':
            # Place to the left (perpendicular to yaw)
            target_x = primary_x - left_right_dist * cos_yaw
            target_y = primary_y + left_right_dist * sin_yaw
            target_yaw = primary_yaw  # Same orientation

        elif arrangement == 'beside_right':
            # Place to the right (perpendicular to yaw)
            target_x = primary_x + left_right_dist * cos_yaw
            target_y = primary_y - left_right_dist * sin_yaw
            target_yaw = primary_yaw  # Same orientation

        elif arrangement == 'on_top':
            keyboard_offset = 0.15  # Push monitor back by 15cm from center
            target_x = primary_x - keyboard_offset * cos_yaw  
            target_y = primary_y - keyboard_offset * sin_yaw
            target_yaw = primary_yaw + math.pi

        else:  # 'around' or unknown
            # Keep current position but adjust orientation to face primary
            target_x = primary_x + front_back_dist * sin_yaw
            target_y = primary_y + front_back_dist * cos_yaw
            target_yaw = primary_yaw + math.pi

        return (target_x, target_y, target_yaw)

    def _optimize_repetitive_objects(self, plan: List[Dict], room: Room) -> List[Dict]:
        """
        Detect patterns of many similar objects and apply grid positioning for systematic layouts.
        
        Use cases:
        - Computer labs with many desks
        - Libraries with multiple tables
        - Classrooms with many workstations
        
        Only applies to 5+ similar objects to avoid interfering with LLM's intentional layouts.
        """
        # Count objects by type
        type_counts = {}
        type_items = {}
        
        for item in plan:
            obj_type = item['type'].lower()
            # Normalize type (remove numbers/suffixes)
            base_type = obj_type.split('_')[0] if '_' in obj_type else obj_type
            
            type_counts[base_type] = type_counts.get(base_type, 0) + 1
            if base_type not in type_items:
                type_items[base_type] = []
            type_items[base_type].append(item)
        
        # Identify types with many instances (5+ for systematic grid)
        repetitive_types = {obj_type: items for obj_type, items in type_items.items() 
                           if type_counts[obj_type] >= 5}
        
        if not repetitive_types:
            logger.debug("  No repetitive object patterns detected (need 5+ of same type)")
            return plan
        
        # Initialize collision detector if needed
        if not self.collision_detector:
            world_metadata = {
                'door_positions': getattr(room, 'doorways', []),
                'room_bounds': {
                    room.name: {
                        'min_x': -room.dimensions['width'] / 2,
                        'max_x': room.dimensions['width'] / 2,
                        'min_y': -room.dimensions['length'] / 2,
                        'max_y': room.dimensions['length'] / 2,
                        'center_x': 0.0,
                        'center_y': 0.0,
                        'width': room.dimensions['width'],
                        'length': room.dimensions['length']
                    }
                }
            }
            self.collision_detector = CollisionDetector(world_metadata)
        
        wall_thickness = self.room_config.wall_thickness
        inner_half_w = (room.dimensions['width'] / 2) - (wall_thickness / 2)
        inner_half_l = (room.dimensions['length'] / 2) - (wall_thickness / 2)
        
        room_bounds = {
            'min_x': -inner_half_w,
            'max_x': inner_half_w,
            'min_y': -inner_half_l,
            'max_y': inner_half_l,
            'center_x': 0.0,
            'center_y': 0.0,
            'width': room.dimensions['width'],
            'length': room.dimensions['length']
        }
        
        # Get existing objects (those not being repositioned)
        all_items_set = set(id(item) for items in repetitive_types.values() for item in items)
        existing_objects = []
        for item in plan:
            if id(item) not in all_items_set:
                pose = item['pose']
                size = self.get_actual_model_dimensions(item['type'], room_type=room.type)
                existing_objects.append({
                    'name': item.get('type', 'obj'),
                    'position': (pose['x'], pose['y'], pose.get('yaw', 0.0)),
                    'dimensions': size
                })
        
        # Process each repetitive type
        for obj_type, items in repetitive_types.items():
            logger.info(f"  🔢 Applying grid layout to {len(items)}x {obj_type}")
            
            # Prepare objects for grid positioning
            grid_objects = []
            for item in items:
                size = self.get_actual_model_dimensions(item['type'], room_type=room.type)
                grid_objects.append({
                    'name': item.get('type', 'obj'),
                    'dimensions': size
                })
            
            # Determine layout pattern based on room type and object type
            if room.type in ['warehouse', 'storage']:
                position_desc = "in rows"
            elif obj_type in ['desk', 'workstation', 'table']:
                position_desc = "in rows"  # Systematic classroom/lab layout
            else:
                position_desc = "center"  # Default grid in center
            
            # Calculate grid positions
            try:
                positions = self.collision_detector.calculate_grid_positions(
                    grid_objects, position_desc, room_bounds, existing_objects
                )
                
                # Apply positions
                for item, position in zip(items, positions):
                    x, y, yaw = position
                    old_x, old_y = item['pose']['x'], item['pose']['y']
                    item['pose']['x'] = x
                    item['pose']['y'] = y
                    item['pose']['yaw'] = yaw
                    item['_grid_optimized'] = True
                    
                    # Add to existing objects for next type
                    size = self.get_actual_model_dimensions(item['type'], room_type=room.type)
                    existing_objects.append({
                        'name': item.get('type', 'obj'),
                        'position': (x, y, yaw),
                        'dimensions': size
                    })
                
                logger.info(f"    ✓ Grid-optimized {len(items)} {obj_type} objects")
                
            except Exception as e:
                logger.warning(f"  ⚠ Failed to apply grid layout to {obj_type}: {e}")
                # Keep LLM positions on failure
        
        return plan

    def _snap_wall_furniture_to_walls(self, plan: List[Dict], room: Room) -> List[Dict]:
        """
        Snap furniture that should be against walls to the nearest wall using grid positioning.
        
        Uses shared CollisionDetector for systematic wall-aligned layouts when multiple
        items target the same wall. Falls back to simple snapping for single items.
        """
        # Define which object types CAN be against walls
        wall_furniture_types = {'bookshelf', 'shelf', 'cabinet', 'shelving_unit', 'wardrobe', 'dresser'}

        wall_thickness = self.room_config.wall_thickness
        snap_threshold = 2.0  # Only snap if within 2m of a wall
        inner_half_w = (room.dimensions['width'] / 2) - (wall_thickness / 2)
        inner_half_l = (room.dimensions['length'] / 2) - (wall_thickness / 2)

        # Group items by their target wall
        wall_groups = {'west': [], 'east': [], 'north': [], 'south': []}
        non_wall_items = []

        # First pass: identify which items go to which wall
        for i, item in enumerate(plan):
            obj_type = item['type'].lower()

            # Check if this object type CAN be against walls
            if not any(wf_type in obj_type for wf_type in wall_furniture_types):
                non_wall_items.append(item)
                continue

            pose = item['pose']
            x, y = pose['x'], pose['y']

            # Calculate distances to each wall
            dist_to_west = abs(x - (-inner_half_w))
            dist_to_east = abs(x - inner_half_w)
            dist_to_south = abs(y - (-inner_half_l))
            dist_to_north = abs(y - inner_half_l)

            min_dist = min(dist_to_west, dist_to_east, dist_to_south, dist_to_north)

            # Only snap if furniture is already close to a wall
            if min_dist > snap_threshold:
                non_wall_items.append(item)
                continue

            # Determine target wall and add to group
            if min_dist == dist_to_west:
                wall_groups['west'].append(item)
            elif min_dist == dist_to_east:
                wall_groups['east'].append(item)
            elif min_dist == dist_to_south:
                wall_groups['south'].append(item)
            else:  # north wall
                wall_groups['north'].append(item)

        # Initialize collision detector if needed
        if not self.collision_detector:
            world_metadata = {
                'door_positions': getattr(room, 'doorways', []),
                'room_bounds': {
                    room.name: {
                        'min_x': -room.dimensions['width'] / 2,
                        'max_x': room.dimensions['width'] / 2,
                        'min_y': -room.dimensions['length'] / 2,
                        'max_y': room.dimensions['length'] / 2,
                        'center_x': 0.0,
                        'center_y': 0.0,
                        'width': room.dimensions['width'],
                        'length': room.dimensions['length']
                    }
                }
            }
            self.collision_detector = CollisionDetector(world_metadata)

        room_bounds = {
            'min_x': -inner_half_w,
            'max_x': inner_half_w,
            'min_y': -inner_half_l,
            'max_y': inner_half_l,
            'center_x': 0.0,
            'center_y': 0.0,
            'width': room.dimensions['width'],
            'length': room.dimensions['length']
        }

        # Convert non-wall items to existing objects for collision avoidance
        existing_objects = []
        for item in non_wall_items:
            pose = item['pose']
            size = self.get_actual_model_dimensions(item['type'], room_type=room.type)
            existing_objects.append({
                'name': item.get('type', 'obj'),
                'position': (pose['x'], pose['y'], pose.get('yaw', 0.0)),
                'dimensions': size
            })

        # Process each wall group with grid positioning
        for wall_name, items in wall_groups.items():
            if not items:
                continue

            # If multiple items on same wall, use grid positioning
            if len(items) >= 3:
                logger.info(f"  📐 Using grid positioning for {len(items)} items on {wall_name} wall")
                
                # Prepare objects for grid positioning
                grid_objects = []
                for item in items:
                    size = self.get_actual_model_dimensions(item['type'], room_type=room.type)
                    grid_objects.append({
                        'name': item.get('type', 'obj'),
                        'dimensions': size
                    })
                
                # Calculate grid positions along wall
                position_desc = f"along {wall_name} wall"
                positions = self.collision_detector.calculate_grid_positions(
                    grid_objects, position_desc, room_bounds, existing_objects
                )
                
                # Apply positions
                for item, position in zip(items, positions):
                    x, y, yaw = position
                    old_x, old_y = item['pose']['x'], item['pose']['y']
                    item['pose']['x'] = x
                    item['pose']['y'] = y
                    item['pose']['yaw'] = yaw
                    item['_wall_snapped'] = True
                    
                    # Add to existing objects for next wall
                    size = self.get_actual_model_dimensions(item['type'], room_type=room.type)
                    existing_objects.append({
                        'name': item.get('type', 'obj'),
                        'position': (x, y, yaw),
                        'dimensions': size
                    })
                    
                    logger.info(f"    📌 Grid-placed {item['type']} on {wall_name} wall: ({old_x:.2f},{old_y:.2f}) → ({x:.2f},{y:.2f})")
            
            else:
                # Single/few items: simple snap to wall
                for item in items:
                    obj_type = item['type'].lower()
                    pose = item['pose']
                    size = self.get_actual_model_dimensions(obj_type, room_type=room.type)
                    half_w, half_l = size[0] / 2, size[1] / 2
                    wall_clearance = 0.02

                    old_x, old_y = pose['x'], pose['y']

                    if wall_name == 'west':
                        pose['x'] = -inner_half_w + half_w + wall_clearance
                        pose['yaw'] = 1.5708  # face east
                    elif wall_name == 'east':
                        pose['x'] = inner_half_w - half_w - wall_clearance
                        pose['yaw'] = -1.5708  # face west
                    elif wall_name == 'south':
                        pose['y'] = -inner_half_l + half_l + wall_clearance
                        pose['yaw'] = 3.14159  # face north
                    else:  # north
                        pose['y'] = inner_half_l - half_l - wall_clearance
                        pose['yaw'] = 0.0  # face south

                    item['_wall_snapped'] = True
                    
                    # Add to existing objects
                    existing_objects.append({
                        'name': item.get('type', 'obj'),
                        'position': (pose['x'], pose['y'], pose.get('yaw', 0.0)),
                        'dimensions': size
                    })
                    
                    logger.info(f"    📌 Snapped {obj_type} to {wall_name} wall: ({old_x:.2f},{old_y:.2f}) → ({pose['x']:.2f},{pose['y']:.2f})")

        return plan

    def _enforce_doorway_clearance(self, plan: List[Dict], room: Room) -> List[Dict]:
        """
        Enforce doorway clearance zones by moving objects that block doorways.
        Objects in clearance zones are pushed away from the doorway into the room.
        """
        if not hasattr(room, 'doorways') or not room.doorways:
            return plan

        import math
        moved_count = 0

        for doorway in room.doorways:
            # Convert doorway position from world to room-relative coordinates
            dx_world, dy_world = doorway['x'], doorway['y']
            dx_rel = dx_world - room.position['x']
            dy_rel = dy_world - room.position['y']

            clearance = doorway['clearance_radius']
            width = doorway['width']
            side = doorway['side']

            # Check each object
            for item in plan:
                pose = item['pose']
                obj_type = item.get('type', 'unknown')
                size = self.get_actual_model_dimensions(obj_type, room_type=room.type)
                half_w, half_l = size[0] / 2, size[1] / 2

                # Calculate object bounding box
                obj_min_x = pose['x'] - half_w
                obj_max_x = pose['x'] + half_w
                obj_min_y = pose['y'] - half_l
                obj_max_y = pose['y'] + half_l

                # Define clearance zone boundaries
                is_blocking = False
                push_direction = None

                if side == 'north':
                    # Clearance zone extends south from north wall
                    zone_min_x = dx_rel - width/2
                    zone_max_x = dx_rel + width/2
                    zone_min_y = dy_rel - clearance
                    zone_max_y = dy_rel

                    # Check if object overlaps with clearance zone
                    if (obj_max_x > zone_min_x and obj_min_x < zone_max_x and
                        obj_max_y > zone_min_y and obj_min_y < zone_max_y):
                        is_blocking = True
                        push_direction = 'south'  # Push away from north wall

                elif side == 'south':
                    # Clearance zone extends north from south wall
                    zone_min_x = dx_rel - width/2
                    zone_max_x = dx_rel + width/2
                    zone_min_y = dy_rel
                    zone_max_y = dy_rel + clearance

                    if (obj_max_x > zone_min_x and obj_min_x < zone_max_x and
                        obj_max_y > zone_min_y and obj_min_y < zone_max_y):
                        is_blocking = True
                        push_direction = 'north'

                elif side == 'east':
                    # Clearance zone extends west from east wall
                    zone_min_x = dx_rel - clearance
                    zone_max_x = dx_rel
                    zone_min_y = dy_rel - width/2
                    zone_max_y = dy_rel + width/2

                    if (obj_max_x > zone_min_x and obj_min_x < zone_max_x and
                        obj_max_y > zone_min_y and obj_min_y < zone_max_y):
                        is_blocking = True
                        push_direction = 'west'

                elif side == 'west':
                    # Clearance zone extends east from west wall
                    zone_min_x = dx_rel
                    zone_max_x = dx_rel + clearance
                    zone_min_y = dy_rel - width/2
                    zone_max_y = dy_rel + width/2

                    if (obj_max_x > zone_min_x and obj_min_x < zone_max_x and
                        obj_max_y > zone_min_y and obj_min_y < zone_max_y):
                        is_blocking = True
                        push_direction = 'east'

                # If blocking, push object away from doorway
                if is_blocking:
                    old_x, old_y = pose['x'], pose['y']

                    if push_direction == 'north':
                        # Push north (positive Y)
                        pose['y'] = zone_max_y + half_l + 0.2  # Add small buffer
                    elif push_direction == 'south':
                        # Push south (negative Y)
                        pose['y'] = zone_min_y - half_l - 0.2
                    elif push_direction == 'east':
                        # Push east (positive X)
                        pose['x'] = zone_max_x + half_w + 0.2
                    elif push_direction == 'west':
                        # Push west (negative X)
                        pose['x'] = zone_min_x - half_w - 0.2

                    moved_count += 1
                    logger.info(f"  ⚠️  Moved {obj_type} away from {side} doorway: ({old_x:.2f},{old_y:.2f}) → ({pose['x']:.2f},{pose['y']:.2f})")

        if moved_count > 0:
            logger.info(f"  ✓ Cleared {moved_count} object(s) from doorway zones")
        else:
            logger.debug("  ✓ No objects blocking doorways")

        return plan

    def _enforce_robust_chair_positioning(self, plan: List[Dict], room: Room, model_info_cache: Dict[str, Dict] = None) -> List[Dict]:
        """
        Robustly enforce desk/table-chair relationships with proper positioning and orientation.
        This ensures chairs are:
        1. Positioned in front of desks/tables (not behind, not to the side)
        2. Facing toward the desk/table surface
        3. At appropriate distance for user access
        4. Properly oriented based on desk/table orientation
        
        Handles both desks (work surfaces) and dining/conference tables.
        """
        # Find all desks, tables, and chairs
        work_surfaces = []  # Desks + tables
        for i, obj in enumerate(plan):
            obj_type = obj['type'].lower()
            if 'desk' in obj_type or 'table' in obj_type:
                # Determine if it's a work surface (desk) or dining surface (table)
                is_work = 'desk' in obj_type
                work_surfaces.append((i, obj, 'desk' if is_work else 'table'))
        
        chairs = [(i, obj) for i, obj in enumerate(plan) if 'chair' in obj['type'].lower()]

        if not work_surfaces or not chairs:
            logger.debug("  ℹ️ No desk/table-chair pairs to enforce")
            return plan

        # Track which chairs have been assigned
        assigned_chairs = set()
        adjustments_made = 0


        for surf_idx, surface, surf_type in work_surfaces:
            surf_pose = surface['pose']
            surf_yaw = surf_pose.get('yaw', 0.0)
            surf_size = self.get_actual_model_dimensions(surface['type'], room_type=room.type)

            logger.debug(f"  Processing {surf_type} at ({surf_pose['x']:.2f}, {surf_pose['y']:.2f})")

            # For tables, we might want multiple chairs around it
            # For desks, typically one chair in front
            max_chairs = 4 if surf_type == 'table' else 1

            # Find nearest unassigned chairs
            chair_distances = []
            for chair_idx, chair in chairs:
                if chair_idx in assigned_chairs:
                    continue

                chair_pose = chair['pose']
                distance = math.sqrt(
                    (surf_pose['x'] - chair_pose['x'])**2 +
                    (surf_pose['y'] - chair_pose['y'])**2
                )
                chair_distances.append((distance, chair_idx, chair))

            # Sort by distance and take closest ones
            chair_distances.sort(key=lambda x: x[0])
            chairs_to_assign = chair_distances[:min(max_chairs, len(chair_distances))]

            logger.debug(f"    Found {len(chairs_to_assign)} chair(s) to assign (max={max_chairs})")
            for dist, c_idx, _ in chairs_to_assign:
                logger.debug(f"      - chair_{c_idx} at distance {dist:.2f}m")

            # Position chairs around the surface
            for assignment_idx, (distance, chair_idx, chair) in enumerate(chairs_to_assign):
                # Mark chair as assigned immediately to prevent it being picked by another desk
                assigned_chairs.add(chair_idx)
                
                chair_size = self.get_actual_model_dimensions(chair['type'], room_type=room.type)

                # Get model-specific positioning if available
                surf_model_name = None
                if model_info_cache and surface['type'] in model_info_cache:
                    surf_model_name = model_info_cache[surface['type']].get('model_name')
                    logger.debug(f"  Retrieved model name for {surface['type']}: '{surf_model_name}'")
                
                if surf_type == 'desk':
                    # Use model-specific corrections if available
                    if surf_model_name and surf_model_name in self.model_orientation_corrections:
                        correction = self.model_orientation_corrections[surf_model_name]
                        offset_x = correction['chair_offset_x']
                        offset_y = correction['chair_offset_y']
                        chair_yaw_offset = correction['chair_yaw_offset']
                        
                        # Transform offsets from desk's local frame to world frame
                        cos_yaw = math.cos(surf_yaw)
                        sin_yaw = math.sin(surf_yaw)
                        
                        # In desk's local frame: +X is right, +Y is forward
                        # Rotate to world frame
                        world_offset_x = offset_x * cos_yaw - offset_y * sin_yaw
                        world_offset_y = offset_x * sin_yaw + offset_y * cos_yaw
                        
                        target_x = surf_pose['x'] + world_offset_x
                        target_y = surf_pose['y'] + world_offset_y
                        target_yaw = surf_yaw + chair_yaw_offset
                        
                        logger.info(f"  🪑 Using '{surf_model_name}' corrections: "
                                   f"offset=({offset_x:.3f}, {offset_y:.3f}), world=({world_offset_x:.3f}, {world_offset_y:.3f})")
                    else:
                        # Fallback: generic positioning
                        logger.debug(f"  No model-specific correction for '{surf_model_name}', using generic positioning")
                        clearance = 0.15
                        offset_distance = (surf_size[1] / 2) + (chair_size[1] / 2) + clearance
                        sin_yaw = math.sin(surf_yaw)
                        cos_yaw = math.cos(surf_yaw)
                        
                        target_x = surf_pose['x'] - offset_distance * sin_yaw
                        target_y = surf_pose['y'] - offset_distance * cos_yaw
                        target_yaw = surf_yaw

                elif surf_type == 'table':
                    # Table: Distribute chairs around it (north, south, east, west)
                    # Position chairs on all 4 sides based on table dimensions and orientation
                    # Positions: 0=north (front, -Y), 1=east (right, +X), 2=south (back, +Y), 3=west (left, -X)
                    
                    clearance = 0.4  # Space between table edge and chair
                    
                    # Get table dimensions (width=X, length=Y)
                    table_width = surf_size[0]
                    table_length = surf_size[1]
                    
                    # Position index determines which side of table (0-3 for 4 sides)
                    side_idx = assignment_idx % 4
                    
                    # Calculate offset distance based on which side
                    # For north/south (along Y axis), use table_length
                    # For east/west (along X axis), use table_width
                    if side_idx in [0, 2]:  # North or South
                        offset_distance = (table_length / 2) + (chair_size[1] / 2) + clearance
                    else:  # East or West
                        offset_distance = (table_width / 2) + (chair_size[0] / 2) + clearance
                    
                    # Position multipliers in table's local frame
                    # (0, -1) = north/front, (1, 0) = east/right, (0, 1) = south/back, (-1, 0) = west/left
                    positions = [
                        (0, -1),   # North (front in table frame)
                        (1, 0),    # East (right in table frame)
                        (0, 1),    # South (back in table frame)
                        (-1, 0),   # West (left in table frame)
                    ]
                    
                    dx_mult, dy_mult = positions[side_idx]
                    
                    sin_yaw = math.sin(surf_yaw)
                    cos_yaw = math.cos(surf_yaw)
                    
                    # Transform from table's local frame to world frame
                    # Local frame: +X is right, +Y is forward (toward user)
                    local_x = dx_mult * offset_distance
                    local_y = dy_mult * offset_distance
                    
                    # Rotate by table's yaw to get world coordinates
                    world_offset_x = local_x * cos_yaw - local_y * sin_yaw
                    world_offset_y = local_x * sin_yaw + local_y * cos_yaw
                    
                    target_x = surf_pose['x'] + world_offset_x
                    target_y = surf_pose['y'] + world_offset_y
                    
                    # Chair faces toward table center
                    # Calculate vector from chair to table, then add pi/2 for chair orientation
                    dx = surf_pose['x'] - target_x
                    dy = surf_pose['y'] - target_y
                    target_yaw = math.atan2(dy, dx) + math.pi/2
                    
                    logger.debug(f"    Positioning chair {assignment_idx} on {['north', 'east', 'south', 'west'][side_idx]} side at ({target_x:.2f}, {target_y:.2f}, yaw={target_yaw:.2f})")

                # Apply the positioning
                old_x, old_y, old_yaw = chair['pose']['x'], chair['pose']['y'], chair['pose'].get('yaw', 0.0)
                
                # Check if adjustment is significant (> 20cm or > 30 degrees)
                dist_change = math.sqrt((target_x - old_x)**2 + (target_y - old_y)**2)
                yaw_change = abs(target_yaw - old_yaw)
                
                if dist_change > 0.2 or yaw_change > 0.5:
                    chair['pose']['x'] = target_x
                    chair['pose']['y'] = target_y
                    chair['pose']['yaw'] = target_yaw
                    adjustments_made += 1
                    logger.info(f"    ✓ Repositioned chair_{chair_idx}: ({old_x:.2f},{old_y:.2f}) → ({target_x:.2f},{target_y:.2f}), yaw: {old_yaw:.2f} → {target_yaw:.2f}")
                else:
                    logger.debug(f"    • Chair_{chair_idx} already in good position ({old_x:.2f},{old_y:.2f}), no adjustment needed")

        if adjustments_made > 0:
            logger.info(f"  ✓ Adjusted {adjustments_made} chair(s) to proper positions")
        else:
            logger.debug("  ✓ All chairs already in good positions")

        return plan


    def _create_fallback_plan(self, room: Room, objects_to_place: List[Dict]) -> List[Dict]:
        """Generate a simple rule-based placement plan."""
        plan, placed_items = [], []
        
        # Expand and categorize objects
        anchors = [] # Desks, tables, large storage items
        wall_items = [] # Shelves, cabinets
        accessories = [] # Chairs, plants, monitors
        for obj in objects_to_place:
            for i in range(obj.get('count', 1)):
                item = {'type': obj['type'], 'unique_name': f"{obj['type']}_{i}"}
                if item['type'] in ['desk', 'table', 'storage_rack', 'shelving_unit']:
                    anchors.append(item)
                elif item['type'] in ['shelf', 'bookshelf', 'cabinet']:
                    wall_items.append(item)
                else:
                    accessories.append(item)

        # Place anchors (desks, tables, storage items) appropriately
        storage_items = [item for item in anchors if item['type'] in ['storage_rack', 'shelving_unit']]
        office_items = [item for item in anchors if item['type'] in ['desk', 'table']]

        if storage_items and room.type == 'warehouse':
            # Use shared collision detector for intelligent grid placement
            if not self.collision_detector:
                # Create world metadata for collision detector
                world_metadata = {
                    'door_positions': [],  # No doors in initial placement
                    'room_bounds': {
                        room.name: {
                            'min_x': -room.dimensions['width'] / 2,
                            'max_x': room.dimensions['width'] / 2,
                            'min_y': -room.dimensions['length'] / 2,
                            'max_y': room.dimensions['length'] / 2,
                            'center_x': 0.0,
                            'center_y': 0.0,
                            'width': room.dimensions['width'],
                            'length': room.dimensions['length']
                        }
                    }
                }
                self.collision_detector = CollisionDetector(world_metadata)
            
            # Prepare objects for grid positioning
            grid_objects = []
            for item in storage_items:
                obj_type = item['type']
                dimensions = self.object_sizes.get(obj_type, self.object_sizes['default'])
                grid_objects.append({
                    'name': item['unique_name'],
                    'dimensions': dimensions
                })
            
            # Calculate grid positions with wall alignment
            room_bounds = {
                'min_x': -room.dimensions['width'] / 2,
                'max_x': room.dimensions['width'] / 2,
                'min_y': -room.dimensions['length'] / 2,
                'max_y': room.dimensions['length'] / 2,
                'center_x': 0.0,
                'center_y': 0.0,
                'width': room.dimensions['width'],
                'length': room.dimensions['length']
            }
            
            # Use "in rows" for warehouse storage
            positions = self.collision_detector.calculate_grid_positions(
                grid_objects, "in rows", room_bounds, []
            )
            
            # Apply positions to storage items
            for item, position in zip(storage_items, positions):
                x, y, yaw = position
                item['pose'] = {'x': x, 'y': y, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': yaw}
                plan.append(item)
                placed_items.append(item)
            
            logger.info(f"Placed {len(storage_items)} warehouse storage items using collision-free grid positioning")

        # Handle office items normally
        for i, item in enumerate(office_items):
            x = (i * 3.0) - (len(office_items)-1) * 1.5  # Space them out by 3m
            y = room.dimensions['length'] / 2 - 1.0  # Against north wall
            item['pose'] = {'x': x, 'y': y, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': -1.57}  # Facing south
            plan.append(item)
            placed_items.append(item)

        # 2. Place wall items along a different wall
        for i, item in enumerate(wall_items):
            x = -room.dimensions['width'] / 2 + 0.5
            y = (i * 1.5) - (len(wall_items)-1) / 2
            item['pose'] = {'x': x, 'y': y, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 1.57}
            plan.append(item)

        # Place accessories (chairs near desks, items in corners)
        for item in accessories:
            if item['type'] == 'chair':
                for anchor in anchors:
                    if not any(p['unique_name'] == f"chair_for_{anchor['unique_name']}" for p in placed_items):
                        item['unique_name'] = f"chair_for_{anchor['unique_name']}"
                        anchor_pose = anchor['pose']
                        item['pose'] = {'x': anchor_pose['x'], 'y': anchor_pose['y'] - 1.0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 1.57} # Facing north
                        plan.append(item)
                        placed_items.append(item)
                        break
            else: 
                item['pose'] = {'x': room.dimensions['width']/2 - 0.5, 'y': -room.dimensions['length']/2 + 0.5, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0}
                plan.append(item)

        logger.info(f"Generated a fallback rule-based plan for {len(plan)} objects.")
        return plan

    def _create_models_from_plan(self, plan: List[Dict], room: Room, name_generator) -> List[GazeboModel]:
        """Create GazeboModel instances from the placement plan."""
        placed_models = []
        
        logger.info(f"Resolving models and creating {len(plan)} objects...")
        
        # Count objects by type in the plan
        type_counts = {}
        for item in plan:
            obj_type = item.get('type', 'unknown')
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        logger.debug(f"Plan contains: {', '.join([f'{count}x {obj_type}' for obj_type, count in type_counts.items()])}")
        
        # Check if we have semantic groups with "on_top" arrangements
        surface_items_plan = []
        ground_items_plan = []
        
        if hasattr(self, '_semantic_groups') and self._semantic_groups:
            # Build set of items that should go on surfaces
            surface_item_types = set()
            for group in self._semantic_groups:
                # groups are dicts with 'spatial_arrangement' key
                spatial_arrangement = group.get('spatial_arrangement', {})
                for obj_type, arrangement in spatial_arrangement.items():
                    if arrangement == 'on_top':
                        surface_item_types.add(obj_type)
            
            # Separate plan based on semantic grouping
            for item in plan:
                if item.get('type') in surface_item_types:
                    surface_items_plan.append(item)
                else:
                    ground_items_plan.append(item)
            
            logger.debug(f"Using semantic groups for item classification: {len(ground_items_plan)} ground, {len(surface_items_plan)} surface")
        else:
            # Fallback: Everything goes on ground (single pass)
            ground_items_plan = plan
            surface_items_plan = []
            logger.debug(f"No semantic groups - all {len(plan)} items treated as ground objects")
        
        # Create all ground-based furniture
        # This establishes the surfaces that other objects can be placed on
        for item in ground_items_plan:
            obj_type = item['type']
            unique_name = name_generator(obj_type)
            
            model_path = self.model_db.find_best_model(obj_type, room.type)
            if not model_path:
                logger.warning(f"No model found for '{obj_type}', skipping.")
                continue
            
            pose = item['pose']
            # Enforce z=0 for ground items, preserve Z for surface items
            if pose.get('z', 0.0) == 0.0:
                pose['z'] = 0.0
            # Otherwise keep the Z value set by semantic grouping (for monitors on desks, etc.)

            # Transform from room-local coordinates to world coordinates
            world_pose = {
                'x': pose['x'] + room.position['x'],
                'y': pose['y'] + room.position['y'],
                'z': pose['z'] + room.position['z'],
                'roll': pose.get('roll', 0.0),
                'pitch': pose.get('pitch', 0.0),
                'yaw': pose.get('yaw', 0.0)
            }

            model = GazeboModel(
                name=unique_name,
                model_path=model_path,
                category="furniture",
                pose=world_pose,
                room=room.name,
                static=True
            )
            placed_models.append(model)
            logger.info(f"  ✅ Placed {model.name} at (x={world_pose['x']:.2f}, y={world_pose['y']:.2f}, z={world_pose['z']:.2f})")

        # Create all surface-based items
        # (Only create surface items that aren't already handled by semantic grouping)
        if hasattr(self, '_semantic_groups') and self._semantic_groups:
            logger.debug("Skipping surface item creation pass - items handled by semantic grouping")
            surface_items_plan = []  # Don't create duplicates

        for item in surface_items_plan:
            obj_type = item['type']
            unique_name = name_generator(obj_type)

            model_path = self.model_db.find_best_model(obj_type, room.type)
            if not model_path:
                logger.warning(f"No model found for '{obj_type}', skipping.")
                continue

            pose = item['pose']

            # Transform to world coordinates for surface detection
            world_x = pose['x'] + room.position['x']
            world_y = pose['y'] + room.position['y']

            # Find the closest valid surface from the models we just created
            potential_surfaces = [m for m in placed_models if m.name.startswith(('desk', 'table', 'shelf', 'bookshelf'))]
            closest_surface = None
            min_dist = float('inf')

            if potential_surfaces:
                closest_surface = min(
                    potential_surfaces,
                    key=lambda s: math.dist((world_x, world_y), (s.pose['x'], s.pose['y']))
                )
                min_dist = math.dist((world_x, world_y), (closest_surface.pose['x'], closest_surface.pose['y']))

            if closest_surface and min_dist < 1.5:
                # Place object on top of the surface
                # Extract the specific resolved model name from the surface's model_path
                surface_model_name = closest_surface.model_path.replace("model://", "")
                surface_type = closest_surface.name.split('_')[0]

                # Try to get actual dimensions and offset from the SPECIFIC resolved model
                try:
                    # Use the specific model name to get its actual dimensions
                    surface_dims = self.get_actual_model_dimensions(surface_type, model_name=surface_model_name, room_type=room.type)

                    # Try to get the cached offset for this specific surface model
                    surface_cache_key = f"{surface_type}_{surface_model_name}_{room.type or 'default'}"
                    surface_simple_key = f"{surface_type}_resolved"
                    surface_offset = self.model_offsets_cache.get(surface_cache_key) or self.model_offsets_cache.get(surface_simple_key)

                    if surface_offset and surface_offset[2] != 0:
                        # The offset[2] now contains max_z (top surface height)
                        surface_top_z = surface_offset[2]
                        logger.debug(f"Using surface '{surface_model_name}' top Z from bounding box: {surface_top_z:.3f}")
                    else:
                        # No offset available, use extracted dimensions height as fallback
                        surface_top_z = surface_dims[2]
                        logger.debug(f"No bounding box offset for '{surface_model_name}', using dimension height {surface_top_z:.3f}")

                    # Place object at surface top height
                    pose['z'] = surface_top_z
                    logger.info(f"Placing '{unique_name}' on '{surface_model_name}' surface '{closest_surface.name}' at z={surface_top_z:.2f}m.")
                except Exception as e:
                    # Fallback to standard sizes
                    logger.debug(f"Could not get dimensions for {surface_type}: {e}")
                    surface_height = self.object_sizes.get(surface_type, (0,0,0.55))[2]
                    pose['z'] = surface_height
                    logger.info(f"Placing '{unique_name}' on nearby surface '{closest_surface.name}' at z={surface_height:.2f}m (fallback).")
            else:
                # If no surface is found, place it on the ground as a fallback
                pose['z'] = 0.0
                logger.warning(f"No suitable surface found for '{unique_name}'. Placing on the ground.")

            # Transform from room-local coordinates to world coordinates
            world_pose = {
                'x': pose['x'] + room.position['x'],
                'y': pose['y'] + room.position['y'],
                'z': pose['z'] + room.position['z'],
                'roll': pose.get('roll', 0.0),
                'pitch': pose.get('pitch', 0.0),
                'yaw': pose.get('yaw', 0.0)
            }

            model = GazeboModel(
                name=unique_name,
                model_path=model_path,
                category="furniture",
                pose=world_pose,
                room=room.name,
                static=True
            )
            placed_models.append(model)
            logger.info(f"  ✅ Placed {model.name} at (x={world_pose['x']:.2f}, y={world_pose['y']:.2f}, z={world_pose['z']:.2f})")
            
        logger.info(f"🏁 Spatial Placement Complete: {len(placed_models)} models created from plan.")
        return placed_models