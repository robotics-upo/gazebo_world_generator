"""
Natural Placement Engine

Furnishes rooms: resolves real models (and their measured sizes) for every
requested object, lets the LLM design the layout through ``LayoutDesigner``
(propose, get measured violations back, revise), then turns the layout into
Gazebo models.
"""

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

from gazebo_world_generator.src.core.data_models import Room, GazeboModel
from gazebo_world_generator.src.core.object_vocabulary import PRIMITIVE_URI, primitive_size
from gazebo_world_generator.src.config.validated_settings import PlacementConfig, RoomConfig
from gazebo_world_generator.src.placement.checks import free_point
from gazebo_world_generator.src.placement.designer import LayoutDesigner
from gazebo_world_generator.src.placement.spatial import SpatialRegistry
from gazebo_world_generator.src.models.visual_quality import model_shape
from gazebo_world_generator.src.prompts.manager import PromptManager
from gazebo_world_generator.src.utils.file_utils import find_gazebo_model_path
from gazebo_world_generator.src.utils.sdf_parser import SDFDimensionExtractor
from gazebo_world_generator.src.utils.sdf_utils import extract_model_metadata, model_origin_height

logger = logging.getLogger(__name__)

DESCRIPTION_CHARS = 120


class NaturalPlacementEngine:
    """Places requested objects in rooms using an LLM-designed, code-checked layout."""

    def __init__(self, model_db, online_db=None, llm_interface=None,
                 placement_config: Optional[PlacementConfig] = None,
                 room_config: Optional[RoomConfig] = None,
                 spatial_registry: Optional[SpatialRegistry] = None):
        self.model_db = model_db
        self.llm_interface = llm_interface
        self.config = placement_config if placement_config else PlacementConfig()
        self.room_config = room_config if room_config else RoomConfig()
        self.rng = random.Random(self.config.random_seed)
        # Original request; the generator sets it per world so prompts keep the user's intent.
        self.user_description = ""

        if llm_interface and hasattr(llm_interface, 'prompt_manager'):
            self.prompt_manager = llm_interface.prompt_manager
        else:
            try:
                self.prompt_manager = PromptManager(template_dir='prompts/', default_version='v1',
                                                    enable_metrics=True)
            except Exception as e:
                logger.warning(f"Failed to initialize PromptManager in NaturalPlacementEngine: {e}")
                self.prompt_manager = None

        self.sdf_extractor = SDFDimensionExtractor()
        self.spatial_registry = spatial_registry or SpatialRegistry(model_db, self.sdf_extractor)
        self.model_dimensions_cache = self.spatial_registry.model_dimensions_cache
        self.model_offsets_cache = self.spatial_registry.model_offsets_cache
        self.object_sizes = self.spatial_registry.default_sizes

        self.designer = LayoutDesigner(
            llm_interface=llm_interface,
            prompt_manager=self.prompt_manager if llm_interface else None,
            rng=self.rng,
            max_rounds=self.config.max_design_rounds,
            wall_thickness=self.room_config.wall_thickness,
            chars_per_token=self.config.chars_per_token,
            review_rounds=self.config.review_rounds,
        )

    def get_actual_model_dimensions(self, object_type: str, model_name: str = None,
                                    room_type: str = None) -> Tuple[float, float, float]:
        """(width, length, height) measured from the model, or a default size."""
        return self.spatial_registry.get_dimensions(object_type, model_name, room_type)

    def place_all_objects(self, rooms: List[Room], name_generator) -> List[GazeboModel]:
        """Furnish every non-corridor room that has objects."""
        rooms_to_furnish = [room for room in rooms if room.type != "corridor" and room.objects]
        placed = []
        for number, room in enumerate(rooms_to_furnish, start=1):
            placed.extend(self.place_objects_in_room(room, "", room.objects, name_generator,
                                                     number, len(rooms_to_furnish)))
        return placed

    def place_objects_in_room(self, room: Room, world_map: str, objects_to_place: List[Dict],
                              name_generator, room_number: int = 0,
                              total_rooms: int = 0) -> List[GazeboModel]:
        """Resolve models, design the layout with the LLM, and create the models."""
        if not objects_to_place:
            logger.debug(f"No objects to place in '{room.name}'. Skipping placement.")
            return []
        progress = f" [{room_number}/{total_rooms}]" if total_rooms > 1 else ""
        logger.info(f"Placing {len(objects_to_place)} objects in '{room.name}' ({room.type}){progress}")

        model_info = self._resolve_models_early(objects_to_place, room)
        items = self._layout_items(objects_to_place, model_info)
        fixed = self._fixed_items(room)

        placed, violations = self.designer.design(room, items, self.user_description, fixed)
        room.free_point = free_point(room, list(placed) + fixed,
                                     wall_thickness=self.room_config.wall_thickness)
        for violation in violations:
            logger.warning(f"Layout issue in '{room.name}': {violation}")
        if not violations:
            logger.info(f"Layout matches request for '{room.name}'")

        return self._create_models_from_plan(placed, room, name_generator, model_info)

    def _resolve_models_early(self, objects_to_place: List[Dict], room: Room) -> Dict[str, Dict]:
        """
        Resolve a model per object type before layout, so the LLM designs with real sizes.

        Returns:
            Dict mapping object_type -> {'uri', 'model_name', 'dimensions', 'offset',
            'description', 'metadata'}. ``offset`` is (x, y) of the footprint centre
            relative to the model origin, and z of the model's top surface.
        """
        model_info_cache = {}
        object_types = sorted(set(obj['type'] for obj in objects_to_place))
        logger.info(f"🔍 Resolving {len(object_types)} model types early...")

        for obj_type in object_types:
            model_uri = self.model_db.find_best_model(obj_type, room.type)
            if not model_uri:
                # Fail before spending LLM calls on a layout that cannot be built.
                raise ValueError(f"No visible model found for requested object '{obj_type}'")

            if model_uri == PRIMITIVE_URI:
                size = primitive_size(obj_type)
                model_info_cache[obj_type] = {
                    'uri': model_uri, 'model_name': None, 'dimensions': size,
                    'offset': (0.0, 0.0, size[2]),
                    'description': f"A plain box standing in for a {obj_type}", 'metadata': {}}
                continue

            model_name = model_uri.replace("model://", "")
            model_path = find_gazebo_model_path(model_name)
            metadata = extract_model_metadata(model_path) if model_path else {}
            dimensions = self.get_actual_model_dimensions(obj_type, model_name, room.type)
            offset = self.model_offsets_cache.get(
                f"{obj_type}_{model_name}_{room.type or 'default'}", (0.0, 0.0, dimensions[2]))
            model_info_cache[obj_type] = {
                'uri': model_uri, 'model_name': model_name, 'dimensions': dimensions,
                'offset': offset, 'description': metadata.get("description", ""),
                'metadata': metadata, 'shape': self._shape(model_path, offset)}
            logger.debug(f"  ✓ {obj_type} -> {model_name} "
                         f"({dimensions[0]:.2f}×{dimensions[1]:.2f}×{dimensions[2]:.2f}m)")

        logger.info("✅ Model resolution complete - all dimensions known")
        return model_info_cache

    def _shape(self, model_path, offset) -> Optional[List[tuple]]:
        """Model triangles relative to the footprint centre, for the layout image."""
        if model_path is None:
            return None
        search_paths = getattr(self.model_db, "_model_search_paths", lambda: [])()
        triangles = model_shape(model_path, search_paths)
        if not triangles:
            return None
        return [tuple((x - offset[0], y - offset[1], z) for x, y, z in triangle)
                for triangle in triangles]

    @staticmethod
    def _layout_items(objects_to_place: List[Dict], model_info: Dict[str, Dict]) -> List[Dict]:
        """One item per instance, with a stable id the LLM refers to."""
        counters: Dict[str, int] = {}
        items = []
        for obj in objects_to_place:
            obj_type = obj['type']
            info = model_info[obj_type]
            for _ in range(obj.get('count', 1)):
                index = counters.get(obj_type, 0)
                counters[obj_type] = index + 1
                description = " ".join(str(info.get('description') or '').split())
                context = obj.get('semantic_context')
                if context:
                    description = f"{description} (user: {context})".strip()
                items.append({
                    'id': f"{obj_type}_{index}",
                    'type': obj_type,
                    'model': info['model_name'] or "plain box",
                    'dims': tuple(info['dimensions']),
                    'front': '+x',
                    'shape': info.get('shape'),
                    'description': description[:DESCRIPTION_CHARS],
                })
        return items

    @staticmethod
    def _fixed_items(room: Room) -> List[Dict]:
        """Objects already in the room (refinement) as immovable items."""
        fixed = []
        for obj in getattr(room, 'objects', None) or []:
            if getattr(obj, 'model_path', None) != 'existing':
                continue
            size = obj.size or [1.0, 1.0, 1.0]
            fixed.append({'id': obj.name, 'type': obj.category, 'fixed': True,
                          'x': obj.pose['x'] - room.position['x'],
                          'y': obj.pose['y'] - room.position['y'],
                          'yaw': obj.pose.get('yaw', 0.0), 'dims': tuple(size)})
        return fixed

    def _create_models_from_plan(self, plan: List[Dict], room: Room, name_generator,
                                 model_info: Optional[Dict[str, Dict]] = None) -> List[GazeboModel]:
        """Turn laid-out items (footprint centres) into world-frame Gazebo models."""
        by_id = {item['id']: item for item in plan}
        heights: Dict[str, float] = {}
        models = []
        # Supports first, so items standing on them know their surface height.
        for item in sorted(plan, key=lambda entry: bool(entry.get('on'))):
            obj_type = item['type']
            info = (model_info or {}).get(obj_type)
            uri = info['uri'] if info else self.model_db.find_best_model(obj_type, room.type)
            if not uri:
                raise ValueError(f"No visible model found for requested object '{obj_type}'")
            offset = info['offset'] if info else (0.0, 0.0, item['dims'][2])

            yaw = item.get('yaw', 0.0)
            # The layout places footprint centres; the model origin may sit elsewhere.
            origin_x = item['x'] - (offset[0] * math.cos(yaw) - offset[1] * math.sin(yaw))
            origin_y = item['y'] - (offset[0] * math.sin(yaw) + offset[1] * math.cos(yaw))
            base = heights.get(item.get('on'), 0.0) if item.get('on') in by_id else 0.0
            origin_z = base
            if uri.startswith("model://"):
                origin_z += model_origin_height(find_gazebo_model_path(uri.removeprefix("model://")))
            heights[item['id']] = origin_z + offset[2]

            model = GazeboModel(
                name=name_generator(obj_type),
                model_path=uri,
                category="furniture",
                pose={'x': origin_x + room.position['x'], 'y': origin_y + room.position['y'],
                      'z': origin_z + room.position['z'], 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw},
                room=room.name,
                static=True,
                size=list(primitive_size(obj_type)) if uri == PRIMITIVE_URI else None,
            )
            models.append(model)
            logger.info(f"  ✅ Placed {model.name} at (x={model.pose['x']:.2f}, "
                        f"y={model.pose['y']:.2f}, z={model.pose['z']:.2f})")
        logger.info(f"🏁 Spatial Placement Complete: {len(models)} models created from plan.")
        return models
