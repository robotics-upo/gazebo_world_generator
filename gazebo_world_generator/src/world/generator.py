"""
World Generator Module

Handles SDF world file generation with rooms, objects, and physics configuration.
Manages room layout, wall creation, and object placement coordination.
"""

import logging
from typing import Dict, List, Optional
from xml.dom import minidom
import xml.etree.ElementTree as ET
import math 

from gazebo_world_generator.src.llm.interface import OpenAICompatibleInterface as LLMInterface
from gazebo_world_generator.src.core.data_models import Room, GazeboModel
from gazebo_world_generator.src.models.resolver import SmartModelResolver
from gazebo_world_generator.src.models.online_database import OnlineModelDatabase
from gazebo_world_generator.src.placement.engine import NaturalPlacementEngine
from gazebo_world_generator.src.config.validated_settings import ValidatedConfig

logger = logging.getLogger(__name__)


class WorldGenerator:
    """Main class for generating Gazebo world files with advanced layout features."""

    def __init__(self, llm_interface: LLMInterface, config: Optional[ValidatedConfig] = None):
        self.llm_interface = llm_interface
        self.rooms: List[Room] = []
        self.models: List[GazeboModel] = []
        self.model_counter: Dict[str, int] = {}

        # Use provided config or create defaults
        self.config = config if config else ValidatedConfig()
        self.room_config = self.config.rooms
        self.placement_config = self.config.placement
        self.physics_config = self.config.physics

        self.online_db = OnlineModelDatabase(llm_interface=llm_interface)
        self.model_db = SmartModelResolver(llm_interface=llm_interface, online_db=self.online_db)
        self.placement_engine = NaturalPlacementEngine(
            model_db=self.model_db,
            llm_interface=llm_interface,
            placement_config=self.placement_config,
            room_config=self.room_config
        )
        # Log to file only
        logger.debug("WorldGenerator initialized with validated configuration")

    def get_unique_model_name(self, base_name: str) -> str:
        """Generates a globally unique name for a model."""
        sane_base_name = base_name.replace(" ", "_")
        count = self.model_counter.get(sane_base_name, 0)
        self.model_counter[sane_base_name] = count + 1
        return f"{sane_base_name}_{count}"

    def generate_world(self, description: str, output_path: str = "generated_world.sdf") -> str:
        """Generate a complete Gazebo world from a natural language description."""
        logger.debug(f"Generating world from description: '{description}'")
        
        parsed_data = self.llm_interface.parse_room_description(description, self.model_db)
        logger.info(f"Parsed {len(parsed_data.get('rooms', []))} rooms from description.")
        
        logger.info("Creating room structure and layout...")
        self._create_rooms_and_layout(parsed_data.get("rooms", []))
        
        logger.info("Populating rooms with objects...")
        self._populate_rooms()
        
        logger.info("Generating walls and doorways...")
        sdf_content = self._generate_sdf()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(sdf_content)
        
        logger.info(f"World file generated at: {output_path}")
        
        return output_path
    
    def _calculate_room_dimensions(self, room_data: Dict) -> Dict:
        """Calculate room dimensions based on type and object requirements."""
        dims = room_data.get("dimensions", {})
        if room_data.get("type") == "corridor":
            dims.setdefault("width", self.placement_config.corridor_width)
            dims.setdefault("length", 8.0)
        elif "width" not in dims or "length" not in dims:
            required_area = sum(
                (self.placement_engine.object_sizes.get(req["type"], (1.0, 1.0))[0] *
                 self.placement_engine.object_sizes.get(req["type"], (1.0, 1.0))[1]) *
                req.get("count", 1)
                for req in room_data.get("objects", [])
            )
            min_room_area = self.room_config.min_room_width * self.room_config.min_room_width
            total_area = max(required_area * 4.0, min_room_area)
            side_length = math.sqrt(total_area)
            dims = {"width": side_length, "length": side_length}
        dims.setdefault("height", self.room_config.default_height)
        return {k: abs(float(v)) for k, v in dims.items()}

    def _create_rooms_and_layout(self, rooms_data: List[Dict]):
        """Creates and positions rooms using two-pass algorithm."""
        if not rooms_data: return

        # Separate corridors from main rooms
        main_rooms_data = [r for r in rooms_data if r.get("type") != "corridor"]
        corridors_data = [r for r in rooms_data if r.get("type") == "corridor"]

        logger.info(f"Positioning rooms: {len(main_rooms_data)} main rooms and {len(corridors_data)} corridors")

        # Place all main rooms using optimized grid layout
        self._place_main_rooms(main_rooms_data)

        # Calculate and place corridors to bridge gaps between connected rooms
        self._place_connecting_corridors(corridors_data)

        # Calculate doorway positions and clearance zones BEFORE object placement
        logger.debug("Calculating doorway positions for all rooms...")
        for room in self.rooms:
            self._calculate_doorway_positions(room)

        # Generate walls and doorways for all rooms
        logger.debug("Generating walls and doorways for all rooms...")
        for room in self.rooms:
            self._add_walls_and_doorways(room)

            # Add floor for corridors to make them visible
            if room.type == "corridor":
                self._add_corridor_floor(room)

    def _place_main_rooms(self, main_rooms_data: List[Dict]):
        """Place all main rooms in an optimized grid layout."""
        if not main_rooms_data:
            return

        # Calculate room dimensions for all rooms first
        room_info = []
        for room_data in main_rooms_data:
            dims = self._calculate_room_dimensions(room_data)
            room_info.append({
                'data': room_data,
                'dimensions': dims,
                'area': dims['width'] * dims['length']
            })

        # Use connection-aware positioning to place rooms
        self._place_rooms_with_connections(room_info)

    def _opposite_side(self, side: str) -> str:
        """Get the opposite side (e.g., 'north' → 'south')."""
        opposites = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }
        return opposites.get(side, side)

    def _place_rooms_with_connections(self, room_info: List[Dict]):
        """Place rooms respecting their connection semantics."""
        logger.info("Placing rooms using connection-aware positioning...")

        # Create a mapping from room names to room data
        room_data_map = {r['data']['name']: r for r in room_info}
        placed_rooms = {}
        
        # Standard corridor width that we'll use for spacing
        self.corridor_spacing = 2.5  # 2.0m corridor + 0.5m clearance
        
        self.corridor_connections = {}
        all_rooms_data = [r['data'] for r in room_info]
        
        logger.info(f"📋 Analyzing {len(all_rooms_data)} rooms for corridor-based connections...")
        
        # Build a map: corridor_name -> list of (room_name, side) tuples
        corridor_to_rooms = {}
        for room_data in all_rooms_data:
            room_name = room_data['name']
            for connected_name, side in room_data.get('connections', {}).items():
                # If the connected_name looks like a corridor (contains "corridor" in name),
                # track which rooms connect through it
                if 'corridor' in connected_name.lower():
                    if connected_name not in corridor_to_rooms:
                        corridor_to_rooms[connected_name] = []
                    corridor_to_rooms[connected_name].append((room_name, side))
        
        # Create direct room-to-room connections through corridors
        for corridor_name, room_connections in corridor_to_rooms.items():
            logger.info(f"🚪 Corridor '{corridor_name}' connects {len(room_connections)} rooms: {[r[0] for r in room_connections]}")
            
            if len(room_connections) == 2:
                room1_name, room1_side = room_connections[0]
                room2_name, room2_side = room_connections[1]
                
                self.corridor_connections[corridor_name] = {
                    room1_name: room1_side,
                    room2_name: room2_side
                }
                logger.info(f"   → Direct connection: {room1_name} ({room1_side}) ←→ {room2_name} ({room2_side})")
        
        logger.info(f"🚪 Found {len(self.corridor_connections)} corridors to create")

        # Find a starting room (preferably one with the most connections or largest area)
        start_room_info = max(room_info, key=lambda r: (
            len(r['data'].get('connections', {})),
            r['area']
        ))

        # Place the starting room at origin
        start_room_data = start_room_info['data']
        start_dims = start_room_info['dimensions']
        start_position = {"x": 0, "y": 0, "z": 0}

        start_room = Room(
            name=start_room_data['name'],
            type=start_room_data.get("type"),
            dimensions=start_dims,
            position=start_position,
            objects=start_room_data.get("objects", []),
            connections=start_room_data.get("connections", {})
        )

        placed_rooms[start_room.name] = start_room
        self.rooms.append(start_room)
        logger.info(f"🎯 Starting room: '{start_room.name}' at origin with connections: {start_room.connections}")

        # Use a queue to place connected rooms
        placement_queue = [(start_room, None, None)]  # (room, parent_room, connection_side)

        while placement_queue:
            current_room, parent_room, parent_side = placement_queue.pop(0)

            # Process all connections from this room
            for connected_room_name, connection_side in current_room.connections.items():
                logger.info(f"📍 Processing: '{current_room.name}' connects to '{connected_room_name}' on '{connection_side}' side")
                
                # Check if this is a corridor reference
                is_corridor = 'corridor' in connected_room_name.lower()
                actual_connected_room = connected_room_name
                has_corridor_between = False
                
                if is_corridor and connected_room_name in self.corridor_connections:
                    # This is a corridor - find the room on the OTHER side
                    has_corridor_between = True
                    corridor_rooms = self.corridor_connections[connected_room_name]
                    
                    # Find the other room (not current room)
                    for room_name in corridor_rooms.keys():
                        if room_name != current_room.name:
                            actual_connected_room = room_name
                            logger.info(f"✅ Resolved through corridor: '{current_room.name}' → '{connected_room_name}' → '{actual_connected_room}'")
                            break
                    
                    # If we still have the corridor name (didn't find another room), skip
                    if actual_connected_room == connected_room_name:
                        logger.warning(f"⚠️ Skipping '{connected_room_name}' - could not resolve to a target room")
                        continue
                
                if actual_connected_room in placed_rooms:
                    continue  # Already placed

                if actual_connected_room not in room_data_map:
                    logger.debug(f"Room '{actual_connected_room}' referenced in connections but not found in room data")
                    continue

                # Calculate position for the connected room
                connected_room_info = room_data_map[actual_connected_room]
                connected_room_data = connected_room_info['data']
                connected_dims = connected_room_info['dimensions']

                # Calculate position based on connection side (with corridor spacing if needed)
                new_position = self._calculate_connected_room_position(
                    current_room, connection_side, connected_dims, add_corridor_spacing=has_corridor_between
                )

                # Create and place the connected room
                connected_room = Room(
                    name=connected_room_data['name'],
                    type=connected_room_data.get("type"),
                    dimensions=connected_dims,
                    position=new_position,
                    objects=connected_room_data.get("objects", []),
                    connections=connected_room_data.get("connections", {})
                )

                placed_rooms[connected_room.name] = connected_room
                self.rooms.append(connected_room)
                placement_queue.append((connected_room, current_room, connection_side))

                logger.info(f"Placed room '{connected_room.name}' at ({new_position['x']:.1f}, {new_position['y']:.1f}) - {connection_side} of '{current_room.name}'")

        # Handle any unconnected rooms by placing them nearby
        for room_info_item in room_info:
            room_name = room_info_item['data']['name']
            if room_name not in placed_rooms:
                room_data = room_info_item['data']
                dims = room_info_item['dimensions']

                # Check if this room references a corridor (even if corridor only connects 1 room)
                # If so, place it adjacent to an existing room rather than far away
                connections = room_data.get("connections", {})
                has_corridor_ref = any('corridor' in conn.lower() for conn in connections.keys())

                if has_corridor_ref and self.rooms:
                    # Place adjacent to the last placed room using simple grid logic
                    logger.info(f"Room '{room_name}' has corridor reference but couldn't be connected. Placing adjacent to existing rooms.")
                    position = self._find_adjacent_position_for_room(dims)
                else:
                    logger.debug(f"Room '{room_name}' has no connections. Placing independently.")
                    position = self._find_empty_position_for_orphan_room(dims)

                room = Room(
                    name=room_data['name'],
                    type=room_data.get("type"),
                    dimensions=dims,
                    position=position,
                    objects=room_data.get("objects", []),
                    connections=connections
                )

                self.rooms.append(room)
                placed_rooms[room_name] = room
                logger.info(f"Placed orphan room '{room.name}' at ({position['x']:.1f}, {position['y']:.1f})")

    def _calculate_connected_room_position(self, base_room: Room, connection_side: str, connected_dims: Dict[str, float], add_corridor_spacing: bool = False) -> Dict[str, float]:
        """Calculate the position for a room connected to a base room on the specified side."""
        base_pos = base_room.position
        base_dims = base_room.dimensions

        # Calculate half-dimensions for positioning
        base_half_w = base_dims['width'] / 2
        base_half_l = base_dims['length'] / 2
        conn_half_w = connected_dims['width'] / 2
        conn_half_l = connected_dims['length'] / 2

        # Add corridor spacing if there's a corridor in between
        spacing = self.corridor_spacing if add_corridor_spacing else 0.0

        if connection_side == "north":
            return {
                "x": base_pos['x'],
                "y": base_pos['y'] + base_half_l + conn_half_l + spacing,
                "z": 0
            }
        elif connection_side == "south":
            return {
                "x": base_pos['x'],
                "y": base_pos['y'] - base_half_l - conn_half_l - spacing,
                "z": 0
            }
        elif connection_side == "east":
            return {
                "x": base_pos['x'] + base_half_w + conn_half_w + spacing,
                "y": base_pos['y'],
                "z": 0
            }
        elif connection_side == "west":
            return {
                "x": base_pos['x'] - base_half_w - conn_half_w - spacing,
                "y": base_pos['y'],
                "z": 0
            }
        else:
            logger.warning(f"Unknown connection side: {connection_side}")
            return {"x": 0, "y": 0, "z": 0}

    def _find_adjacent_position_for_room(self, dims: Dict[str, float]) -> Dict[str, float]:
        """Find a compact adjacent position for a room (used for corridor-referenced rooms)."""
        if not self.rooms:
            return {"x": 0, "y": 0, "z": 0}

        # Try to place adjacent to existing rooms in a grid pattern
        min_gap = 2.5  # Small gap for corridor space

        for existing_room in self.rooms:
            if existing_room.type == "corridor":
                continue  # Don't position relative to corridors

            # Try east side first (most common)
            candidate_x = existing_room.position['x'] + existing_room.dimensions['width']/2 + dims['width']/2 + min_gap
            candidate_y = existing_room.position['y']

            # Check if this position is free (no collision with other rooms)
            if self._is_position_free(candidate_x, candidate_y, dims):
                return {"x": candidate_x, "y": candidate_y, "z": 0}

            # Try south side
            candidate_x = existing_room.position['x']
            candidate_y = existing_room.position['y'] - existing_room.dimensions['length']/2 - dims['length']/2 - min_gap

            if self._is_position_free(candidate_x, candidate_y, dims):
                return {"x": candidate_x, "y": candidate_y, "z": 0}

        # Fallback: use orphan positioning if no adjacent spot found
        return self._find_empty_position_for_orphan_room(dims)

    def _is_position_free(self, x: float, y: float, dims: Dict[str, float]) -> bool:
        """Check if a position is free from collisions with existing rooms."""
        half_w, half_l = dims['width']/2, dims['length']/2

        for room in self.rooms:
            if room.type == "corridor":
                continue  # Ignore corridors for collision

            room_half_w = room.dimensions['width']/2
            room_half_l = room.dimensions['length']/2

            # Check AABB collision
            if (abs(x - room.position['x']) < half_w + room_half_w + 0.5 and
                abs(y - room.position['y']) < half_l + room_half_l + 0.5):
                return False

        return True

    def _find_empty_position_for_orphan_room(self, dims: Dict[str, float]) -> Dict[str, float]:
        """Find an empty position for a room that has no connections."""
        if not self.rooms:
            return {"x": 0, "y": 0, "z": 0}

        max_x = max(room.position['x'] + room.dimensions['width']/2 for room in self.rooms)
        return {
            "x": max_x + dims['width'],
            "y": 0,
            "z": 0
        }

    def _place_connecting_corridors(self, corridors_data: List[Dict]):
        """Adjust room positions to create space for corridors and place them."""
        if not corridors_data:
            return

        placed_rooms = {r.name: r for r in self.rooms}

        for corridor_data in corridors_data:
            corridor_name = corridor_data['name']
            connections = corridor_data.get('connections', {})

            if len(connections) < 2:
                logger.warning(f"Corridor '{corridor_name}' has fewer than 2 connections. Skipping.")
                continue

            # Get the two rooms this corridor should connect
            connected_room_names = list(connections.keys())
            room1_name, room2_name = connected_room_names[0], connected_room_names[1]

            if room1_name not in placed_rooms or room2_name not in placed_rooms:
                logger.error(f"Corridor '{corridor_name}' references non-existent rooms. Skipping.")
                continue

            room1, room2 = placed_rooms[room1_name], placed_rooms[room2_name]

            # Convert corridor perspective to room perspective (opposite sides)
            room1_side_facing_corridor = self._opposite_side(connections[room1_name])
            room2_side_facing_corridor = self._opposite_side(connections[room2_name])

            # Calculate the corridor geometry to fill the existing gap between rooms
            corridor_pos, corridor_dims = self._calculate_corridor_geometry(
                room1, room2, room1_side_facing_corridor, room2_side_facing_corridor
            )

            corridor = Room(
                name=corridor_name,
                type="corridor",
                dimensions=corridor_dims,
                position=corridor_pos,
                objects=[],
                connections=connections
            )

            self.rooms.append(corridor)
            placed_rooms[corridor_name] = corridor
            logger.debug(f"Placed corridor '{corridor_name}' between '{room1_name}' and '{room2_name}'")


    def _calculate_corridor_geometry(self, room1: Room, room2: Room, room1_side: str, room2_side: str) -> tuple:
        """Calculate corridor position and dimensions to exactly fill the gap between rooms."""
        # Get the boundary points where the corridor should connect
        boundary1 = self._get_room_boundary_point(room1, room1_side)
        boundary2 = self._get_room_boundary_point(room2, room2_side)

        logger.info(f"Corridor calculation: {room1.name} {room1_side} side at {boundary1} -> {room2.name} {room2_side} side at {boundary2}")

        # Calculate corridor center point
        corridor_center_x = (boundary1[0] + boundary2[0]) / 2
        corridor_center_y = (boundary1[1] + boundary2[1]) / 2

        # Calculate the actual distance between rooms
        actual_distance = math.dist(boundary1, boundary2)
        corridor_width = 2.0  # Standard corridor width (cross-sectional width)

        # Determine if corridor is horizontal or vertical
        dx = boundary2[0] - boundary1[0]
        dy = boundary2[1] - boundary1[1]

        # In SDF, width=X-axis and length=Y-axis
        # For horizontal corridor (east-west): width should be long, length should be narrow
        # For vertical corridor (north-south): width should be narrow, length should be long
        if abs(dx) > abs(dy):  # Horizontal corridor
            dimensions = {
                "width": actual_distance,
                "length": corridor_width,
                "height": 3.0
            }
        else:  # Vertical corridor
            dimensions = {
                "width": corridor_width,
                "length": actual_distance,
                "height": 3.0
            }

        position = {"x": corridor_center_x, "y": corridor_center_y, "z": 0}

        logger.info(f"Corridor: {actual_distance:.1f}m long x {corridor_width}m wide at ({corridor_center_x:.1f}, {corridor_center_y:.1f})")

        return position, dimensions

    def _get_room_boundary_point(self, room: Room, side: str) -> tuple:
        """
        Get the point beyond the room's edge to overlap with doorway.
        Returns (x, y) coordinates of the boundary point on the specified side.
        """
        x, y = room.position['x'], room.position['y']
        w, l = room.dimensions['width'], room.dimensions['length']
        
        # No extension needed - corridor floor should span exactly the gap between rooms
        extension = 0.0
        
        if side == 'north': return (x, y + l/2 - extension)
        elif side == 'south': return (x, y - l/2 + extension)
        elif side == 'east': return (x + w/2 - extension, y)
        elif side == 'west': return (x - w/2 + extension, y)
        else: return (x, y)

    def _calculate_doorway_positions(self, room: Room):
        """
        Calculate doorway positions and clearance zones for a room.
        """
        if not room.connections:
            return  # No doorways in this room

        w, l = room.dimensions["width"], room.dimensions["length"]
        cx, cy = room.position["x"], room.position["y"]
        door_width = 1.6  # Standard doorway width

        connected_sides = list(room.connections.values())

        # For each connected side, calculate the doorway center position
        for side in connected_sides:
            doorway_info = {
                'name': f'{room.name}_{side}_door',
                'side': side,
                'width': door_width,
                'clearance_radius': 1.5  # Clearance radius around doorway (meters)
            }

            # Calculate the doorway center position in world coordinates
            if side == 'north':
                doorway_info['x'] = cx  # Center of north wall
                doorway_info['y'] = cy + l/2  # At north wall position
            elif side == 'south':
                doorway_info['x'] = cx
                doorway_info['y'] = cy - l/2
            elif side == 'east':
                doorway_info['x'] = cx + w/2
                doorway_info['y'] = cy
            elif side == 'west':
                doorway_info['x'] = cx - w/2
                doorway_info['y'] = cy

            room.doorways.append(doorway_info)
            logger.debug(f"Room '{room.name}': Doorway on {side} wall at ({doorway_info['x']:.2f}, {doorway_info['y']:.2f})")

    def _populate_rooms(self):
        """Populate each room with objects using the placement engine."""
        self.models.append(GazeboModel(name="ground_plane", model_path="model://ground_plane", category="ground",
                                       pose={"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}, static=True))
        
        for room in self.rooms:
            if room.objects:
                expanded_objects = []
                for obj in room.objects:
                    count = obj.get('count', 1)
                    for _ in range(count):
                        expanded_objects.append({
                            'type': obj['type'],
                            'count': 1,
                            'semantic_context': obj.get('semantic_context', '')
                        })
                room.objects = expanded_objects
                logger.debug(f"Expanded {len(room.objects)} object instances for room '{room.name}'")
        
        placed_models = self.placement_engine.place_all_objects(self.rooms, self.get_unique_model_name)
        self.models.extend(placed_models)

    def _add_walls_and_doorways(self, room: Room):
        """Generates wall segments for a room, creating doorway openings for connections."""
        w, l, h = room.dimensions["width"], room.dimensions["length"], room.dimensions["height"]
        cx, cy = room.position["x"], room.position["y"]
        thickness = 0.15
        door_width = 1.6  

        # For corridors, we need to handle dimensions correctly based on orientation
        if room.type == "corridor":
            if room.dimensions["length"] > room.dimensions["width"]:
                # Horizontal corridor (longer in X direction)
                wall_specs = {
                    "north": {'p1': (cx - l/2, cy + w/2), 'p2': (cx + l/2, cy + w/2), 'axis': 'x', 'length': l},
                    "east":  {'p1': (cx + l/2, cy + w/2), 'p2': (cx + l/2, cy - w/2), 'axis': 'y', 'length': w},
                    "south": {'p1': (cx + l/2, cy - w/2), 'p2': (cx - l/2, cy - w/2), 'axis': 'x', 'length': l},
                    "west":  {'p1': (cx - l/2, cy - w/2), 'p2': (cx - l/2, cy + w/2), 'axis': 'y', 'length': w}
                }
            else:
                # Vertical corridor (longer in Y direction)
                wall_specs = {
                    "north": {'p1': (cx - w/2, cy + l/2), 'p2': (cx + w/2, cy + l/2), 'axis': 'x', 'length': w},
                    "east":  {'p1': (cx + w/2, cy + l/2), 'p2': (cx + w/2, cy - l/2), 'axis': 'y', 'length': l},
                    "south": {'p1': (cx + w/2, cy - l/2), 'p2': (cx - w/2, cy - l/2), 'axis': 'x', 'length': w},
                    "west":  {'p1': (cx - w/2, cy - l/2), 'p2': (cx - w/2, cy + l/2), 'axis': 'y', 'length': l}
                }
        else:
            # Regular rooms use traditional width/length meaning
            wall_specs = {
                "north": {'p1': (cx - w/2, cy + l/2), 'p2': (cx + w/2, cy + l/2), 'axis': 'x', 'length': w},
                "east":  {'p1': (cx + w/2, cy + l/2), 'p2': (cx + w/2, cy - l/2), 'axis': 'y', 'length': l},
                "south": {'p1': (cx + w/2, cy - l/2), 'p2': (cx - w/2, cy - l/2), 'axis': 'x', 'length': w},
                "west":  {'p1': (cx - w/2, cy - l/2), 'p2': (cx - w/2, cy + l/2), 'axis': 'y', 'length': l}
            }

        connected_sides = list(room.connections.values())
        sane_name = room.name.replace(" ", "_")

        # Special handling for corridors
        if room.type == "corridor":
            self._create_corridor_walls(room, wall_specs, connected_sides, h, thickness, door_width)
        else:
            # Regular room wall generation
            for side, spec in wall_specs.items():
                if side not in connected_sides:
                    # Build a complete wall on this side
                    self._create_complete_wall(sane_name, side, spec, h, thickness)
                else:
                    # Build wall segments with a doorway opening in the middle
                    self._create_wall_with_doorway(sane_name, side, spec, h, thickness, door_width)
                    logger.debug(f"Created doorway on '{room.name}'s {side} wall")

    def _create_corridor_walls(self, corridor: Room, wall_specs: Dict, connected_sides: List[str], height: float, thickness: float, door_width: float):
        """Create corridor walls that don't extend beyond room doorway boundaries."""
        sane_name = corridor.name.replace(" ", "_")

        for side, spec in wall_specs.items():
            if side not in connected_sides:
                # Build walls for sides that don't connect to rooms, but limit their extent
                self._create_limited_corridor_wall(corridor, sane_name, side, spec, height, thickness, door_width)
            else:
                logger.info(f"Leaving corridor '{corridor.name}' {side} side open for room connection")

    def _create_limited_corridor_wall(self, corridor: Room, sane_name: str, side: str, spec: Dict, height: float, thickness: float, door_width: float):
        """Create corridor walls only where actually needed - avoiding all room connection areas."""
        # Find connected rooms to determine where NOT to place walls
        connected_room_names = list(corridor.connections.keys())
        connected_rooms = [r for r in self.rooms if r.name in connected_room_names and r.type != "corridor"]

        if len(connected_rooms) < 2:
            # If we can't find both connected rooms, skip wall creation entirely
            logger.info(f"Skipping corridor wall on {side} - insufficient connected rooms found")
            return

        # Check if this corridor side faces any connected rooms
        rooms_on_this_side = []

        for room in connected_rooms:
            if self._room_faces_corridor_side(room, corridor, side):
                rooms_on_this_side.append(room)

        if rooms_on_this_side:
            logger.info(f"Corridor {side} side faces {len(rooms_on_this_side)} rooms - creating minimal edge walls only")
            self._create_minimal_edge_walls(corridor, sane_name, side, spec, height, thickness, door_width, rooms_on_this_side)
        else:
            logger.info(f"Corridor {side} side faces no rooms - creating limited full wall")
            self._create_limited_full_wall(corridor, sane_name, side, spec, height, thickness)

    def _room_faces_corridor_side(self, room: Room, corridor: Room, corridor_side: str) -> bool:
        """Check if a room is positioned to face a specific side of the corridor."""
        # Determine room position relative to corridor
        room_x, room_y = room.position['x'], room.position['y']
        corridor_x, corridor_y = corridor.position['x'], corridor.position['y']

        if corridor_side == 'north':
            return room_y > corridor_y + 0.5  
        elif corridor_side == 'south':
            return room_y < corridor_y - 0.5 
        elif corridor_side == 'east':
            return room_x > corridor_x + 0.5  
        elif corridor_side == 'west':
            return room_x < corridor_x - 0.5 

        return False

    def _create_minimal_edge_walls(self, corridor: Room, sane_name: str, side: str, spec: Dict, height: float, thickness: float, door_width: float, rooms_on_this_side: list):
        """Create very small wall segments only at the edges beyond room boundaries."""

        if spec['axis'] == 'x':  # North/South corridor walls (horizontal)
            corridor_start_x = corridor.position['x'] - corridor.dimensions['width']/2
            corridor_end_x = corridor.position['x'] + corridor.dimensions['width']/2
            wall_y = spec['p1'][1]

            # Find the leftmost and rightmost room extents
            room_extends = []
            for room in rooms_on_this_side:
                room_left = room.position['x'] - room.dimensions['width']/2
                room_right = room.position['x'] + room.dimensions['width']/2
                room_extends.extend([room_left, room_right])

            if room_extends:
                leftmost_room = min(room_extends)
                rightmost_room = max(room_extends)

                if leftmost_room > corridor_start_x + 0.3:
                    wall_length = min(1.0, leftmost_room - corridor_start_x)
                    wall_center_x = corridor_start_x + wall_length/2
                    self._create_wall_segment(f"{sane_name}_wall_{side}_left", height, wall_length, thickness,
                                            (wall_center_x, wall_y, height/2))
                    logger.info(f"Created minimal left edge wall on {side} (length: {wall_length:.1f}m)")

                if rightmost_room < corridor_end_x - 0.3:
                    wall_length = min(1.0, corridor_end_x - rightmost_room)
                    wall_center_x = rightmost_room + wall_length/2
                    self._create_wall_segment(f"{sane_name}_wall_{side}_right", height, wall_length, thickness,
                                            (wall_center_x, wall_y, height/2))
                    logger.info(f"Created minimal right edge wall on {side} (length: {wall_length:.1f}m)")

        else:  # East/West corridor walls (vertical)
            corridor_start_y = corridor.position['y'] - corridor.dimensions['length']/2
            corridor_end_y = corridor.position['y'] + corridor.dimensions['length']/2
            wall_x = spec['p1'][0]

            room_extends = []
            for room in rooms_on_this_side:
                room_bottom = room.position['y'] - room.dimensions['length']/2
                room_top = room.position['y'] + room.dimensions['length']/2
                room_extends.extend([room_bottom, room_top])

            if room_extends:
                bottommost_room = min(room_extends)
                topmost_room = max(room_extends)

                if bottommost_room > corridor_start_y + 0.3:
                    wall_length = min(1.0, bottommost_room - corridor_start_y)
                    wall_center_y = corridor_start_y + wall_length/2
                    self._create_wall_segment(f"{sane_name}_wall_{side}_bottom", height, thickness, wall_length,
                                            (wall_x, wall_center_y, height/2))
                    logger.info(f"Created minimal bottom edge wall on {side} (length: {wall_length:.1f}m)")

                if topmost_room < corridor_end_y - 0.3:
                    wall_length = min(1.0, corridor_end_y - topmost_room)
                    wall_center_y = topmost_room + wall_length/2
                    self._create_wall_segment(f"{sane_name}_wall_{side}_top", height, thickness, wall_length,
                                            (wall_x, wall_center_y, height/2))
                    logger.info(f"Created minimal top edge wall on {side} (length: {wall_length:.1f}m)")

    def _create_limited_full_wall(self, corridor: Room, sane_name: str, side: str, spec: Dict, height: float, thickness: float):
        """Create corridor walls that extend to align with the room wall edges."""

        # Find all connected rooms
        connected_room_names = list(corridor.connections.keys())
        connected_rooms = [r for r in self.rooms if r.name in connected_room_names and r.type != "corridor"]
        
        door_width = 1.6  # Standard doorway width

        if spec['axis'] == 'x':  # North/South corridor walls (walls running east-west along X-axis)
            # Position wall at the doorway edge, not the corridor edge
            # For north wall, position at +door_width/2; for south wall, at -door_width/2
            if side == 'north':
                wall_y = corridor.position['y'] + door_width/2
            elif side == 'south':
                wall_y = corridor.position['y'] - door_width/2
            else:
                wall_y = spec['p1'][1]  # Fallback
            
            # For walls running along X-axis, extend to the CENTER of perpendicular room walls
            # This creates proper corner joints without gaps or excessive overlap
            if connected_rooms:
                room_x_positions = []
                wall_thickness = self.room_config.wall_thickness
                for room in connected_rooms:
                    # Extend to wall center (half thickness AWAY from room interior)
                    if room.position['x'] < corridor.position['x']:
                        # Room is west, extend away from room (westward, negative direction)
                        facing_edge = room.position['x'] + room.dimensions['width']/2 - wall_thickness/2
                    else:
                        # Room is east, extend away from room (eastward, positive direction)
                        facing_edge = room.position['x'] - room.dimensions['width']/2 + wall_thickness/2
                    room_x_positions.append(facing_edge)
                
                if room_x_positions:
                    corridor_start_x = min(room_x_positions)
                    corridor_end_x = max(room_x_positions)
                else:
                    # Fallback to corridor dimensions
                    corridor_start_x = corridor.position['x'] - corridor.dimensions['width']/2 
                    corridor_end_x = corridor.position['x'] + corridor.dimensions['width']/2
            else:
                # Fallback to corridor dimensions
                corridor_start_x = corridor.position['x'] - corridor.dimensions['width']/2 
                corridor_end_x = corridor.position['x'] + corridor.dimensions['width']/2

            # No gaps needed - north/south walls run continuously along X-axis
            room_gaps = []

            # Create wall segments
            self._create_wall_segments_with_gaps(corridor_start_x, corridor_end_x, room_gaps,
                                                sane_name, side, height, thickness, wall_y, True)

        else:  # East/West corridor walls (walls running north-south along Y-axis)
            # Position wall at the doorway edge, not the corridor edge
            if side == 'east':
                wall_x = corridor.position['x'] + door_width/2
            elif side == 'west':
                wall_x = corridor.position['x'] - door_width/2
            else:
                wall_x = spec['p1'][0]  # Fallback
            
            # For walls running along Y-axis, extend to the CENTER of perpendicular room walls
            # This creates proper corner joints without gaps or excessive overlap
            if connected_rooms:
                room_y_positions = []
                wall_thickness = self.room_config.wall_thickness
                for room in connected_rooms:
                    # Extend to wall center (half thickness AWAY from room interior)
                    if room.position['y'] < corridor.position['y']:
                        # Room is south, extend away from room (southward, negative direction)
                        facing_edge = room.position['y'] + room.dimensions['length']/2 - wall_thickness/2
                    else:
                        # Room is north, extend away from room (northward, positive direction)
                        facing_edge = room.position['y'] - room.dimensions['length']/2 + wall_thickness/2
                    room_y_positions.append(facing_edge)
                
                if room_y_positions:
                    corridor_start_y = min(room_y_positions)
                    corridor_end_y = max(room_y_positions)
                else:
                    corridor_start_y = corridor.position['y'] - corridor.dimensions['width']/2
                    corridor_end_y = corridor.position['y'] + corridor.dimensions['width']/2
            else:
                corridor_start_y = corridor.position['y'] - corridor.dimensions['width']/2
                corridor_end_y = corridor.position['y'] + corridor.dimensions['width']/2

            # No gaps needed - east/west walls run continuously along Y-axis
            room_gaps = []

            self._create_wall_segments_with_gaps(corridor_start_y, corridor_end_y, room_gaps,
                                                sane_name, side, height, thickness, wall_x, False)

    def _create_wall_segments_with_gaps(self, start_pos: float, end_pos: float, gaps: list,
                                       sane_name: str, side: str, height: float, thickness: float,
                                       fixed_coord: float, is_horizontal: bool):
        """Create wall segments along a line, leaving gaps for room connections."""
        if not gaps:
            if end_pos > start_pos + 0.5:  # Only create if meaningful length
                wall_length = end_pos - start_pos
                center_pos = start_pos + wall_length/2

                if is_horizontal:
                    self._create_wall_segment(f"{sane_name}_wall_{side}_limited", height, wall_length, thickness,
                                            (center_pos, fixed_coord, height/2))
                else:
                    self._create_wall_segment(f"{sane_name}_wall_{side}_limited", height, thickness, wall_length,
                                            (fixed_coord, center_pos, height/2))
                logger.debug(f"Created corridor wall on {side} (length: {wall_length:.1f}m)")
            return

        # Sort gaps by position
        gaps = sorted(gaps)

        # Merge overlapping gaps
        merged_gaps = []
        for gap_start, gap_end in gaps:
            if merged_gaps and gap_start <= merged_gaps[-1][1] + 0.5:
                # Overlapping or close gaps, merge them
                merged_gaps[-1] = (merged_gaps[-1][0], max(merged_gaps[-1][1], gap_end))
            else:
                merged_gaps.append((gap_start, gap_end))

        # Create wall segments between gaps
        current_pos = start_pos
        segment_count = 0

        for gap_start, gap_end in merged_gaps:
            # Create wall before this gap
            if current_pos < gap_start - 0.1:  # Leave small margin
                wall_length = gap_start - current_pos
                center_pos = current_pos + wall_length/2

                if wall_length > 0.3:  
                    segment_count += 1
                    if is_horizontal:
                        self._create_wall_segment(f"{sane_name}_wall_{side}_{segment_count}", height, wall_length, thickness,
                                                (center_pos, fixed_coord, height/2))
                    else:
                        self._create_wall_segment(f"{sane_name}_wall_{side}_{segment_count}", height, thickness, wall_length,
                                                (fixed_coord, center_pos, height/2))
                    logger.info(f"Created corridor wall segment {segment_count} on {side} (length: {wall_length:.1f}m)")

            current_pos = gap_end

        # Create wall after the last gap
        if current_pos < end_pos - 0.1:  # Leave small margin
            wall_length = end_pos - current_pos
            center_pos = current_pos + wall_length/2

            if wall_length > 0.3:  
                segment_count += 1
                if is_horizontal:
                    self._create_wall_segment(f"{sane_name}_wall_{side}_{segment_count}", height, wall_length, thickness,
                                            (center_pos, fixed_coord, height/2))
                else:
                    self._create_wall_segment(f"{sane_name}_wall_{side}_{segment_count}", height, thickness, wall_length,
                                            (fixed_coord, center_pos, height/2))
                logger.info(f"Created corridor wall segment {segment_count} on {side} (length: {wall_length:.1f}m)")

        logger.info(f"Created {segment_count} wall segments on {side} with {len(merged_gaps)} gaps for room connections")


    def _add_corridor_floor(self, corridor: Room):
        """Add a visible floor plane for the corridor to make it visible in Gazebo."""
        w, l = corridor.dimensions["width"], corridor.dimensions["length"]
        cx, cy = corridor.position["x"], corridor.position["y"]

        floor_name = f"{corridor.name.replace(' ', '_')}_floor"

        # Create a thin floor plane at ground level
        self.models.append(GazeboModel(
            name=floor_name,
            model_path="built_in_wall",
            category="structure",
            pose={'x': cx, 'y': cy, 'z': 0.01, 'roll': 0, 'pitch': 0, 'yaw': 0},
            size=[w, l, 0.02],
            static=True
        ))
        logger.info(f"Added floor plane for corridor '{corridor.name}' ({w:.1f}m x {l:.1f}m)")

    def _create_complete_wall(self, room_name: str, side: str, spec: Dict, height: float, thickness: float):
        """Create a complete wall segment that extends to meet perpendicular walls at their centers."""
        # Extend wall by half thickness on each end to meet perpendicular wall centers
        wall_extension = thickness / 2
        wall_length = spec['length'] + wall_extension * 2  # Extend both ends
        
        mid_point = ((spec['p1'][0] + spec['p2'][0]) / 2, (spec['p1'][1] + spec['p2'][1]) / 2)

        if spec['axis'] == 'x':  # North/South walls (horizontal)
            self._create_wall_segment(f"{room_name}_wall_{side}", height, wall_length, thickness,
                                    (mid_point[0], mid_point[1], height/2))
        else:  # East/West walls (vertical)
            self._create_wall_segment(f"{room_name}_wall_{side}", height, thickness, wall_length,
                                    (mid_point[0], mid_point[1], height/2))

    def _create_wall_with_doorway(self, room_name: str, side: str, spec: Dict, height: float, thickness: float, door_width: float):
        """Create wall segments with a doorway opening in the center."""
        wall_length = spec['length']

        # Calculate doorway position (center of the wall)
        if wall_length <= door_width + 1.0:
            logger.info(f"Wall {side} too short for doorway segments, leaving completely open")
            return

        # Calculate the length of each wall segment on either side of the doorway
        wall_extension = thickness  # Full thickness for proper corner coverage
        segment_length = (wall_length - door_width) / 2 + wall_extension

        if segment_length < 0.5:
            logger.info(f"Wall segments too short on {side}, leaving completely open")
            return

        # Calculate positions for the two wall segments
        if spec['axis'] == 'x':
            wall_y = spec['p1'][1]
            wall_start_x = spec['p1'][0]
            wall_end_x = spec['p2'][0]

            # Ensure proper ordering
            if wall_start_x > wall_end_x:
                wall_start_x, wall_end_x = wall_end_x, wall_start_x

            # Left segment - extends slightly beyond wall start to meet perpendicular wall
            left_center_x = wall_start_x - wall_extension/2 + segment_length / 2
            self._create_wall_segment(f"{room_name}_wall_{side}_left", height, segment_length, thickness,
                                    (left_center_x, wall_y, height/2))

            # Right segment - extends slightly beyond wall end to meet perpendicular wall
            right_center_x = wall_end_x + wall_extension/2 - segment_length / 2
            self._create_wall_segment(f"{room_name}_wall_{side}_right", height, segment_length, thickness,
                                    (right_center_x, wall_y, height/2))

        else:  # East/West walls (vertical walls)
            wall_x = spec['p1'][0]
            wall_start_y = spec['p1'][1]
            wall_end_y = spec['p2'][1]

            # Ensure proper ordering
            if wall_start_y > wall_end_y:
                wall_start_y, wall_end_y = wall_end_y, wall_start_y

            # Bottom segment - extends slightly beyond wall start to meet perpendicular wall
            bottom_center_y = wall_start_y - wall_extension/2 + segment_length / 2
            self._create_wall_segment(f"{room_name}_wall_{side}_bottom", height, thickness, segment_length,
                                    (wall_x, bottom_center_y, height/2))

            # Top segment - extends slightly beyond wall end to meet perpendicular wall
            top_center_y = wall_end_y + wall_extension/2 - segment_length / 2
            self._create_wall_segment(f"{room_name}_wall_{side}_top", height, thickness, segment_length,
                                    (wall_x, top_center_y, height/2))

    def _create_wall_segment(self, name: str, h: float, w: float, l: float, pose_xyz: tuple, yaw: float = 0.0):
        """Helper to create a single wall model."""
        sane_name = name.replace(" ", "_")
        self.models.append(GazeboModel(
            name=sane_name, model_path="built_in_wall", category="structure",
            pose={'x': pose_xyz[0], 'y': pose_xyz[1], 'z': pose_xyz[2], 'roll': 0, 'pitch': 0, 'yaw': yaw},
            size=[w, l, h], static=True
        ))

    def _generate_sdf(self) -> str:
        """Generate the SDF XML content using xml.etree.ElementTree for correctness."""
        sdf = ET.Element("sdf", version="1.7")
        world = ET.SubElement(sdf, "world", name="generated_world")
        
        # Add Header (GUI, Physics, Scene, Light)
        world.append(ET.fromstring('''
        <gui>
          <camera name="user_camera"><pose>7.1031 -0.036058 23.3852 0 1.239 3.1267</pose></camera>
        </gui>'''))
        world.append(ET.fromstring('''
        <physics type="ode" name="default_physics">
          <gravity>0 0 -9.8</gravity>
          <real_time_update_rate>1000</real_time_update_rate>
          <max_step_size>0.001</max_step_size>
        </physics>'''))
        world.append(ET.fromstring('''
        <scene>
          <ambient>0.4 0.4 0.4 1</ambient>
          <background>0.7 0.7 0.7 1</background>
          <shadows>true</shadows>
        </scene>'''))
        world.append(ET.fromstring('''
        <light name="sun" type="directional">
          <cast_shadows>1</cast_shadows>
          <pose>0 0 10 0 0 0</pose>
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
          <attenuation><range>1000</range></attenuation>
          <direction>0 0.5 -0.9</direction>
        </light>'''))

        # Add all models
        for model in self.models:
            p = model.pose
            pose_text = (f"{p.get('x', 0.0):.3f} {p.get('y', 0.0):.3f} {p.get('z', 0.0):.3f} "
                         f"{p.get('roll', 0.0):.3f} {p.get('pitch', 0.0):.3f} {p.get('yaw', 0.0):.3f}")

            if model.model_path.startswith("model://"):
                include = ET.SubElement(world, "include")
                ET.SubElement(include, "uri").text = model.model_path
                ET.SubElement(include, "name").text = model.name
                ET.SubElement(include, "pose").text = pose_text
                if model.static:
                    ET.SubElement(include, "static").text = "true"
                    
            elif model.model_path == "built_in_wall":
                model_elem = ET.SubElement(world, "model", name=model.name)
                ET.SubElement(model_elem, "static").text = "true"
                ET.SubElement(model_elem, "pose").text = pose_text
                link = ET.SubElement(model_elem, "link", name="wall_link")

                size_str = f"{model.size[0]:.3f} {model.size[1]:.3f} {model.size[2]:.3f}"
                
                # Visual with material
                visual = ET.SubElement(link, "visual", name="wall_visual")
                box_geom_v = ET.SubElement(ET.SubElement(visual, "geometry"), "box")
                ET.SubElement(box_geom_v, "size").text = size_str
                mat = ET.SubElement(visual, "material")
                script = ET.SubElement(mat, "script")
                ET.SubElement(script, "name").text = "Gazebo/Grey"

                # Collision with surface properties
                collision = ET.SubElement(link, "collision", name="wall_collision")
                box_geom_c = ET.SubElement(ET.SubElement(collision, "geometry"), "box")
                ET.SubElement(box_geom_c, "size").text = size_str
                surface = ET.SubElement(collision, "surface")
                friction = ET.SubElement(surface, "friction")
                ode = ET.SubElement(friction, "ode")
                ET.SubElement(ode, "mu").text = "1.0"
                ET.SubElement(ode, "mu2").text = "1.0"
        
        # Format XML to be human-readable
        xml_string = ET.tostring(sdf, encoding='unicode')
        dom = minidom.parseString(xml_string)
        return dom.toprettyxml(indent="  ")

    def _validate_sdf(self, sdf_content: str) -> bool:
        """Basic XML validation for the generated SDF content."""
        try:
            ET.fromstring(sdf_content)
            logger.info("Generated SDF content is well-formed XML.")
            return True
        except ET.ParseError as e:
            logger.error(f"SDF Validation Error: Failed to parse XML. Details: {e}")
            return False

    def _print_summary(self, output_path: str):
        """Prints a summary of the generated world."""
        print("\n✓ World generated successfully!")
        print(f"  File: {output_path}")
        print(f"  Rooms: {len(self.rooms)}")
        print(f"  Total objects: {sum(1 for m in self.models if m.category != 'structure' and m.name != 'ground_plane')}")
        for room in self.rooms:
            obj_count = sum(1 for m in self.models if getattr(m, 'room', None) == room.name and m.category != "structure")
            print(f"    • {room.name} ({room.type}): {obj_count} objects")
        print(f"\nRun with: gazebo {output_path}")