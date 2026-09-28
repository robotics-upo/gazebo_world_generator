"""
Measure and check a room layout.

The LLM designs layouts; this module only states facts about them. Each check
returns violations with a concrete hint ("move east >= 0.30 m") that is fed
back to the LLM, so it can fix its own layout rather than having code
rearrange it.

Coordinates are room-local: origin at the room centre, +X east, +Y north,
yaw counter-clockwise in radians. An item is a dict with ``id``, ``type``,
``x``, ``y``, ``yaw``, ``dims`` (width, length, height), an optional ``on``
(id of the item it stands on) and ``front`` (its local front axis).
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# (min_x, max_x, min_y, max_y)
Box = Tuple[float, float, float, float]

GRID_RESOLUTION = 0.1
# Footprints are axis-aligned bounds of rotated boxes; ignore contact noise.
OVERLAP_TOLERANCE = 0.02
WALL_CONTACT = 0.15
FACING_TOLERANCE = math.radians(35)
FRONT_AXES = {"+x": 0.0, "+y": math.pi / 2, "-x": math.pi, "-y": -math.pi / 2}

ALWAYS_ON = ("all_placed", "inside_room", "no_overlap", "doorways_clear")
# Requirement catalogue shown to the LLM: check name -> (arguments, meaning).
CATALOGUE = {
    "path_between_doors": ("width", "a route at least `width` m wide connects every doorway "
                                    "(single door: the free floor is one connected region)"),
    "min_spacing": ("width", "every pair of objects, and every object and wall, is at least "
                             "`width` m apart; pairs you tied with near/faces and objects you put "
                             "against_wall are exempt"),
    "near": ("a, b, max", "footprints of `a` and `b` are at most `max` m apart"),
    "against_wall": ("id", "object `id` touches a wall (within 0.15 m)"),
    "faces": ("a, b", "the front of `a` points at `b` (within 35 degrees)"),
}


@dataclass
class Violation:
    check: str
    ids: List[str]
    message: str
    hint: str = ""
    always_on: bool = field(default=False)

    def __str__(self) -> str:
        return f"[{self.check}] {self.message}" + (f" -> {self.hint}" if self.hint else "")


# --- geometry ------------------------------------------------------------------------------------

def footprint(x: float, y: float, dims: Sequence[float], yaw: float) -> Box:
    """Axis-aligned bounds of a width x length box rotated by yaw about (x, y)."""
    half_w, half_l = dims[0] / 2, dims[1] / 2
    cos_yaw, sin_yaw = abs(math.cos(yaw)), abs(math.sin(yaw))
    extent_x = half_w * cos_yaw + half_l * sin_yaw
    extent_y = half_w * sin_yaw + half_l * cos_yaw
    return (x - extent_x, x + extent_x, y - extent_y, y + extent_y)


def item_box(item: Dict) -> Box:
    return footprint(item["x"], item["y"], item["dims"], item.get("yaw", 0.0))


def box_gap(a: Box, b: Box) -> float:
    """Shortest distance between two boxes (0 when they touch or overlap)."""
    dx = max(b[0] - a[1], a[0] - b[1], 0.0)
    dy = max(b[2] - a[3], a[2] - b[3], 0.0)
    return math.hypot(dx, dy)


def penetration(a: Box, b: Box) -> Tuple[float, float]:
    """Overlap depth along x and y (both positive when the boxes overlap)."""
    return (min(a[1], b[1]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[2], b[2]))


def inner_half_extents(room, wall_thickness: float) -> Tuple[float, float]:
    return (room.dimensions["width"] / 2 - wall_thickness / 2,
            room.dimensions["length"] / 2 - wall_thickness / 2)


def doorway_zones(room) -> List[Tuple[str, Box]]:
    """Clearance rectangle in front of each doorway, room-local."""
    zones = []
    half_w, half_l = room.dimensions["width"] / 2, room.dimensions["length"] / 2
    for door in getattr(room, "doorways", None) or []:
        dx = door["x"] - room.position["x"]
        dy = door["y"] - room.position["y"]
        half_door = door.get("width", 1.6) / 2
        depth = door.get("clearance_radius", 1.5)
        side = door.get("side")
        if side == "north":
            zones.append((side, (dx - half_door, dx + half_door, half_l - depth, half_l)))
        elif side == "south":
            zones.append((side, (dx - half_door, dx + half_door, -half_l, -half_l + depth)))
        elif side == "east":
            zones.append((side, (half_w - depth, half_w, dy - half_door, dy + half_door)))
        elif side == "west":
            zones.append((side, (-half_w, -half_w + depth, dy - half_door, dy + half_door)))
    return zones


def front_direction(item: Dict) -> float:
    """World heading of an item's front."""
    return item.get("yaw", 0.0) + FRONT_AXES.get(item.get("front", "+x"), 0.0)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _direction(dx: float, dy: float) -> str:
    return ("east" if dx > 0 else "west") if abs(dx) >= abs(dy) else ("north" if dy > 0 else "south")


# --- checks --------------------------------------------------------------------------------------

def check_layout(room, items: List[Dict], requirements: Iterable[Dict] = (),
                 expected_ids: Iterable[str] = (), fixed: Sequence[Dict] = (),
                 wall_thickness: float = 0.2) -> List[Violation]:
    """Check the always-on rules plus the LLM's requested requirements."""
    by_id = {item["id"]: item for item in items}
    half_x, half_y = inner_half_extents(room, wall_thickness)
    violations: List[Violation] = []

    missing = [item_id for item_id in expected_ids if item_id not in by_id]
    if missing:
        violations.append(Violation("all_placed", missing, f"not placed: {', '.join(missing)}",
                                    "give every listed id a pose", True))

    for item in items:
        box = item_box(item)
        push_x = max(-half_x - box[0], 0.0) - max(box[1] - half_x, 0.0)
        push_y = max(-half_y - box[2], 0.0) - max(box[3] - half_y, 0.0)
        if abs(push_x) > OVERLAP_TOLERANCE or abs(push_y) > OVERLAP_TOLERANCE:
            dx, dy = (push_x, 0.0) if abs(push_x) >= abs(push_y) else (0.0, push_y)
            violations.append(Violation(
                "inside_room", [item["id"]], f"{item['id']} crosses a wall",
                f"move {item['id']} {_direction(dx, dy)} >= {max(abs(push_x), abs(push_y)):.2f} m", True))

    solid = [item for item in items if not item.get("on")] + list(fixed)
    for index, a in enumerate(solid):
        for b in solid[index + 1:]:
            if a.get("fixed") and b.get("fixed"):
                continue
            over_x, over_y = penetration(item_box(a), item_box(b))
            if over_x > OVERLAP_TOLERANCE and over_y > OVERLAP_TOLERANCE:
                mover, other = (b, a) if a.get("fixed") else (a, b)
                if over_x <= over_y:
                    hint = (f"move {mover['id']} {_direction(mover['x'] - other['x'] or 1, 0)} "
                            f">= {over_x:.2f} m")
                else:
                    hint = (f"move {mover['id']} {_direction(0, mover['y'] - other['y'] or 1)} "
                            f">= {over_y:.2f} m")
                violations.append(Violation("no_overlap", [a["id"], b["id"]],
                                            f"{a['id']} overlaps {b['id']}", hint, True))

    for item in items:
        support = by_id.get(item.get("on")) if item.get("on") else None
        if item.get("on") and support is None:
            violations.append(Violation("no_overlap", [item["id"]],
                                        f"{item['id']} stands on unknown id {item['on']}",
                                        "set `on` to a placed id or remove it", True))
        elif support is not None:
            inner, outer = item_box(item), item_box(support)
            if inner[0] < outer[0] - 0.05 or inner[1] > outer[1] + 0.05 or \
                    inner[2] < outer[2] - 0.05 or inner[3] > outer[3] + 0.05:
                violations.append(Violation(
                    "no_overlap", [item["id"], support["id"]],
                    f"{item['id']} overhangs {support['id']}",
                    f"move {item['id']} to ({support['x']:.2f}, {support['y']:.2f})", True))

    for side, zone in doorway_zones(room):
        for item in items:
            over_x, over_y = penetration(item_box(item), zone)
            if over_x > OVERLAP_TOLERANCE and over_y > OVERLAP_TOLERANCE:
                away = {"north": "south", "south": "north", "east": "west", "west": "east"}[side]
                depth = over_y if side in ("north", "south") else over_x
                violations.append(Violation(
                    "doorways_clear", [item["id"]], f"{item['id']} blocks the {side} doorway",
                    f"move {item['id']} {away} >= {depth:.2f} m or sideways out of the doorway", True))

    requirements = [r for r in requirements]
    for requirement in requirements:
        violations.extend(_check_requirement(room, requirement, items, by_id, list(fixed),
                                             half_x, half_y, requirements))
    return violations


def _tied(requirements) -> Tuple[set, set]:
    """Pairs the LLM wants close (near/faces) and objects it wants against a wall."""
    pairs, walled = set(), set()
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        if requirement.get("check") in ("near", "faces"):
            pairs.add(frozenset((requirement.get("a"), requirement.get("b"))))
        elif requirement.get("check") == "against_wall":
            walled.add(requirement.get("id"))
    return pairs, walled


def _check_requirement(room, requirement, items, by_id, fixed, half_x, half_y,
                       requirements=()) -> List[Violation]:
    name = requirement.get("check") if isinstance(requirement, dict) else None
    if name not in CATALOGUE:
        return [Violation("requirement", [], f"unknown requirement {requirement!r}",
                          f"use one of: {', '.join(CATALOGUE)}")]
    try:
        if name == "path_between_doors":
            width = float(requirement.get("width", 0.8))
            obstacles = [item_box(item) for item in items + fixed if not item.get("on")]
            blocked_at = route_blockage(room, obstacles, width, half_x, half_y)
            if blocked_at is None:
                return []
            return [Violation(name, [], f"no {width:.2f} m route connects the doorways",
                              f"open a gap of >= {width:.2f} m near ({blocked_at[0]:.1f}, {blocked_at[1]:.1f})")]

        if name == "min_spacing":
            width = float(requirement["width"])
            tied_pairs, walled = _tied(requirements)
            found = []
            solid = [item for item in items if not item.get("on")]
            for index, a in enumerate(solid):
                box_a = item_box(a)
                wall_gap = min(box_a[0] + half_x, half_x - box_a[1], box_a[2] + half_y, half_y - box_a[3])
                if a["id"] not in walled and wall_gap < width - OVERLAP_TOLERANCE:
                    found.append(Violation(name, [a["id"]], f"{a['id']} is {max(wall_gap, 0):.2f} m from a wall",
                                           f"move {a['id']} >= {width - max(wall_gap, 0):.2f} m away from the wall"))
                for b in solid[index + 1:]:
                    if frozenset((a["id"], b["id"])) in tied_pairs:
                        continue
                    gap = box_gap(box_a, item_box(b))
                    if gap < width - OVERLAP_TOLERANCE:
                        found.append(Violation(name, [a["id"], b["id"]],
                                               f"{a['id']} and {b['id']} are {gap:.2f} m apart",
                                               f"separate them by another {width - gap:.2f} m"))
            return found

        if name == "near":
            a, b, limit = by_id[requirement["a"]], by_id[requirement["b"]], float(requirement["max"])
            gap = box_gap(item_box(a), item_box(b))
            if gap <= limit + OVERLAP_TOLERANCE:
                return []
            return [Violation(name, [a["id"], b["id"]], f"{a['id']} is {gap:.2f} m from {b['id']}",
                              f"move {a['id']} {_direction(b['x'] - a['x'], b['y'] - a['y'])} "
                              f"by {gap - limit:.2f} m")]

        if name == "against_wall":
            item = by_id[requirement["id"]]
            box = item_box(item)
            gaps = {"west": box[0] + half_x, "east": half_x - box[1],
                    "south": box[2] + half_y, "north": half_y - box[3]}
            wall, gap = min(gaps.items(), key=lambda entry: entry[1])
            if gap <= WALL_CONTACT:
                return []
            return [Violation(name, [item["id"]], f"{item['id']} is {gap:.2f} m from the nearest wall",
                              f"move {item['id']} {wall} by {gap:.2f} m")]

        if name == "faces":
            a, b = by_id[requirement["a"]], by_id[requirement["b"]]
            wanted = math.atan2(b["y"] - a["y"], b["x"] - a["x"])
            error = _wrap(wanted - front_direction(a))
            if abs(error) <= FACING_TOLERANCE:
                return []
            target_yaw = _wrap(a.get("yaw", 0.0) + error)
            return [Violation(name, [a["id"], b["id"]], f"{a['id']} does not face {b['id']}",
                              f"set yaw of {a['id']} to {target_yaw:.2f}")]
    except (KeyError, TypeError, ValueError) as exc:
        return [Violation("requirement", [], f"malformed {name} requirement {requirement!r} ({exc})",
                          f"arguments: {CATALOGUE[name][0]}; ids must be listed objects")]
    return []


def route_blockage(room, obstacles: List[Box], width: float,
                   half_x: float, half_y: float) -> Optional[Tuple[float, float]]:
    """None if a disc of diameter `width` can travel between all doorways.

    Otherwise, the reachable point closest to the first unreachable doorway,
    which is where the route is blocked. With fewer than two doorways the free
    floor must be one connected region.
    """
    # Cells only sample the free space; allow one cell of discretisation slack.
    radius = max(width / 2 - GRID_RESOLUTION, GRID_RESOLUTION / 2)
    columns = max(1, int(2 * half_x / GRID_RESOLUTION))
    rows = max(1, int(2 * half_y / GRID_RESOLUTION))

    def centre(cell):
        return (-half_x + (cell[0] + 0.5) * GRID_RESOLUTION, -half_y + (cell[1] + 0.5) * GRID_RESOLUTION)

    def free(cell) -> bool:
        x, y = centre(cell)
        if abs(x) > half_x - radius or abs(y) > half_y - radius:
            return False
        return all(max(b[0] - x, x - b[1], 0.0) ** 2 + max(b[2] - y, y - b[3], 0.0) ** 2 >= radius ** 2
                   for b in obstacles)

    free_cells = {(c, r) for c in range(columns) for r in range(rows) if free((c, r))}
    if not free_cells:
        return (0.0, 0.0)

    def nearest(point, cells):
        return min(cells, key=lambda cell: (centre(cell)[0] - point[0]) ** 2 + (centre(cell)[1] - point[1]) ** 2)

    doors = [((zone[0] + zone[1]) / 2, (zone[2] + zone[3]) / 2) for _, zone in doorway_zones(room)]
    start = nearest(doors[0], free_cells) if doors else next(iter(sorted(free_cells)))
    seen = {start}
    queue = deque([start])
    while queue:
        column, row = queue.popleft()
        for neighbour in ((column + 1, row), (column - 1, row), (column, row + 1), (column, row - 1)):
            if neighbour in free_cells and neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)

    for door in doors[1:]:
        target = nearest(door, free_cells)
        if target not in seen:
            return centre(nearest(centre(target), seen))
    if len(doors) < 2 and len(seen) != len(free_cells):
        return centre(nearest(centre(next(iter(sorted(free_cells - seen)))), seen))
    return None


def free_point(room, items: Sequence[Dict], clearance: float = 0.5,
               wall_thickness: float = 0.2) -> Optional[Tuple[float, float]]:
    """Room-local point nearest the centre with `clearance` to every object and wall."""
    half_x, half_y = inner_half_extents(room, wall_thickness)
    boxes = [item_box(item) for item in items if not item.get("on")]
    step = 0.25
    candidates = [(i * step, j * step)
                  for i in range(-int(half_x / step), int(half_x / step) + 1)
                  for j in range(-int(half_y / step), int(half_y / step) + 1)]
    for x, y in sorted(candidates, key=lambda point: point[0] ** 2 + point[1] ** 2):
        if abs(x) > half_x - clearance or abs(y) > half_y - clearance:
            continue
        if all(max(b[0] - x, x - b[1], 0.0) ** 2 + max(b[2] - y, y - b[3], 0.0) ** 2 >= clearance ** 2
               for b in boxes):
            return (x, y)
    return None


def score(violations: List[Violation]) -> int:
    """Lower is better; the always-on rules weigh most."""
    return sum(3 if violation.always_on else 1 for violation in violations)
