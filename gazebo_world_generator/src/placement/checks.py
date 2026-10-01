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
FACING_REACH = 4.0  # metres a front may point across to count as facing
FRONT_AXES = {"+x": 0.0, "+y": math.pi / 2, "-x": math.pi, "-y": -math.pi / 2}

IN_FRONT_MAX_GAP = 0.5  # metres; in_front_of means usable, e.g. sitting at a desk
ALWAYS_ON = ("all_placed", "inside_room", "no_overlap", "doorways_clear")
# Requirement catalogue shown to the LLM: check name -> (arguments, meaning).
CATALOGUE = {
    "path_between_doors": ("width", "a route at least `width` m wide connects every doorway "
                                    "(single door: the free floor is one connected region)"),
    "min_spacing": ("width", "every pair of objects, and every object and wall, is at least "
                             "`width` m apart; pairs you tied with near/faces/in_front_of and objects you put "
                             "against_wall are exempt"),
    "near": ("a, b, max", "footprints of `a` and `b` are at most `max` m apart"),
    "against_wall": ("id", "object `id` touches a wall (within 0.15 m); an object with a front "
                           "stands with its back on the nearest wall, front to the room"),
    "faces": ("a, b", "the front of `a` points at `b` (a line straight out of `a`'s front "
                      "hits `b`'s footprint)"),
    "in_front_of": ("a, b, max_gap", "`a` stands on `b`'s front side (beyond its front edge, within "
                                     "its width) at most `max_gap` m (<= 0.5) away, facing `b` if `a` has a "
                                     "front, e.g. a chair at a desk; "
                                     "for `b` without a front (a table) any side counts, so seats "
                                     "can go around it"),
}


@dataclass
class Violation:
    check: str
    ids: List[str]
    message: str
    hint: str = ""
    always_on: bool = field(default=False)
    # Machine-applicable fix: {item id: {"x": .., "y": .., "yaw": ..}}
    fix: Dict[str, Dict[str, float]] = field(default_factory=dict)

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


def _ray_hits(x: float, y: float, heading: float, target: Dict, reach: float) -> bool:
    """Does a ray from (x, y) along heading cross target's (rotated) footprint within reach?"""
    yaw = target.get("yaw", 0.0)
    cos_t, sin_t = math.cos(-yaw), math.sin(-yaw)
    ox, oy = x - target["x"], y - target["y"]
    ox, oy = ox * cos_t - oy * sin_t, ox * sin_t + oy * cos_t
    dx, dy = math.cos(heading - yaw), math.sin(heading - yaw)
    t_min, t_max = 0.0, reach
    for origin, direction, half in ((ox, dx, target["dims"][0] / 2), (oy, dy, target["dims"][1] / 2)):
        if abs(direction) < 1e-9:
            if abs(origin) > half:
                return False
            continue
        t1, t2 = (-half - origin) / direction, (half - origin) / direction
        t_min, t_max = max(t_min, min(t1, t2)), min(t_max, max(t1, t2))
        if t_min > t_max:
            return False
    return True


def front_extents(item: Dict) -> Tuple[float, float]:
    """(half depth along the front axis, half width across it) in the item's own frame."""
    width, length = item["dims"][0], item["dims"][1]
    return (width / 2, length / 2) if item.get("front", "+x") in ("+x", "-x") else (length / 2, width / 2)


def usable_side(b: Dict, a: Dict) -> Tuple[float, float, float]:
    """(world heading, half depth, half width) of the side of ``b`` that ``a`` uses.

    ``b``'s front when it has one; for an object without a front (a table) the
    side ``a`` is nearest, so seats may go around it.
    """
    if b.get("front_known") is not False:
        return (front_direction(b),) + front_extents(b)
    yaw = b.get("yaw", 0.0)
    half_w, half_l = b["dims"][0] / 2, b["dims"][1] / 2
    dx, dy = a["x"] - b["x"], a["y"] - b["y"]
    local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
    local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
    if abs(local_x) - half_w >= abs(local_y) - half_l:
        return (yaw + (0.0 if local_x >= 0 else math.pi), half_w, half_l)
    return (yaw + (math.pi / 2 if local_y >= 0 else -math.pi / 2), half_l, half_w)


def relations(room, items: Sequence[Dict], wall_thickness: float = 0.2) -> List[str]:
    """How each object relates to its nearest neighbour and wall, from declared fronts.

    Read-only facts for the LLM: e.g. "chair_0 (front -y): nearest desk_0 0.05 m
    north; its front points AWAY from desk_0". Makes orientation mistakes visible
    even for requirements the LLM did not ask to check.
    """
    half_x, half_y = inner_half_extents(room, wall_thickness)
    lines = []
    for item in items:
        heading = front_direction(item)
        parts = [f"{item['id']} (front {item.get('front', '+x')}, facing {_compass(heading)}, "
                 f"{'on ' + item['on'] if item.get('on') else 'on the floor'})"]
        half_front, half_side = front_extents(item)
        if half_side * 1.3 < half_front:
            long_axes = "+x/-x" if item.get("front", "+x") in ("+y", "-y") else "+y/-y"
            parts.append(f"its declared front is a SHORT side ({2 * half_side:.2f} m wide; "
                         f"the long sides are {long_axes})")
        others = [other for other in items if other is not item]
        if others:
            nearest = min(others, key=lambda other: box_gap(item_box(item), item_box(other)))
            gap = box_gap(item_box(item), item_box(nearest))
            bearing = math.atan2(nearest["y"] - item["y"], nearest["x"] - item["x"])
            relative = _wrap(bearing - heading)
            if abs(relative) <= math.radians(35):
                aim = f"its front points AT {nearest['id']}"
            elif abs(relative) >= math.radians(145):
                aim = f"its front points AWAY from {nearest['id']} (its back is toward it)"
            else:
                aim = f"{nearest['id']} is to its {'left' if relative > 0 else 'right'}"
            parts.append(f"nearest {nearest['id']} {gap:.2f} m to the {_compass(bearing)}; {aim}")
        box = item_box(item)
        walls = {"west": box[0] + half_x, "east": half_x - box[1], "south": box[2] + half_y, "north": half_y - box[3]}
        wall, gap = min(walls.items(), key=lambda entry: entry[1])
        if gap <= 0.3:
            facing_wall = _compass(heading) == wall
            parts.append(f"against the {wall} wall" + (" and FACING it" if facing_wall else ""))
        lines.append("; ".join(parts))
    return lines


def space_use(room, items: Sequence[Dict], wall_thickness: float = 0.2) -> List[str]:
    """How the layout uses the floor: the span of the objects and the empty margins.

    Read-only facts for the LLM, so it can judge whether the arrangement fits the
    request (a warehouse filling the floor with rows, scattered clutter, a course
    using the hall's length) instead of everything packed into one corner.
    """
    floor = [item_box(item) for item in items if not item.get("on")]
    if not floor:
        return []
    half_x, half_y = inner_half_extents(room, wall_thickness)
    west, east = min(box[0] for box in floor), max(box[1] for box in floor)
    south, north = min(box[2] for box in floor), max(box[3] for box in floor)
    width, length = 2 * half_x, 2 * half_y
    covered = sum((box[1] - box[0]) * (box[3] - box[2]) for box in floor) / (width * length)
    lines = [f"objects span x [{west:.1f}, {east:.1f}] ({(east - west) / width:.0%} of the room's width) and "
             f"y [{south:.1f}, {north:.1f}] ({(north - south) / length:.0%} of its length); "
             f"they cover {covered:.0%} of the floor"]
    margins = {"west": (west + half_x, width, f"x < {west:.1f}"), "east": (half_x - east, width, f"x > {east:.1f}"),
               "south": (south + half_y, length, f"y < {south:.1f}"), "north": (half_y - north, length, f"y > {north:.1f}")}
    for side, (gap, size, where) in margins.items():
        if gap > 0.25 * size and gap > 1.0:
            lines.append(f"the {side} {gap / size:.0%} of the room ({where}, {gap:.1f} m deep) is empty")
    return lines


def _shifted(item: Dict, direction: Tuple[int, int], distance: float, half_x: float, half_y: float) -> Dict:
    """Pose of ``item`` moved ``distance`` along an axis direction, kept inside the room."""
    box = item_box(item)
    x = item["x"] + direction[0] * distance
    y = item["y"] + direction[1] * distance
    x = min(max(x, item["x"] - (box[0] + half_x)), item["x"] + (half_x - box[1]))
    y = min(max(y, item["y"] - (box[2] + half_y)), item["y"] + (half_y - box[3]))
    return {"x": round(x, 3), "y": round(y, 3)}


def _compass(angle: float) -> str:
    # % 4: for an angle a hair below a multiple of 2 pi, the float modulo returns 2 pi itself.
    return ("east", "north", "west", "south")[int(((angle + math.pi / 4) % (2 * math.pi)) // (math.pi / 2)) % 4]


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
    # A requirement may name an object by its type ("chair") when there is only one of it.
    types = [item.get("type") for item in items]
    named = {item["type"]: item for item in items if item.get("type") and types.count(item["type"]) == 1}
    named.update(by_id)
    for requirement in requirements:
        violations.extend(_check_requirement(room, requirement, items, named, list(fixed),
                                             half_x, half_y, requirements))
    return violations


def tied(requirements) -> Tuple[set, set]:
    """Pairs the LLM wants close (near/faces/in_front_of) and objects it wants against a wall."""
    pairs, walled = set(), set()
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        if requirement.get("check") in ("near", "faces", "in_front_of"):
            pairs.add(frozenset((requirement.get("a"), requirement.get("b"))))
        elif requirement.get("check") == "against_wall":
            walled.add(requirement.get("id"))
    return pairs, walled


def _check_requirement(room, requirement, items, by_id, fixed, half_x, half_y,
                       requirements=()) -> List[Violation]:
    name = requirement.get("check") if isinstance(requirement, dict) else None
    if name in ALWAYS_ON or name in ("all_ids_placed", "no_overlaps", "inside", "doorway_clear"):
        return []  # restating a rule that is always checked is harmless
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
            tied_pairs, walled = tied(requirements)
            found = []
            solid = [item for item in items if not item.get("on")]
            for index, a in enumerate(solid):
                box_a = item_box(a)
                walls = {(1, 0): box_a[0] + half_x, (-1, 0): half_x - box_a[1],
                         (0, 1): box_a[2] + half_y, (0, -1): half_y - box_a[3]}
                away, wall_gap = min(walls.items(), key=lambda entry: entry[1])
                if a["id"] not in walled and wall_gap < width - OVERLAP_TOLERANCE:
                    target = _shifted(a, away, width - max(wall_gap, 0) + 0.01, half_x, half_y)
                    found.append(Violation(name, [a["id"]], f"{a['id']} is {max(wall_gap, 0):.2f} m from a wall",
                                           f"move {a['id']} to ({target['x']:.2f}, {target['y']:.2f})",
                                           fix={a["id"]: target}))
                for b in solid[index + 1:]:
                    if frozenset((a["id"], b["id"])) in tied_pairs:
                        continue
                    box_b = item_box(b)
                    gap = box_gap(box_a, box_b)
                    if gap < width - OVERLAP_TOLERANCE:
                        # Push a away from b along the axis they are already apart on.
                        apart_x = max(box_b[0] - box_a[1], box_a[0] - box_b[1])
                        apart_y = max(box_b[2] - box_a[3], box_a[2] - box_b[3])
                        if apart_x >= apart_y:
                            away = (1 if a["x"] >= b["x"] else -1, 0)
                            push = width - apart_x
                        else:
                            away = (0, 1 if a["y"] >= b["y"] else -1)
                            push = width - apart_y
                        target = _shifted(a, away, push + 0.01, half_x, half_y)
                        found.append(Violation(name, [a["id"], b["id"]],
                                               f"{a['id']} and {b['id']} are {gap:.2f} m apart",
                                               f"move {a['id']} to ({target['x']:.2f}, {target['y']:.2f}) "
                                               f"(another {width - gap:.2f} m apart)",
                                               fix={a["id"]: target}))
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
            if item.get("front_known") is False or "front_known" not in item:
                if gap <= WALL_CONTACT:
                    return []
                return [Violation(name, [item["id"]], f"{item['id']} is {gap:.2f} m from the nearest wall",
                                  f"move {item['id']} {wall} by {gap:.2f} m")]
            # With a front, "against the wall" means its back on the wall, front to the room
            # (a shelf, not a shelf sticking out of the wall end-on).
            centre_gaps = {"west": item["x"] + half_x, "east": half_x - item["x"],
                           "south": item["y"] + half_y, "north": half_y - item["y"]}
            wall = min(centre_gaps, key=centre_gaps.get)
            back = _compass(front_direction(item) + math.pi)
            if back == wall and gaps[wall] <= WALL_CONTACT:
                return []
            outward = {"east": 0.0, "north": math.pi / 2, "west": math.pi, "south": -math.pi / 2}[wall]
            yaw = _wrap(outward + math.pi - FRONT_AXES.get(item.get("front", "+x"), 0.0))
            half_depth = front_extents(item)[0]
            x, y = item["x"], item["y"]
            if wall in ("east", "west"):
                x = (half_x - half_depth) * (1 if wall == "east" else -1)
            else:
                y = (half_y - half_depth) * (1 if wall == "north" else -1)
            problem = (f"{item['id']} is {gaps[wall]:.2f} m from the {wall} wall" if back == wall
                       else f"{item['id']}'s back is not to the {wall} wall (its back faces {back})")
            return [Violation(name, [item["id"]], problem,
                              f"put {item['id']} at ({x:.2f}, {y:.2f}) with yaw {yaw:.2f}, back on the {wall} wall",
                              fix={item["id"]: {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 4)}})]

        if name == "in_front_of":
            a, b = by_id[requirement["a"]], by_id[requirement["b"]]
            if a.get("on") == b["id"]:
                return []  # a stands on b (a monitor on a desk): it is not beside it
            # "At" b means close enough to use it, however generous the gap the LLM asked for.
            limit = min(float(requirement.get("max_gap", 0.4)), IN_FRONT_MAX_GAP)
            heading, half_front, half_side = usable_side(b, a)
            depth = min(a["dims"][0], a["dims"][1]) / 2
            dx, dy = a["x"] - b["x"], a["y"] - b["y"]
            along = dx * math.cos(heading) + dy * math.sin(heading)
            side = -dx * math.sin(heading) + dy * math.cos(heading)
            gap = along - half_front - depth
            # Standing there facing b: the opposite of b's heading, through a's own front axis.
            facing_yaw = _wrap(heading + math.pi - FRONT_AXES.get(a.get("front", "+x"), 0.0))
            if abs(side) <= half_side and -OVERLAP_TOLERANCE - depth <= gap <= limit:
                # At b's front to use it means facing it too (a chair turned sideways at a desk
                # is not "at" it); only checked when a's front is known.
                if a.get("front_known") is not True or _ray_hits(a["x"], a["y"], front_direction(a), b,
                                                                   FACING_REACH):
                    return []
                return [Violation(name, [a["id"], b["id"]],
                                  f"{a['id']} is at {b['id']}'s front but does not face it",
                                  f"set yaw of {a['id']} to {facing_yaw:.2f}",
                                  fix={a["id"]: {"yaw": round(facing_yaw, 4)}})]
            reach = half_front + depth + min(0.1, limit)
            # Keep a's place along that side (seats spread along a table edge stay apart);
            # only pull it to the edge, within the side's width.
            room_along = max(half_side - max(a["dims"][0], a["dims"][1]) / 2, 0.0)
            offset = max(-room_along, min(room_along, side))
            target = (b["x"] + reach * math.cos(heading) - offset * math.sin(heading),
                      b["y"] + reach * math.sin(heading) + offset * math.cos(heading))
            where = ("behind or beside it" if along <= half_front else
                     "off to the side of its front" if abs(side) > half_side else "too far away")
            fix = {"x": round(target[0], 3), "y": round(target[1], 3)}
            if a.get("front_known") is not False:
                fix["yaw"] = round(facing_yaw, 4)
            return [Violation(name, [a["id"], b["id"]],
                              f"{a['id']} is not at {b['id']}'s front ({where})",
                              f"move {a['id']} to ({target[0]:.2f}, {target[1]:.2f}), in front of {b['id']}",
                              fix={a["id"]: fix})]

        if name == "faces":
            a, b = by_id[requirement["a"]], by_id[requirement["b"]]
            if a.get("front_known") is False:
                return []  # a has no front to point
            if _ray_hits(a["x"], a["y"], front_direction(a), b, FACING_REACH):
                return []
            # Squarely at b's usable side when a stands there (a chair at a desk edge, not
            # necessarily its centre; a seat at any side of a table); otherwise toward b's centre.
            wanted = math.atan2(b["y"] - a["y"], b["x"] - a["x"])
            b_heading, half_front, _ = usable_side(b, a)
            beyond = 0.0 if b.get("front_known") is not False else half_front
            square = b_heading + math.pi
            if ((a["x"] - b["x"]) * math.cos(b_heading) + (a["y"] - b["y"]) * math.sin(b_heading) > beyond
                    and _ray_hits(a["x"], a["y"], square, b, FACING_REACH)):
                wanted = square  # level with that side, so the square heading reaches b
            target_yaw = _wrap(a.get("yaw", 0.0) + _wrap(wanted - front_direction(a)))
            return [Violation(name, [a["id"], b["id"]], f"{a['id']} does not face {b['id']}",
                              f"set yaw of {a['id']} to {target_yaw:.2f}",
                              fix={a["id"]: {"yaw": round(target_yaw, 4)}})]
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
