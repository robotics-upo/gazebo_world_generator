"""Shared vocabulary that maps what users say to things the pipeline can build.

Users describe intent ("some obstacles", "office-like rooms") rather than
Gazebo model names. These tables turn category words into concrete, resolvable
object types, give function-only rooms a minimal furniture set, and define
which generic objects may fall back to a primitive box.
"""

from typing import Dict, List, Optional, Tuple

# Category words -> concrete types to try, in order of preference.
ABSTRACT_TYPE_SUBSTITUTES: Dict[str, List[str]] = {
    "obstacle": ["cardboard_box", "construction_cone", "construction_barrel",
                 "cinder_block", "jersey_barrier"],
    "barrier": ["jersey_barrier", "construction_barrel", "construction_cone"],
    "clutter": ["cardboard_box", "box"],
    "block": ["cinder_block", "cardboard_box", "box"],
    "crate": ["cardboard_box", "wooden_crate", "box"],
    "box": ["cardboard_box"],
    "cone": ["construction_cone", "traffic_cone"],
    "traffic_cone": ["construction_cone"],
    "wall_segment": ["jersey_barrier"],
}

# Generic objects that may become an inline box when no model resolves.
# Specific furniture never does: a grey box is not a sofa.
PRIMITIVE_SIZES: Dict[str, Tuple[float, float, float]] = {
    "obstacle": (0.5, 0.5, 0.5),
    "block": (0.5, 0.5, 0.5),
    "box": (0.5, 0.5, 0.5),
    "crate": (0.6, 0.4, 0.4),
    "cardboard_box": (0.5, 0.4, 0.4),
    "barrier": (1.2, 0.3, 0.8),
    "jersey_barrier": (1.8, 0.6, 1.0),
    "wall_segment": (1.5, 0.2, 1.0),
}
PRIMITIVE_URI = "primitive://box"

# Minimal furniture for rooms described by function but with no objects listed.
IMPLIED_OBJECTS: Dict[str, List[Dict]] = {
    "office": [
        {"type": "desk", "count": 1, "semantic_context": "Implied workstation"},
        {"type": "chair", "count": 1, "semantic_context": "Implied desk chair"},
    ],
    "meeting_room": [
        {"type": "table", "count": 1, "semantic_context": "Implied meeting table"},
        {"type": "chair", "count": 4, "semantic_context": "Implied meeting chairs"},
    ],
    "lab": [
        {"type": "table", "count": 1, "semantic_context": "Implied lab bench"},
    ],
}


def normalize_type(name: str) -> str:
    return "_".join(str(name or "").strip().lower().replace("-", " ").split())


def singular(name: str) -> str:
    if name.endswith("ies") and len(name) > 4:
        return name[:-3] + "y"
    if name.endswith(("ches", "shes", "xes")):
        return name[:-2]
    if name.endswith("s") and not name.endswith("ss"):
        return name[:-1]
    return name


def substitutes_for(object_type: str) -> List[str]:
    key = singular(normalize_type(object_type))
    return list(ABSTRACT_TYPE_SUBSTITUTES.get(key, []))


def primitive_size(object_type: str) -> Optional[Tuple[float, float, float]]:
    return PRIMITIVE_SIZES.get(singular(normalize_type(object_type)))
