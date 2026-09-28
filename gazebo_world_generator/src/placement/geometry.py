"""
Relative positions used by world refinement ("put a chair in front of desk_0").

Generation lets the LLM design layouts (see ``designer``); refinement still
places single objects relative to existing ones with these rules.
"""

import logging
import math
from typing import Tuple

logger = logging.getLogger(__name__)

# Chair offsets for desk models whose geometry is known.
MODEL_ORIENTATION_CORRECTIONS = {
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
        'chair_yaw_offset': -math.pi / 2  # Face East (yaw=-pi/2), perpendicular to desk
    },
    'default': {
        'placement_direction': 'right_side',
        'chair_offset_x': 0.75,
        'chair_offset_y': 0.0,
        'chair_yaw_offset': -math.pi / 2
    }
}


def calculate_relative_position(primary_x: float, primary_y: float, primary_yaw: float,
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
    if arrangement == 'in_front' and primary_model_name and primary_model_name in MODEL_ORIENTATION_CORRECTIONS:
        correction = MODEL_ORIENTATION_CORRECTIONS[primary_model_name]
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
