"""
SDF utilities for robustly parsing and manipulating SDF files.
"""

import xml.etree.ElementTree as ET
import logging
from typing import Optional, Dict, Tuple
from pathlib import Path
logger = logging.getLogger(__name__)


def parse_sdf_file(sdf_path: str) -> Optional[ET.Element]:
    """Parse an SDF file and return the root element."""
    try:
        tree = ET.parse(sdf_path)
        return tree.getroot()
    except ET.ParseError as e:
        logger.error(f"XML parsing error in {sdf_path}: {e}")
        return None
    except FileNotFoundError:
        logger.error(f"SDF file not found at {sdf_path}")
        return None


def get_model_dimensions(sdf_root: ET.Element) -> Tuple[float, float, float]:
    """
    Extract model dimensions (width, length, height) from an SDF root element.
    """
    # Search for any geometry tag within either collision or visual tags
    for geometry_parent_tag in ['collision', 'visual']:
        for geometry in sdf_root.iterfind(f".//{geometry_parent_tag}/geometry"):
            # Box: <size>width length height</size>
            if (box := geometry.find('box')) is not None and (size_elem := box.find('size')) is not None:
                try:
                    dims = [float(d) for d in size_elem.text.split()]
                    return dims[0], dims[1], dims[2]
                except (ValueError, IndexError):
                    continue
            
            # Cylinder: <radius>, <length> (height)
            if (cyl := geometry.find('cylinder')) is not None:
                try:
                    radius = float(cyl.find('radius').text)
                    length = float(cyl.find('length').text)
                    return radius * 2, radius * 2, length # width, length, height
                except (AttributeError, ValueError):
                    continue

    logger.warning("Could not find valid geometry to determine dimensions. Falling back.")
    return 1.0, 1.0, 1.0 # Default fallback dimensions


def extract_model_info(sdf_path: str) -> Optional[Dict]:
    """
    Extract comprehensive information about a model from its SDF in a single pass.
    """
    root = parse_sdf_file(sdf_path)
    if root is None:
        return None

    info = {
        'name': 'unknown',
        'dimensions': (1.0, 1.0, 1.0),
        'version': root.get('version', '1.4'),
        'is_static': False,
        'has_collision': False,
        'has_visual': False,
    }

    model_elem = root.find('.//model')
    if model_elem is not None:
        info['name'] = model_elem.get('name', 'unknown')
        info['is_static'] = model_elem.findtext('static', 'false').lower() == 'true'
        info['has_collision'] = model_elem.find('.//collision') is not None
        info['has_visual'] = model_elem.find('.//visual') is not None
        info['dimensions'] = get_model_dimensions(model_elem)
    
    return info

def extract_model_metadata(model_path: Path) -> Dict:
    """Extracts rich metadata from a model's .sdf and .config files."""
    metadata = {"description": "A Gazebo model.", "tags": []}
    
    # Try to parse model.config first for description and author
    try:
        config_tree = ET.parse(model_path / "model.config")
        description_elem = config_tree.find('.//description')
        if description_elem is not None and description_elem.text:
            metadata["description"] = description_elem.text.strip()
    except Exception:
        pass # Ignore if config doesn't exist or is malformed

    return metadata


def create_simple_box_sdf(name: str, size: Tuple[float, float, float],
                          pose: Dict[str, float], static: bool = True) -> str:
    """Create a simple box SDF model as a formatted string."""
    pose_str = f"{pose['x']} {pose['y']} {pose['z']} {pose['roll']} {pose['pitch']} {pose['yaw']}"
    size_str = f"{size[0]} {size[1]} {size[2]}"
    
    return f"""<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="{name}">
    <static>{str(static).lower()}</static>
    <pose>{pose_str}</pose>
    <link name="link">
      <visual name="visual">
        <geometry><box><size>{size_str}</size></box></geometry>
        <material><script><name>Gazebo/Grey</name></script></material>
      </visual>
      <collision name="collision">
        <geometry><box><size>{size_str}</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>"""