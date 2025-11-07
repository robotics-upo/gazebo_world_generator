#!/usr/bin/env python3
"""
SDF Parser for extracting model dimensions and metadata
Extracts collision geometry dimensions from Gazebo SDF model files.
"""

import xml.etree.ElementTree as ET
import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class SDFDimensionExtractor:
    """Extracts dimensions and metadata from Gazebo SDF model files."""

    def __init__(self):
        # Standard mesh dimensions for common models (when SDF uses meshes without explicit sizes)
        self.mesh_dimensions = {
            'euro_pallet': (1.2, 0.8, 0.144),  
            'pallet': (1.2, 0.8, 0.144),
            'bookshelf': (0.8, 0.3, 1.8),
            'desk': (1.2, 0.6, 0.55),  
            'chair': (0.6, 0.6, 0.9),
            'table': (1.5, 0.8, 0.75),
        }

    def extract_model_bounding_box(self, model_path: str) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Extract complete bounding box information from SDF file.

        Args:
            model_path: Path to the model directory or SDF file

        Returns:
            Tuple of ((width, length, height), (offset_x, offset_y, offset_z)) or None
            - Dimensions: size of the bounding box
            - Offset: For furniture with surfaces, offset_z is the top surface height
        """
        try:
            sdf_path = self._find_sdf_file(model_path)
            if not sdf_path:
                logger.warning(f"No SDF file found in {model_path}")
                return None

            tree = ET.parse(sdf_path)
            root = tree.getroot()

            # Try to extract surface-specific height for furniture first
            surface_result = self._extract_surface_height(root)
            if surface_result:
                logger.info(f"Extracted surface height from {sdf_path}: dims={surface_result[0]}, surface_z={surface_result[1][2]:.3f}")
                return surface_result

            # Fallback to complete bounding box with offsets
            result = self._extract_complete_bounding_box_with_offset(root)
            if result:
                logger.info(f"Extracted bounding box from {sdf_path}: dims={result[0]}, offset={result[1]}")
                return result

            return None

        except Exception as e:
            logger.error(f"Error extracting bounding box from {model_path}: {e}")
            return None

    def extract_model_dimensions(self, model_path: str) -> Optional[Tuple[float, float, float]]:
        """
        Extract model dimensions from SDF file (backward compatibility).

        Args:
            model_path: Path to the model directory or SDF file

        Returns:
            Tuple of (width, length, height) in meters, or None if extraction fails
        """
        try:
            sdf_path = self._find_sdf_file(model_path)
            if not sdf_path:
                logger.warning(f"No SDF file found in {model_path}")
                return None

            tree = ET.parse(sdf_path)
            root = tree.getroot()

            # Extract complete bounding box with pose transformations
            bbox = self._extract_complete_bounding_box(root)
            if bbox:
                logger.info(f"Extracted complete bounding box from {sdf_path}: {bbox}")
                return bbox

            # Fallback: try basic collision extraction
            dimensions = self._extract_from_collision(root)
            if dimensions:
                logger.info(f"Extracted collision dimensions from {sdf_path}: {dimensions}")
                return dimensions

            # Fallback: try to extract from visual geometry
            dimensions = self._extract_from_visual(root)
            if dimensions:
                logger.info(f"Extracted visual dimensions from {sdf_path}: {dimensions}")
                return dimensions

            # Final fallback: use model name to guess dimensions
            model_name = os.path.basename(str(model_path).rstrip('/'))
            dimensions = self._guess_from_model_name(model_name)
            if dimensions:
                logger.info(f"Using standard dimensions for {model_name}: {dimensions}")
                return dimensions

            logger.info(f"Could not extract dimensions from {sdf_path}")
            return None

        except Exception as e:
            logger.error(f"Error extracting dimensions from {model_path}: {e}")
            return None

    def _find_sdf_file(self, model_path: str) -> Optional[str]:
        """Find the SDF file in a model directory."""
        # Convert to string if it's a Path object
        model_path_str = str(model_path)

        if model_path_str.endswith('.sdf'):
            return model_path_str if os.path.exists(model_path_str) else None

        # Look for model.sdf or any .sdf file in the directory
        model_dir = Path(model_path_str)
        if not model_dir.exists():
            return None

        # First try model.sdf
        model_sdf = model_dir / 'model.sdf'
        if model_sdf.exists():
            return str(model_sdf)

        # Then try any .sdf file
        sdf_files = list(model_dir.glob('*.sdf'))
        if sdf_files:
            return str(sdf_files[0])

        return None

    def _parse_pose(self, pose_elem: ET.Element) -> Optional[List[float]]:
        """Parse a pose element and return [x, y, z, roll, pitch, yaw]."""
        if pose_elem is None:
            return None
        try:
            pose_text = pose_elem.text.strip()
            pose_values = [float(x) for x in pose_text.split()]
            if len(pose_values) >= 6:
                return pose_values[:6]
            elif len(pose_values) == 3:
                return pose_values + [0.0, 0.0, 0.0]  # Add zero rotation
        except:
            pass
        return None

    def _extract_surface_height(self, root: ET.Element) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Extract furniture surface height by finding the topmost thin horizontal collision box.
        This is specifically for desk/table models where we want the actual surface, not the full height.

        Returns:
            Tuple of ((width, length, height), (offset_x, offset_y, surface_top_z)) or None
        """
        try:
            import math
            best_surface = None
            best_surface_top_z = -float('inf')

            # Find all collision geometries in all links
            for model in root.iter('model'):
                model_pose = self._parse_pose(model.find('pose')) or [0, 0, 0, 0, 0, 0]

                for link in model.iter('link'):
                    link_pose = self._parse_pose(link.find('pose')) or [0, 0, 0, 0, 0, 0]

                    for collision in link.findall('collision'):
                        collision_pose = self._parse_pose(collision.find('pose')) or [0, 0, 0, 0, 0, 0]

                        # Get the geometry
                        geometry = collision.find('geometry')
                        if geometry is None:
                            continue

                        box = geometry.find('box')
                        if box is not None:
                            size_elem = box.find('size')
                            if size_elem is not None:
                                size = [float(x) for x in size_elem.text.strip().split()]
                                if len(size) == 3:
                                    width, length, height = size

                                    # Look for thin horizontal boxes (likely surfaces)
                                    if height < 0.1 and width > 0.3 and length > 0.3:
                                        # Calculate the center Z of this box
                                        box_center_z = collision_pose[2]
                                        # Transform through link and model poses
                                        box_center_z += link_pose[2] + model_pose[2]
                                        # Calculate top surface Z
                                        surface_top_z = box_center_z + (height / 2)

                                        # Keep track of the highest thin horizontal surface
                                        if surface_top_z > best_surface_top_z and surface_top_z < 2.0:  
                                            best_surface_top_z = surface_top_z
                                            best_surface = (width, length, surface_top_z)

            if best_surface:
                # Return dimensions (from the surface box) and offset with surface_top_z
                dims = (best_surface[0], best_surface[1], best_surface[2])  # width, length, surface_height
                offset = (0, 0, best_surface[2])  
                return (dims, offset)

            return None

        except Exception as e:
            logger.debug(f"Failed to extract surface height: {e}")
            return None

    def _extract_complete_bounding_box_with_offset(self, root: ET.Element) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Extract the complete axis-aligned bounding box WITH its offset from model origin.

        Returns:
            Tuple of ((width, length, height), (offset_x, offset_y, offset_z)) or None
        """
        try:
            import math
            all_corners = []

            # Find all collision geometries in all links
            for model in root.iter('model'):
                model_pose = self._parse_pose(model.find('pose')) or [0, 0, 0, 0, 0, 0]

                for link in model.iter('link'):
                    link_pose = self._parse_pose(link.find('pose')) or [0, 0, 0, 0, 0, 0]

                    for collision in link.findall('collision'):
                        collision_pose = self._parse_pose(collision.find('pose')) or [0, 0, 0, 0, 0, 0]

                        # Get the geometry
                        geometry = collision.find('geometry')
                        if geometry is None:
                            continue

                        box = geometry.find('box')
                        if box is not None:
                            size_elem = box.find('size')
                            if size_elem is not None:
                                size = [float(x) for x in size_elem.text.strip().split()]
                                if len(size) == 3:
                                    # Calculate the 8 corners of the box
                                    w, l, h = size[0] / 2, size[1] / 2, size[2] / 2
                                    corners = [
                                        [-w, -l, -h], [w, -l, -h], [-w, l, -h], [w, l, -h],
                                        [-w, -l, h], [w, -l, h], [-w, l, h], [w, l, h]
                                    ]

                                    # Transform corners through pose hierarchy
                                    for corner in corners:
                                        # Apply collision pose -> link pose -> model pose
                                        p = self._apply_pose_2d(corner, collision_pose)
                                        p = self._apply_pose_2d(p, link_pose)
                                        p = self._apply_pose_2d(p, model_pose)
                                        all_corners.append(p)

            if not all_corners:
                return None

            # Calculate axis-aligned bounding box from all transformed corners
            xs = [c[0] for c in all_corners]
            ys = [c[1] for c in all_corners]
            zs = [c[2] for c in all_corners]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            min_z, max_z = min(zs), max(zs)

            width = max_x - min_x
            length = max_y - min_y
            height = max_z - min_z

            # Calculate the center of the bounding box (offset from model origin)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            center_z = (min_z + max_z) / 2

            dimensions = (width, length, height)
            # Store max_z as the Z component since that represents the top surface
            offset = (center_x, center_y, max_z)

            return (dimensions, offset)

        except Exception as e:
            logger.debug(f"Failed to extract complete bounding box with offset: {e}")
            return None

    def _extract_complete_bounding_box(self, root: ET.Element) -> Optional[Tuple[float, float, float]]:
        """
        Extract the complete axis-aligned bounding box by traversing the pose hierarchy.
        This accounts for model pose, link poses, and collision/visual poses.
        """
        try:
            import math
            all_corners = []

            # Find all collision geometries in all links
            for model in root.iter('model'):
                model_pose = self._parse_pose(model.find('pose')) or [0, 0, 0, 0, 0, 0]

                for link in model.iter('link'):
                    link_pose = self._parse_pose(link.find('pose')) or [0, 0, 0, 0, 0, 0]

                    for collision in link.findall('collision'):
                        collision_pose = self._parse_pose(collision.find('pose')) or [0, 0, 0, 0, 0, 0]

                        # Get the geometry
                        geometry = collision.find('geometry')
                        if geometry is None:
                            continue

                        box = geometry.find('box')
                        if box is not None:
                            size_elem = box.find('size')
                            if size_elem is not None:
                                size = [float(x) for x in size_elem.text.strip().split()]
                                if len(size) == 3:
                                    # Calculate the 8 corners of the box
                                    w, l, h = size[0] / 2, size[1] / 2, size[2] / 2
                                    corners = [
                                        [-w, -l, -h], [w, -l, -h], [-w, l, -h], [w, l, -h],
                                        [-w, -l, h], [w, -l, h], [-w, l, h], [w, l, h]
                                    ]

                                    # Transform corners through pose hierarchy
                                    for corner in corners:
                                        # Apply collision pose -> link pose -> model pose
                                        p = self._apply_pose_2d(corner, collision_pose)
                                        p = self._apply_pose_2d(p, link_pose)
                                        p = self._apply_pose_2d(p, model_pose)
                                        all_corners.append(p)

            if not all_corners:
                return None

            # Calculate axis-aligned bounding box from all transformed corners
            xs = [c[0] for c in all_corners]
            ys = [c[1] for c in all_corners]
            zs = [c[2] for c in all_corners]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            min_z, max_z = min(zs), max(zs)

            width = max_x - min_x
            length = max_y - min_y
            height = max_z - min_z

            return (width, length, height)

        except Exception as e:
            logger.debug(f"Failed to extract complete bounding box: {e}")
            return None

    def _apply_pose_2d(self, point: List[float], pose: List[float]) -> List[float]:
        """
        Apply a 2D pose transformation (ignoring full 3D rotation for simplicity).
        This is sufficient for most ground-based models where yaw is the primary rotation.

        Args:
            point: [x, y, z]
            pose: [x, y, z, roll, pitch, yaw]

        Returns:
            Transformed [x, y, z]
        """
        import math

        x, y, z = point
        px, py, pz, roll, pitch, yaw = pose

        # Apply yaw rotation (rotation around Z axis)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        x_rot = x * cos_yaw - y * sin_yaw
        y_rot = x * sin_yaw + y * cos_yaw
        z_rot = z

        # Apply translation
        x_final = x_rot + px
        y_final = y_rot + py
        z_final = z_rot + pz

        return [x_final, y_final, z_final]

    def _extract_from_collision(self, root: ET.Element) -> Optional[Tuple[float, float, float]]:
        """Extract dimensions from collision geometry, accounting for model and link poses."""
        # Find collision elements
        for collision in root.iter('collision'):
            geometry = collision.find('geometry')
            if geometry is None:
                continue

            # Check for box geometry
            box = geometry.find('box')
            if box is not None:
                size_elem = box.find('size')
                if size_elem is not None:
                    size_text = size_elem.text.strip()
                    dimensions = [float(x) for x in size_text.split()]
                    if len(dimensions) == 3:
                        # Return the basic dimensions - handle pose offsets in placement engine
                        return tuple(dimensions)

            # Check for cylinder geometry
            cylinder = geometry.find('cylinder')
            if cylinder is not None:
                radius_elem = cylinder.find('radius')
                length_elem = cylinder.find('length')
                if radius_elem is not None and length_elem is not None:
                    radius = float(radius_elem.text)
                    length = float(length_elem.text)
                    # For cylinder: diameter as width and length, height as height
                    return (radius * 2, radius * 2, length)

            # Check for sphere geometry
            sphere = geometry.find('sphere')
            if sphere is not None:
                radius_elem = sphere.find('radius')
                if radius_elem is not None:
                    radius = float(radius_elem.text)
                    return (radius * 2, radius * 2, radius * 2)

        return None

    def _extract_from_visual(self, root: ET.Element) -> Optional[Tuple[float, float, float]]:
        """Extract dimensions from visual geometry (similar to collision)."""
        for visual in root.iter('visual'):
            geometry = visual.find('geometry')
            if geometry is None:
                continue

            # Check for box geometry
            box = geometry.find('box')
            if box is not None:
                size_elem = box.find('size')
                if size_elem is not None:
                    size_text = size_elem.text.strip()
                    dimensions = [float(x) for x in size_text.split()]
                    if len(dimensions) == 3:
                        return tuple(dimensions)

        return None

    def _guess_from_model_name(self, model_name: str) -> Optional[Tuple[float, float, float]]:
        """Guess dimensions based on model name."""
        model_name_lower = model_name.lower()

        # Direct matches
        for key, dims in self.mesh_dimensions.items():
            if key in model_name_lower:
                return dims

        # Fuzzy matches
        if 'pallet' in model_name_lower or 'euro' in model_name_lower:
            return self.mesh_dimensions['euro_pallet']
        elif 'shelf' in model_name_lower or 'aruco' in model_name_lower:
            return (3.6, 0.6, 1.8)  
        elif 'desk' in model_name_lower:
            return self.mesh_dimensions['desk']
        elif 'chair' in model_name_lower:
            return self.mesh_dimensions['chair']
        elif 'table' in model_name_lower:
            return self.mesh_dimensions['table']
        elif 'book' in model_name_lower:
            return self.mesh_dimensions['bookshelf']

        return None