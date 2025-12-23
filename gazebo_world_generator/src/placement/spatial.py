"""
Spatial Registry Module

Handles model dimension extraction, caching, and geometric calculations
for object placement. Provides a clean interface for spatial operations
used throughout the placement system.
"""

import logging
import math
from typing import Dict, Tuple, Optional
from gazebo_world_generator.src.utils.sdf_parser import SDFDimensionExtractor
from gazebo_world_generator.src.utils.file_utils import find_gazebo_model_path

logger = logging.getLogger(__name__)


class SpatialRegistry:
    """
    Registry for model dimensions and geometric calculations.
    
    Provides:
    - Model dimension extraction from SDF files
    - Dimension caching for performance
    - Bounding box offset tracking
    - Effective bounds calculation with rotation
    """
    
    def __init__(self, model_db, sdf_extractor: Optional[SDFDimensionExtractor] = None):
        """
        Initialize spatial registry.
        
        Args:
            model_db: Model database for resolving model names
            sdf_extractor: Optional SDF dimension extractor (creates default if None)
        """
        self.model_db = model_db
        self.sdf_extractor = sdf_extractor or SDFDimensionExtractor()
        
        # Caches for performance
        self.model_dimensions_cache: Dict[str, Tuple[float, float, float]] = {}
        self.model_offsets_cache: Dict[str, Tuple[float, float, float]] = {}
        
        # Default object sizes (fallback when extraction fails)
        self.default_sizes = {
            'desk': (1.2, 0.7, 0.75),
            'chair': (0.5, 0.5, 0.9),
            'table': (1.5, 0.8, 0.75),
            'shelf': (0.8, 0.3, 1.8),
            'bookshelf': (0.8, 0.3, 1.8),
            'cabinet': (0.6, 0.5, 1.5),
            'shelving_unit': (2.0, 0.8, 2.5),
            'storage_rack': (2.0, 0.8, 2.5),
            'pallet': (1.2, 0.8, 0.15),
            'box': (0.5, 0.5, 0.5),
            'plant': (0.3, 0.3, 0.5),
            'monitor': (0.5, 0.15, 0.4),
            'lamp': (0.2, 0.2, 0.5),
            'bed': (2.0, 1.5, 0.6),
            'nightstand': (0.5, 0.4, 0.5),
            'wardrobe': (1.2, 0.6, 2.0),
            'sofa': (2.0, 0.9, 0.8),
            'coffee_table': (1.2, 0.6, 0.4),
            'tv_stand': (1.5, 0.4, 0.5),
            'dresser': (1.2, 0.5, 1.0),
            'default': (1.0, 1.0, 1.0)
        }
    
    def get_dimensions(self, object_type: str, model_name: Optional[str] = None, 
                       room_type: Optional[str] = None) -> Tuple[float, float, float]:
        """
        Get actual model dimensions from SDF files or fallback to defaults.
        
        Args:
            object_type: The object type (e.g., 'shelving_unit')
            model_name: The specific model name (e.g., 'Shelf with ARUco boxes')
            room_type: The room type for context-aware model resolution
            
        Returns:
            Tuple of (width, length, height) in meters
        """
        cache_key = f"{object_type}_{model_name or 'default'}_{room_type or 'default'}"
        
        # Check cache first
        if cache_key in self.model_dimensions_cache:
            return self.model_dimensions_cache[cache_key]
        
        dimensions = None
        resolved_model_name = model_name
        
        # Try to extract from specific model
        if model_name:
            dimensions = self._extract_from_model(model_name, cache_key, object_type)
        
        # If no specific model, try to find the best available model
        if not dimensions and not model_name:
            dimensions = self._extract_from_best_model(object_type, room_type, cache_key)
        
        # Fallback to hardcoded values
        if not dimensions:
            dimensions = self.default_sizes.get(object_type, self.default_sizes['default'])
            logger.debug(f"Using fallback dimensions for '{object_type}': {dimensions}")
        
        # Cache the result
        self.model_dimensions_cache[cache_key] = dimensions
        return dimensions
    
    def _extract_from_model(self, model_name: str, cache_key: str, 
                           object_type: str) -> Optional[Tuple[float, float, float]]:
        """
        Extract dimensions from a specific model.
        
        Args:
            model_name: Model name to extract from
            cache_key: Cache key for storing offsets
            object_type: Object type for logging
            
        Returns:
            Dimensions tuple or None if extraction failed
        """
        try:
            model_path = find_gazebo_model_path(model_name)
            if model_path:
                # Try to get complete bounding box with offset
                bbox_result = self.sdf_extractor.extract_model_bounding_box(model_path)
                if bbox_result:
                    dimensions, offset = bbox_result
                    # Store offset with multiple cache keys for easier retrieval
                    self._store_offset(cache_key, offset, object_type)
                    logger.info(f"Extracted dims for '{model_name}': {dimensions}, offset: {offset}")
                    return dimensions
                else:
                    # Fallback to simple dimension extraction
                    dimensions = self.sdf_extractor.extract_model_dimensions(model_path)
                    if dimensions:
                        logger.info(f"Extracted dimensions for '{model_name}': {dimensions}")
                        return dimensions
        except Exception as e:
            logger.warning(f"Failed to extract dimensions for '{model_name}': {e}")
        
        return None
    
    def _extract_from_best_model(self, object_type: str, room_type: Optional[str], 
                                 cache_key: str) -> Optional[Tuple[float, float, float]]:
        """
        Extract dimensions from the best available model for an object type.
        
        Args:
            object_type: Object type to find model for
            room_type: Room type for context
            cache_key: Cache key for storing offsets
            
        Returns:
            Dimensions tuple or None if extraction failed
        """
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
                        self._store_offset(cache_key, offset, object_type)
                        logger.info(f"Extracted dims from '{resolved_model_name}' for '{object_type}': {dimensions}, offset: {offset}")
                        return dimensions
                    else:
                        dimensions = self.sdf_extractor.extract_model_dimensions(model_path)
                        if dimensions:
                            logger.info(f"Extracted dimensions from resolved model '{resolved_model_name}' for '{object_type}': {dimensions}")
                            return dimensions
        except Exception as e:
            logger.debug(f"Failed to resolve and extract dimensions for '{object_type}': {e}")
        
        return None
    
    def _store_offset(self, cache_key: str, offset: Tuple[float, float, float], 
                     object_type: str):
        """
        Store offset in cache with multiple keys for easier retrieval.
        
        Args:
            cache_key: Primary cache key
            offset: Offset tuple (x, y, z)
            object_type: Object type for generating alternate keys
        """
        self.model_offsets_cache[cache_key] = offset
        # Store with simplified keys as well
        simple_key = f"{object_type}_resolved"
        self.model_offsets_cache[simple_key] = offset
    
    def get_effective_bounds(self, object_type: str, center_x: float, center_y: float,
                            room_type: Optional[str] = None, 
                            yaw: float = 0.0) -> Tuple[float, float, float, float]:
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
        # Get the dimensions
        dimensions = self.get_dimensions(object_type, room_type=room_type)
        half_w, half_l = dimensions[0] / 2, dimensions[1] / 2
        
        # Get the bounding box offset (if available)
        offset = self._find_offset(object_type, room_type)
        
        # Rotate the offset by the yaw angle
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotated_offset_x = offset[0] * cos_yaw - offset[1] * sin_yaw
        rotated_offset_y = offset[0] * sin_yaw + offset[1] * cos_yaw
        
        # Calculate the actual center of the bounding box after rotation
        bbox_center_x = center_x + rotated_offset_x
        bbox_center_y = center_y + rotated_offset_y
        
        # Calculate axis-aligned bounding box (AABB) for rotated rectangle
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
    
    def _find_offset(self, object_type: str, room_type: Optional[str]) -> Tuple[float, float, float]:
        """
        Find the bounding box offset for an object type.
        
        Args:
            object_type: Object type to find offset for
            room_type: Room type for context
            
        Returns:
            Offset tuple (x, y, z), defaults to (0, 0, 0) if not found
        """
        # Try different cache key formats
        possible_keys = [
            f"{object_type}_resolved",
            f"{object_type}_default_{room_type or 'default'}",
            f"{object_type}_None_{room_type or 'default'}",
            f"{object_type}_default_default",
            f"{object_type}_None_default",
        ]
        
        logger.debug(f"Looking for offset for '{object_type}', room_type='{room_type}'")
        
        for key in possible_keys:
            if key in self.model_offsets_cache:
                offset = self.model_offsets_cache[key]
                logger.info(f"Found offset {offset} for '{object_type}' using cache_key: {key}")
                return offset
        
        logger.debug(f"No offset found for '{object_type}', using default (0, 0, 0)")
        return (0, 0, 0)
    
    def clear_cache(self):
        """Clear all caches."""
        self.model_dimensions_cache.clear()
        self.model_offsets_cache.clear()
        logger.debug("Spatial registry caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache sizes
        """
        return {
            'dimensions_cache_size': len(self.model_dimensions_cache),
            'offsets_cache_size': len(self.model_offsets_cache)
        }
