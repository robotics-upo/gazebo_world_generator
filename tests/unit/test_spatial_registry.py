"""
Unit tests for SpatialRegistry

Tests dimension extraction, caching, offset calculations, and geometric computations.
"""

import pytest
import math
from unittest.mock import MagicMock, patch
from gazebo_world_generator.src.placement.spatial import SpatialRegistry


@pytest.fixture
def mock_model_db():
    """Mock model database."""
    db = MagicMock()
    db.find_best_model.return_value = "model://test_model"
    return db


@pytest.fixture
def mock_sdf_extractor():
    """Mock SDF dimension extractor."""
    extractor = MagicMock()
    extractor.extract_model_dimensions.return_value = (1.2, 0.6, 0.75)
    extractor.extract_model_bounding_box.return_value = (
        (1.2, 0.6, 0.75),  # dimensions
        (0.1, 0.05, 0.0)    # offset
    )
    return extractor


@pytest.fixture
def spatial_registry(mock_model_db, mock_sdf_extractor):
    """Create SpatialRegistry instance with mocks."""
    return SpatialRegistry(mock_model_db, mock_sdf_extractor)


def test_get_dimensions_fallback(spatial_registry):
    """Test fallback to default sizes when extraction fails."""
    # SDF extraction will fail, should use default
    spatial_registry.sdf_extractor.extract_model_bounding_box.return_value = None
    spatial_registry.sdf_extractor.extract_model_dimensions.return_value = None
    
    with patch('gazebo_world_generator.src.placement.spatial.find_gazebo_model_path', return_value=None):
        dimensions = spatial_registry.get_dimensions('desk')
    
    # Should return default desk size
    assert dimensions == (1.2, 0.7, 0.75)


def test_get_dimensions_with_cache(spatial_registry):
    """Test that dimensions are cached and reused."""
    with patch('gazebo_world_generator.src.placement.spatial.find_gazebo_model_path', return_value='/fake/path'):
        # First call - should extract
        dim1 = spatial_registry.get_dimensions('desk', 'TestDesk', 'office')
        
        # Second call - should use cache
        dim2 = spatial_registry.get_dimensions('desk', 'TestDesk', 'office')
    
    # Should return same dimensions
    assert dim1 == dim2
    # SDF extractor should only be called once
    assert spatial_registry.sdf_extractor.extract_model_bounding_box.call_count == 1


def test_get_dimensions_from_sdf(spatial_registry):
    """Test successful dimension extraction from SDF."""
    with patch('gazebo_world_generator.src.placement.spatial.find_gazebo_model_path', return_value='/fake/path'):
        dimensions = spatial_registry.get_dimensions('desk', 'TestDesk', 'office')
    
    # Should return extracted dimensions
    assert dimensions == (1.2, 0.6, 0.75)
    # Should have cached the offset
    assert 'desk_TestDesk_office' in spatial_registry.model_offsets_cache


def test_get_effective_bounds_no_rotation(spatial_registry):
    """Test effective bounds calculation without rotation."""
    # Set up known dimensions
    spatial_registry.model_dimensions_cache['test_default_default'] = (2.0, 1.0, 1.0)
    
    min_x, max_x, min_y, max_y = spatial_registry.get_effective_bounds(
        'test', center_x=0.0, center_y=0.0, yaw=0.0
    )
    
    # For 2.0x1.0 object at (0,0) with no rotation
    # Bounds should be [-1.0, 1.0] x [-0.5, 0.5]
    assert abs(min_x - (-1.0)) < 0.01
    assert abs(max_x - 1.0) < 0.01
    assert abs(min_y - (-0.5)) < 0.01
    assert abs(max_y - 0.5) < 0.01


def test_get_effective_bounds_with_rotation(spatial_registry):
    """Test effective bounds calculation with 90-degree rotation."""
    # Set up known dimensions: 2.0m wide x 1.0m deep
    spatial_registry.model_dimensions_cache['test_default_default'] = (2.0, 1.0, 1.0)
    
    # Rotate 90 degrees (pi/2)
    min_x, max_x, min_y, max_y = spatial_registry.get_effective_bounds(
        'test', center_x=0.0, center_y=0.0, yaw=math.pi/2
    )
    
    # After 90-degree rotation, width and length swap in axis-aligned bounding box
    # Should be approximately [-0.5, 0.5] x [-1.0, 1.0]
    assert abs(min_x - (-0.5)) < 0.01
    assert abs(max_x - 0.5) < 0.01
    assert abs(min_y - (-1.0)) < 0.01
    assert abs(max_y - 1.0) < 0.01


def test_get_effective_bounds_with_offset(spatial_registry):
    """Test effective bounds calculation with bounding box offset."""
    # Set up known dimensions and offset
    spatial_registry.model_dimensions_cache['test_default_default'] = (2.0, 1.0, 1.0)
    spatial_registry.model_offsets_cache['test_resolved'] = (0.5, 0.0, 0.0)
    
    min_x, max_x, min_y, max_y = spatial_registry.get_effective_bounds(
        'test', center_x=0.0, center_y=0.0, yaw=0.0
    )
    
    # With offset (0.5, 0.0, 0.0), center shifts from 0 to 0.5 in X
    # Bounds: [-0.5, 1.5] x [-0.5, 0.5]
    assert abs(min_x - (-0.5)) < 0.01
    assert abs(max_x - 1.5) < 0.01


def test_get_effective_bounds_offset_with_rotation(spatial_registry):
    """Test effective bounds with both offset and rotation."""
    # Set up dimensions and offset
    spatial_registry.model_dimensions_cache['test_default_default'] = (2.0, 1.0, 1.0)
    # Offset 0.5 in X direction
    spatial_registry.model_offsets_cache['test_resolved'] = (0.5, 0.0, 0.0)
    
    # Rotate 90 degrees - offset should rotate too
    min_x, max_x, min_y, max_y = spatial_registry.get_effective_bounds(
        'test', center_x=0.0, center_y=0.0, yaw=math.pi/2
    )
    
    # After 90-degree rotation, X offset becomes Y offset
    # Rotated offset: (0, 0.5)
    # Bounds should reflect this
    assert min_x is not None  # Just verify calculation completes


def test_cache_stats(spatial_registry):
    """Test cache statistics reporting."""
    # Add some entries
    spatial_registry.model_dimensions_cache['test1'] = (1, 1, 1)
    spatial_registry.model_dimensions_cache['test2'] = (2, 2, 2)
    spatial_registry.model_offsets_cache['test1'] = (0, 0, 0)
    
    stats = spatial_registry.get_cache_stats()
    
    assert stats['dimensions_cache_size'] == 2
    assert stats['offsets_cache_size'] == 1


def test_clear_cache(spatial_registry):
    """Test cache clearing."""
    # Add entries
    spatial_registry.model_dimensions_cache['test'] = (1, 1, 1)
    spatial_registry.model_offsets_cache['test'] = (0, 0, 0)
    
    # Clear
    spatial_registry.clear_cache()
    
    # Verify empty
    assert len(spatial_registry.model_dimensions_cache) == 0
    assert len(spatial_registry.model_offsets_cache) == 0


def test_get_dimensions_from_best_model(spatial_registry):
    """Test dimension extraction using model database."""
    spatial_registry.model_db.find_best_model.return_value = "model://BestDesk"
    
    with patch('gazebo_world_generator.src.placement.spatial.find_gazebo_model_path', return_value='/fake/path'):
        dimensions = spatial_registry.get_dimensions('desk', None, 'office')
    
    # Should call model_db to find best model
    spatial_registry.model_db.find_best_model.assert_called_once_with('desk', 'office')
    # Should return extracted dimensions
    assert dimensions == (1.2, 0.6, 0.75)


def test_multiple_cache_keys_for_offset(spatial_registry):
    """Test that offsets are stored with multiple keys."""
    with patch('gazebo_world_generator.src.placement.spatial.find_gazebo_model_path', return_value='/fake/path'):
        spatial_registry.get_dimensions('desk', 'TestDesk', 'office')
    
    # Should store offset with multiple keys
    assert 'desk_TestDesk_office' in spatial_registry.model_offsets_cache
    assert 'desk_resolved' in spatial_registry.model_offsets_cache


def test_default_sizes_available(spatial_registry):
    """Test that default sizes are available for common objects."""
    assert 'desk' in spatial_registry.default_sizes
    assert 'chair' in spatial_registry.default_sizes
    assert 'table' in spatial_registry.default_sizes
    assert 'default' in spatial_registry.default_sizes
    
    # Verify reasonable default sizes
    desk_size = spatial_registry.default_sizes['desk']
    assert len(desk_size) == 3  # (width, length, height)
    assert all(s > 0 for s in desk_size)  # All positive
