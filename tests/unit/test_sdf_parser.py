import pytest
from gazebo_world_generator.src.utils.sdf_parser import SDFDimensionExtractor

def test_extract_model_dimensions_basic(sample_sdf_file):
    extractor = SDFDimensionExtractor()
    dims = extractor.extract_model_dimensions(sample_sdf_file)
    assert dims == (1.0, 2.0, 3.0)

def test_guess_from_model_name():
    extractor = SDFDimensionExtractor()
    dims = extractor._guess_from_model_name("euro_pallet")
    assert dims == (1.2, 0.8, 0.144)
    
    dims = extractor._guess_from_model_name("some_random_chair")
    assert dims == (0.6, 0.6, 0.9)

def test_extract_model_bounding_box(sample_sdf_file):
    extractor = SDFDimensionExtractor()
    result = extractor.extract_model_bounding_box(sample_sdf_file)
    assert result is not None
    dims, offset = result
    assert dims == (1.0, 2.0, 3.0)
    # The current implementation of _extract_complete_bounding_box_with_offset
    # returns max_z as the offset_z
    assert offset == (0.0, 0.0, 1.5) 
