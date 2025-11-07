"""
Utility modules for the Gazebo World Generator
This file defines the public API of the 'utils' package.
"""

# Import the corrected list of functions from file_utils
from .file_utils import (
    find_gazebo_model_path,
    read_file_safe,
    write_file_safe,
    is_valid_model_directory,
    get_model_directories
)

# Import the corrected list of functions from sdf_utils
from .sdf_utils import (
    parse_sdf_file,
    get_model_dimensions,
    extract_model_info,
    create_simple_box_sdf
)

# Import the corrected list of functions from llm_utils
from .llm_utils import (
    extract_json_from_response,
    validate_json_with_schema,
    validate_coordinates
)

# The __all__ list defines what gets imported with 'from src.utils import *'
# It should match the functions imported above.
__all__ = [
    # file_utils
    'find_gazebo_model_path',
    'read_file_safe',
    'write_file_safe',
    'is_valid_model_directory',
    'get_model_directories',
    
    # sdf_utils
    'parse_sdf_file',
    'get_model_dimensions',
    'extract_model_info',
    'create_simple_box_sdf',
    
    # llm_utils
    'extract_json_from_response',
    'validate_json_with_schema',
    'validate_coordinates',
]