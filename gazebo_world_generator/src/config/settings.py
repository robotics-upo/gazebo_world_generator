"""
Configuration settings and constants for the Gazebo World Generator.

This module centralizes all configuration parameters, default values,
and system constants used throughout the application.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List

# Load user configuration
def load_config():
    """Load configuration from YAML file."""
    config_locations = []
    
    try:
        # Try to import ament_index_python only if available
        import importlib
        ament_spec = importlib.util.find_spec("ament_index_python.packages")
        if ament_spec is not None:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = Path(get_package_share_directory('gazebo_world_generator'))
            config_locations.append(pkg_share / "config" / "generator_config.yaml")
        else:
            # ament_index_python not available
            pass
    except Exception:
        pass
    
    # Development/source location
    config_locations.append(Path(__file__).parent.parent.parent.parent / "config" / "generator_config.yaml")
    
    # User home directory
    config_locations.append(Path.home() / ".config" / "gazebo_world_generator" / "generator_config.yaml")
    
    # System-wide location
    config_locations.append(Path("/etc/gazebo_world_generator/generator_config.yaml"))
    
    for config_path in config_locations:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")
    
    # Fallback to defaults if no config file found
    print("Warning: No config file found. Using hardcoded defaults.")
    print(f"Searched locations: {[str(p) for p in config_locations]}")
    return {
        'llm': {
            'server_url': 'http://localhost:1234/v1',
            'model_name': 'default-model'
        }
    }

# Load configuration
_config = load_config()

# Module-level constants for easy import (for backward compatibility)
DEFAULT_LLM_SERVER = _config.get('llm', {}).get('server_url', 'http://localhost:1234/v1')
DEFAULT_MODEL = _config.get('llm', {}).get('model_name', 'default-model')


class Config:
    """Main configuration class containing all system settings"""
    
    # LLM Configuration
    DEFAULT_LLM_SERVER = _config.get('llm', {}).get('server_url', 'http://localhost:1234/v1')
    DEFAULT_LLM_MODEL = _config.get('llm', {}).get('model_name', 'default-model')
    
    # Model Resolution Configuration
    GAZEBO_MODEL_PATHS = [
        os.path.expanduser(path) 
        for path in _config.get('models', {}).get('search_paths', [
            '~/.gazebo/models',
            '/usr/share/gazebo-11/models',
            '/usr/local/share/gazebo-11/models'
        ])
    ]
    
    MODEL_CACHE_DIR = os.path.expanduser(
        _config.get('models', {}).get('cache_directory', "~/.gazebo/models")
    )
    
    # Online Model Database Configuration
    ONLINE_REPOS = {
        "ignition_fuel": {
            "base_url": "https://fuel.ignitionrobotics.org",
            "api_url": "https://fuel.ignitionrobotics.org/1.0/models",
            "search_endpoint": "/search"
        },
        "gazebo_models": {
            "github_url": "https://github.com/osrf/gazebo_models",
            "raw_url": "https://raw.githubusercontent.com/osrf/gazebo_models/master"
        },
        "gazebo_models_collection": {
            "github_url": "https://github.com/leonhartyao/gazebo_models_worlds_collection",
            "raw_url": "https://raw.githubusercontent.com/leonhartyao/gazebo_models_worlds_collection/master",
            "models_path": "models"
        }
    }
    
    # Model Resolution Configuration
    MAX_CONCURRENT_MODEL_RESOLUTION = _config.get('models', {}).get('max_concurrent_downloads', 3)
    MAX_ONLINE_SEARCH_TERMS = 3
    MAX_ONLINE_MODELS_PER_SEARCH = _config.get('models', {}).get('max_search_results', 15)
    
    # Placement Configuration
    PLACEMENT_GRID_RESOLUTION = _config.get('placement', {}).get('grid_resolution', 0.25)
    MIN_OBJECT_DISTANCE = _config.get('placement', {}).get('min_object_distance', 0.5)
    WALL_CLEARANCE = _config.get('placement', {}).get('wall_clearance', 0.1)
    VALIDATION_MARGIN = _config.get('placement', {}).get('validation_margin', 0.1)
    TRAFFIC_CORRIDOR_WIDTH = _config.get('placement', {}).get('corridor_width', 1.2)
    
    # Essential Models
    ESSENTIAL_MODELS = {
        "ground_plane": "model://ground_plane",
        "sun": "model://sun"
    }
    
    # Default Object Sizes (width, length, height)
    DEFAULT_OBJECT_SIZES = {
        'desk': [1.5, 0.8, 0.75],
        'table': [1.2, 0.8, 0.75], 
        'chair': [0.6, 0.6, 0.9],
        'plant': [0.4, 0.4, 1.2],
        'shelf': [0.8, 0.3, 1.8],
        'bookshelf': [0.8, 0.4, 2.0],
        'monitor': [0.5, 0.2, 0.4],
        'lamp': [0.3, 0.3, 1.4],
        'box': [0.5, 0.5, 0.5],
        'cabinet': [0.8, 0.4, 1.8],
    }
    
    # Model Selection Configuration
    CORE_SYNONYMS = {
        'chair': ['seat', 'office_chair', 'officechair'],
        'seat': ['chair', 'office_chair', 'officechair'],
        'desk': ['office_desk', 'officedesk', 'table'],
        'table': ['desk', 'cafe_table', 'cafetable'],
        'shelf': ['bookshelf', 'shelving', 'storage'],
        'bookshelf': ['shelf', 'book_case', 'bookcase'],
        'cabinet': ['cupboard', 'storage'],
        'wardrobe': ['cabinet', 'closet', 'armoire', 'storage', 'metalcabinet'],
        'closet': ['wardrobe', 'cabinet', 'storage'],
        'lamp': ['light', 'desk_lamp', 'desklamp'],
        'light': ['lamp', 'desk_lamp', 'desklamp'],
        'computer': ['pc', 'desktop'],
        'laptop': ['computer', 'notebook'],
        'monitor': ['display', 'screen', 'keyboard'],
        'screen': ['monitor', 'display'],
        'keyboard': ['computer_keyboard'],
        'plant': ['potted_plant', 'indoor_plant', 'potted', 'pot'],
        'potted_plant': ['plant', 'potted', 'indoor_plant'],
        'tree': ['plant', 'tree_small'],
        'box': ['container', 'crate'],
        'bin': ['container', 'trash_bin'],
    }
    
    # Model Fallbacks
    FALLBACK_MODELS = {
        "table": "model://cafe_table",
        "desk": "model://cafe_table",
        "chair": "model://cafe_table",
        # Warehouse/Industrial fallbacks
        "forklift": "model://Euro_pallet",  # Use pallet as placeholder
        "hand_truck": "model://Euro_pallet",  # Pallet represents movable cargo
        "pallet_jack": "model://Euro_pallet",
        "workstation": "model://cafe_table",  # Table as work surface
        "workbench": "model://cafe_table",
        "staging_table": "model://cafe_table",
        "conveyor_system": None,  # No reasonable fallback
        "conveyor_belt": None,
        # Storage alternatives
        "storage_unit": "model://StorageRack",
        "warehouse_rack": "model://StorageRack",
    }
    
    # Height Correction Configuration
    SURFACE_HEIGHT_OFFSET = 0.02
    DESK_DETECTION_RADIUS = 1.5

    # SDF Generation Configuration
    DEFAULT_WORLD_NAME = "generated_world"
    DEFAULT_PHYSICS_ENGINE = _config.get('physics', {}).get('engine', 'ode')
    DEFAULT_GRAVITY = _config.get('physics', {}).get('gravity', -9.8)
    DEFAULT_REAL_TIME_UPDATE_RATE = _config.get('physics', {}).get('real_time_update_rate', 1000)
    DEFAULT_MAX_STEP_SIZE = _config.get('physics', {}).get('max_step_size', 0.001)
    
    # Room Configuration
    DEFAULT_ROOM_HEIGHT = _config.get('rooms', {}).get('default_height', 3.0)
    DEFAULT_WALL_THICKNESS = _config.get('rooms', {}).get('wall_thickness', 0.2)
    DEFAULT_WALL_HEIGHT = _config.get('rooms', {}).get('wall_height', 3.0)


class LoggingConfig:
    """Logging configuration settings"""
    
    LOG_LEVEL = _config.get('logging', {}).get('level', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Module-specific log levels
    MODULE_LOG_LEVELS = _config.get('logging', {}).get('modules', {
        'models.resolver': 'INFO',
        'placement.engine': 'INFO', 
        'llm.interface': 'INFO',
        'core.generator': 'INFO'
    })
