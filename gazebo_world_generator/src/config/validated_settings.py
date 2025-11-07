"""
Configuration validation using Pydantic.

This module provides type-safe, validated configuration models that replace
the previous dictionary-based configuration system.
"""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings
import yaml

from gazebo_world_generator.src.exceptions import (
    ConfigValidationError,
    InvalidConfigError,
    MissingConfigError
)


# =============================================================================
# LLM Configuration
# =============================================================================

class LLMConfig(BaseModel):
    """LLM server configuration with validation."""

    server_url: str = Field(
        default="http://localhost:1234/v1",
        description="LLM server URL (OpenAI-compatible endpoint)"
    )
    model_name: str = Field(
        default="default-model",
        min_length=1,
        description="Name of the LLM model to use"
    )
    timeout: float = Field(
        default=180.0,
        gt=0,
        le=600.0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of retry attempts"
    )
    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM sampling"
    )

    @field_validator('server_url')
    @classmethod
    def validate_server_url(cls, v: str) -> str:
        """Ensure server URL is properly formatted."""
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            v = f"http://{v}"
        if not v.endswith('/v1'):
            v = f"{v.rstrip('/')}/v1"
        return v

    model_config = {
        "protected_namespaces": (),  # Allow model_name field
        "json_schema_extra": {
            "examples": [
                {
                    "server_url": "http://localhost:1234/v1",
                    "model_name": "gpt-4",
                    "timeout": 180.0,
                    "max_retries": 3,
                    "temperature": 0.6
                }
            ]
        }
    }


# =============================================================================
# Placement Configuration
# =============================================================================

class PlacementConfig(BaseModel):
    """Object placement configuration with validation."""

    grid_resolution: float = Field(
        default=0.25,
        gt=0.01,
        le=1.0,
        description="Grid resolution for placement calculations (meters)"
    )
    min_object_distance: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="Minimum distance between objects (meters)"
    )
    wall_clearance: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Clearance from walls (meters)"
    )
    validation_margin: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Validation margin for collision detection (meters)"
    )
    corridor_width: float = Field(
        default=1.2,
        ge=0.5,
        le=5.0,
        description="Width of traffic corridors between furniture (meters)"
    )
    max_placement_attempts: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum attempts for object placement"
    )


# =============================================================================
# Room Configuration
# =============================================================================

class RoomConfig(BaseModel):
    """Room generation configuration with validation."""

    default_height: float = Field(
        default=3.0,
        ge=2.0,
        le=10.0,
        description="Default room height (meters)"
    )
    wall_thickness: float = Field(
        default=0.15, 
        gt=0.01,
        le=1.0,
        description="Wall thickness (meters)"
    )
    wall_height: float = Field(
        default=3.0,
        ge=2.0,
        le=10.0,
        description="Wall height (meters)"
    )
    min_room_width: float = Field(
        default=2.0,
        gt=0.0,
        description="Minimum room width (meters)"
    )
    min_room_length: float = Field(
        default=2.0,
        gt=0.0,
        description="Minimum room length (meters)"
    )
    max_room_width: float = Field(
        default=100.0,
        gt=0.0,
        description="Maximum room width (meters)"
    )
    max_room_length: float = Field(
        default=100.0,
        gt=0.0,
        description="Maximum room length (meters)"
    )

    @model_validator(mode='after')
    def validate_room_sizes(self):
        """Ensure max dimensions are larger than min dimensions."""
        if self.max_room_width <= self.min_room_width:
            raise ConfigValidationError(
                f"max_room_width ({self.max_room_width}) must be greater than "
                f"min_room_width ({self.min_room_width})"
            )
        if self.max_room_length <= self.min_room_length:
            raise ConfigValidationError(
                f"max_room_length ({self.max_room_length}) must be greater than "
                f"min_room_length ({self.min_room_length})"
            )
        return self


# =============================================================================
# Physics Configuration
# =============================================================================

class PhysicsConfig(BaseModel):
    """Gazebo physics configuration with validation."""

    engine: Literal["ode", "bullet", "simbody", "dart"] = Field(
        default="ode",
        description="Physics engine to use"
    )
    gravity: float = Field(
        default=-9.8,
        ge=-20.0,
        le=0.0,
        description="Gravity acceleration (m/s²)"
    )
    real_time_update_rate: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Real-time update rate (Hz)"
    )
    max_step_size: float = Field(
        default=0.001,
        gt=0.0,
        le=0.1,
        description="Maximum simulation step size (seconds)"
    )


# =============================================================================
# Main Configuration
# =============================================================================

class ValidatedConfig(BaseSettings):
    """
    Main validated configuration class.

    This class loads configuration from YAML files and validates all settings
    using Pydantic. It replaces the previous dictionary-based Config class.
    """

    llm: LLMConfig = Field(default_factory=LLMConfig)
    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    rooms: RoomConfig = Field(default_factory=RoomConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)

    model_config = {
        "env_prefix": "GAZEBO_WORLD_GEN_",
        "env_nested_delimiter": "__",
        "case_sensitive": False
    }

    @classmethod
    def from_yaml(cls, config_path: Path) -> "ValidatedConfig":
        """
        Load and validate configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Validated configuration instance

        Raises:
            InvalidConfigError: If YAML file is malformed
            ConfigValidationError: If validation fails
        """
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except FileNotFoundError:
            raise MissingConfigError(
                f"Configuration file not found: {config_path}"
            )
        except yaml.YAMLError as e:
            raise InvalidConfigError(
                f"Failed to parse YAML configuration: {e}"
            )

        try:
            return cls(**config_dict)
        except Exception as e:
            raise ConfigValidationError(
                f"Configuration validation failed: {e}"
            )

    @classmethod
    def from_multiple_sources(cls) -> "ValidatedConfig":
        """
        Load configuration from multiple sources in priority order.

        Priority order:
        1. Environment variables (highest priority)
        2. ROS2 package share directory
        3. Development directory (./config/)
        4. User home (~/.config/gazebo_world_generator/)
        5. System-wide (/etc/gazebo_world_generator/)
        6. Default values (lowest priority)

        Returns:
            Validated configuration instance
        """
        config_locations = []

        # Try ROS2 package share directory
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = Path(get_package_share_directory('gazebo_world_generator'))
            config_locations.append(pkg_share / "config" / "generator_config.yaml")
        except Exception:
            pass

        # Development directory (relative to this file)
        config_locations.append(
            Path(__file__).parent.parent.parent.parent / "config" / "generator_config.yaml"
        )

        # User home directory
        config_locations.append(
            Path.home() / ".config" / "gazebo_world_generator" / "generator_config.yaml"
        )

        # System-wide location
        config_locations.append(
            Path("/etc/gazebo_world_generator/generator_config.yaml")
        )

        # Try each location
        for config_path in config_locations:
            if config_path.exists():
                try:
                    return cls.from_yaml(config_path)
                except Exception as e:
                    # Log warning but continue to next location
                    print(f"Warning: Failed to load config from {config_path}: {e}")

        # Fall back to defaults with environment variable overrides
        print("Warning: No config file found. Using defaults with environment overrides.")
        return cls()

    def validate_for_generation(self, description: str, num_rooms: int, total_objects: int):
        """
        Validate that a generation request is within resource limits.

        Args:
            description: World description
            num_rooms: Number of rooms
            total_objects: Total number of objects

        Raises:
            ResourceLimitExceededError: If limits are exceeded
        """
        from gazebo_world_generator.src.exceptions import ResourceLimitExceededError

        if len(description) > self.resource_limits.max_description_length:
            raise ResourceLimitExceededError(
                "description_length",
                len(description),
                self.resource_limits.max_description_length
            )

        if num_rooms > self.resource_limits.max_rooms:
            raise ResourceLimitExceededError(
                "rooms",
                num_rooms,
                self.resource_limits.max_rooms
            )

        if total_objects > self.resource_limits.max_total_objects:
            raise ResourceLimitExceededError(
                "total_objects",
                total_objects,
                self.resource_limits.max_total_objects
            )

    def to_dict(self) -> dict:
        """Export configuration as dictionary with Path objects converted to strings."""
        data = self.model_dump(mode='python')
        # Recursively convert Path objects to strings
        return self._convert_paths_to_strings(data)

    def _convert_paths_to_strings(self, obj):
        """Recursively convert Path objects to strings in a data structure."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_paths_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_paths_to_strings(item) for item in obj]
        else:
            return obj

    def to_yaml(self, output_path: Path):
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Singleton instance
# =============================================================================

_config_instance: Optional[ValidatedConfig] = None


def get_config() -> ValidatedConfig:
    """
    Get the global configuration instance (singleton pattern).

    Returns:
        Validated configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ValidatedConfig.from_multiple_sources()
    return _config_instance


def reload_config(config_path: Optional[Path] = None):
    """
    Reload configuration from file or sources.

    Args:
        config_path: Optional path to specific config file
    """
    global _config_instance
    if config_path:
        _config_instance = ValidatedConfig.from_yaml(config_path)
    else:
        _config_instance = ValidatedConfig.from_multiple_sources()


if __name__ == "__main__":
    # Test configuration loading
    print("Loading configuration...")
    config = get_config()
    print("\n✓ Configuration loaded successfully!")
    print(f"\nLLM Server: {config.llm.server_url}")
    print(f"Model: {config.llm.model_name}")
    print(f"Max Placement Attempts: {config.placement.max_placement_attempts}")
    print(f"Room Types: {', '.join(config.rooms.room_types)}")
