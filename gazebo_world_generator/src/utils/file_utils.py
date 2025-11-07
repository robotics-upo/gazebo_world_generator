"""
File utilities for handling SDF files and model operations using modern pathlib.
"""

import os
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def find_gazebo_model_path(model_name: str) -> Optional[Path]:
    """
    Find the path to a Gazebo model in standard locations.
    Searches by directory name first, then by model name in model.config files.

    Returns:
        A Path object to the model directory if found, otherwise None.
    """
    # Return None silently if model_name is empty or whitespace (expected in some cases)
    if not model_name or not model_name.strip():
        return None
    
    search_paths = [
        Path.home() / ".gazebo" / "models",
        Path("/usr/share/gazebo/models"),
        Path("/usr/share/gazebo-11/models"),
    ]

    # First try: search by directory name (fast path)
    for base_path in search_paths:
        if not base_path.is_dir():
            continue
        model_path = base_path / model_name
        if (model_path / "model.sdf").is_file() and (model_path / "model.config").is_file():
            logger.debug(f"Found model '{model_name}' at: {model_path}")
            return model_path

    # Second try: search by model name in model.config files (slower but handles name mismatches)
    for base_path in search_paths:
        if not base_path.is_dir():
            continue
        for model_dir in base_path.iterdir():
            if not model_dir.is_dir():
                continue
            config_file = model_dir / "model.config"
            if config_file.is_file() and (model_dir / "model.sdf").is_file():
                try:
                    tree = ET.parse(config_file)
                    root = tree.getroot()
                    name_elem = root.find('name')
                    if name_elem is not None and name_elem.text and name_elem.text.strip() == model_name:
                        logger.debug(f"Found model '{model_name}' at: {model_dir} (via model.config)")
                        return model_dir
                except Exception:
                    pass

    logger.warning(f"Could not find a valid model directory for '{model_name}'")
    return None


def read_file_safe(file_path: Path) -> Optional[str]:
    """Safely read a file, returning its content or None on failure."""
    try:
        return file_path.read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError) as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None


def write_file_safe(file_path: Path, content: str) -> bool:
    """Safely write content to a file, creating parent directories if needed."""
    try:
        # Ensure the parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        return True
    except IOError as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        return False


def is_valid_model_directory(directory_path: Path) -> bool:
    """Check if a directory contains a valid Gazebo model."""
    if not directory_path.is_dir():
        return False
    
    has_sdf = (directory_path / 'model.sdf').is_file()
    has_config = (directory_path / 'model.config').is_file()
    
    return has_sdf and has_config


def get_model_directories(base_path: Path) -> list[Path]:
    """Get all valid model directories within a base path."""
    if not base_path.is_dir():
        return []
    
    return [
        item for item in base_path.iterdir()
        if item.is_dir() and is_valid_model_directory(item)
    ]