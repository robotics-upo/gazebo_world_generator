"""
Locate the package data directory holding config/ and prompts/.

The same files live in different places depending on how the package runs:
a source checkout, a colcon install (ament share directory), or a plain
pip install (data files under <prefix>/share/gazebo_world_generator).
"""

import sys
import sysconfig
from pathlib import Path
from typing import List, Optional

PACKAGE_NAME = "gazebo_world_generator"


def _candidate_dirs() -> List[Path]:
    candidates = [Path(__file__).resolve().parents[2]]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(Path(get_package_share_directory(PACKAGE_NAME)))
    except Exception:
        pass
    for prefix in (sysconfig.get_paths()["data"], sys.prefix):
        candidates.append(Path(prefix) / "share" / PACKAGE_NAME)
    return candidates


def find_data_path(relative: str) -> Optional[Path]:
    """Return the first existing <data dir>/<relative>, or None."""
    for base in _candidate_dirs():
        path = base / relative
        if path.exists():
            return path
    return None


def user_state_dir() -> Path:
    """Writable per-user directory for caches and metrics."""
    return Path.home() / ".gazebo_world_generator"
