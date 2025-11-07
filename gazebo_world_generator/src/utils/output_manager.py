"""
Output Manager
Handles organized file output with dedicated folders for worlds and logs.
"""

import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages organized output of generated worlds and their logs."""

    def __init__(self, base_output_dir: Path = None):
        """
        Initialize the output manager.

        Args:
            base_output_dir: Base directory for outputs. Defaults to 'generated_worlds' in current directory.
        """
        self.base_output_dir = base_output_dir or Path.cwd() / "generated_worlds"
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        self.worlds_dir = self.base_output_dir / "worlds"
        self.logs_dir = self.base_output_dir / "logs"

        self.worlds_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Output manager initialized:")
        logger.debug(f"  Worlds: {self.worlds_dir}")
        logger.debug(f"  Logs: {self.logs_dir}")

    def get_output_paths(self, world_name: str = None) -> dict:
        """
        Get organized output paths for a world generation.

        Args:
            world_name: Name for the world (without extension). If None, generates timestamp-based name.

        Returns:
            Dictionary with 'world_file', 'log_file', and 'world_name' keys.
        """
        if world_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            world_name = f"world_{timestamp}"
        else:
            world_name = Path(world_name).stem

        world_file = self.worlds_dir / f"{world_name}.sdf"
        log_file = self.logs_dir / f"{world_name}.log"

        return {
            'world_file': world_file,
            'log_file': log_file,
            'world_name': world_name
        }

    def setup_logging_for_world(self, log_file: Path, debug: bool = False) -> logging.FileHandler:
        """
        Set up file logging for a specific world generation.

        Args:
            log_file: Path to the log file
            debug: Enable debug level logging

        Returns:
            FileHandler instance that can be removed later
        """
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)

        logger.debug(f"Logging to file: {log_file}")

        return file_handler

    def finalize_world_output(self, temp_world_path: Path, final_world_path: Path,
                             file_handler: Optional[logging.FileHandler] = None) -> Path:
        """
        Move generated world file to final location and clean up logging.

        Args:
            temp_world_path: Temporary path where world was generated
            final_world_path: Final destination path
            file_handler: File handler to remove from logging (optional)

        Returns:
            Path to the final world file
        """
        # Move world file to organized location
        if temp_world_path.exists():
            shutil.move(str(temp_world_path), str(final_world_path))
            logger.debug(f"World file moved to: {final_world_path}")

        # Remove file handler from logging
        if file_handler:
            file_handler.close()
            logging.getLogger().removeHandler(file_handler)

        return final_world_path

    def get_summary(self) -> str:
        """Get a summary of the output directory structure."""
        world_count = len(list(self.worlds_dir.glob("*.sdf")))
        log_count = len(list(self.logs_dir.glob("*.log")))

        return f"""Output Directory: {self.base_output_dir}
├── worlds/ ({world_count} files)
└── logs/ ({log_count} files)"""
