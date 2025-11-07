#!/usr/bin/env python3
"""
Gazebo World Generator - Main Entry Point

Generates Gazebo simulation worlds from natural language descriptions using LLMs.
Supports both command-line arguments and an interactive user mode.

"""

import sys
import argparse
import logging
import re
import time
import threading
import subprocess
import os
from pathlib import Path

from datetime import datetime

# Import from package structure
from gazebo_world_generator.src.llm.interface import OpenAICompatibleInterface
from gazebo_world_generator.src.world.generator import WorldGenerator
from gazebo_world_generator.src.config.settings import DEFAULT_LLM_SERVER, DEFAULT_MODEL
from gazebo_world_generator.src.config.validated_settings import ValidatedConfig, LLMConfig
from gazebo_world_generator.src.utils.output_manager import OutputManager

# Configure root logger
logger = logging.getLogger()


class Style:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"


class ConsoleHandler(logging.Handler):
    """Custom handler that shows minimal, user-friendly messages on console."""

    def __init__(self):
        super().__init__()
        self.shown_messages = set()
        self.phase_start_time = {}
        self.current_phase = None
        self.model_selections = []
        self.spinner_active = False
        self.spinner_thread = None
        self.current_spinner_message = None
        self.current_spinner_indent = "  "
        self.gear_message_shown = (
            False  
        )
        self.current_room = None 

    def _start_phase(self, phase_name):
        """Start timing a phase."""
        self.current_phase = phase_name
        self.phase_start_time[phase_name] = time.time()

    def _end_phase(self, phase_name=None):
        """End timing a phase and return duration."""
        phase = phase_name or self.current_phase
        if phase and phase in self.phase_start_time:
            duration = time.time() - self.phase_start_time[phase]
            del self.phase_start_time[phase]
            if phase == self.current_phase:
                self.current_phase = None
            return duration
        return None

    def _format_duration(self, seconds):
        """Format duration in human-readable form."""
        if seconds < 1:
            return f"{int(seconds * 1000)}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"

    def _stop_spinner(self, show_gear=False):
        """Stop the spinner if active.

        Args:
            show_gear: If True, show the spinner message with a gear icon before clearing
        """
        self.spinner_active = False
        if self.spinner_thread:
            self.spinner_thread.join()
            self.spinner_thread = None
            # Clear spinner line
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

            if (
                show_gear
                and self.current_spinner_message
                and not self.gear_message_shown
            ):
                print(
                    f"{self.current_spinner_indent}{Style.DIM}├─{Style.ENDC} {Style.CYAN}⚙{Style.ENDC}  {self.current_spinner_message}"
                )
                self.gear_message_shown = True
                self.current_spinner_message = None

    def _start_spinner(self, message, indent="  "):
        """Start a spinner for long operations."""
        self._stop_spinner() 
        self.spinner_active = True
        self.current_spinner_message = message
        self.current_spinner_indent = indent
        self.gear_message_shown = False

        def spin():
            spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            i = 0
            while self.spinner_active:
                sys.stdout.write(
                    f"\r{indent}{Style.CYAN}{spinner_chars[i]}{Style.ENDC} {message}"
                )
                sys.stdout.flush()
                time.sleep(0.1)
                i = (i + 1) % len(spinner_chars)

        self.spinner_thread = threading.Thread(target=spin, daemon=True)
        self.spinner_thread.start()

    def emit(self, record):
        """Only show important user-facing messages."""
        # Skip debug messages entirely
        if record.levelno == logging.DEBUG:
            return

        msg = record.getMessage()

        # Handle warnings and errors immediately
        if record.levelno == logging.WARNING:
            self._stop_spinner(show_gear=True)
            msg_clean = msg.lstrip("⚠️⚠ ").strip()
            print(f"    {Style.DIM}|  ├─ ⚠ {msg_clean}{Style.ENDC}")
            return
        elif record.levelno >= logging.ERROR:
            self._stop_spinner()
            print(f"\n{Style.RED}✗ {msg}{Style.ENDC}")
            return

        if "Could not extract dimensions" in msg or "Could not find" in msg:
            self._stop_spinner(show_gear=True)
            print(f"    {Style.DIM}|  ├─ ℹ {msg}{Style.ENDC}")
            return

        # Skip overly detailed informational messages
        if "Using standard dimensions" in msg:
            return

        # World generation start
        if "Generating world from description:" in msg:
            desc = msg.split('"')[1] if '"' in msg else msg.split(":")[1].strip()
            print(f"\n{Style.CYAN}▶ Generating world: {Style.BOLD}{desc}{Style.ENDC}")
            self._start_phase("total")

        # Room parsing phase
        elif "Parsing room description" in msg:
            self._start_spinner("Parsing room description with LLM...")
            self._start_phase("parsing")

        elif "Parsed" in msg and "rooms from description" in msg:
            # Skip duplicate message
            msg_key = msg[:50]
            if msg_key in self.shown_messages:
                return
            self.shown_messages.add(msg_key)

            duration = self._end_phase("parsing")
            self._stop_spinner()
            num_rooms = msg.split()[1] if len(msg.split()) > 1 else "?"
            time_str = (
                f" {Style.DIM}({self._format_duration(duration)}){Style.ENDC}"
                if duration
                else ""
            )
            print(f"{Style.GREEN}  ✓ Parsed {num_rooms} room(s){time_str}{Style.ENDC}")

        # Room structure creation
        elif "Creating room structure" in msg:
            print(f"  {Style.CYAN}⚙{Style.ENDC}  Creating room structures...")

        elif "Positioning rooms" in msg:
            print(f"  {Style.CYAN}⚙{Style.ENDC}  Positioning rooms...")
            print(f"  {Style.GREEN}✓ Rooms positioned{Style.ENDC}")

        # Room furnishing
        elif "Placing" in msg and "objects in" in msg:
            self._stop_spinner()
            self.shown_messages.clear()
            room_name = msg.split("'")[1] if "'" in msg else ""
            room_type = (
                msg.split("(")[1].split(")")[0] if "(" in msg and ")" in msg else ""
            )
            room_number_info = ""
            if "[" in msg and "/" in msg and "]" in msg:
                start = msg.index("[")
                end = msg.index("]")
                room_number_info = f" {Style.DIM}{msg[start:end+1]}{Style.ENDC}"
            print(
                f"\n  {Style.CYAN}▶{Style.ENDC} Furnishing: {Style.BOLD}{room_name}{Style.ENDC} ({room_type}){room_number_info}"
            )
            self.model_selections = []
            self.current_room = room_name

        # Early model resolution
        elif "Resolving" in msg and "model types early" in msg:
            parts = msg.split()
            count = parts[1] if len(parts) > 1 else "?"
            self._start_spinner(
                f"Resolving model dimensions ({count} types)...", indent="    "
            )

        elif "Model resolution complete" in msg:
            self._stop_spinner()
            print(
                f"    {Style.DIM}├─{Style.ENDC} {Style.GREEN}✓{Style.ENDC} Model dimensions ready"
            )

        # Semantic grouping
        elif (
            "Requesting semantic grouping" in msg
            or "Analyzing object relationships" in msg
        ):
            self._start_spinner(
                "Analyzing object relationships with LLM...", indent="    "
            )
            self._start_phase("grouping")

        elif re.search(r"Created.*functional groups", msg):
            self._stop_spinner()
            if not self.gear_message_shown:
                print(
                    f"    {Style.DIM}├─{Style.ENDC} {Style.CYAN}⚙{Style.ENDC}  Analyzing object relationships with LLM..."
                )
            duration = self._end_phase("grouping")
            parts = msg.split()
            created_idx = parts.index("Created") if "Created" in parts else -1
            count = (
                parts[created_idx + 1]
                if created_idx >= 0 and created_idx + 1 < len(parts)
                else "?"
            )
            time_str = (
                f" {Style.DIM}({self._format_duration(duration)}){Style.ENDC}"
                if duration
                else ""
            )
            print(
                f"    {Style.DIM}├─{Style.ENDC} {Style.GREEN}✓{Style.ENDC} Identified {count} functional group(s){time_str}"
            )

        # Model selections
        elif "LLM selected local model" in msg or ("Final selection for" in msg):
            obj_type = msg.split("'")[1] if "'" in msg else ""
            model = msg.split(":")[-1].strip() if ":" in msg else ""
            if obj_type and model:
                self.model_selections.append((obj_type, model))

        # LLM placement
        elif "Requesting LLM to place" in msg or "Requesting LLM to position" in msg:
            self._start_spinner("Generating placement plan with LLM...", indent="    ")
            self._start_phase("placement")

        elif "✅ LLM placement successful" in msg or "LLM generated a valid" in msg:
            self._stop_spinner()
            if not self.gear_message_shown:
                print(
                    f"    {Style.DIM}├─{Style.ENDC} {Style.CYAN}⚙{Style.ENDC}  Generating placement plan with LLM..."
                )
            duration = self._end_phase("placement")
            time_str = (
                f" {Style.DIM}({self._format_duration(duration)}){Style.ENDC}"
                if duration
                else ""
            )
            print(
                f"    {Style.DIM}├─{Style.ENDC} {Style.GREEN}✓{Style.ENDC} Placement plan generated{time_str}"
            )

        # LLM self-correction
        elif "LLM self-correction" in msg:
            self._start_spinner("LLM reviewing placement...", indent="    ")
            self._start_phase("self_correction")

        elif "LLM review: No issues" in msg or "Applied" in msg and "correction" in msg:
            self._stop_spinner()
            if not self.gear_message_shown:
                print(
                    f"    {Style.DIM}├─{Style.ENDC} {Style.CYAN}⚙{Style.ENDC}  LLM reviewing placement..."
                )
            duration = self._end_phase("self_correction")
            time_str = (
                f" {Style.DIM}({self._format_duration(duration)}){Style.ENDC}"
                if duration
                else ""
            )
            if "No issues" in msg:
                print(
                    f"    {Style.DIM}├─{Style.ENDC} {Style.GREEN}✓{Style.ENDC} Self-correction: No issues found{time_str}"
                )
            else:
                # Extract number of corrections from message
                match = re.search(r"(\d+)\s+correction", msg)
                count = match.group(1) if match else "?"
                print(
                    f"    {Style.DIM}├─{Style.ENDC} {Style.GREEN}✓{Style.ENDC} Applied {count} correction(s){time_str}"
                )

        # Validation
        elif "Validating placement" in msg:
            self._start_spinner("Validating and correcting positions...", indent="    ")
            self._start_phase("validation")

        # Collision detection (sub-message of validation)
        elif "Collision detection converged" in msg:
            self._stop_spinner()
            if not self.gear_message_shown:
                print(
                    f"    {Style.DIM}├─{Style.ENDC} {Style.CYAN}⚙{Style.ENDC}  Validating and correcting positions..."
                )
                self.gear_message_shown = True
            match = re.search(r"after (\d+) iteration", msg)
            iterations = match.group(1) if match else "?"
            print(
                f"    {Style.DIM}|  ├─ ✓ Collision detection converged after {iterations} iteration(s){Style.ENDC}"
            )

        elif (
            "🚪 Enforcing doorway clearance" in msg
            or "🚪 Ensuring doorway clearance" in msg
        ):
            if "doorway_clearance" in self.shown_messages:
                return
            self.shown_messages.add("doorway_clearance")
            self._stop_spinner(show_gear=True)
            print(f"    {Style.DIM}|  ├─ ⚙ Enforcing doorway clearance...{Style.ENDC}")
            return

        elif "🪑 Enforcing desk/table-chair relationships" in msg:
            if "chair_positioning" in self.shown_messages:
                return
            self.shown_messages.add("chair_positioning")
            self._stop_spinner(show_gear=True)
            print(f"    {Style.DIM}|  ├─ ⚙ Positioning chairs at desks...{Style.ENDC}")
            return

        # Validation sub-phase completions
        elif "✓ Cleared" in msg and "doorway zones" in msg:
            match = re.search(r"(\d+)", msg)
            count = match.group(1) if match else "?"
            print(
                f"    {Style.DIM}|  ├─ ✓ Cleared {count} object(s) from doorways{Style.ENDC}"
            )
            return

        elif "✓ Adjusted" in msg and "chair(s)" in msg:
            match = re.search(r"(\d+)", msg)
            count = match.group(1) if match else "?"
            print(
                f"    {Style.DIM}|  ├─ ✓ Positioned {count} chair(s) at desks{Style.ENDC}"
            )
            return

        elif "Validation complete" in msg:
            self._stop_spinner()
            if not self.gear_message_shown:
                print(
                    f"    {Style.DIM}├─{Style.ENDC} {Style.CYAN}⚙{Style.ENDC}  Validating and correcting positions..."
                )
            duration = self._end_phase("validation")
            time_str = (
                f" {Style.DIM}({self._format_duration(duration)}){Style.ENDC}"
                if duration
                else ""
            )
            match = re.search(r"(\d+)\s+objects positioned", msg)
            count = match.group(1) if match else "?"
            print(
                f"    {Style.DIM}├─{Style.ENDC} {Style.GREEN}✓{Style.ENDC} Validation complete ({count} objects){time_str}"
            )

        # Semantic enforcement 
        elif "Enforcing semantic group" in msg:
            print(f"    {Style.DIM}├─{Style.ENDC} Applying semantic constraints...")

        # Wall furniture snapping 
        elif "Positioning wall furniture" in msg:
            print(f"    {Style.DIM}├─{Style.ENDC} Snapping wall furniture...")

        # Model resolution
        elif "Resolving models and creating" in msg:
            count = msg.split()[4] if len(msg.split()) > 4 else "?"
            print(
                f"    {Style.DIM}├─{Style.ENDC} Resolving 3D models ({count} objects)..."
            )
            # Show buffered model selections
            if self.model_selections:
                for obj_type, model in self.model_selections[:3]:  # Show max 3
                    print(f"      {Style.DIM}• {obj_type}: {model}{Style.ENDC}")
                if len(self.model_selections) > 3:
                    print(
                        f"      {Style.DIM}• ... and {len(self.model_selections) - 3} more{Style.ENDC}"
                    )
                self.model_selections = []

        # Online search (if needed)
        elif "Searching online" in msg or "No suitable local model" in msg:
            print(
                f"      {Style.YELLOW}↓{Style.ENDC} {Style.DIM}Searching online repositories...{Style.ENDC}"
            )

        elif "Downloaded" in msg and "model" in msg:
            print(
                f"      {Style.GREEN}✓{Style.ENDC} {Style.DIM}Downloaded from online source{Style.ENDC}"
            )

        # Completion
        elif "Spatial Placement Complete" in msg:
            parts = msg.split()
            count_idx = parts.index("models") - 1 if "models" in parts else None
            count = parts[count_idx] if count_idx and count_idx >= 0 else "?"
            print(
                f"    {Style.DIM}├─{Style.ENDC} {Style.GREEN}✓{Style.ENDC} 3D models resolved"
            )
            print(
                f"    {Style.DIM}├─{Style.ENDC} {Style.GREEN}✓{Style.ENDC} Placed {count} object(s) successfully"
            )

        # Wall generation
        elif "Generating walls" in msg:
            print(f"  {Style.CYAN}⚙{Style.ENDC}  Generating walls and doorways...")

        elif "World file generated" in msg:
            duration = self._end_phase("total")
            time_str = (
                f" {Style.DIM}in {self._format_duration(duration)}{Style.ENDC}"
                if duration
                else ""
            )
            print(f"  {Style.GREEN}✓ World file created{time_str}{Style.ENDC}")

        # Collision warnings 
        elif "Separated overlapping objects" in msg or "Collision detection" in msg:
            print(f"{Style.YELLOW}{Style.DIM}  ⚙ {msg}{Style.ENDC}")

        # ==================== REFINEMENT PHASE ====================

        # Refinement parsing
        elif "Parsing refinement request" in msg:
            self._start_spinner("Parsing refinement request with LLM...")
            self._start_phase("refinement_parsing")

        elif "Successfully parsed refinement" in msg:
            duration = self._end_phase("refinement_parsing")
            self._stop_spinner()
            operation = msg.split(":")[-1].strip() if ":" in msg else "unknown"
            time_str = (
                f" {Style.DIM}({self._format_duration(duration)}){Style.ENDC}"
                if duration
                else ""
            )
            print(f"\n{Style.CYAN}⚙ Phase 1: Parsing refinement request...{Style.ENDC}")
            print(
                f"  {Style.GREEN}✓ Parsed operation: {Style.BOLD}{operation}{Style.ENDC}{time_str}"
            )

        elif "Objects affected:" in msg or "objects specified" in msg:
            count = msg.split()[0] if msg.split() else "?"
            print(f"    Objects affected: {count}")

        # Add operation phase
        elif "Applying add objects refinement" in msg:
            print(
                f"\n{Style.CYAN}⚙ Phase 2: Applying {Style.BOLD}add{Style.ENDC}{Style.CYAN} operation...{Style.ENDC}"
            )
            self._start_phase("refinement_operation")

        elif re.search(r"Adding \d+ .+\(s\) with hint", msg):
            parts = msg.split()
            if len(parts) >= 2:
                count = parts[1]
                obj_type = parts[2] if len(parts) > 2 else "object"
                print(f"    Adding {count} {obj_type}(s)...")

        # Remove operation phase
        elif "Applying remove objects refinement" in msg:
            print(
                f"\n{Style.CYAN}⚙ Phase 2: Applying {Style.BOLD}remove{Style.ENDC}{Style.CYAN} operation...{Style.ENDC}"
            )
            self._start_phase("refinement_operation")

        elif re.search(r"Removed \d+ object\(s\), saved", msg):
            duration = self._end_phase("refinement_operation")
            count = msg.split()[1] if len(msg.split()) > 1 else "?"
            time_str = (
                f" {Style.DIM}({self._format_duration(duration)}){Style.ENDC}"
                if duration
                else ""
            )
            print(
                f"      {Style.GREEN}✓ Removed {count} object(s){Style.ENDC}{time_str}"
            )

        # Modify operation phase
        elif "Applying modify objects refinement" in msg:
            print(
                f"\n{Style.CYAN}⚙ Phase 2: Applying {Style.BOLD}modify{Style.ENDC}{Style.CYAN} operation...{Style.ENDC}"
            )
            self._start_phase("refinement_operation")

        elif re.search(r"Modified \d+ object\(s\), saved", msg):
            duration = self._end_phase("refinement_operation")
            count = msg.split()[1] if len(msg.split()) > 1 else "?"
            time_str = (
                f" {Style.DIM}({self._format_duration(duration)}){Style.ENDC}"
                if duration
                else ""
            )
            print(
                f"      {Style.GREEN}✓ Modified {count} object(s){Style.ENDC}{time_str}"
            )

        # Reposition operation phase
        elif "Applying reposition objects refinement" in msg:
            print(
                f"\n{Style.CYAN}⚙ Phase 2: Applying {Style.BOLD}reposition{Style.ENDC}{Style.CYAN} operation...{Style.ENDC}"
            )
            self._start_phase("refinement_operation")

        elif re.search(r"Repositioned \d+ object\(s\), saved", msg):
            duration = self._end_phase("refinement_operation")
            count = msg.split()[1] if len(msg.split()) > 1 else "?"
            time_str = (
                f" {Style.DIM}({self._format_duration(duration)}){Style.ENDC}"
                if duration
                else ""
            )
            print(
                f"      {Style.GREEN}✓ Repositioned {count} object(s){Style.ENDC}{time_str}"
            )

        # Resize room operation
        elif "Applying resize room refinement" in msg:
            print(
                f"\n{Style.CYAN}⚙ Phase 2: Applying {Style.BOLD}resize_room{Style.ENDC}{Style.CYAN} operation...{Style.ENDC}"
            )
            self._start_phase("refinement_operation")

        elif "Resized room" in msg and "saved to" not in msg.lower():
            duration = self._end_phase("refinement_operation")
            if "'" in msg:
                room_match = re.search(r"'([^']+)'", msg)
                room_name = room_match.group(1) if room_match else "room"
                dims_match = re.search(r"(\d+\.?\d*)m × (\d+\.?\d*)m", msg)
                if dims_match:
                    time_str = (
                        f" {Style.DIM}({self._format_duration(duration)}){Style.ENDC}"
                        if duration
                        else ""
                    )
                    print(
                        f"      {Style.GREEN}✓ Resized {room_name} to {dims_match.group(1)}m × {dims_match.group(2)}m{Style.ENDC}{time_str}"
                    )
                else:
                    print(f"      {Style.GREEN}✓ Room resized successfully{Style.ENDC}")
            else:
                print(f"      {Style.GREEN}✓ Room resized successfully{Style.ENDC}")

        # Successfully added objects 
        elif "Successfully added" in msg and "with placement engine" in msg:
            duration = self._end_phase("refinement_operation")
            parts = msg.split()
            count_idx = 2 if len(parts) > 2 else -1
            count = parts[count_idx] if count_idx >= 0 else "?"
            time_str = (
                f" {Style.DIM}({self._format_duration(duration)}){Style.ENDC}"
                if duration
                else ""
            )
            print(f"      {Style.GREEN}✓ Added {count} object(s){Style.ENDC}{time_str}")

        # Refinement completion
        elif (
            "Refinement completed successfully" in msg
            or "saved to:" in msg.lower()
            and "_refined" in msg
        ):
            print(f"  {Style.GREEN}✓ Refinement completed successfully{Style.ENDC}")
            if "Output:" in msg or "saved to:" in msg.lower():
                output = msg.split(":")[-1].strip() if ":" in msg else ""
                if output:
                    print(f"    Output: {output}")

        # Group detection in refinement
        elif "Detected grouped object" in msg and "✓" in msg:
            pass

        # Added individual object (show progress)
        elif re.search(r"Added .+ at \(.+\)", msg) and "with yaw" in msg:
            # Extract model name
            model_name = (
                msg.split("Added")[1].split("at")[0].strip()
                if "Added" in msg
                else "object"
            )
            # Show subtle progress indicator
            sys.stdout.write(f"\r      {Style.DIM}• Adding {model_name}...{Style.ENDC}")
            sys.stdout.flush()


def setup_logging(debug: bool = False):
    """Configures a dual logging system: minimal console + detailed file."""
    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set root logger to DEBUG to capture everything
    logger.setLevel(logging.DEBUG)

    # Add custom console handler
    console_handler = ConsoleHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gazebo World Generator - Generate Gazebo worlds from natural language descriptions."
    )

    parser.add_argument(
        "--description",
        type=str,
        default=None,  # Default to None to trigger interactive mode if not provided
        help="Natural language description of the world. If omitted, runs in interactive mode.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated_world.sdf"),
        help="Output SDF file path (default: generated_world.sdf)",
    )

    parser.add_argument(
        "--llm-server",
        type=str,
        default=DEFAULT_LLM_SERVER,
        help=f"LLM server URL (default: {DEFAULT_LLM_SERVER})",
    )

    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="LLM model name to use"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging for verbose output."
    )

    return parser.parse_args()


def run_generation(
    description: str,
    output_path: Path,
    llm_server: str,
    model: str,
    debug: bool = False,
):
    """Encapsulates the core world generation logic."""
    # Validate LLM configuration
    try:
        llm_config = LLMConfig(
            server_url=llm_server,
            model_name=model
        )
        logger.info(f"✓ LLM configuration validated: {llm_config.server_url}")
    except Exception as e:
        logger.error(f"✗ LLM configuration validation failed: {e}")
        print(f"{Style.RED}✗ Invalid LLM configuration: {e}{Style.ENDC}\n")
        return 1

    # Initialize output manager
    output_manager = OutputManager()

    # Get organized output paths
    world_name = (
        output_path.stem if output_path != Path("generated_world.sdf") else None
    )
    paths = output_manager.get_output_paths(world_name)

    # Set up file logging for this world (detailed logs go here)
    file_handler = output_manager.setup_logging_for_world(paths["log_file"], debug)

    # Show header when running with arguments
    if description:
        print(
            f"\n{Style.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.ENDC}"
        )
        print(f"{Style.BOLD}🌍 Gazebo World Generator{Style.ENDC}")
        print(
            f"{Style.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.ENDC}"
        )

    # Detailed logging (file only)
    logger.info(f"Initializing LLM interface for model '{llm_config.model_name}' at {llm_config.server_url}")
    logger.info(f"World name: {paths['world_name']}")

    # Create full validated config
    validated_config = ValidatedConfig(llm=llm_config)

    # Create interfaces using validated config (silent to user)
    llm_interface = OpenAICompatibleInterface(
        server_url=validated_config.llm.server_url,
        model_name=validated_config.llm.model_name,
        timeout=validated_config.llm.timeout
    )
    generator = WorldGenerator(llm_interface, config=validated_config)

    logger.info(f'Generating world from description: "{description}"')

    # Generate to temporary location first
    temp_output = str(output_path)

    try:
        output_file = generator.generate_world(description, temp_output)
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        file_handler.close()
        logging.getLogger().removeHandler(file_handler)
        print(
            f"\n{Style.RED}✗ Generation failed. Check log: {paths['log_file']}{Style.ENDC}\n"
        )
        return 1

    if "invalid" in str(output_file):
        logger.error("World generation resulted in an invalid SDF file.")
        logger.error(f"Please inspect the output file for errors: {output_file}")
        file_handler.close()
        logging.getLogger().removeHandler(file_handler)
        print(
            f"\n{Style.RED}✗ Invalid SDF generated. Check log: {paths['log_file']}{Style.ENDC}\n"
        )
        return 1

    # Move to organized location
    final_world = output_manager.finalize_world_output(
        Path(output_file), paths["world_file"], file_handler
    )

    # Detailed logging
    logger.info("✅ World generation completed successfully!")
    logger.info(f"   Output file: {final_world}")
    logger.info(f"   Log file: {paths['log_file']}")

    # Generate occupancy map
    map_files = None
    try:
        logger.info("Generating occupancy map...")
        print(f"  {Style.CYAN}⚙{Style.ENDC}  Generating occupancy map...")
        map_files = generate_occupancy_map(
            str(final_world), final_world.parent.parent / "occupancy_maps"
        )
        print(f"{Style.GREEN}  ✓ Occupancy map generated successfully{Style.ENDC}")
        logger.info(f"✅ Occupancy map generated:")
        logger.info(f"   YAML: {map_files['yaml']}")
        logger.info(f"   PGM:  {map_files['pgm']}")
    except FileNotFoundError:
        logger.warning(
            "⚠️  gazebo_ros2_2dmap_plugin not found. Skipping map generation."
        )
        print(
            f"{Style.YELLOW}   ⚠️  Map generation skipped (plugin not available){Style.ENDC}"
        )
    except RuntimeError as e:
        logger.error(f"Map generation failed: {str(e)}")
        print(f"{Style.YELLOW}   ⚠️  Map generation failed: {str(e)}{Style.ENDC}")
    except Exception as e:
        logger.warning(f"Map generation error: {str(e)}")
        print(
            f"{Style.YELLOW}   ⚠️  Map generation error (continuing anyway){Style.ENDC}"
        )

    # Get world statistics for summary
    room_count = len(generator.rooms)
    object_count = sum(
        1
        for m in generator.models
        if m.category != "structure" and m.name != "ground_plane"
    )

    # User-friendly success message
    print(f"\n{Style.GREEN}{'━' * 50}{Style.ENDC}")
    print(f"{Style.GREEN}{Style.BOLD}✓ World Generated Successfully!{Style.ENDC}")
    print(f"{Style.GREEN}{'━' * 50}{Style.ENDC}")
    print(f"\n📊 {Style.BOLD}Summary:{Style.ENDC}")
    print(f"   Rooms:   {room_count}")
    print(f"   Objects: {object_count}")
    print(f"\n📁 {Style.BOLD}Output:{Style.ENDC}")
    print(f"   World: {Style.CYAN}{final_world}{Style.ENDC}")
    print(f"   Log:   {Style.CYAN}{paths['log_file']}{Style.ENDC}")
    if map_files:
        print(f"   Map:   {Style.CYAN}{map_files['yaml']}{Style.ENDC}")
        print(f"          {Style.CYAN}{map_files['pgm']}{Style.ENDC}")
    print(f"\n🚀 {Style.BOLD}Launch:{Style.ENDC}")
    print(f"   {Style.YELLOW}gazebo {final_world}{Style.ENDC}\n")

    return 0


def generate_occupancy_map(world_file, output_dir):
    """
    Generate an occupancy map from a Gazebo world file using gazebo_ros2_2dmap_plugin.

    Args:
        world_file: Path to the .sdf world file
        output_dir: Directory to save the occupancy map

    Returns:
        dict: Paths to generated map files {'yaml': path, 'pgm': path}

    Raises:
        RuntimeError: If map generation fails
    """
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run(
        [
            "ros2",
            "run",
            "gazebo_ros2_2dmap_plugin",
            "generate_map.sh",
            world_file,
            output_dir,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode == 0:
        map_name = os.path.splitext(os.path.basename(world_file))[0]
        return {
            "yaml": f"{output_dir}/{map_name}.yaml",
            "pgm": f"{output_dir}/{map_name}.pgm",
        }
    raise RuntimeError(result.stderr)


def test_llm_connection(llm_server, model):
    """Test connection to the LLM server."""
    print(f"\n{Style.CYAN}Testing LLM connection...{Style.ENDC}")
    print(f"  Server: {Style.YELLOW}{llm_server}{Style.ENDC}")
    print(f"  Model:  {Style.YELLOW}{model}{Style.ENDC}\n")

    try:
        llm = OpenAICompatibleInterface(server_url=llm_server, model_name=model)

        # Send a simple test query
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Respond with 'Connection successful' if you receive this.",
            },
        ]

        response = llm.query(test_messages, temperature=0.1)

        if response:
            print(f"{Style.GREEN}✓ Connection successful!{Style.ENDC}")
            print(f"{Style.DIM}Response: {response[:100]}...{Style.ENDC}\n")
            return True
        else:
            print(f"{Style.RED}✗ Connection failed: Empty response{Style.ENDC}\n")
            return False

    except Exception as e:
        print(f"{Style.RED}✗ Connection failed: {e}{Style.ENDC}\n")
        return False


def refine_world_interactive(
    llm_server, model, debug=False, preloaded_refiner=None, current_world_path=None
):
    """Interactive world refinement mode.

    Args:
        llm_server: LLM server URL
        model: Model name
        debug: Debug mode flag
        preloaded_refiner: Optional pre-loaded WorldRefiner instance (skips world selection)
        current_world_path: Optional path to the current world file being refined
    """
    from gazebo_world_generator.src.world.refiner import WorldRefiner

    print(
        f"\n{Style.BLUE}{Style.BOLD}╔════════════════════════════════════════════════╗{Style.ENDC}"
    )
    print(
        f"{Style.BLUE}║          {Style.ENDC}🔧 {Style.BOLD}World Refinement Mode{Style.ENDC}              {Style.BLUE}║{Style.ENDC}"
    )
    print(
        f"{Style.BLUE}{Style.BOLD}╚════════════════════════════════════════════════╝{Style.ENDC}\n"
    )

    worlds_dir = Path("generated_worlds/worlds")

    # If a refiner is already loaded, skip world selection
    if preloaded_refiner and current_world_path:
        refiner = preloaded_refiner
        selected_world = current_world_path
        print(f"{Style.GREEN}✓ Refining: {Path(selected_world).name}{Style.ENDC}")
    else:
        # List available world files
        if not worlds_dir.exists():
            print(f"{Style.RED}✗ No generated worlds found.{Style.ENDC}")
            print(
                f"{Style.YELLOW}  Generate a world first using option 1.{Style.ENDC}\n"
            )
            return 1

        world_files = sorted(
            worlds_dir.glob("*.sdf"), key=lambda p: p.stat().st_mtime, reverse=True
        )

        if not world_files:
            print(f"{Style.RED}✗ No world files found in {worlds_dir}{Style.ENDC}\n")
            return 1

        # Show available worlds
        print(f"{Style.CYAN}Available worlds:{Style.ENDC}\n")
        for i, world_file in enumerate(world_files[:10], 1):  # Show last 10
            mtime = datetime.fromtimestamp(world_file.stat().st_mtime)
            print(f"  {Style.BOLD}{i}.{Style.ENDC} {world_file.name}")
            print(
                f"     {Style.DIM}Created: {mtime.strftime('%Y-%m-%d %H:%M:%S')}{Style.ENDC}"
            )

        # Select world
        print(
            f"\n{Style.CYAN}Select world to refine (1-{len(world_files[:10])}), or press Enter for most recent:{Style.ENDC}"
        )
        selection = input(f"{Style.YELLOW}> {Style.ENDC}").strip()

        if selection == "":
            selected_world = world_files[0]
        else:
            try:
                idx = int(selection) - 1
                if 0 <= idx < len(world_files[:10]):
                    selected_world = world_files[idx]
                else:
                    print(f"{Style.RED}✗ Invalid selection{Style.ENDC}\n")
                    return 1
            except ValueError:
                print(f"{Style.RED}✗ Invalid input{Style.ENDC}\n")
                return 1

        print(f"\n{Style.GREEN}✓ Selected: {selected_world.name}{Style.ENDC}")

        # Initialize refiner
        try:
            llm_config = LLMConfig(server_url=llm_server, model_name=model)
            validated_config = ValidatedConfig(llm=llm_config)
            llm = OpenAICompatibleInterface(server_url=llm_server, model_name=model)
            generator = WorldGenerator(llm, config=validated_config)
            refiner = WorldRefiner(llm, generator)

            # Load world
            if not refiner.load_world(selected_world):
                print(f"{Style.RED}✗ Failed to load world file{Style.ENDC}\n")
                return 1
        except Exception as e:
            print(f"\n{Style.RED}✗ Error loading world: {e}{Style.ENDC}")
            logger.error(f"Error loading world: {e}", exc_info=True)
            return 1

    # Show world summary
    metadata = refiner.world_metadata
    print(f"\n{Style.CYAN}World Summary:{Style.ENDC}")
    print(f"  Rooms:   {metadata.get('rooms', 'Unknown')}")
    print(f"  Objects: {metadata.get('objects', 'Unknown')}")

    # Get refinement request
    print(f"\n{Style.CYAN}Describe the changes you want to make:{Style.ENDC}")
    print(f"{Style.DIM}Examples:{Style.ENDC}")
    print(f"{Style.DIM}  - Add 2 more desks and chairs{Style.ENDC}")
    print(f"{Style.DIM}  - Remove all bookshelves{Style.ENDC}")
    print(f"{Style.DIM}  - Move desks closer to the entrance{Style.ENDC}\n")

    refinement_request = input(f"{Style.YELLOW}> {Style.ENDC}").strip()

    if not refinement_request:
        print(f"{Style.RED}✗ Refinement request cannot be empty{Style.ENDC}\n")
        return 1

    # Generate output filename based on original world name
    original_stem = Path(selected_world).stem
    if original_stem.endswith("_refined"):
        original_stem = original_stem[:-8]
    output_file = worlds_dir / f"{original_stem}_refined.sdf"

    # Apply refinement
    print(f"\n{Style.CYAN}Processing refinement...{Style.ENDC}")
    success, message = refiner.refine(refinement_request, output_file)

    if success:
        print(f"\n{Style.GREEN}✓ {message}{Style.ENDC}\n")

        # Ask if user wants to refine again
        print(f"{Style.CYAN}Refine this world again? (y/n):{Style.ENDC}")
        again = input(f"{Style.YELLOW}> {Style.ENDC}").strip().lower()

        if again == "y":
            # Reload the refined world and continue
            refiner.load_world(output_file)
            return refine_world_interactive(
                llm_server,
                model,
                debug,
                preloaded_refiner=refiner,
                current_world_path=output_file,
            )

        return 0
    else:
        print(f"\n{Style.RED}✗ {message}{Style.ENDC}\n")
        return 1


def show_main_menu():
    """Display the main interactive menu."""
    print(
        f"\n{Style.BLUE}{Style.BOLD}╔════════════════════════════════════════════════╗{Style.ENDC}"
    )
    print(
        f"{Style.BLUE}║           {Style.ENDC}🌍 {Style.BOLD}Gazebo World Generator{Style.ENDC}            {Style.BLUE}║{Style.ENDC}"
    )
    print(
        f"{Style.BLUE}{Style.BOLD}╚════════════════════════════════════════════════╝{Style.ENDC}\n"
    )

    print(f"{Style.CYAN}Select an option:{Style.ENDC}\n")
    print(f"  {Style.BOLD}1.{Style.ENDC} Generate new world")
    print(f"  {Style.BOLD}2.{Style.ENDC} Refine existing world")
    print(f"  {Style.BOLD}3.{Style.ENDC} Test LLM connection")
    print(f"  {Style.BOLD}4.{Style.ENDC} Exit\n")


def interactive_mode(llm_server, model, debug=False):
    """Main interactive mode with menu."""
    while True:
        try:
            show_main_menu()
            choice = input(f"{Style.YELLOW}> {Style.ENDC}").strip()

            if choice == "1":
                # Generate new world
                print(
                    f"\n{Style.CYAN}Describe the world you want to generate:{Style.ENDC}"
                )
                print(f"{Style.DIM}Examples:{Style.ENDC}")
                print(f"{Style.DIM}  - Office with 6 desks and chairs{Style.ENDC}")
                print(
                    f"{Style.DIM}  - A 12m x 10m warehouse with 8 storage racks{Style.ENDC}"
                )
                print(
                    f"{Style.DIM}  - Two offices connected by a corridor{Style.ENDC}\n"
                )

                description = input(f"{Style.YELLOW}> {Style.ENDC}").strip()
                if not description:
                    print(f"{Style.RED}✗ Description cannot be empty{Style.ENDC}")
                    continue

                # Ask for world name
                print(
                    f"\n{Style.CYAN}Enter a name for this world (press Enter for default timestamp name):{Style.ENDC}"
                )
                print(
                    f"{Style.DIM}  Examples: office_layout, warehouse_v2, my_test_world{Style.ENDC}\n"
                )
                world_name_input = input(f"{Style.YELLOW}> {Style.ENDC}").strip()

                if world_name_input:
                    # Sanitize the world name (remove invalid characters)
                    world_name = re.sub(r"[^\w\-]", "_", world_name_input)
                    output_path = Path(f"{world_name}.sdf")
                else:
                    # Use timestamp as default
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = Path(f"world_{timestamp}.sdf")

                result = run_generation(
                    description, output_path, llm_server, model, debug
                )

                if result == 0:
                    # Ask if user wants to refine
                    print(
                        f"\n{Style.CYAN}Would you like to refine this world? (y/n):{Style.ENDC}"
                    )
                    refine_choice = (
                        input(f"{Style.YELLOW}> {Style.ENDC}").strip().lower()
                    )

                    if refine_choice == "y":
                        # Find the just-created world
                        worlds_dir = Path("generated_worlds/worlds")
                        world_files = sorted(
                            worlds_dir.glob("*.sdf"),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )
                        if world_files:
                            # Initialize refiner with the new world
                            from gazebo_world_generator.src.world.refiner import (
                                WorldRefiner,
                            )

                            llm_config = LLMConfig(server_url=llm_server, model_name=model)
                            validated_config = ValidatedConfig(llm=llm_config)
                            llm = OpenAICompatibleInterface(
                                server_url=llm_server, model_name=model
                            )
                            generator = WorldGenerator(llm, config=validated_config)
                            refiner = WorldRefiner(llm, generator)

                            # Load the just-generated world
                            if refiner.load_world(world_files[0]):
                                # Enter refinement loop directly (skip world selection)
                                refine_world_interactive(
                                    llm_server,
                                    model,
                                    debug,
                                    preloaded_refiner=refiner,
                                    current_world_path=world_files[0],
                                )
                            else:
                                print(
                                    f"{Style.RED}✗ Failed to load generated world for refinement{Style.ENDC}\n"
                                )

            elif choice == "2":
                # Refine existing world
                refine_world_interactive(llm_server, model, debug)

            elif choice == "3":
                # Test LLM connection
                test_llm_connection(llm_server, model)

            elif choice == "4":
                # Exit
                print(f"\n{Style.GREEN}Goodbye!{Style.ENDC}\n")
                return 0

            else:
                print(f"{Style.RED}✗ Invalid option. Please select 1-4.{Style.ENDC}")

        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{Style.YELLOW}⚠ Session cancelled by user.{Style.ENDC}")
            return 1


def main():
    """Main entry point for the world generator."""
    args = parse_arguments()
    setup_logging(args.debug)

    try:
        if args.description is None:
            # --- INTERACTIVE MODE WITH MENU ---
            return interactive_mode(args.llm_server, args.model, args.debug)
        else:
            # --- ARGUMENT MODE (Direct generation) ---
            return run_generation(
                args.description, args.output, args.llm_server, args.model, args.debug
            )

    except KeyboardInterrupt:
        print(f"\n{Style.YELLOW}⚠ Generation cancelled by user.{Style.ENDC}")
        logger.info("\nGeneration cancelled by user.")
        return 1
    except Exception as e:
        print(
            f"\n{Style.RED}✗ Critical error occurred. Check logs for details.{Style.ENDC}"
        )
        logger.error(
            "A critical error occurred during world generation.", exc_info=True
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
