"""Console formatting for generator progress messages."""

import logging
import re
import sys
import threading
import time

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

        elif msg.startswith("Interpreted room"):
            print(f"    {Style.DIM}•{Style.ENDC} {msg.removeprefix('Interpreted room ')}")

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
                f"    {Style.DIM}├─{Style.ENDC} {Style.GREEN}✓{Style.ENDC} Models resolved"
            )
            for obj_type, model in self.model_selections:
                print(f"      {Style.DIM}• {obj_type}: {model}{Style.ENDC}")
            self.model_selections = []

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

        # LLM layout design rounds
        elif msg.startswith("Layout design round"):
            match = re.search(r"round (\d+)/(\d+)", msg)
            label = f" (round {match.group(1)}/{match.group(2)})" if match else ""
            self._start_spinner(f"Designing layout with LLM{label}...", indent="    ")
            self._start_phase("layout")

        elif msg.startswith("Layout round"):
            self._stop_spinner()
            duration = self._end_phase("layout")
            time_str = (
                f" {Style.DIM}({self._format_duration(duration)}){Style.ENDC}"
                if duration
                else ""
            )
            clean = re.search(r": 0 problem", msg) is not None
            icon = f"{Style.GREEN}✓{Style.ENDC}" if clean else f"{Style.YELLOW}·{Style.ENDC}"
            print(f"    {Style.DIM}├─{Style.ENDC} {icon} {msg.removeprefix('Layout ')}{time_str}")

        elif msg.startswith("Layout matches request"):
            print(f"    {Style.DIM}├─{Style.ENDC} {Style.GREEN}✓{Style.ENDC} Layout matches request")

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


