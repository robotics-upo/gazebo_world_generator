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
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import os
from pathlib import Path

from datetime import datetime

# Import from package structure
from gazebo_world_generator.src.llm.interface import OpenAICompatibleInterface
from gazebo_world_generator.src.world.generator import WorldGenerator
from gazebo_world_generator.src.config.validated_settings import ValidatedConfig, LLMConfig
from gazebo_world_generator.src.utils.output_manager import OutputManager
from gazebo_world_generator.src.utils.console import ConsoleHandler, Style

# Configure root logger
logger = logging.getLogger()

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
        default=None,
        help="Write the world to this exact SDF path; otherwise use the configured output directory",
    )

    parser.add_argument(
        "--llm-server",
        type=str,
        default=None,
        help="Override configured LLM server URL",
    )

    parser.add_argument(
        "--model", type=str, default=None, help="Override configured LLM model name"
    )
    parser.add_argument("--simulator", choices=["classic", "harmonic"], default=None)
    parser.add_argument("--seed", type=int, default=None, help="Placement random seed")

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging for verbose output."
    )

    return parser.parse_args()


def run_generation(
    description: str,
    output_path: Path,
    llm_server: str = None,
    model: str = None,
    debug: bool = False,
    config: ValidatedConfig = None,
):
    """Encapsulates the core world generation logic."""
    overrides = {"llm": {}}
    if llm_server is not None:
        overrides["llm"]["server_url"] = llm_server
    if model is not None:
        overrides["llm"]["model_name"] = model
    validated_config = config or ValidatedConfig.from_multiple_sources(overrides)

    # Validate LLM configuration
    try:
        llm_config = validated_config.llm
        logger.info(f"✓ LLM configuration validated: {llm_config.server_url}")
    except Exception as e:
        logger.error(f"✗ LLM configuration validation failed: {e}")
        print(f"{Style.RED}✗ Invalid LLM configuration: {e}{Style.ENDC}\n")
        return 1

    # Initialize output manager
    output_manager = OutputManager(validated_config.output.base_directory)

    # Get organized output paths
    world_name = output_path.stem if output_path is not None else None
    paths = output_manager.get_output_paths(world_name)
    if output_path is not None:
        paths["world_file"] = Path(output_path).expanduser()

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

    # Create interfaces using validated config (silent to user)
    llm_interface = OpenAICompatibleInterface(
        server_url=validated_config.llm.server_url,
        model_name=validated_config.llm.model_name,
        timeout=validated_config.llm.timeout
    )
    generator = WorldGenerator(llm_interface, config=validated_config)

    logger.info(f'Generating world from description: "{description}"')

    try:
        output_file = generator.generate_world(description, str(paths["world_file"]))
    except Exception as e:
        logger.exception(f"Generation failed: {str(e)}")
        file_handler.close()
        logging.getLogger().removeHandler(file_handler)
        print(
            f"\n{Style.RED}✗ Generation failed. Check log: {paths['log_file']}{Style.ENDC}\n"
        )
        return 1

    final_world = Path(output_file)

    # Detailed logging
    logger.info("✅ World generation completed successfully!")
    logger.info(f"   Output file: {final_world}")
    logger.info(f"   Log file: {paths['log_file']}")

    # Generate occupancy map
    map_files = None
    if validated_config.output.auto_generate_map:
        try:
            logger.info("Generating occupancy map...")
            print(f"  {Style.CYAN}⚙{Style.ENDC}  Generating occupancy map...")
            map_files = generate_occupancy_map(
                str(final_world), output_manager.base_output_dir / "occupancy_maps",
                simulator=validated_config.simulator,
                model_paths=[*validated_config.models.search_paths,
                             validated_config.models.cache_directory],
                seed=generator.map_seed()
            )
            print(f"{Style.GREEN}  ✓ Occupancy map generated successfully{Style.ENDC}")
            logger.info("✅ Occupancy map generated:")
            logger.info(f"   YAML: {map_files['yaml']}")
            logger.info(f"   PGM:  {map_files['pgm']}")
        except FileNotFoundError:
            logger.warning("⚠️  gazebo_ros2_2dmap_plugin not found. Skipping map generation.")
            print(f"{Style.YELLOW}   ⚠️  Map generation skipped (plugin not available){Style.ENDC}")
        except RuntimeError as e:
            logger.error(f"Map generation failed: {str(e)}")
            print(f"{Style.YELLOW}   ⚠️  Map generation failed: {str(e)}{Style.ENDC}")
        except Exception as e:
            logger.warning(f"Map generation error: {str(e)}")
            print(f"{Style.YELLOW}   ⚠️  Map generation error (continuing anyway){Style.ENDC}")
    file_handler.close()
    logging.getLogger().removeHandler(file_handler)

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
    launch = "gazebo" if validated_config.simulator == "classic" else "gz sim"
    print(f"   {Style.YELLOW}{launch} {final_world}{Style.ENDC}\n")

    return 0


def _classic_server_running():
    """The Classic companion script kills all gzserver processes on exit."""
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            if (process / "comm").read_text().strip() == "gzserver" and \
                    "State:\tZ" not in (process / "status").read_text():
                return True
        except OSError:
            continue
    return False


# Classic's map plugin marks a cell occupied by casting rays along its edges.
# Faces lying exactly on those edges (common with round coordinates) can be
# missed, leaving objects open on one side. The map copy is shifted by a
# quarter cell and the saved map origin shifted back, so no face coincides.
# Harmonic gets the same shift so both simulators produce the same map grid.
MAP_SHIFT = 0.0125


def _world_with_map_plugin(world_file, output_dir, simulator, seed, directory, shift=0.0):
    """Copy of the world, moved by `shift` in x and y, with the map plugin configured.

    The plugin flood-fills from `seed` instead of world (0, 0), which may be
    inside an object. The copy keeps the world's file name, so the map is
    named after it.
    """
    x, y = seed[0] + shift, seed[1] + shift
    if simulator == "harmonic":
        output_path = Path(output_dir).resolve() / Path(world_file).stem
        plugin = (f'<plugin filename="gz_2Dmap_system" name="gz::sim::systems::OccupancyMapFromWorld">'
                  f'<map_resolution>0.05</map_resolution><map_height>0.2</map_height>'
                  f'<init_robot_x>{x:.4f}</init_robot_x><init_robot_y>{y:.4f}</init_robot_y>'
                  f'<output_path>{output_path}</output_path></plugin>')
    else:
        plugin = (f"<plugin name='gazebo_occupancy_map' filename='libgazebo_2Dmap_plugin.so'>"
                  f"<init_robot_x>{x:.4f}</init_robot_x><init_robot_y>{y:.4f}</init_robot_y></plugin>")
    root = ET.parse(world_file).getroot()
    world = root.find("world")
    if shift:
        for element in world.findall("model") + world.findall("include"):
            pose = element.find("pose")
            if pose is None:
                pose = ET.SubElement(element, "pose")
            values = [float(value) for value in (pose.text or "").split()] or [0.0] * 6
            values += [0.0] * (6 - len(values))
            values[0] += shift
            values[1] += shift
            pose.text = " ".join(f"{value:.4f}" for value in values)
    world.append(ET.fromstring(plugin))
    copy = Path(directory) / Path(world_file).name
    ET.ElementTree(root).write(copy, encoding="unicode")
    return str(copy)


def _shift_map_origin(yaml_path, shift):
    """Undo the map copy's shift in the saved map's origin."""
    text = Path(yaml_path).read_text()
    match = re.search(r"origin:\s*\[([^\]]*)\]", text)
    if not match:
        return
    values = [float(value) for value in match.group(1).split(",")]
    values[0] -= shift
    values[1] -= shift
    origin = ", ".join(f"{value:.4f}".rstrip("0").rstrip(".") for value in values)
    Path(yaml_path).write_text(text[:match.start()] + f"origin: [{origin}]" + text[match.end():])


def generate_occupancy_map(world_file, output_dir, simulator="classic", model_paths=None,
                           seed=None):
    """
    Generate an occupancy map from a Gazebo world file using gazebo_ros2_2dmap_plugin.

    Args:
        world_file: Path to the .sdf world file
        output_dir: Directory to save the occupancy map
        seed: World (x, y) on open floor where the map's flood fill starts

    Returns:
        dict: Paths to generated map files {'yaml': path, 'pgm': path}

    Raises:
        RuntimeError: If map generation fails
    """
    if simulator == "classic" and _classic_server_running():
        raise RuntimeError("Close the active Gazebo Classic server before generating a map; "
                           "the companion map script terminates gzserver processes")
    os.makedirs(output_dir, exist_ok=True)
    map_name = Path(world_file).stem
    paths = {
        "yaml": str(Path(output_dir) / f"{map_name}.yaml"),
        "pgm": str(Path(output_dir) / f"{map_name}.pgm"),
    }
    prior_mtimes = {key: Path(path).stat().st_mtime_ns if Path(path).exists() else 0
                    for key, path in paths.items()}

    map_environment = os.environ.copy()
    if model_paths:
        variable = "GZ_SIM_RESOURCE_PATH" if simulator == "harmonic" else "GAZEBO_MODEL_PATH"
        configured = os.pathsep.join(str(Path(path).expanduser()) for path in model_paths)
        map_environment[variable] = os.pathsep.join(
            part for part in (configured, map_environment.get(variable, "")) if part)
    shift = MAP_SHIFT
    with tempfile.TemporaryDirectory() as directory:
        map_world = _world_with_map_plugin(world_file, output_dir, simulator,
                                           seed or (0.0, 0.0), directory, shift)
        result = subprocess.run(
            [
                "ros2",
                "run",
                "gazebo_ros2_2dmap_plugin",
                "generate_map.sh",
                map_world,
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=240,
            env=map_environment,
        )

    if result.returncode == 0:
        if all(Path(path).is_file() and Path(path).stat().st_size > 0 and
               Path(path).stat().st_mtime_ns > prior_mtimes[key]
               for key, path in paths.items()):
            if shift:
                _shift_map_origin(paths["yaml"], shift)
            return paths
        raise RuntimeError("Map command succeeded but fresh PGM or YAML output is missing or empty")
    raise RuntimeError((result.stderr or result.stdout).strip() or
                       f"Map command exited with status {result.returncode}")


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


def interactive_mode(llm_server, model, debug=False, config=None):
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
                    base_dir = (config or ValidatedConfig.from_multiple_sources()).output.base_directory
                    output_path = base_dir / "worlds" / f"{world_name}.sdf"
                else:
                    output_path = None

                result = run_generation(
                    description, output_path, llm_server, model, debug, config=config
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
                        worlds_dir = (config or ValidatedConfig.from_multiple_sources()).output.base_directory / "worlds"
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
    overrides = {"llm": {}, "placement": {}}
    if args.llm_server is not None:
        overrides["llm"]["server_url"] = args.llm_server
    if args.model is not None:
        overrides["llm"]["model_name"] = args.model
    if args.simulator is not None:
        overrides["simulator"] = args.simulator
    if args.seed is not None:
        overrides["placement"]["random_seed"] = args.seed
    config = ValidatedConfig.from_multiple_sources(overrides)

    try:
        if args.description is None:
            # --- INTERACTIVE MODE WITH MENU ---
            return interactive_mode(config.llm.server_url, config.llm.model_name, args.debug,
                                    config=config)
        else:
            # --- ARGUMENT MODE (Direct generation) ---
            return run_generation(
                args.description, args.output, args.llm_server, args.model, args.debug,
                config=config
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
