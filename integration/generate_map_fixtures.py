"""Generate and map deterministic worlds inside a Classic or Harmonic test image."""

import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from gazebo_world_generator.src.config.validated_settings import ValidatedConfig
from gazebo_world_generator.src.core.data_models import GazeboModel
from gazebo_world_generator.src.world.generator import WorldGenerator
from gazebo_world_generator.src.world.refiner import WorldRefiner
from gazebo_world_generator.src.world.refinement import OperationExecutor


class FixtureLLM:
    def __init__(self, rooms):
        self.rooms = rooms

    def parse_room_description(self, _description, _model_db):
        return {"rooms": self.rooms}


def generate_world(layout: str, simulator: str, output: Path) -> WorldGenerator:
    rooms = [{"name": "Office", "type": "office",
              "dimensions": {"width": 6, "length": 6}, "objects": []}]
    if layout == "connected":
        rooms.append({"name": "Storage", "type": "storage",
                      "dimensions": {"width": 5, "length": 5}, "objects": [],
                      "connections": {"Office": "west"}})
    elif layout == "warehouse":
        rooms = [{"name": "Warehouse", "type": "warehouse",
                  "dimensions": {"width": 12, "length": 10},
                  "objects": [{"type": "shelf", "count": 1}]}]
    generator = WorldGenerator(FixtureLLM(rooms),
                               config=ValidatedConfig(simulator=simulator))
    if layout == "warehouse":
        generator.placement_engine.place_all_objects = lambda *_: [GazeboModel(
            name="shelf_0", model_path="model://test_crate", category="furniture",
            pose={"x": 1, "y": 1, "z": 0.9, "roll": 0, "pitch": 0, "yaw": 0},
            size=[0.8, 0.4, 1.8], static=True)]
    generator.generate_world(layout, str(output))
    return generator


def validate_world(world: Path) -> None:
    result = subprocess.run(["gz", "sdf", "-k", str(world)], capture_output=True,
                            text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"SDF validation failed for {world}: {result.stdout}{result.stderr}")


def smoke_world(world: Path, simulator: str) -> None:
    command = (["gzserver", "--verbose", str(world)] if simulator == "classic" else
               ["gz", "sim", "-s", "-r", "-v", "4", str(world)])
    result = subprocess.run(["timeout", "-k", "2s", "10s", *command],
                            capture_output=True, text=True, timeout=15)
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)
    marker = ("Loading world file [" if simulator == "classic" else
              "World [generated_world] initialized")
    if (result.returncode not in (0, 124) or marker not in output or
            any(error in output for error in ("Unable to find uri", "Unable to find file",
                                               "Error parsing XML", "Failed to load"))):
        raise RuntimeError(f"{simulator} failed to load {world}:\n{output[-4000:]}")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("classic", "harmonic"):
        raise SystemExit("Usage: generate_map_fixtures.py {classic,harmonic} OUTPUT_DIR")
    simulator, output_dir = sys.argv[1], Path(sys.argv[2]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path.home() / ".gazebo/models"
    variable = "GAZEBO_MODEL_PATH" if simulator == "classic" else "GZ_SIM_RESOURCE_PATH"
    os.environ[variable] = os.pathsep.join(
        part for part in (str(model_dir), os.environ.get(variable, "")) if part)
    os.environ["SDF_PATH"] = os.pathsep.join(
        part for part in (str(model_dir), os.environ.get("SDF_PATH", "")) if part)
    source = Path(__file__).resolve().parents[1] / "tests/fixtures/map_room.sdf"
    shutil.copyfile(source, output_dir / "map_room.sdf")
    for layout in ("empty", "connected", "warehouse"):
        generator = generate_world(layout, simulator, output_dir / f"{layout}.sdf")
        if layout == "warehouse":
            refined = output_dir / "refined.sdf"
            shutil.copyfile(output_dir / "warehouse.sdf", refined)
            refiner = WorldRefiner(generator.llm_interface, generator)
            refiner.current_world = ET.parse(refined)
            refiner.executor = OperationExecutor(refiner)
            if not refiner.executor._apply_remove_objects({
                    "operation": "remove", "objects": [{"target": "shelf_0"}]}, refined):
                raise RuntimeError("Refinement did not remove the shelf")
    for name in ("empty", "map_room", "connected", "warehouse", "refined"):
        validate_world(output_dir / f"{name}.sdf")
    for name in ("warehouse", "refined"):
        smoke_world(output_dir / f"{name}.sdf", simulator)
    for name in ("map_room", "connected", "warehouse"):
        world = output_dir / f"{name}.sdf"
        subprocess.run(["ros2", "run", "gazebo_ros2_2dmap_plugin", "generate_map.sh",
                        str(world), str(output_dir)], check=True, timeout=120)
        for extension in (".pgm", ".yaml"):
            result = output_dir / f"{name}{extension}"
            if not result.is_file() or result.stat().st_size == 0:
                raise RuntimeError(f"Missing map output: {result}")


if __name__ == "__main__":
    main()
