"""
Render a generated world in Gazebo from above, one image per room plus an overview.

Used to check results visually (object orientation, blocked doors, sunken or
floating models) against what Gazebo really shows, rather than against the
generator's own assumptions. Cameras are added to a temporary copy of the
world; the world file itself is never changed.

    ros2 run gazebo_world_generator render_world generated_worlds/worlds/lab.sdf --out renders/
"""

import argparse
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image

# (min_x, max_x, min_y, max_y)
Bounds = Tuple[float, float, float, float]

IMAGE_WIDTH, IMAGE_HEIGHT = 1024, 768
HORIZONTAL_FOV = 0.7  # narrow, from higher up, to limit perspective lean of walls and furniture
MARGIN = 1.15
FRAMES_BEFORE_CAPTURE = 3  # the first frames can predate the scene being fully loaded


def room_bounds(world: ET.Element) -> Dict[str, Bounds]:
    """Room extents from the generator's wall models (`<Room>_wall_<side>...`)."""
    rooms: Dict[str, list] = {}
    for model in world.findall("model"):
        name = model.get("name", "")
        if "_wall_" not in name:
            continue
        room = name.split("_wall_")[0]
        values = [float(v) for v in (model.findtext("pose") or "0 0 0 0 0 0").split()] + [0.0] * 6
        size_text = model.findtext(".//collision/geometry/box/size") or model.findtext(".//box/size")
        if not size_text:
            continue
        sx, sy, _ = (float(v) for v in size_text.split())
        yaw = values[5]
        ex = abs(sx / 2 * math.cos(yaw)) + abs(sy / 2 * math.sin(yaw))
        ey = abs(sx / 2 * math.sin(yaw)) + abs(sy / 2 * math.cos(yaw))
        rooms.setdefault(room, []).append((values[0] - ex, values[0] + ex, values[1] - ey, values[1] + ey))
    return {room: (min(b[0] for b in boxes), max(b[1] for b in boxes),
                   min(b[2] for b in boxes), max(b[3] for b in boxes))
            for room, boxes in rooms.items()}


def _camera_height(bounds: Bounds) -> float:
    width, length = bounds[1] - bounds[0], bounds[3] - bounds[2]
    vertical_fov = 2 * math.atan(math.tan(HORIZONTAL_FOV / 2) * IMAGE_HEIGHT / IMAGE_WIDTH)
    return MARGIN * max(width / 2 / math.tan(HORIZONTAL_FOV / 2),
                        length / 2 / math.tan(vertical_fov / 2)) + 1.0


OBLIQUE_PITCH = 0.85  # radians below horizontal for the angled view


def _camera_pose(bounds: Bounds, oblique: bool) -> str:
    """Top view: straight down, image up = north (+y), right = east (+x).

    Oblique view: from the south, looking north and down, so objects show
    their sides (a chair's backrest, a desk's legs) as well as their tops.
    """
    cx, cy = (bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2
    height = _camera_height(bounds)
    if not oblique:
        return f"{cx:.3f} {cy:.3f} {height:.3f} 0 1.5708 1.5708"
    distance = height * 0.9
    back = distance * math.cos(OBLIQUE_PITCH)
    up = distance * math.sin(OBLIQUE_PITCH)
    return f"{cx:.3f} {cy - back:.3f} {up:.3f} 0 {OBLIQUE_PITCH:.4f} 1.5708"


def _camera_model(name: str, bounds: Bounds, save_dir: Path, simulator: str,
                  oblique: bool = False) -> ET.Element:
    model = ET.Element("model", name=f"render_camera_{name}")
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = _camera_pose(bounds, oblique)
    link = ET.SubElement(model, "link", name="link")
    sensor = ET.SubElement(link, "sensor", name="camera", type="camera")
    ET.SubElement(sensor, "always_on").text = "true"
    ET.SubElement(sensor, "update_rate").text = "2"
    if simulator == "harmonic":
        ET.SubElement(sensor, "topic").text = f"render/{name}"
    camera = ET.SubElement(sensor, "camera")
    ET.SubElement(camera, "horizontal_fov").text = str(HORIZONTAL_FOV)
    image = ET.SubElement(camera, "image")
    ET.SubElement(image, "width").text = str(IMAGE_WIDTH)
    ET.SubElement(image, "height").text = str(IMAGE_HEIGHT)
    ET.SubElement(image, "format").text = "R8G8B8"
    clip = ET.SubElement(camera, "clip")
    ET.SubElement(clip, "near").text = "0.1"
    ET.SubElement(clip, "far").text = "200"
    save = ET.SubElement(camera, "save", enabled="true")
    ET.SubElement(save, "path").text = str(save_dir)
    return model


def _lower_walls(world: ET.Element, height: float = 0.1) -> None:
    """Cut the generator's walls down in the render copy so they outline rooms without hiding them."""
    for model in world.findall("model"):
        if "_wall_" not in model.get("name", ""):
            continue
        pose = model.find("pose")
        if pose is not None:
            values = (pose.text or "").split()
            if len(values) >= 3:
                values[2] = f"{height / 2:.3f}"
                pose.text = " ".join(values)
        for size in model.iter("size"):
            values = (size.text or "").split()
            if len(values) == 3:
                values[2] = f"{height:.3f}"
                size.text = " ".join(values)


def _frames(directory: Path):
    """Saved frames, oldest first (Classic writes JPEG, gz-sensors PNG)."""
    frames = [path for path in directory.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return sorted(frames, key=lambda path: (path.stat().st_mtime, path.name))


def render_world(world_file, out_dir, simulator: str = "classic",
                 timeout: float = 90.0, rooms: Optional[Dict[str, Bounds]] = None) -> Dict[str, str]:
    """Render `world_file`; return {view name: PNG path} for 'overview' and each room."""
    world_file = Path(world_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = ET.parse(world_file).getroot()
    world = root.find("world")
    rooms = dict(rooms or room_bounds(world))
    overview = ((min(b[0] for b in rooms.values()), max(b[1] for b in rooms.values()),
                 min(b[2] for b in rooms.values()), max(b[3] for b in rooms.values()))
                if rooms else (-10.0, 10.0, -10.0, 10.0))
    # name -> (bounds, oblique)
    views = {"overview": (overview, False)}
    for room, bounds in rooms.items():
        views[room] = (bounds, False)
        views[f"{room}_angled"] = (bounds, True)
    _lower_walls(world)

    with tempfile.TemporaryDirectory(prefix="gwg_render_") as scratch:
        scratch = Path(scratch)
        save_dirs = {}
        for name, (bounds, oblique) in views.items():
            save_dirs[name] = scratch / name
            save_dirs[name].mkdir()
            world.append(_camera_model(name, bounds, save_dirs[name], simulator, oblique))
        if simulator == "harmonic" and not world.findall("plugin[@name='gz::sim::systems::Sensors']"):
            plugin = ET.SubElement(world, "plugin", filename="gz-sim-sensors-system",
                                   name="gz::sim::systems::Sensors")
            ET.SubElement(plugin, "render_engine").text = "ogre2"
        copy = scratch / world_file.name
        ET.ElementTree(root).write(copy, encoding="unicode")

        log_path = scratch / "simulator.log"
        process = _start_simulator(copy, simulator, log_path)
        try:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if all(len(_frames(d)) >= FRAMES_BEFORE_CAPTURE for d in save_dirs.values()):
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.5)
        finally:
            _stop(process)

        log_tail = log_path.read_text(errors="replace")[-2000:] if log_path.exists() else ""
        rendered = {}
        for name, directory in save_dirs.items():
            frames = _frames(directory)
            if not frames:
                continue
            target = out_dir / f"{world_file.stem}_{name}.png"
            Image.open(frames[-1]).save(target)
            rendered[name] = str(target)
    missing = sorted(set(views) - set(rendered))
    if missing:
        raise RuntimeError(f"No frames rendered for: {', '.join(missing)} (timeout {timeout:.0f}s)\n"
                           f"Simulator output:\n{log_tail}")
    return rendered


def _start_simulator(world: Path, simulator: str, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    models = str(Path.home() / ".gazebo" / "models")
    if simulator == "harmonic":
        env["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(filter(None, [env.get("GZ_SIM_RESOURCE_PATH"), models]))
        command = ["gz", "sim", "-s", "-r", "--headless-rendering", str(world)]
    else:
        env["GAZEBO_MODEL_PATH"] = os.pathsep.join(filter(None, [env.get("GAZEBO_MODEL_PATH"), models]))
        # A private master port, so a Gazebo the user has open is left alone.
        env["GAZEBO_MASTER_URI"] = f"http://127.0.0.1:{_free_port()}"
        command = ["gzserver", "--verbose", str(world)]
    if not env.get("DISPLAY") and simulator == "classic" and shutil.which("xvfb-run"):
        command = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24"] + command
    log = open(log_path, "w")
    return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render a generated world from above in Gazebo.")
    parser.add_argument("world", help="SDF world file")
    parser.add_argument("--out", default="renders", help="output directory for PNG files")
    parser.add_argument("--simulator", choices=("classic", "harmonic"),
                        default="harmonic" if os.environ.get("ROS_DISTRO") == "jazzy" else "classic")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args(argv)
    try:
        for name, path in render_world(args.world, args.out, args.simulator, args.timeout).items():
            print(f"{name}: {path}")
    except (RuntimeError, OSError, ET.ParseError) as exc:
        print(f"Render failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
