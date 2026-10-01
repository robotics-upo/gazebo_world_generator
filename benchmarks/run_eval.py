"""
Generate a suite of worlds with the real LLM, render them in Gazebo, and report.

    python3 benchmarks/run_eval.py --seeds 1 2 [--only office_2 robotics_lab] [--simulator classic]
        [--wait-for-server 600]

For each prompt and seed it runs `generate_world`, parses the design rounds and
remaining layout issues from the world's log, measures a few probes from the
world file, and renders overview/per-room images with `render_world`. The
report (report.md + results.json) links the renders; the renders, inspected by
eye, are the acceptance signal. Probes only measure: they use model geometry,
never the fronts the LLM declared.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gazebo_world_generator.render_world import render_world  # noqa: E402
from gazebo_world_generator.src.models.visual_quality import model_shape  # noqa: E402
from gazebo_world_generator.src.placement.checks import _ray_hits  # noqa: E402

SEATS = ("chair",)
WORK_SURFACES = ("desk", "table")
MAX_SEAT_GAP = 0.45  # metres between a seat's footprint and its desk edge


def model_dirs():
    paths = [Path.home() / ".gazebo" / "models", Path("/data/models"), Path("/usr/share/gazebo-11/models")]
    for variable in ("GAZEBO_MODEL_PATH", "GZ_SIM_RESOURCE_PATH"):
        paths += [Path(p) for p in os.environ.get(variable, "").split(os.pathsep) if p]
    return [p for p in paths if p.is_dir()]


def placed_objects(world_file):
    """Includes with world pose and measured geometry (footprint centre, extents, backrest)."""
    world = ET.parse(world_file).getroot().find("world")
    bases = model_dirs()
    shapes = {}
    objects = []
    for include in world.findall("include"):
        uri = include.findtext("uri") or ""
        directory = uri.removeprefix("model://")
        if directory not in shapes:
            model_dir = next((b / directory for b in bases if (b / directory / "model.sdf").is_file()), None)
            shapes[directory] = model_shape(model_dir, bases) if model_dir else None
        triangles = shapes[directory]
        pose = [float(v) for v in (include.findtext("pose") or "0 0 0 0 0 0").split()]
        entry = {"name": include.findtext("name"), "x": pose[0], "y": pose[1], "yaw": pose[5]}
        if triangles:
            points = [p for t in triangles for p in t]
            xs, ys, zs = zip(*points)
            entry["centre"] = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
            entry["half"] = ((max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2)
            top = max(zs)
            tall = [(x, y) for x, y, z in points if z > 0.75 * top]
            entry["tall"] = (sum(x for x, _ in tall) / len(tall), sum(y for _, y in tall) / len(tall))
        objects.append(entry)
    return objects


def _to_world(obj, local):
    c, s = math.cos(obj["yaw"]), math.sin(obj["yaw"])
    return (obj["x"] + local[0] * c - local[1] * s, obj["y"] + local[0] * s + local[1] * c)


def _to_local(obj, point):
    c, s = math.cos(obj["yaw"]), math.sin(obj["yaw"])
    dx, dy = point[0] - obj["x"], point[1] - obj["y"]
    return (dx * c + dy * s, -dx * s + dy * c)


def seat_probe(objects):
    """Per seat: at a long side of the nearest work surface, backrest away from it."""
    surfaces = [o for o in objects if o["name"].startswith(WORK_SURFACES) and "half" in o]
    results = []
    for seat in (o for o in objects if o["name"].startswith(SEATS) and "half" in o):
        centre = _to_world(seat, seat["centre"])
        if not surfaces:
            results.append({"seat": seat["name"], "ok": False, "why": "no desk/table"})
            continue
        surface = min(surfaces, key=lambda s: math.dist(_to_world(s, s["centre"]), centre))
        local = _to_local(surface, centre)
        local = (local[0] - surface["centre"][0], local[1] - surface["centre"][1])
        hx, hy = surface["half"]
        seat_depth = min(seat["half"])
        # Which side of the surface the seat is on, and whether that side is a long one.
        long_axis = 0 if hx >= hy else 1
        across = abs(local[1 - long_axis])
        along = abs(local[long_axis])
        half_long, half_short = (hx, hy) if long_axis == 0 else (hy, hx)
        at_long_side = along <= half_long and across - half_short - seat_depth <= MAX_SEAT_GAP
        # Only a desk has a user side; seats may stand at any side of a table (meetings).
        if not surface["name"].startswith("desk"):
            gap_x = abs(local[0]) - hx - seat_depth
            gap_y = abs(local[1]) - hy - seat_depth
            at_long_side = max(gap_x, gap_y) <= MAX_SEAT_GAP and min(abs(local[0]) - hx, abs(local[1]) - hy) <= 0.3
        # Backrest direction from measured geometry: seat centre -> tall part, rotated into the world.
        back_local = (seat["tall"][0] - seat["centre"][0], seat["tall"][1] - seat["centre"][1])
        back_heading = math.atan2(back_local[1], back_local[0]) + seat["yaw"]
        surface_centre = _to_world(surface, surface["centre"])
        to_surface = math.atan2(surface_centre[1] - centre[1], surface_centre[0] - centre[0])
        facing_error = abs((to_surface - (back_heading + math.pi) + math.pi) % (2 * math.pi) - math.pi)
        # Facing = a line straight out of the seat side (opposite the measured backrest) hits the surface.
        target = {"x": surface_centre[0], "y": surface_centre[1], "yaw": surface["yaw"],
                  "dims": (2 * surface["half"][0], 2 * surface["half"][1])}
        facing = (_ray_hits(centre[0], centre[1], back_heading + math.pi, target, 3.0)
                  if math.hypot(*back_local) > 0.05 else None)
        ok = at_long_side and facing is not False
        why = [] if ok else [w for w, bad in (("not at a usable side", not at_long_side),
                                              (f"faces {math.degrees(facing_error):.0f} deg off", facing is False)) if bad]
        results.append({"seat": seat["name"], "surface": surface["name"], "ok": ok, "why": ", ".join(why)})
    return results


def log_metrics(log_file):
    text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""
    rooms = {}
    current = None
    for line in text.splitlines():
        match = re.search(r"Placing \d+ objects in '([^']+)'", line)
        if match:
            current = match.group(1)
            rooms[current] = {"rounds": 0, "issues": []}
        match = re.search(r"Layout round (\d+)(?:/\d+)?: (\d+) problem", line)
        if match and current:
            rooms[current]["rounds"] = int(match.group(1))
        match = re.search(r"Layout issue in '([^']+)': (.*)", line)
        if match:
            rooms.setdefault(match.group(1), {"rounds": 0, "issues": []})["issues"].append(match.group(2))
    return rooms


def run_one(prompt, seed, out_dir, simulator, timeout):
    name = f"{prompt['id']}_s{seed}"
    world = out_dir / "worlds" / f"{name}.sdf"
    command = [sys.executable, "-m", "gazebo_world_generator.gazebo_world_generator",
               "--description", prompt["description"], "--output", str(world), "--seed", str(seed)]
    if simulator:
        command += ["--simulator", simulator]
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    entry = {"id": prompt["id"], "seed": seed, "world": str(world),
             "seconds": round(time.time() - started, 1), "generated": result.returncode == 0 and world.exists()}
    base = Path(os.environ.get("GAZEBO_WORLD_GEN_OUTPUT__BASE_DIRECTORY", "generated_worlds"))
    entry["rooms"] = log_metrics(base / "logs" / f"{name}.log")
    if not entry["generated"]:
        entry["error"] = (result.stderr or result.stdout)[-800:]
        return entry
    entry["seats"] = seat_probe(placed_objects(world))
    try:
        # Headless Harmonic rendering (ogre2 without a GPU) does not produce frames in the
        # container; placement does not depend on the renderer, so Classic renders both.
        renderer = "classic" if shutil.which("gzserver") else (simulator or "classic")
        entry["renders"] = render_world(world, out_dir / "renders", renderer)
    except (RuntimeError, OSError) as exc:
        entry["render_error"] = str(exc)[-500:]
    return entry


def write_report(results, out_dir):
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    lines = ["# Evaluation report", "",
             "| world | ok | s | rooms (rounds / issues) | seats ok | renders |", "|---|---|---|---|---|---|"]
    for r in results:
        rooms = "; ".join(f"{room}: {m['rounds']}/{len(m['issues'])}" for room, m in r.get("rooms", {}).items())
        seats = r.get("seats", [])
        seat_text = f"{sum(s['ok'] for s in seats)}/{len(seats)}" if seats else "-"
        renders = " ".join(f"[{k}]({Path(v).relative_to(out_dir)})" for k, v in r.get("renders", {}).items())
        lines.append(f"| {r['id']} s{r['seed']} | {'yes' if r['generated'] else 'NO'} | {r['seconds']} | "
                     f"{rooms} | {seat_text} | {renders or r.get('render_error', '')[:60]} |")
    failures = [(r, s) for r in results for s in r.get("seats", []) if not s["ok"]]
    if failures:
        lines += ["", "## Seat problems", ""]
        lines += [f"- {r['id']} s{r['seed']}: {s['seat']} -> {s.get('surface', '?')}: {s['why']}" for r, s in failures]
    issues = [(r, room, i) for r in results for room, m in r.get("rooms", {}).items() for i in m["issues"]]
    if issues:
        lines += ["", "## Remaining layout issues", ""]
        lines += [f"- {r['id']} s{r['seed']} {room}: {i}" for r, room, i in issues]
    errors = [r for r in results if not r["generated"]]
    if errors:
        lines += ["", "## Generation failures", ""]
        lines += [f"- {r['id']} s{r['seed']}: `{r.get('error', '').strip()[-300:]}`" for r in errors]
    generated = [r for r in results if r["generated"]]
    seats = [s for r in results for s in r.get("seats", [])]
    lines += ["", "## Summary", "",
              f"- generated: {len(generated)}/{len(results)}",
              f"- seats at a desk, facing it: {sum(s['ok'] for s in seats)}/{len(seats)}",
              f"- rooms with remaining issues: "
              f"{sum(1 for r in results for m in r.get('rooms', {}).values() if m['issues'])}"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def server_ready(probe_timeout=60) -> bool:
    """True when the configured LLM answers a one-token completion (/health misses engine hangs)."""
    from gazebo_world_generator.src.config.validated_settings import ValidatedConfig
    llm = ValidatedConfig.from_multiple_sources().llm
    body = json.dumps({"model": llm.model_name, "max_tokens": 1,
                       "messages": [{"role": "user", "content": "ok"}]}).encode()
    request = urllib.request.Request(f"{llm.server_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=probe_timeout) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


def wait_for_server(max_wait) -> bool:
    """Poll the LLM until it answers or max_wait seconds pass."""
    deadline = time.time() + max_wait
    waiting = False
    while not server_ready():
        if time.time() >= deadline:
            return False
        if not waiting:
            print(f"LLM server not answering; waiting up to {max_wait:.0f}s", flush=True)
            waiting = True
        time.sleep(20)
    if waiting:
        print("LLM server is back", flush=True)
    return True


def _consecutive_timeouts(results) -> int:
    count = 0
    for result in reversed(results):
        if result.get("error") != "timeout" and "timed out" not in str(result.get("error", "")):
            break
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prompts", default=str(REPO / "benchmarks" / "eval_prompts.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--only", nargs="*", help="prompt ids to run")
    parser.add_argument("--simulator", choices=("classic", "harmonic"))
    parser.add_argument("--out", default=None, help="default: generated_worlds/eval/<timestamp>")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--wait-for-server", type=float, default=0, metavar="SECONDS",
                        help="before each world, wait up to SECONDS for the LLM to answer; "
                             "a world that times out while the server is down is retried once")
    args = parser.parse_args()

    prompts = yaml.safe_load(Path(args.prompts).read_text())
    if args.only:
        prompts = [p for p in prompts if p["id"] in args.only]
    out_dir = Path(args.out or f"generated_worlds/eval/{time.strftime('%Y%m%d-%H%M%S')}")
    (out_dir / "worlds").mkdir(parents=True, exist_ok=True)
    results = []
    total = len(prompts) * len(args.seeds)
    for prompt in prompts:
        if _consecutive_timeouts(results) >= 2:
            print("Stopping: the last two worlds timed out (is the LLM server reachable?)", flush=True)
            break
        for seed in args.seeds:
            print(f"[{len(results) + 1}/{total}] {prompt['id']} seed {seed}", flush=True)
            attempts = 2 if args.wait_for_server else 1
            for attempt in range(attempts):
                if args.wait_for_server and not wait_for_server(args.wait_for_server):
                    print(f"Stopping: LLM server down for over {args.wait_for_server:.0f}s", flush=True)
                    write_report(results, out_dir)
                    print(f"Report: {out_dir / 'report.md'}")
                    return
                try:
                    result = run_one(prompt, seed, out_dir, args.simulator, args.timeout)
                except subprocess.TimeoutExpired:
                    result = {"id": prompt["id"], "seed": seed, "generated": False,
                              "seconds": args.timeout, "error": "timeout"}
                # A timeout caused by a server outage says nothing about the generator: retry it.
                if result.get("error") != "timeout" or attempt + 1 == attempts or server_ready():
                    break
                print("World timed out while the server was down; retrying", flush=True)
            results.append(result)
            write_report(results, out_dir)
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
