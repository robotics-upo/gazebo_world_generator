"""Deterministic, offline placement baseline for 10, 50, and 100 objects.

Without an LLM the layout designer falls back to its seeded sampler, so this
measures the sampler, the checks and model creation.
"""

import json
import math
import os
import statistics
import time
from pathlib import Path

from gazebo_world_generator.src.core.data_models import Room
from gazebo_world_generator.src.placement.engine import NaturalPlacementEngine
from gazebo_world_generator.src.config.validated_settings import PlacementConfig
from gazebo_world_generator.src.placement.checks import OVERLAP_TOLERANCE, footprint, penetration


class FixtureCatalog:
    """Resolve every requested rack to the local fixture model."""

    def find_best_model(self, _object_type, _room_type):
        return "model://test_crate"


def overlaps(first, second, dims, offset):
    """True when the rotated footprints of two placed models intersect."""
    def box(model):
        yaw = model.pose["yaw"]
        x = model.pose["x"] + offset[0] * math.cos(yaw) - offset[1] * math.sin(yaw)
        y = model.pose["y"] + offset[0] * math.sin(yaw) + offset[1] * math.cos(yaw)
        return footprint(x, y, dims, yaw)
    return min(penetration(box(first), box(second))) > OVERLAP_TOLERANCE


def measure(count):
    room = Room(name="Warehouse", type="warehouse",
                dimensions={"width": 30.0, "length": 30.0, "height": 3.0},
                position={"x": 0.0, "y": 0.0, "z": 0.0},
                objects=[{"type": "storage_rack", "count": count}])
    timings = []
    placed = []
    for _ in range(3):
        engine = NaturalPlacementEngine(FixtureCatalog(),
                                        placement_config=PlacementConfig(random_seed=7))
        names = iter(f"rack_{index}" for index in range(count))
        start = time.perf_counter()
        placed = engine.place_all_objects([room], lambda _type: next(names))
        timings.append(time.perf_counter() - start)
    dims = engine.get_actual_model_dimensions("storage_rack", "test_crate", room.type)
    offset = engine.model_offsets_cache.get(f"storage_rack_test_crate_{room.type}", (0.0, 0.0, 0.0))
    collisions = sum(overlaps(first, second, dims, offset)
                     for index, first in enumerate(placed)
                     for second in placed[index + 1:])
    if len(placed) != count or collisions:
        raise AssertionError(f"Placement regression: {len(placed)}/{count} placed, "
                             f"{collisions} overlaps")
    return {"requested": count, "placed": len(placed),
            "median_seconds": round(statistics.median(timings), 6),
            "llm_calls": 0, "overlaps": collisions}


if __name__ == "__main__":
    fixture_models = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "models"
    os.environ["GAZEBO_MODEL_PATH"] = os.pathsep.join(
        part for part in (str(fixture_models), os.environ.get("GAZEBO_MODEL_PATH", "")) if part)
    print(json.dumps([measure(count) for count in (10, 50, 100)], indent=2))
