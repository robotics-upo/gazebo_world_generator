"""Deterministic, offline placement baseline for 10, 50, and 100 objects."""

import json
import os
import statistics
import time
from pathlib import Path

from gazebo_world_generator.src.core.data_models import Room
from gazebo_world_generator.src.placement.engine import NaturalPlacementEngine
from gazebo_world_generator.src.config.validated_settings import PlacementConfig


class FixtureCatalog:
    """Resolve every requested rack to the local fixture model."""

    def find_best_model(self, _object_type, _room_type):
        return "model://test_crate"


def overlaps(first, second, width=0.8, length=0.4):
    return (abs(first.pose["x"] - second.pose["x"]) < width and
            abs(first.pose["y"] - second.pose["y"]) < length)


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
    collisions = sum(overlaps(first, second)
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
