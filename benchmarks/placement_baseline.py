"""Reproducible collision-validation baseline for 10, 50 and 100 objects."""

import json
import time

from gazebo_world_generator.src.utils.collision_detection import CollisionDetector


def measure(count):
    detector = CollisionDetector()
    objects = []
    overlaps = 0
    dimensions = (0.5, 0.5, 0.5)
    start = time.perf_counter()
    for index in range(count):
        position = ((index % 10) * 0.8, (index // 10) * 0.8, 0.0)
        if not detector.validate_position(position, dimensions, objects):
            overlaps += 1
        objects.append({"name": f"object_{index}", "position": position,
                        "dimensions": dimensions})
    elapsed = time.perf_counter() - start
    return {"objects": count, "seconds": round(elapsed, 6),
            "llm_calls": 0, "overlaps": overlaps}


if __name__ == "__main__":
    print(json.dumps([measure(count) for count in (10, 50, 100)], indent=2))
