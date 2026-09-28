"""Checks, rendering and the propose -> check -> revise loop."""

import io
import json
import random

import pytest
from PIL import Image

from gazebo_world_generator.src.core.data_models import Room
from gazebo_world_generator.src.placement.checks import check_layout, item_box, penetration
from gazebo_world_generator.src.placement.designer import LayoutDesigner
from gazebo_world_generator.src.placement.render import render_layout
from gazebo_world_generator.src.prompts.manager import PromptManager
from gazebo_world_generator.src.utils.token_estimator import estimate_tokens

DOORS = [{"side": "west", "x": -6.0, "y": 0.0, "width": 1.6, "clearance_radius": 1.5},
         {"side": "east", "x": 6.0, "y": 0.0, "width": 1.6, "clearance_radius": 1.5}]


def _room(doorways=DOORS):
    room = Room(name="Lab", type="lab", dimensions={"width": 12.0, "length": 8.0, "height": 3},
                position={"x": 0.0, "y": 0.0, "z": 0.0}, purpose="obstacle course")
    room.doorways = list(doorways)
    return room


def _item(item_id, x, y, yaw=0.0, dims=(0.6, 0.6, 0.6), **extra):
    return dict(id=item_id, type=item_id.rsplit("_", 1)[0], x=x, y=y, yaw=yaw, dims=dims,
                front="+x", **extra)


def _names(violations):
    return sorted(v.check for v in violations)


# --- checks --------------------------------------------------------------------------------------

def test_clean_layout_has_no_violations():
    items = [_item("box_0", 0, 2), _item("box_1", 2, -2)]
    assert check_layout(_room(), items, expected_ids=["box_0", "box_1"]) == []


def test_always_on_checks_report_concrete_hints():
    items = [_item("box_0", 0, 0), _item("box_1", 0.3, 0), _item("box_2", 5.8, 3.8),
             _item("box_3", -5.2, 0)]
    violations = check_layout(_room(), items, expected_ids=["box_0", "box_1", "box_2", "box_3", "box_4"])
    assert _names(violations) == ["all_placed", "doorways_clear", "inside_room", "no_overlap"]
    overlap = next(v for v in violations if v.check == "no_overlap")
    assert "move box_0 west >= 0.30 m" == overlap.hint
    assert all(v.always_on for v in violations)


def test_item_on_support_must_fit_on_it():
    desk = _item("desk_0", 0, 2, dims=(1.4, 0.7, 0.75))
    monitor = _item("monitor_0", 0, 2, dims=(0.5, 0.2, 0.4), on="desk_0")
    assert check_layout(_room(), [desk, monitor]) == []
    monitor["x"] = 1.0
    assert _names(check_layout(_room(), [desk, monitor])) == ["no_overlap"]


def test_path_between_doors_detects_a_wall_of_objects():
    wall = [_item(f"barrier_{i}", 0.0, y, dims=(0.6, 1.8, 1.0)) for i, y in enumerate((-3.1, -1.4, 0.3, 2.0, 3.1))]
    violations = check_layout(_room(), wall, [{"check": "path_between_doors", "width": 0.8}])
    assert [v.check for v in violations if not v.always_on] == ["path_between_doors"]
    staggered = [_item(f"box_{i}", x, y) for i, (x, y) in enumerate(((-2, 2), (0, -2), (2, 2)))]
    assert check_layout(_room(), staggered, [{"check": "path_between_doors", "width": 0.8}]) == []


@pytest.mark.parametrize("requirement, items, fails", [
    ({"check": "min_spacing", "width": 1.0}, [_item("a_0", 0, 2), _item("a_1", 1, 2)], True),
    ({"check": "min_spacing", "width": 1.0}, [_item("a_0", 0, 2), _item("a_1", 2, 2)], False),
    ({"check": "near", "a": "chair_0", "b": "desk_0", "max": 0.3},
     [_item("chair_0", 0, 0), _item("desk_0", 2, 0)], True),
    ({"check": "against_wall", "id": "shelf_0"}, [_item("shelf_0", 0, 3.6)], False),
    ({"check": "against_wall", "id": "shelf_0"}, [_item("shelf_0", 0, 2)], True),
    ({"check": "faces", "a": "chair_0", "b": "desk_0"}, [_item("chair_0", 0, 0), _item("desk_0", 2, 0)], False),
    ({"check": "faces", "a": "chair_0", "b": "desk_0"}, [_item("chair_0", 0, 0, yaw=3.14), _item("desk_0", 2, 0)], True),
    ({"check": "teleport"}, [], True),
    ({"check": "near", "a": "ghost_0"}, [], True),
])
def test_optional_requirements(requirement, items, fails):
    room = _room(doorways=[])
    optional = [v for v in check_layout(room, items, [requirement]) if not v.always_on]
    assert bool(optional) is fails


def test_faces_hint_gives_the_yaw_to_use():
    violations = check_layout(_room(), [_item("chair_0", 0, 2, yaw=3.14), _item("desk_0", 0, -2)],
                              [{"check": "faces", "a": "chair_0", "b": "desk_0"}])
    assert violations[0].hint == "set yaw of chair_0 to -1.57"


# --- rendering -----------------------------------------------------------------------------------

def test_render_produces_a_png_of_the_room():
    png = render_layout(_room(), [_item("box_0", 0, 0)], check_layout(_room(), [_item("box_0", -5.5, 0)]))
    image = Image.open(io.BytesIO(png))
    assert image.format == "PNG" and image.width > image.height


# --- designer loop -------------------------------------------------------------------------------

class ScriptedLLM:
    def __init__(self, *answers, reject_images=False):
        self.answers = list(answers)
        self.requests = []
        self.reject_images = reject_images
        self.context_window = 32768

    def query(self, messages, max_tokens=4096, temperature=0.6):
        self.requests.append(messages)
        has_image = any(isinstance(m["content"], list) for m in messages)
        if self.reject_images and has_image:
            return ""
        return self.answers.pop(0) if self.answers else ""


def _designer(llm, rounds=3):
    manager = PromptManager(template_dir="prompts", default_version="v1", enable_metrics=False)
    return LayoutDesigner(llm, manager, random.Random(1), max_rounds=rounds)


ITEMS = [dict(id="box_0", type="box", dims=(0.6, 0.6, 0.6), front="+x", model="box", description=""),
         dict(id="box_1", type="box", dims=(0.6, 0.6, 0.6), front="+x", model="box", description="")]


def _layout(*poses, requirements=()):
    return json.dumps({"requirements": list(requirements),
                       "objects": [{"id": f"box_{i}", "x": x, "y": y, "yaw": 0} for i, (x, y) in enumerate(poses)]})


def test_violations_are_fed_back_and_loop_stops_when_clean():
    llm = ScriptedLLM(_layout((0, 0), (0.2, 0)), _layout((0, 2), (2, -2)), _layout((9, 9), (9, 9)))
    placed, violations = _designer(llm).design(_room(), ITEMS, "two boxes")
    assert violations == []
    assert len(llm.requests) == 2
    revision = llm.requests[1][-1]["content"][0]["text"]
    assert "box_0 overlaps box_1" in revision
    assert llm.requests[1][-1]["content"][1]["type"] == "image_url"
    assert {(p["x"], p["y"]) for p in placed} == {(0, 2), (2, -2)}


def test_best_round_is_kept_and_nudged_when_llm_never_fixes_it():
    rounds = [_layout((0, 0), (0.2, 0)), _layout((0, 0), (0.1, 0)), _layout((0, 0), (0, 0))]
    placed, violations = _designer(ScriptedLLM(*rounds)).design(_room(), ITEMS, "two boxes")
    assert [v for v in violations if v.always_on] == []
    a, b = (item_box(p) for p in placed)
    assert min(penetration(a, b)) <= 0.02


def test_llm_requirements_are_checked():
    answer = _layout((-2, 0.2), (2, -0.2), requirements=[{"check": "min_spacing", "width": 5.0}])
    fixed = _layout((-2, 2.5), (2.9, -2.5), requirements=[{"check": "min_spacing", "width": 1.0}])
    llm = ScriptedLLM(answer, fixed)
    _, violations = _designer(llm).design(_room(), ITEMS, "spread out")
    assert len(llm.requests) == 2 and violations == []


def test_text_only_endpoint_falls_back_without_images():
    llm = ScriptedLLM(_layout((0, 2), (2, -2)), reject_images=True)
    designer = _designer(llm)
    _, violations = designer.design(_room(), ITEMS, "two boxes")
    assert violations == [] and designer.use_images is False
    assert isinstance(llm.requests[-1][0]["content"], str)


def test_missing_ids_and_garbage_are_sampled():
    llm = ScriptedLLM("not json", json.dumps({"objects": [{"id": "box_0", "x": 0, "y": 2, "yaw": 0},
                                                           {"id": "intruder", "x": 0, "y": 0}]}))
    placed, violations = _designer(llm, rounds=2).design(_room(), ITEMS, "two boxes")
    assert sorted(p["id"] for p in placed) == ["box_0", "box_1"]
    assert violations == []


def test_sampler_places_many_items_without_llm():
    items = [dict(ITEMS[0], id=f"box_{i}") for i in range(40)]
    placed, violations = LayoutDesigner(rng=random.Random(3)).design(_room(), items)
    assert len(placed) == 40 and [v for v in violations if v.always_on] == []


def test_image_messages_are_budgeted():
    text = [{"role": "user", "content": "hello"}]
    image = [{"role": "user", "content": [{"type": "text", "text": "hello"},
                                          {"type": "image_url", "image_url": {"url": "data:"}}]}]
    assert estimate_tokens(image) - estimate_tokens(text) >= 1000


def test_revision_answers_list_only_moved_objects():
    first = _layout((0, 2), (0.2, 2))
    moved_only = json.dumps({"objects": [{"id": "box_1", "x": 2.0, "y": -2.0, "yaw": 0}]})
    llm = ScriptedLLM(first, moved_only)
    placed, violations = _designer(llm).design(_room(), ITEMS, "two boxes")
    assert violations == []
    assert {p["id"]: (p["x"], p["y"]) for p in placed} == {"box_0": (0, 2), "box_1": (2.0, -2.0)}
    assert "box_0, box_1" in llm.requests[1][-1]["content"][0]["text"]


def test_min_spacing_exempts_pairs_the_llm_tied_together():
    items = [_item("desk_0", 0, 2, dims=(1.4, 0.7, 0.75)), _item("chair_0", 0, 1.3),
             _item("shelf_0", -5.6, 0, dims=(0.4, 1.0, 1.8))]
    requirements = [{"check": "min_spacing", "width": 0.8},
                    {"check": "near", "a": "chair_0", "b": "desk_0", "max": 0.3},
                    {"check": "against_wall", "id": "shelf_0"}]
    assert check_layout(_room(doorways=[]), items, requirements) == []


def test_free_point_avoids_objects_at_the_centre():
    from gazebo_world_generator.src.placement.checks import free_point
    x, y = free_point(_room(), [_item("barrel_0", 0, 0)], clearance=0.5)
    assert max(abs(x), abs(y)) >= 0.75


@pytest.mark.parametrize("simulator, marker", [("classic", "libgazebo_2Dmap_plugin.so"),
                                              ("harmonic", "gz_2Dmap_system")])
def test_map_world_copy_starts_flood_fill_at_seed(tmp_path, simulator, marker):
    import xml.etree.ElementTree as ET
    from gazebo_world_generator.gazebo_world_generator import _world_with_map_plugin
    world = tmp_path / "lab.sdf"
    world.write_text("<sdf version='1.7'><world name='w'></world></sdf>")
    copy_dir = tmp_path / "copy"
    copy_dir.mkdir()
    copy = _world_with_map_plugin(str(world), tmp_path / "maps", simulator, (2.5, -1.0), copy_dir)
    plugin = ET.parse(copy).getroot().find("world/plugin")
    assert copy.endswith("lab.sdf") and marker in plugin.get("filename")
    assert plugin.findtext("init_robot_x") == "2.5000" and plugin.findtext("init_robot_y") == "-1.0000"


def test_classic_map_copy_is_shifted_and_origin_restored(tmp_path):
    import xml.etree.ElementTree as ET
    from gazebo_world_generator.gazebo_world_generator import (
        _shift_map_origin, _world_with_map_plugin)
    world = tmp_path / "lab.sdf"
    world.write_text("<sdf version='1.7'><world name='w'><include><uri>model://box</uri>"
                     "<pose>1 2 0.15 0 0 0.5</pose></include><model name='wall'/></world></sdf>")
    copy_dir = tmp_path / "copy"
    copy_dir.mkdir()
    copy = ET.parse(_world_with_map_plugin(str(world), tmp_path, "classic", (0, 0), copy_dir, 0.0125))
    assert copy.find("world/include/pose").text == "1.0125 2.0125 0.1500 0.0000 0.0000 0.5000"
    assert copy.find("world/model/pose").text.startswith("0.0125 0.0125")
    assert copy.find("world/plugin").findtext("init_robot_x") == "0.0125"
    yaml = tmp_path / "lab.yaml"
    yaml.write_text("image: lab.pgm\norigin: [-16, -13, 0]\nnegate: 0\n")
    _shift_map_origin(yaml, 0.0125)
    assert "origin: [-16.0125, -13.0125, 0]" in yaml.read_text()


def test_item_that_does_not_fit_on_support_moves_to_the_floor():
    items = [dict(ITEMS[0], id="desk_0", dims=(1.2, 0.6, 0.75)), dict(ITEMS[0], id="chair_0")]
    answer = json.dumps({"objects": [{"id": "desk_0", "x": 0, "y": 2, "yaw": 0},
                                     {"id": "chair_0", "x": 0.6, "y": 2, "yaw": 0, "on": "desk_0"}]})
    placed, violations = _designer(ScriptedLLM(answer, answer, answer)).design(_room(), items, "desk")
    assert violations == [] and all(item["on"] is None for item in placed)
