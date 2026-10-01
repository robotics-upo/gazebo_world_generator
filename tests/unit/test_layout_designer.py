"""Checks, rendering and the propose -> check -> revise loop."""

import io
import math
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


def _designer(llm, rounds=3, review_rounds=0):
    manager = PromptManager(template_dir="prompts", default_version="v1", enable_metrics=False)
    return LayoutDesigner(llm, manager, random.Random(1), max_rounds=rounds,
                          review_rounds=review_rounds)


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


# --- orientation: in_front_of, relations, review, front memory ----------------------------------

DESK = dict(id="desk_0", type="desk", x=0.0, y=1.3, yaw=0.0, dims=(0.5, 0.84, 0.55), front="+x",
            model="Desk", description="")
CHAIR = dict(id="chair_0", type="chair", x=0.0, y=0.5, yaw=3.14159, dims=(0.74, 0.75, 1.25), front="-y",
             model="Chair", description="")
AT_DESK = [{"check": "in_front_of", "a": "chair_0", "b": "desk_0", "max_gap": 0.3}]


@pytest.mark.parametrize("pose, ok", [
    ((0.7, 1.3), True),    # at the desk's long (+x) side
    ((0.0, 0.5), False),   # at its short end: the Office B screenshot
    ((-0.7, 1.3), False),  # behind it
    ((1.8, 1.3), False),   # too far away
])
def test_in_front_of_uses_the_front_side_and_width(pose, ok):
    chair = dict(CHAIR, x=pose[0], y=pose[1])
    violations = check_layout(_room(doorways=[]), [dict(DESK), chair], AT_DESK)
    assert (not violations) is ok
    if not ok:
        assert "move chair_0 to (" in violations[0].hint


def test_relations_expose_backwards_chair_and_short_side_front():
    from gazebo_world_generator.src.placement.checks import relations
    desk = dict(DESK, front="+y")
    chair = dict(CHAIR, yaw=0.0)  # front -y at yaw 0 faces south, away from the desk to its north
    lines = relations(_room(doorways=[]), [desk, chair])
    assert "SHORT side" in lines[0]
    assert "AWAY from desk_0" in lines[1]


def _office_answer(chair_pose, yaw, fronts=None):
    answer = {"requirements": [], "objects": [
        {"id": "desk_0", "x": 0.0, "y": 1.3, "yaw": 0.0},
        {"id": "chair_0", "x": chair_pose[0], "y": chair_pose[1], "yaw": yaw}]}
    if fronts:
        answer["fronts"] = fronts
    return json.dumps(answer)


def test_review_adds_requirements_and_fixes_the_chair():
    items = [dict(DESK), dict(CHAIR)]
    wrong = _office_answer((0.0, 0.5), 0.0, {"desk": "+y", "chair": "-y"})
    review = json.dumps({"fronts": {"desk": "+x"},
                         "requirements": AT_DESK + [{"check": "faces", "a": "chair_0", "b": "desk_0"}],
                         "objects": [{"id": "chair_0", "x": 0.72, "y": 1.3, "yaw": -1.5708}]})
    llm = ScriptedLLM(wrong, review)
    placed, violations = _designer(llm, review_rounds=1).design(_room(doorways=[]), items, "office")
    chair = next(p for p in placed if p["id"] == "chair_0")
    assert violations == [] and (chair["x"], chair["y"]) == (0.72, 1.3)
    assert "MEASURED RELATIONS" in llm.requests[1][-1]["content"][0]["text"]


def test_review_that_breaks_basic_rules_is_discarded():
    items = [dict(DESK), dict(CHAIR)]
    good = _office_answer((0.72, 1.3), -1.5708, {"desk": "+x", "chair": "-y"})
    review = json.dumps({"objects": [{"id": "chair_0", "x": 0.0, "y": 1.3, "yaw": 0.0}]})  # into the desk
    placed, violations = _designer(ScriptedLLM(good, review), review_rounds=1).design(
        _room(doorways=[]), items, "office")
    assert violations == [] and next(p for p in placed if p["id"] == "chair_0")["x"] == 0.72


def test_restated_always_on_rules_are_ignored():
    requirements = [{"check": "no_overlap"}, {"check": "all_ids_placed"}, {"check": "inside_room"}]
    assert check_layout(_room(doorways=[]), [_item("box_0", 0, 0)], requirements) == []


def test_front_is_measured_from_the_open_side():
    from gazebo_world_generator.src.placement.designer import measured_front

    def box(x0, x1, y0, y1, z0, z1):
        c = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
        faces = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
        return [t for a, b, cc, d in faces for t in ((c[a], c[b], c[cc]), (c[a], c[cc], c[d]))]

    seat = box(-0.25, 0.25, -0.25, 0.25, 0.4, 0.45)
    backrest = box(-0.25, 0.25, 0.22, 0.25, 0.45, 1.1)
    front, closed = measured_front(seat + backrest)
    assert front == "-y" and closed["+y"] > closed["-y"]
    assert measured_front(box(-0.4, 0.4, -0.4, 0.4, 0.0, 0.7))[0] is None  # a plain block


def test_faces_needs_a_front_to_point():
    table = dict(DESK, id="table_0", type="table", front_known=False)
    box = _item("box_0", 2, 2, front_known=False)
    assert check_layout(_room(doorways=[]), [table, box],
                        [{"check": "faces", "a": "box_0", "b": "table_0"}]) == []


TABLE = dict(DESK, id="table_0", type="table", dims=(1.2, 0.6, 0.75), front_known=False)
AT_TABLE = [{"check": "in_front_of", "a": "chair_0", "b": "table_0", "max_gap": 0.3},
            {"check": "faces", "a": "chair_0", "b": "table_0"}]


@pytest.mark.parametrize("pose, yaw, ok", [
    ((0.2, 2.05), 0.0, True),           # north long side, front -y faces the table
    ((0.2, 0.55), 3.14159, True),       # south long side
    ((1.15, 1.3), -1.5708, True),       # east short end: a table has no "wrong" side
    ((0.0, 2.9), 0.0, False),           # a ring of chairs 1 m out: too far to use it
    ((0.2, 2.05), 3.14159, False),      # at the table, back to it
])
def test_seats_go_around_a_table_without_a_front(pose, yaw, ok):
    chair = dict(CHAIR, x=pose[0], y=pose[1], yaw=yaw)
    violations = check_layout(_room(doorways=[]), [dict(TABLE), chair], AT_TABLE)
    assert (not violations) is ok


def test_solver_pulls_a_far_seat_up_to_the_nearest_table_side():
    answer = json.dumps({"requirements": AT_TABLE, "objects": [
        {"id": "table_0", "x": 0.0, "y": 1.3, "yaw": 0.0},
        {"id": "chair_0", "x": 0.1, "y": 2.9, "yaw": 3.14159}]})  # 1.3 m north, turned away
    placed, violations = _designer(ScriptedLLM(answer, answer, answer)).design(
        _room(doorways=[]), [dict(TABLE), dict(CHAIR)], "meeting")
    chair = next(p for p in placed if p["id"] == "chair_0")
    assert violations == [] and chair["y"] - 1.3 < 0.3 + 0.3 + 0.375 + 0.01  # at the north side


def test_objects_a_crowded_layout_cannot_fit_are_re_placed():
    items = [_item(f"rack_{i}", 0, 0, dims=(1.0, 2.0, 2.0)) for i in range(8)]
    # One column of 8 x 2 m racks in a 10 m room: overlapping and through the wall.
    answer = json.dumps({"requirements": [], "objects": [
        {"id": f"rack_{i}", "x": -4.0, "y": -4.0 + 1.5 * i, "yaw": 0.0} for i in range(8)]})
    placed, violations = _designer(ScriptedLLM(answer, answer, answer)).design(
        _room(doorways=[]), items, "warehouse")
    assert len(placed) == 8 and not [v for v in violations if v.always_on]


def test_requirement_may_name_the_only_object_of_a_type():
    chair = dict(CHAIR, x=0.0, y=0.5)  # at the desk's short end
    violations = check_layout(_room(doorways=[]), [dict(DESK), chair],
                              [{"check": "in_front_of", "a": "chair", "b": "desk", "max_gap": 0.3}])
    assert [v.check for v in violations] == ["in_front_of"] and "chair_0" in violations[0].fix


SHELF = dict(id="shelf_0", type="shelf", dims=(0.92, 0.4, 1.8), front="-y", front_known=True,
             model="Bookshelf", description="")


@pytest.mark.parametrize("pose, yaw, ok", [
    ((0.0, 3.7), 0.0, True),         # front -y faces south into the room, back on the north wall
    ((0.0, 3.7), 3.14159, False),    # touching the north wall but facing it
    ((0.0, 3.44), 1.5708, False),    # end-on: short end touches the wall, sticks into the room
    ((0.0, 2.9), 0.0, False),        # right way round, 0.8 m off the wall
])
def test_against_wall_puts_the_back_of_a_fronted_object_on_the_wall(pose, yaw, ok):
    shelf = dict(SHELF, x=pose[0], y=pose[1], yaw=yaw)
    violations = check_layout(_room(doorways=[]), [shelf], [{"check": "against_wall", "id": "shelf_0"}])
    assert (not violations) is ok
    if not ok:
        fixed = dict(shelf, **violations[0].fix["shelf_0"])
        assert check_layout(_room(doorways=[]), [fixed], [{"check": "against_wall", "id": "shelf_0"}]) == []


def test_measured_no_front_is_not_overridden_by_the_llm():
    table = dict(TABLE, front_measured=True)
    designer = _designer(ScriptedLLM())
    placed = designer._apply_answer({"fronts": {"table": "+x"}, "objects": [{"id": "table_0", "x": 0, "y": 0}]},
                                    {"table_0": table}, correct_fronts=True)
    assert placed[0]["front_known"] is False


def test_space_use_reports_an_empty_part_of_the_room():
    from gazebo_world_generator.src.placement.checks import space_use
    racks = [_item(f"rack_{i}", -5.0, -3.0 + 1.5 * i, dims=(1.0, 1.0, 2.0)) for i in range(4)]
    lines = space_use(_room(doorways=[]), racks)
    assert "of the room's width" in lines[0]
    assert any(line.startswith("the east") and "empty" in line for line in lines)


def test_solver_spreads_objects_closer_than_the_requested_spacing():
    cones = [_item(f"cone_{i}", -2.0 + 0.9 * i, 0.0, dims=(0.3, 0.3, 0.5)) for i in range(4)]  # 0.6 m gaps
    spacing = [{"check": "min_spacing", "width": 0.8}]
    answer = json.dumps({"requirements": spacing, "objects": [
        {"id": c["id"], "x": c["x"], "y": c["y"], "yaw": 0.0} for c in cones]})
    placed, violations = _designer(ScriptedLLM(answer, answer, answer)).design(
        _room(doorways=[]), cones, "slalom")
    assert violations == []


def test_invented_check_names_are_not_reported_as_layout_problems():
    answer = json.dumps({"requirements": [{"check": "placed", "ids": ["box_0"]}],
                         "objects": [{"id": "box_0", "x": 0.0, "y": 0.0, "yaw": 0.0}]})
    placed, violations = _designer(ScriptedLLM(answer, answer, answer)).design(
        _room(doorways=[]), [_item("box_0", 0, 0)], "room")
    assert violations == []


def test_fix_moves_what_stands_on_the_moved_item():
    from gazebo_world_generator.src.placement.designer import _apply_fix
    desk = dict(DESK)
    monitor = _item("monitor_0", 0.1, 1.3, dims=(0.3, 0.5, 0.4), on="desk_0")
    moved = _apply_fix([desk, monitor], {"desk_0": {"x": 1.0, "y": 1.3, "yaw": math.pi / 2}})
    assert abs(moved[1]["x"] - 1.0) < 1e-9 and abs(moved[1]["y"] - 1.4) < 1e-9
    assert abs(moved[1]["yaw"] - math.pi / 2) < 1e-9


def test_in_front_of_requires_a_measured_seat_to_face_the_desk():
    sideways = dict(CHAIR, x=0.7, y=1.3, yaw=0.0, front_known=True)  # at the +x side, front -y: south
    violations = check_layout(_room(doorways=[]), [dict(DESK), sideways], AT_DESK)
    assert [v.check for v in violations] == ["in_front_of"] and "does not face" in violations[0].message
    turned = dict(sideways, **violations[0].fix["chair_0"])
    assert check_layout(_room(doorways=[]), [dict(DESK), turned], AT_DESK) == []


def test_faces_fix_aims_at_the_target_when_not_level_with_its_side():
    chair = dict(CHAIR, x=0.0, y=1.3, yaw=0.0, front_known=True)
    monitor = dict(_item("monitor_0", -0.6, 0.2, dims=(0.3, 0.5, 0.4)), front="-x", front_known=True)
    requirement = [{"check": "faces", "a": "monitor_0", "b": "chair_0"}]
    violations = check_layout(_room(doorways=[]), [chair, monitor], requirement)
    aimed = dict(monitor, **violations[0].fix["monitor_0"])
    assert check_layout(_room(doorways=[]), [chair, aimed], requirement) == []


def test_on_stated_as_a_requirement_puts_the_item_on_its_support():
    monitor = _item("monitor_0", 0, 0, dims=(0.3, 0.4, 0.4))
    answer = json.dumps({"requirements": [{"check": "on", "a": "monitor_0", "b": "desk_0"}], "objects": [
        {"id": "desk_0", "x": 0.0, "y": 1.3, "yaw": 0.0},
        {"id": "monitor_0", "x": -1.0, "y": 0.4, "yaw": 0.0}]})  # on the floor beside the desk
    placed, violations = _designer(ScriptedLLM(answer, answer, answer)).design(
        _room(doorways=[]), [dict(DESK), monitor], "office")
    monitor = next(p for p in placed if p["id"] == "monitor_0")
    assert violations == [] and monitor["on"] == "desk_0" and (monitor["x"], monitor["y"]) == (0.0, 1.3)


def test_review_moves_stand_even_if_they_cost_a_soft_requirement():
    near = [{"check": "near", "a": "box_0", "b": "box_1", "max": 0.5}]
    first = json.dumps({"requirements": near, "objects": [
        {"id": "box_0", "x": -4.0, "y": 0.0, "yaw": 0.0}, {"id": "box_1", "x": -3.0, "y": 0.0, "yaw": 0.0}]})
    review = json.dumps({"objects": [{"id": "box_0", "x": 3.0, "y": 0.0, "yaw": 0.0}]})  # spread out
    llm = ScriptedLLM(first, review, review)
    placed, _ = _designer(llm, review_rounds=1).design(
        _room(doorways=[]), [_item("box_0", 0, 0), _item("box_1", 0, 0)], "clutter")
    assert next(p for p in placed if p["id"] == "box_0")["x"] == 3.0


def test_seat_tucked_into_a_table_ends_up_beside_it_facing_it():
    # The LLM tucks the chair into the table, turned away: inside the table every ray "hits" it,
    # so only after nudging it out does the turn show; relations are solved again then.
    answer = json.dumps({"requirements": AT_TABLE, "objects": [
        {"id": "table_0", "x": 0.0, "y": 1.3, "yaw": 0.0},
        {"id": "chair_0", "x": 0.5, "y": 1.3, "yaw": 1.5708}]})
    chair = dict(CHAIR, front_known=True)
    placed, violations = _designer(ScriptedLLM(answer, answer, answer)).design(
        _room(doorways=[]), [dict(TABLE), chair], "library")
    assert violations == []


def test_compass_handles_angles_just_below_a_full_turn():
    from gazebo_world_generator.src.placement.checks import _compass
    # The float just below -pi/4 (a diagonal yaw) made the sector index 4 and crashed a run.
    assert _compass(math.nextafter(-math.pi / 4, -1.0)) in ("east", "south")


def test_in_front_of_fix_keeps_seats_apart_along_a_long_table():
    table = dict(TABLE, dims=(2.4, 1.0, 0.75))  # long side 2.4 m along x
    small = (0.5, 0.5, 1.0)
    chairs = [dict(CHAIR, id=f"chair_{i}", x=x, y=2.6, dims=small) for i, x in enumerate((-0.8, 0.8))]
    requirements = [{"check": "in_front_of", "a": c["id"], "b": "table_0", "max_gap": 0.2} for c in chairs]
    violations = check_layout(_room(doorways=[]), [table] + chairs, requirements)
    targets = [v.fix[v.ids[0]] for v in violations]
    assert len(targets) == 2 and abs(targets[0]["x"] - targets[1]["x"]) > 1.0


def test_scatter_places_ids_irregularly_inside_the_requested_area():
    boxes = [_item(f"box_{i}", 0, 0, dims=(0.5, 0.5, 0.5)) for i in range(6)]
    answer = json.dumps({"requirements": [], "objects": [{"id": b["id"], "x": 0, "y": 0, "yaw": 0} for b in boxes],
                         "scatter": [{"ids": [b["id"] for b in boxes], "area": [-4, 4, -3, 3], "gap": 0.6}]})
    placed, violations = _designer(ScriptedLLM(answer, answer, answer)).design(
        _room(doorways=[]), boxes, "clutter")
    assert violations == []
    xs, ys = [p["x"] for p in placed], [p["y"] for p in placed]
    assert all(-4.4 <= x <= 4.4 for x in xs) and all(-3.4 <= y <= 3.4 for y in ys)
    assert len({round(x, 1) for x in xs}) == 6 and len({round(p["yaw"], 2) for p in placed}) > 2
    gaps = [item_box(a) for a in placed]
    from gazebo_world_generator.src.placement.checks import box_gap
    assert min(box_gap(a, b) for i, a in enumerate(gaps) for b in gaps[i + 1:]) >= 0.6 - 1e-9


def test_in_front_of_caps_a_generous_gap():
    chair = dict(CHAIR, x=0.25 + 0.375 + 0.8, y=1.3, yaw=-1.5708)  # 0.8 m off the desk's front
    loose = [{"check": "in_front_of", "a": "chair_0", "b": "desk_0", "max_gap": 1.0}]
    assert [v.check for v in check_layout(_room(doorways=[]), [dict(DESK), chair], loose)] == ["in_front_of"]


def test_item_on_its_support_is_not_beside_it():
    monitor = _item("monitor_0", 0.0, 1.3, dims=(0.3, 0.5, 0.4), on="desk_0")
    requirement = [{"check": "in_front_of", "a": "monitor_0", "b": "desk_0", "max_gap": 0.3}]
    assert check_layout(_room(doorways=[]), [dict(DESK), monitor], requirement) == []


def test_solver_seats_the_chair_at_the_desk_front_facing_it():
    requirements = AT_DESK + [{"check": "faces", "a": "chair_0", "b": "desk_0"}]
    answer = json.dumps({"requirements": requirements, "objects": [
        {"id": "desk_0", "x": 0.0, "y": 1.3, "yaw": 0.0},
        {"id": "chair_0", "x": -0.1, "y": 0.2, "yaw": 0.0}]})  # at the short end, turned away
    placed, violations = _designer(ScriptedLLM(answer, answer, answer)).design(
        _room(doorways=[]), [dict(DESK), dict(CHAIR)], "office")
    chair = next(p for p in placed if p["id"] == "chair_0")
    assert violations == []
    assert chair["x"] > 0.25 and abs(chair["y"] - 1.3) < 0.05  # desk's +x (long) side
    assert abs(((chair["yaw"] + 1.5708 + math.pi) % (2 * math.pi)) - math.pi) < 0.01  # front -y points west


def test_symmetric_models_get_no_front():
    from gazebo_world_generator.src.placement.designer import LayoutDesigner as Designer
    block = [((-0.4, -0.4, 0.0), (0.4, -0.4, 0.0), (0.4, 0.4, 0.7)), ((-0.4, -0.4, 0.0), (0.4, 0.4, 0.7), (-0.4, 0.4, 0.7))]
    items = [dict(_item("cone_0", 0, 0), shape=block)]
    Designer._determine_fronts(items)
    assert items[0]["front_known"] is False


def test_faces_accepts_a_chair_off_centre_along_the_desk_front():
    desk = dict(DESK, dims=(0.5, 1.4, 0.75))
    chair = dict(CHAIR, x=0.7, y=1.8, yaw=-1.5708)  # at +x front, 0.5 m north of centre, facing west
    assert check_layout(_room(doorways=[]), [desk, chair],
                        [{"check": "faces", "a": "chair_0", "b": "desk_0"}]) == []


def test_solver_turns_a_backwards_chair_in_place():
    requirements = AT_DESK + [{"check": "faces", "a": "chair_0", "b": "desk_0"}]
    answer = json.dumps({"requirements": requirements, "objects": [
        {"id": "desk_0", "x": 0.0, "y": 1.3, "yaw": 0.0},
        {"id": "chair_0", "x": 0.72, "y": 1.45, "yaw": 1.5708}]})  # right place, backrest to the desk
    placed, violations = _designer(ScriptedLLM(answer, answer, answer)).design(
        _room(doorways=[]), [dict(DESK), dict(CHAIR)], "office")
    chair = next(p for p in placed if p["id"] == "chair_0")
    assert violations == [] and abs(chair["y"] - 1.45) < 1e-6
    assert abs(((chair["yaw"] + 1.5708 + math.pi) % (2 * math.pi)) - math.pi) < 0.01
