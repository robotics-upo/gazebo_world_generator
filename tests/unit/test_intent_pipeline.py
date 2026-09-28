"""Tests for carrying user intent from parsing through placement and model resolution."""

import os
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

from gazebo_world_generator.src.config.validated_settings import ValidatedConfig
from gazebo_world_generator.src.core.data_models import GazeboModel, Room
from gazebo_world_generator.src.core.object_vocabulary import PRIMITIVE_URI
from gazebo_world_generator.src.llm.interface import OpenAICompatibleInterface
from gazebo_world_generator.src.models import resolver as resolver_module
from gazebo_world_generator.src.models.resolver import SmartModelResolver
from gazebo_world_generator.src.models.visual_quality import inspect_model_visuals
from gazebo_world_generator.src.placement.engine import NaturalPlacementEngine
from gazebo_world_generator.src.world.generator import WorldGenerator


@pytest.fixture(autouse=True)
def no_env_model_paths(monkeypatch):
    monkeypatch.delenv("GAZEBO_MODEL_PATH", raising=False)
    monkeypatch.delenv("GZ_SIM_RESOURCE_PATH", raising=False)


def _write_box_model(base, name, dimensions="0.5 0.5 0.5"):
    directory = base / name
    directory.mkdir()
    (directory / "model.config").write_text(f"<model><name>{name}</name></model>")
    (directory / "model.sdf").write_text(
        f"<sdf version='1.7'><model name='{name}'><link name='link'>"
        f"<visual name='visual'><geometry><box><size>{dimensions}</size></box>"
        f"</geometry></visual></link></model></sdf>")
    return directory


def _room(width=12.0, length=8.0, doorways=None):
    room = Room(name="Test", type="lab", dimensions={"width": width, "length": length, "height": 3},
                position={"x": 10.0, "y": -5.0, "z": 0.0})
    room.doorways = doorways or []
    return room


# --- model resolution ---------------------------------------------------------------------------

def test_texture_in_materials_textures_is_found(tmp_path):
    directory = tmp_path / "crate"
    (directory / "meshes").mkdir(parents=True)
    (directory / "materials" / "textures").mkdir(parents=True)
    (directory / "materials" / "textures" / "crate.png").write_bytes(b"png")
    (directory / "model.config").write_text("<model><name>crate</name></model>")
    (directory / "model.sdf").write_text(
        "<sdf version='1.7'><model name='crate'><link name='link'><visual name='v'>"
        "<geometry><mesh><uri>meshes/crate.dae</uri></mesh></geometry></visual></link></model></sdf>")
    (directory / "meshes" / "crate.dae").write_text(
        "<COLLADA xmlns='http://www.collada.org/2005/11/COLLADASchema'>"
        "<library_images><image id='i'><init_from>crate.png</init_from></image></library_images>"
        "<library_geometries><geometry id='g'><mesh><source id='p'>"
        "<float_array id='a' count='6'>0 0 0 1 1 1</float_array></source>"
        "<vertices id='v'><input semantic='POSITION' source='#p'/></vertices>"
        "</mesh></geometry></library_geometries></COLLADA>")
    inspection = inspect_model_visuals(directory)
    assert inspection.error is None


def test_dangling_absolute_symlink_is_replaced_with_relative(tmp_path):
    _write_box_model(tmp_path, "Euro pallet")
    link = tmp_path / "Euro_pallet"
    os.symlink("/nonexistent/host/path/Euro pallet", link)

    resolver = SmartModelResolver(search_paths=[tmp_path], enable_cache=False)
    assert "Euro pallet" in resolver.local_models
    assert os.readlink(link) == "Euro pallet"
    assert resolver.get_directory_for_model("Euro pallet") == "Euro_pallet"


def test_symlink_to_other_valid_directory_is_left_alone(tmp_path):
    _write_box_model(tmp_path, "Oak tree")
    _write_box_model(tmp_path, "Pine")
    os.symlink("Pine", tmp_path / "Oak_tree")
    resolver = SmartModelResolver(search_paths=[tmp_path], enable_cache=False)
    resolver.local_models
    assert os.readlink(tmp_path / "Oak_tree") == "Pine"


def test_model_inspection_is_memoized_and_failures_reused(tmp_path, caplog):
    _write_box_model(tmp_path, "cardboard_box")
    resolver = SmartModelResolver(search_paths=[tmp_path], enable_cache=False)
    with patch.object(resolver_module, "inspect_model_visuals",
                      wraps=resolver_module.inspect_model_visuals) as inspect:
        assert resolver._uri_suitable("model://cardboard_box", "box")
        assert resolver._uri_suitable("model://cardboard_box", "box")
    assert inspect.call_count == 1

    assert resolver.find_best_model("sofa") is None
    assert resolver.find_best_model("sofa") is None
    assert caplog.text.count("No visible, suitably sized model found for 'sofa'") == 1


def test_category_word_resolves_to_concrete_substitute(tmp_path):
    _write_box_model(tmp_path, "cardboard_box")
    resolver = SmartModelResolver(search_paths=[tmp_path], enable_cache=False)
    assert resolver.find_best_model("obstacle") == "model://cardboard_box"


def test_generic_object_falls_back_to_primitive_but_furniture_does_not(tmp_path):
    resolver = SmartModelResolver(search_paths=[tmp_path], enable_cache=False)
    assert resolver.find_best_model("obstacle") == PRIMITIVE_URI
    assert resolver.find_best_model("sofa") is None


def test_unresolvable_furniture_fails_before_placement():
    database = MagicMock()
    database.find_best_model.return_value = None
    llm = MagicMock()
    engine = NaturalPlacementEngine(model_db=database, llm_interface=llm)
    with pytest.raises(ValueError, match="No visible model found"):
        engine.place_objects_in_room(_room(), "", [{"type": "sofa", "count": 1}], lambda n: n)
    llm.query.assert_not_called()


# --- parsing ------------------------------------------------------------------------------------

def test_normalize_rooms_tidies_types_and_adds_implied_furniture():
    llm = OpenAICompatibleInterface(server_url="localhost:8000")
    parsed = {"rooms": [
        {"name": "Big Testing Room", "type": "testing_room",
         "purpose": "Obstacle course for robot navigation testing",
         "objects": [{"type": "Cardboard Box", "count": 5}]},
        {"name": "Office-1", "type": "office", "objects": []},
        {"name": "Hall", "type": "corridor", "objects": []},
    ]}
    llm._normalize_rooms(parsed, "A lab with a testing room and office-like rooms")
    testing, office, hall = parsed["rooms"]
    assert testing["type"] == "testing_room"
    assert testing["purpose"] == "Obstacle course for robot navigation testing"
    assert testing["objects"][0]["type"] == "cardboard_box"
    assert [(o["type"], o["count"], o["inferred"]) for o in office["objects"]] == [
        ("desk", 1, True), ("chair", 1, True)]
    assert hall["objects"] == []


def test_empty_room_request_is_not_furnished():
    llm = OpenAICompatibleInterface(server_url="localhost:8000")
    parsed = {"rooms": [{"name": "Office", "type": "office", "objects": []}]}
    llm._normalize_rooms(parsed, "An empty office")
    assert parsed["rooms"][0]["objects"] == []


def test_balance_keeps_inferred_objects_in_place():
    llm = OpenAICompatibleInterface(server_url="localhost:8000")
    parsed = {"rooms": [
        {"name": "A", "type": "office", "objects": [{"type": "desk", "count": 4}]},
        {"name": "B", "type": "office", "objects": [
            {"type": "chair", "count": 1, "inferred": True}]},
    ]}
    llm._balance_object_distribution(parsed)
    b_objects = parsed["rooms"][1]["objects"]
    assert {"type": "chair", "count": 1, "inferred": True} in b_objects
    total_desks = sum(o["count"] for room in parsed["rooms"] for o in room["objects"]
                      if o["type"] == "desk")
    assert total_desks == 4


# --- SDF ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("simulator", ["classic", "harmonic"])
def test_primitive_box_is_written_as_inline_model(mock_llm_interface, tmp_path, simulator):
    mock_llm_interface.parse_room_description.return_value = {"rooms": [
        {"name": "Lab", "type": "lab", "dimensions": {"width": 6, "length": 6},
         "objects": [{"type": "obstacle", "count": 1}]}]}
    generator = WorldGenerator(mock_llm_interface, config=ValidatedConfig(simulator=simulator))
    generator.placement_engine.place_all_objects = lambda *_: [GazeboModel(
        name="obstacle_0", model_path=PRIMITIVE_URI, category="furniture",
        pose={"x": 1, "y": 1, "z": 0, "roll": 0, "pitch": 0, "yaw": 0.5},
        size=[0.5, 0.5, 0.5], static=True)]
    output = tmp_path / "world.sdf"
    generator.generate_world("lab", str(output))
    model = ET.parse(output).getroot().find("world/model[@name='obstacle_0']")
    assert model is not None
    assert model.findtext("link/visual/geometry/box/size") == "0.500 0.500 0.500"
    assert model.findtext("link/collision/geometry/box/size") == "0.500 0.500 0.500"
    assert model.findtext("link/pose") == "0 0 0.250 0 0 0"


def test_floor_models_keep_their_declared_origin_height(tmp_path, monkeypatch):
    directory = _write_box_model(tmp_path, "lifted_box", "0.5 0.4 0.3")
    sdf = (directory / "model.sdf").read_text()
    (directory / "model.sdf").write_text(sdf.replace("<link", "<pose>0 0 0.15 0 0 0</pose><link", 1))
    monkeypatch.setenv("GAZEBO_MODEL_PATH", str(tmp_path))
    database = MagicMock()
    database.find_best_model.return_value = "model://lifted_box"
    engine = NaturalPlacementEngine(model_db=database)
    plan = [{"id": "box_0", "type": "box", "x": 0.0, "y": 0.0, "yaw": 0.0, "dims": (0.5, 0.4, 0.3)}]
    models = engine._create_models_from_plan(plan, _room(), lambda name: name + "_0")
    assert models[0].pose["z"] == pytest.approx(0.15)
