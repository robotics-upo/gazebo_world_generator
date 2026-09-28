import json
from unittest.mock import MagicMock, patch

import pytest

from gazebo_world_generator.src.core.data_models import GazeboModel, Room
from gazebo_world_generator.src.placement.engine import NaturalPlacementEngine
from gazebo_world_generator.src.prompts.manager import PromptManager


def _room():
    return Room(name="Office", type="office", dimensions={"width": 6, "length": 5, "height": 3},
                position={"x": 10, "y": 0, "z": 0})


class ScriptedLLM:
    """Returns canned answers and records every request."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []
        self.context_window = 32768
        self.prompt_manager = PromptManager(template_dir="prompts", default_version="v1",
                                            enable_metrics=False)

    def query(self, messages, max_tokens=4096, temperature=0.6):
        self.requests.append(messages)
        return self.answers.pop(0) if self.answers else ""


def test_missing_requested_model_fails_before_llm_design():
    database = MagicMock()
    database.find_best_model.return_value = None
    llm = ScriptedLLM()
    engine = NaturalPlacementEngine(model_db=database, llm_interface=llm)
    with pytest.raises(ValueError, match="No visible model found"):
        engine.place_objects_in_room(_room(), "", [{"type": "desk", "count": 1}], lambda n: n)
    assert llm.requests == []


def test_resolve_models_early():
    model_db = MagicMock()
    model_db.find_best_model.side_effect = lambda t, c: f"model://{t}"
    engine = NaturalPlacementEngine(model_db=model_db)
    room = Room(name="Test Room", type="office", dimensions={"width": 10, "length": 10, "height": 3},
                position={"x": 0, "y": 0, "z": 0})
    with patch.object(NaturalPlacementEngine, 'get_actual_model_dimensions', return_value=(1.5, 0.8, 0.75)):
        resolved = engine._resolve_models_early([{"type": "desk"}, {"type": "chair"}], room)
    assert resolved["desk"]["uri"] == "model://desk"
    assert resolved["desk"]["dimensions"] == (1.5, 0.8, 0.75)
    assert resolved["desk"]["offset"] == (0.0, 0.0, 0.75)


def test_llm_layout_becomes_world_frame_models():
    database = MagicMock()
    database.find_best_model.side_effect = lambda t, c: f"model://{t}"
    answer = {"requirements": [{"check": "near", "a": "chair_0", "b": "desk_0", "max": 0.5}],
              "objects": [{"id": "desk_0", "x": 1.0, "y": 1.0, "yaw": 0.0},
                          {"id": "chair_0", "x": 1.0, "y": 0.2, "yaw": 1.57},
                          {"id": "monitor_0", "x": 1.0, "y": 1.1, "yaw": 0.0, "on": "desk_0"}]}
    llm = ScriptedLLM(json.dumps(answer))
    engine = NaturalPlacementEngine(model_db=database, llm_interface=llm)
    engine.user_description = "office with a desk, chair and monitor"
    sizes = {"desk": (1.4, 0.7, 0.75), "chair": (0.6, 0.6, 1.0), "monitor": (0.5, 0.2, 0.4)}
    with patch.object(NaturalPlacementEngine, 'get_actual_model_dimensions',
                      side_effect=lambda t, *_: sizes[t]):
        models = engine.place_objects_in_room(
            _room(), "", [{"type": "desk"}, {"type": "chair"}, {"type": "monitor"}],
            lambda n: n + "_w")

    assert len(llm.requests) == 1  # clean on the first round
    prompt = llm.requests[0][0]["content"][0]["text"]
    assert "office with a desk, chair and monitor" in prompt
    assert "desk_0" in prompt and "1.40 x 0.70 x 0.75" in prompt
    poses = {model.name: model.pose for model in models}
    assert poses["desk_w"]["x"] == pytest.approx(11.0)  # room offset applied
    assert poses["chair_w"]["yaw"] == pytest.approx(1.57)
    assert poses["monitor_w"]["z"] == pytest.approx(0.75)  # on the desk top


def test_without_llm_objects_are_sampled_without_overlap():
    database = MagicMock()
    database.find_best_model.side_effect = lambda t, c: f"model://{t}"
    engine = NaturalPlacementEngine(model_db=database)
    with patch.object(NaturalPlacementEngine, 'get_actual_model_dimensions', return_value=(0.8, 0.4, 1.8)):
        models = engine.place_objects_in_room(_room(), "", [{"type": "rack", "count": 6}], lambda n: n)
    assert len(models) == 6
    boxes = [(m.pose["x"], m.pose["y"], m.pose["yaw"]) for m in models]
    assert len(set(boxes)) == 6


def test_existing_objects_become_fixed_obstacles():
    room = _room()
    room.objects = [GazeboModel(name="old_desk", model_path="existing", category="desk",
                                pose={"x": 10.0, "y": 0.0, "z": 0, "yaw": 0.0}, size=[1.0, 0.6, 0.7])]
    fixed = NaturalPlacementEngine._fixed_items(room)
    assert fixed == [{"id": "old_desk", "type": "desk", "fixed": True, "x": 0.0, "y": 0.0,
                      "yaw": 0.0, "dims": (1.0, 0.6, 0.7)}]
