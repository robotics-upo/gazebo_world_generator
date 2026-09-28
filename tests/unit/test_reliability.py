import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gazebo_world_generator.gazebo_world_generator import generate_occupancy_map, run_generation
from gazebo_world_generator.src.config.validated_settings import ValidatedConfig, ModelsConfig
from gazebo_world_generator.src.models.online_database import OnlineModelDatabase
from gazebo_world_generator.src.utils.cache import PersistentCache
from gazebo_world_generator.src.utils.output_manager import OutputManager
from gazebo_world_generator.src.utils.token_estimator import estimate_tokens
from gazebo_world_generator.src.world.generator import WorldGenerator


def test_config_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    user = tmp_path / ".config/gazebo_world_generator/generator_config.yaml"
    user.parent.mkdir(parents=True)
    user.write_text("llm:\n  model_name: user-model\noutput:\n  auto_generate_map: false\n")
    monkeypatch.setenv("GAZEBO_WORLD_GEN_LLM__MODEL_NAME", "env-model")
    config = ValidatedConfig.from_multiple_sources({"llm": {"model_name": "cli-model"}})
    assert config.llm.model_name == "cli-model"
    assert config.output.auto_generate_map is False
    assert config.placement.grid_resolution > 0


def test_default_simulator_follows_ros_distro(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GAZEBO_WORLD_GEN_SIMULATOR", raising=False)
    monkeypatch.setenv("ROS_DISTRO", "humble")
    assert ValidatedConfig.from_multiple_sources().simulator == "classic"
    monkeypatch.setenv("ROS_DISTRO", "jazzy")
    assert ValidatedConfig.from_multiple_sources().simulator == "harmonic"
    assert ValidatedConfig.from_multiple_sources({"simulator": "classic"}).simulator == "classic"


def test_output_directory_expands_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    manager = OutputManager(Path("~/generated"))
    assert manager.base_output_dir == tmp_path / "generated"


def test_token_estimator_accounts_for_multibyte_text():
    ascii_count = estimate_tokens([{"content": "aaaa"}])
    unicode_count = estimate_tokens([{"content": "猫猫猫猫"}])
    assert unicode_count > ascii_count


def test_explicit_output_is_exact(tmp_path):
    destination = tmp_path / "custom/world.sdf"
    config = ValidatedConfig(output={"base_directory": tmp_path / "organized",
                                     "auto_generate_map": False})
    with patch("gazebo_world_generator.gazebo_world_generator.WorldGenerator") as generator_type, \
         patch("gazebo_world_generator.gazebo_world_generator.OpenAICompatibleInterface"):
        generator = generator_type.return_value
        generator.generate_world.return_value = str(destination)
        generator.rooms = []
        generator.models = []
        assert run_generation("empty room", destination, config=config) == 0
        generator.generate_world.assert_called_once_with("empty room", str(destination))


def test_repeated_generation_and_failed_llm(mock_llm_interface, tmp_path):
    generator = WorldGenerator(mock_llm_interface)
    mock_llm_interface.parse_room_description.return_value = {
        "rooms": [{"name": "Room", "type": "office",
                   "dimensions": {"width": 4, "length": 4}, "objects": []}]}
    output = tmp_path / "world.sdf"
    generator.generate_world("room", str(output))
    first_model_count = len(generator.models)
    generator.generate_world("room", str(output))
    assert len(generator.models) == first_model_count
    prior = output.read_bytes()
    mock_llm_interface.parse_room_description.return_value = None
    with pytest.raises(ValueError):
        generator.generate_world("bad response", str(output))
    assert output.read_bytes() == prior


def test_seed_resets_between_runs(mock_llm_interface, tmp_path):
    mock_llm_interface.parse_room_description.return_value = {"rooms": [{
        "name": "Room", "type": "office", "dimensions": {"width": 4, "length": 4},
        "objects": []}]}
    generator = WorldGenerator(mock_llm_interface,
                               config=ValidatedConfig(placement={"random_seed": 19}))
    expected = generator.placement_engine.rng.random()
    generator.generate_world("room", str(tmp_path / "world.sdf"))
    assert generator.placement_engine.rng.random() == expected


def test_map_requires_both_files(tmp_path):
    with patch("gazebo_world_generator.gazebo_world_generator.subprocess.run") as run:
        run.return_value.returncode = 0
        with pytest.raises(RuntimeError, match="missing or empty"):
            generate_occupancy_map("world.sdf", tmp_path)


def test_map_failure_reports_script_output(tmp_path):
    with patch("gazebo_world_generator.gazebo_world_generator.subprocess.run") as run:
        run.return_value.returncode = 1
        run.return_value.stderr = ""
        run.return_value.stdout = "plugin failed to load"
        with pytest.raises(RuntimeError, match="plugin failed to load"):
            generate_occupancy_map("world.sdf", tmp_path)


def test_map_does_not_interrupt_active_classic_simulator(tmp_path):
    with patch("gazebo_world_generator.gazebo_world_generator._classic_server_running",
               return_value=True), \
         patch("gazebo_world_generator.gazebo_world_generator.subprocess.run") as run:
        with pytest.raises(RuntimeError, match="Close the active Gazebo Classic server"):
            generate_occupancy_map("world.sdf", tmp_path)
        run.assert_not_called()


def test_hostile_archive_is_rejected(tmp_path):
    config = ModelsConfig(cache_directory=tmp_path)
    database = OnlineModelDatabase(config=config)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "bad")
    response = MagicMock()
    response.url = "https://fuel.gazebosim.org/model.zip"
    response.iter_content.return_value = [archive.getvalue()]
    database._make_request_with_retry = MagicMock(return_value=response)
    with pytest.raises(ValueError, match="Unsafe"):
        database.download_model({"name": "model", "download_url": response.url})
    assert not (tmp_path.parent / "outside.txt").exists()


def test_invalid_model_preserves_existing_directory(tmp_path):
    database = OnlineModelDatabase(config=ModelsConfig(cache_directory=tmp_path))
    existing = tmp_path / "model"
    existing.mkdir()
    (existing / "marker").write_text("keep")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("model/model.sdf", "<not-sdf/>")
        handle.writestr("model/model.config", "<model/>")
    response = MagicMock()
    response.url = "https://fuel.gazebosim.org/model.zip"
    response.iter_content.return_value = [archive.getvalue()]
    database._make_request_with_retry = MagicMock(return_value=response)
    with pytest.raises(ValueError, match="model.sdf"):
        database.download_model({"name": "model", "download_url": response.url})
    assert (existing / "marker").read_text() == "keep"


def test_unsafe_model_config_reference_preserves_existing_directory(tmp_path):
    database = OnlineModelDatabase(config=ModelsConfig(cache_directory=tmp_path))
    existing = tmp_path / "model"
    existing.mkdir()
    (existing / "marker").write_text("keep")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("model/model.sdf", "<sdf version='1.7'><model name='x'><link name='l'><visual name='v'><geometry><box><size>1 1 1</size></box></geometry></visual></link></model></sdf>")
        handle.writestr("model/model.config", "<model><name>x</name><sdf>../other.sdf</sdf></model>")
    response = MagicMock()
    response.url = "https://fuel.gazebosim.org/model.zip"
    response.iter_content.return_value = [archive.getvalue()]
    database._make_request_with_retry = MagicMock(return_value=response)
    with pytest.raises(ValueError, match="Unsafe SDF reference"):
        database.download_model({"name": "model", "download_url": response.url})
    assert (existing / "marker").read_text() == "keep"


def test_valid_download_replaces_invalid_existing_model(tmp_path):
    database = OnlineModelDatabase(config=ModelsConfig(cache_directory=tmp_path))
    existing = tmp_path / "model"
    existing.mkdir()
    (existing / "model.sdf").write_text("<broken/>")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("model/model.sdf", "<sdf version='1.7'><model name='x'><link name='l'><visual name='v'><geometry><box><size>1 1 1</size></box></geometry></visual></link></model></sdf>")
        handle.writestr("model/model.config", "<model><name>x</name><sdf>model.sdf</sdf></model>")
    response = MagicMock()
    response.url = "https://fuel.gazebosim.org/model.zip"
    response.iter_content.return_value = [archive.getvalue()]
    database._make_request_with_retry = MagicMock(return_value=response)

    assert database.download_model({"name": "model", "download_url": response.url}) == existing
    assert "<sdf" in (existing / "model.sdf").read_text()
    assert database.download_model({"name": "model", "download_url": response.url}) == existing
    database._make_request_with_retry.assert_called_once()


def test_download_without_visual_preserves_existing_model(tmp_path):
    database = OnlineModelDatabase(config=ModelsConfig(cache_directory=tmp_path))
    existing = tmp_path / "model"
    existing.mkdir()
    (existing / "marker").write_text("keep")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("model/model.sdf", "<sdf version='1.7'><model name='x'><link name='l'/></model></sdf>")
        handle.writestr("model/model.config", "<model><name>x</name><sdf>model.sdf</sdf></model>")
    response = MagicMock()
    response.url = "https://fuel.gazebosim.org/model.zip"
    response.iter_content.return_value = [archive.getvalue()]
    database._make_request_with_retry = MagicMock(return_value=response)
    with pytest.raises(ValueError, match="no visual geometry"):
        database.download_model({"name": "model", "download_url": response.url})
    assert (existing / "marker").read_text() == "keep"


def test_archive_link_and_size_limits(tmp_path):
    database = OnlineModelDatabase(config=ModelsConfig(cache_directory=tmp_path,
                                                         max_archive_bytes=10))
    with pytest.raises(ValueError, match="Unsafe"):
        database._validate_archive_members([("model/link", 1, True)])
    with pytest.raises(ValueError, match="size limit"):
        database._validate_archive_members([("model/model.sdf", 11, False)])
    with pytest.raises(ValueError, match="not allowed"):
        database.download_model({"name": "model", "download_url": "https://evil.example/a.zip"})


def test_cache_recovers_from_corrupt_data(tmp_path):
    cache = PersistentCache("models", cache_dir=tmp_path, auto_save=False)
    cache.set("item", {"names": {"a", "b"}})
    cache.save()
    loaded = PersistentCache("models", cache_dir=tmp_path, auto_save=False)
    assert loaded.get("item") == {"names": ["a", "b"]}
    cache.cache_file.write_text("not json")
    recovered = PersistentCache("models", cache_dir=tmp_path, auto_save=False)
    assert recovered.get("item") is None
