import pytest
import os
import tempfile
from pathlib import Path
from gazebo_world_generator.src.prompts.manager import PromptManager

@pytest.fixture
def mock_template_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        v1_dir = Path(tmp_dir) / "v1"
        v1_dir.mkdir()
        (v1_dir / "test_template.j2").write_text("Hello {{ name }}!")
        yield Path(tmp_dir)

def test_prompt_manager_initialization(mock_template_dir):
    pm = PromptManager(template_dir=mock_template_dir)
    assert pm.default_version == 'v1'
    assert pm.template_dir == mock_template_dir

def test_prompt_manager_render(mock_template_dir):
    pm = PromptManager(template_dir=mock_template_dir)
    rendered = pm.render("test_template", name="World")
    assert rendered == "Hello World!"

def test_prompt_manager_metrics(mock_template_dir):
    pm = PromptManager(template_dir=mock_template_dir, enable_metrics=True)
    pm.log_success("test_template", version="v1", tokens_used=10, response_time=1.0)
    metrics = pm.get_metrics("test_template")
    assert metrics["v1/test_template"]['successes'] == 1
    assert metrics["v1/test_template"]['total_tokens'] == 10
