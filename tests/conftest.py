import pytest
import os
import tempfile
from unittest.mock import MagicMock

@pytest.fixture
def sample_sdf_content():
    return """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="sample_model">
    <link name="link">
      <collision name="collision">
        <geometry>
          <box>
            <size>1.0 2.0 3.0</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""

@pytest.fixture
def sample_sdf_file(sample_sdf_content):
    with tempfile.NamedTemporaryFile(suffix=".sdf", mode="w", delete=False) as tmp:
        tmp.write(sample_sdf_content)
        tmp_path = tmp.name
    yield tmp_path
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

@pytest.fixture
def mock_llm_interface():
    mock = MagicMock()
    mock.parse_room_description.return_value = {
        "rooms": [
            {
                "name": "Living Room",
                "type": "living_room",
                "dimensions": {"width": 5.0, "length": 5.0, "height": 3.0},
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "objects": [
                    {"type": "sofa", "count": 1},
                    {"type": "coffee_table", "count": 1}
                ]
            }
        ]
    }
    return mock
