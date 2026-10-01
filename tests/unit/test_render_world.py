"""render_world helpers that do not need Gazebo."""

import math
import xml.etree.ElementTree as ET

import pytest

from gazebo_world_generator.render_world import (
    _camera_height, _camera_model, _camera_pose, _lower_walls, room_bounds)

WORLD = """<sdf version='1.7'><world name='w'>
  <model name='Office_wall_north'><pose>0 3 1.5 0 0 0</pose><link name='l'><collision name='c'>
    <geometry><box><size>6 0.2 3</size></box></geometry></collision></link></model>
  <model name='Office_wall_east'><pose>3 0 1.5 0 0 1.5708</pose><link name='l'><collision name='c'>
    <geometry><box><size>6 0.2 3</size></box></geometry></collision></link></model>
  <model name='Office_wall_south_left'><pose>-2 -3 1.5 0 0 0</pose><link name='l'><collision name='c'>
    <geometry><box><size>2 0.2 3</size></box></geometry></collision></link></model>
  <model name='ground_plane'><pose>0 0 0 0 0 0</pose></model>
</world></sdf>"""


def test_room_bounds_come_from_wall_models():
    world = ET.fromstring(WORLD).find("world")
    (name, bounds), = room_bounds(world).items()
    assert name == "Office"
    assert [round(v, 2) for v in bounds] == [-3.0, 3.1, -3.1, 3.1]


def test_walls_are_lowered_only_in_the_render_copy():
    world = ET.fromstring(WORLD).find("world")
    _lower_walls(world)
    wall = world.find("model[@name='Office_wall_north']")
    assert wall.findtext("pose").split()[2] == "0.050"
    assert wall.findtext(".//size") == "6 0.2 0.100"
    assert world.find("model[@name='ground_plane']").findtext("pose") == "0 0 0 0 0 0"


def test_top_camera_covers_the_room_and_points_down(tmp_path):
    bounds = (-3.0, 3.0, -2.0, 2.0)
    x, y, z, roll, pitch, yaw = (float(v) for v in _camera_pose(bounds, oblique=False).split())
    assert (x, y) == (0.0, 0.0) and pitch == pytest.approx(math.pi / 2, abs=1e-3)
    assert z == pytest.approx(_camera_height(bounds), abs=1e-3) and z > 3 / math.tan(0.35)
    sensor = _camera_model("Office", bounds, tmp_path, "classic").find("link/sensor")
    assert sensor.get("type") == "camera"
    assert sensor.find("camera/save").get("enabled") == "true"
    assert sensor.findtext("camera/save/path") == str(tmp_path)


def test_angled_camera_looks_north_from_the_south():
    x, y, z, _, pitch, yaw = (float(v) for v in _camera_pose((-3.0, 3.0, -2.0, 2.0), oblique=True).split())
    assert y < -2.0 and z > 0 and 0 < pitch < math.pi / 2 and yaw == pytest.approx(math.pi / 2, abs=1e-3)
