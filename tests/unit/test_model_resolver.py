import pytest
from unittest.mock import MagicMock, patch
from gazebo_world_generator.src.models.resolver import SmartModelResolver
from gazebo_world_generator.src.models.visual_quality import inspect_model_visuals
from gazebo_world_generator.src.utils.sdf_parser import SDFDimensionExtractor


def _write_box_model(base, name, dimensions):
    directory = base / name
    directory.mkdir()
    (directory / "model.config").write_text(
        f"<model><name>{name}</name><sdf>model.sdf</sdf></model>")
    (directory / "model.sdf").write_text(
        f"<sdf version='1.7'><model name='{name}'><link name='link'>"
        f"<visual name='visual'><geometry><box><size>{dimensions}</size></box>"
        f"</geometry></visual></link></model></sdf>")
    return directory

def test_model_resolver_fallback():
    resolver = SmartModelResolver()
    # Test fallback for unknown object
    fallback = resolver._get_fallback_model("completely_unknown_type_xyz")
    assert fallback is None
    
    # Test fallback for known object
    fallback = resolver._get_fallback_model("desk")
    assert "desk" in fallback or "table" in fallback

def test_find_best_model_keyword_match(tmp_path):
    _write_box_model(tmp_path, "OfficeChair", "0.6 0.6 0.9")
    _write_box_model(tmp_path, "WoodenTable", "1.2 0.8 0.75")
    resolver = SmartModelResolver(search_paths=[tmp_path], enable_cache=False)
    model = resolver.find_best_model("chair")
    assert "chair" in model.lower()


def test_rejects_small_accessory_and_stale_cached_choice(tmp_path):
    _write_box_model(tmp_path, "DeskPortrait", "0.17 0.10 0.24")
    _write_box_model(tmp_path, "ReadingDesk", "1.1 0.6 0.74")
    resolver = SmartModelResolver(search_paths=[tmp_path], enable_cache=False)
    resolver.model_cache["desk_office"] = "model://DeskPortrait"
    assert resolver.find_best_model("desk", "office") == "model://ReadingDesk"
    assert resolver.model_cache["desk_office"] == "model://ReadingDesk"


def test_missing_visual_mesh_is_rejected(tmp_path):
    directory = _write_box_model(tmp_path, "BrokenDesk", "1 0.6 0.7")
    (directory / "model.sdf").write_text(
        "<sdf version='1.7'><model name='BrokenDesk'><link name='link'>"
        "<visual name='visual'><geometry><mesh><uri>meshes/missing.dae</uri>"
        "</mesh></geometry></visual></link></model></sdf>")
    assert "missing visual mesh" in inspect_model_visuals(directory).error
    resolver = SmartModelResolver(search_paths=[tmp_path], enable_cache=False)
    assert resolver.find_best_model("desk") is None


def test_invalid_visual_size_is_reported(tmp_path):
    directory = _write_box_model(tmp_path, "BrokenDesk", "1 1 1")
    (directory / "model.sdf").write_text(
        "<sdf version='1.7'><model name='BrokenDesk'><link name='link'>"
        "<visual name='v'><geometry><box/></geometry></visual></link></model></sdf>")
    assert "invalid visual geometry" in inspect_model_visuals(directory).error


def test_partial_object_name_can_select_shelf(tmp_path):
    _write_box_model(tmp_path, "Shelf", "1.0 0.5 1.8")
    resolver = SmartModelResolver(search_paths=[tmp_path], enable_cache=False)
    assert resolver.find_best_model("shelving_unit") == "model://Shelf"


def test_collada_mesh_dimensions_replace_name_guess(tmp_path):
    directory = _write_box_model(tmp_path, "DeskPortrait", "1 1 1")
    meshes = directory / "meshes"
    meshes.mkdir()
    (meshes / "portrait.dae").write_text("""
<COLLADA xmlns='http://www.collada.org/2005/11/COLLADASchema'>
  <asset><unit meter='0.01'/><up_axis>Z_UP</up_axis></asset>
  <library_geometries><geometry><mesh>
    <source id='positions'><float_array id='array' count='9'>0 0 0 17 0 0 0 10 24</float_array>
      <technique_common><accessor source='#array' count='3' stride='3'/></technique_common>
    </source>
    <vertices id='vertices'><input semantic='POSITION' source='#positions'/></vertices>
  </mesh></geometry></library_geometries>
</COLLADA>""")
    (directory / "model.sdf").write_text(
        "<sdf version='1.7'><model name='DeskPortrait'><link name='link'>"
        "<visual name='visual'><geometry><mesh><uri>meshes/portrait.dae</uri>"
        "</mesh></geometry></visual></link></model></sdf>")
    dimensions = SDFDimensionExtractor().extract_model_dimensions(directory)
    assert dimensions == pytest.approx((0.17, 0.10, 0.24))


def _write_collada_model(base, name, scene):
    """Model whose mesh has one 2 x 1 x 1 box geometry and the given scene."""
    directory = _write_box_model(base, name, "1 1 1")
    (directory / "meshes").mkdir()
    (directory / "meshes" / "mesh.dae").write_text(f"""
<COLLADA xmlns='http://www.collada.org/2005/11/COLLADASchema'>
  <asset><unit meter='0.01'/><up_axis>Z_UP</up_axis></asset>
  <library_geometries><geometry id='box'><mesh>
    <source id='positions'><float_array id='array' count='6'>0 0 0 2 1 1</float_array>
      <technique_common><accessor source='#array' count='2' stride='3'/></technique_common>
    </source>
    <vertices id='vertices'><input semantic='POSITION' source='#positions'/></vertices>
  </mesh></geometry></library_geometries>
  {scene}
</COLLADA>""")
    (directory / "model.sdf").write_text(
        f"<sdf version='1.7'><model name='{name}'><link name='link'>"
        "<visual name='visual'><geometry><mesh><uri>meshes/mesh.dae</uri>"
        "</mesh></geometry></visual></link></model></sdf>")
    return directory


def test_collada_node_scale_is_applied(tmp_path):
    # Fuel's Office Desk: centimetre units, but nodes scale geometry by 100.
    directory = _write_collada_model(tmp_path, "ScaledDesk", """
  <library_visual_scenes><visual_scene id='scene'><node>
    <matrix>100 0 0 0 0 100 0 0 0 0 100 0 0 0 0 1</matrix>
    <instance_geometry url='#box'/>
  </node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url='#scene'/></scene>""")
    assert inspect_model_visuals(directory).dimensions == pytest.approx((2.0, 1.0, 1.0))


def test_collada_nested_rotation_and_instance_node(tmp_path):
    directory = _write_collada_model(tmp_path, "Composite", """
  <library_nodes><node id='shifted'><translate>0 3 0</translate>
    <instance_geometry url='#box'/></node></library_nodes>
  <library_visual_scenes><visual_scene id='scene'><node>
    <scale>100 100 100</scale>
    <node><rotate>0 0 1 90</rotate><instance_geometry url='#box'/></node>
    <instance_node url='#shifted'/>
  </node></visual_scene></library_visual_scenes>""")
    # Rotated box spans x in [-1, 0]; shifted box spans x in [0, 2], y in [3, 4].
    assert inspect_model_visuals(directory).dimensions == pytest.approx((3.0, 4.0, 1.0))

def test_model_name_from_config(tmp_path):
    resolver = SmartModelResolver()
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    config_file = model_dir / "model.config"
    config_file.write_text("""<?xml version="1.0"?>
<model>
  <name>Test Model Name</name>
</model>""")
    
    name = resolver._get_model_name_from_config(model_dir)
    assert name == "Test Model Name"


def test_model_scan_cache_changes_with_resource_path(tmp_path, monkeypatch):
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    primary.mkdir()
    secondary.mkdir()
    resolver = SmartModelResolver(search_paths=[primary], enable_cache=False)
    first_key = resolver._scan_cache_key()
    monkeypatch.setenv("GZ_SIM_RESOURCE_PATH", str(secondary))
    assert resolver._scan_cache_key() != first_key


def test_duplicate_config_name_prefers_matching_directory(tmp_path):
    _write_box_model(tmp_path, "Desk", "1.2 0.7 0.75")
    duplicate = _write_box_model(tmp_path, "Office_Desk", "1.2 0.7 0.75")
    (duplicate / "model.config").write_text(
        "<model><name>Desk</name><sdf>model.sdf</sdf></model>")
    resolver = SmartModelResolver(search_paths=[tmp_path], enable_cache=False)
    assert resolver.get_directory_for_model("Desk") == "Desk"
    assert resolver.find_best_model("desk") == "model://Desk"


def test_spaced_model_directory_gets_relative_symlink(tmp_path):
    _write_box_model(tmp_path, "Office Desk", "1.2 0.7 0.75")
    resolver = SmartModelResolver(search_paths=[tmp_path], enable_cache=False)
    assert "Office Desk" in resolver.local_models
    link = tmp_path / "Office_Desk"
    assert link.is_symlink() and str(link.readlink()) == "Office Desk"


def test_model_shape_reads_collada_polylist_through_scene(tmp_path):
    from gazebo_world_generator.src.models.visual_quality import model_shape
    directory = _write_box_model(tmp_path, "Quad", "1 1 1")
    (directory / "meshes").mkdir()
    (directory / "meshes" / "quad.dae").write_text("""
<COLLADA xmlns='http://www.collada.org/2005/11/COLLADASchema'>
  <library_geometries><geometry id='quad'><mesh>
    <source id='positions'><float_array id='array' count='12'>0 0 0 1 0 0 1 1 0 0 1 1</float_array>
      <technique_common><accessor source='#array' count='4' stride='3'/></technique_common></source>
    <vertices id='vertices'><input semantic='POSITION' source='#positions'/></vertices>
    <polylist count='1'><input semantic='VERTEX' source='#vertices' offset='0'/>
      <input semantic='NORMAL' source='#normals' offset='1'/>
      <vcount>4</vcount><p>0 0 1 0 2 0 3 0</p></polylist>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id='scene'><node><translate>10 0 0</translate>
    <instance_geometry url='#quad'/></node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url='#scene'/></scene>
</COLLADA>""")
    (directory / "model.sdf").write_text(
        "<sdf version='1.7'><model name='Quad'><link name='link'><pose>0 0 1 0 0 0</pose>"
        "<visual name='v'><geometry><mesh><uri>meshes/quad.dae</uri><scale>2 1 1</scale></mesh>"
        "</geometry></visual></link></model></sdf>")
    triangles = model_shape(directory)
    assert len(triangles) == 2
    assert triangles[0] == ((20.0, 0.0, 1.0), (22.0, 0.0, 1.0), (22.0, 1.0, 1.0))


def test_model_shape_of_primitive_box(tmp_path):
    from gazebo_world_generator.src.models.visual_quality import model_shape
    triangles = model_shape(_write_box_model(tmp_path, "Crate", "0.6 0.4 0.5"))
    points = {point for triangle in triangles for point in triangle}
    assert points == {(-0.3, -0.2, 0.25), (0.3, -0.2, 0.25), (0.3, 0.2, 0.25), (-0.3, 0.2, 0.25)}
