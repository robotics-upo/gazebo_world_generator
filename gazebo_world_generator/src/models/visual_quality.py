"""Check that a Gazebo model has usable visual geometry and measure its mesh."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import struct
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class VisualInspection:
    dimensions: tuple[float, float, float] | None = None
    error: str | None = None


def _mesh_path(uri: str, model_dir: Path, search_paths: tuple[Path, ...]) -> Path | None:
    if not uri or ".." in Path(uri).parts and not uri.startswith("model://"):
        return None
    if uri.startswith("model://"):
        parts = uri.removeprefix("model://").split("/", 1)
        if len(parts) != 2 or ".." in Path(parts[1]).parts:
            return None
        for base in (model_dir.parent, *search_paths):
            candidate = base / parts[0] / parts[1]
            if candidate.is_file():
                return candidate
        return None
    if "://" in uri or Path(uri).is_absolute():
        return None
    candidate = model_dir / uri
    return candidate if candidate.is_file() else None


_IDENTITY = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
# Guards against cyclic or absurdly deep <instance_node> references.
_MAX_NODE_DEPTH = 64


def _matmul(a, b):
    return tuple(tuple(sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4))
                 for row in range(4))


def _node_transform(node) -> tuple:
    """Compose a COLLADA node's transform elements in document order."""
    matrix = _IDENTITY
    for element in node:
        tag = element.tag.rsplit("}", 1)[-1]
        values = [float(value) for value in (element.text or "").split()]
        if tag == "matrix" and len(values) == 16:
            local = tuple(tuple(values[row * 4:row * 4 + 4]) for row in range(4))
        elif tag == "translate" and len(values) == 3:
            x, y, z = values
            local = ((1, 0, 0, x), (0, 1, 0, y), (0, 0, 1, z), (0, 0, 0, 1))
        elif tag == "scale" and len(values) == 3:
            x, y, z = values
            local = ((x, 0, 0, 0), (0, y, 0, 0), (0, 0, z, 0), (0, 0, 0, 1))
        elif tag == "rotate" and len(values) == 4:
            x, y, z, degrees = values
            norm = math.sqrt(x * x + y * y + z * z)
            if norm == 0:
                continue
            x, y, z = x / norm, y / norm, z / norm
            c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
            t = 1 - c
            local = ((t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0),
                     (t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0),
                     (t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0),
                     (0, 0, 0, 1))
        else:
            continue
        matrix = _matmul(matrix, local)
    return matrix


def _collada_geometry_vertices(root) -> dict:
    """Map each geometry id to its untransformed POSITION vertices."""
    geometries = {}
    for geometry in root.findall(".//{*}library_geometries/{*}geometry"):
        mesh = geometry.find("{*}mesh")
        if mesh is None:
            continue
        position_ids = {
            item.get("source", "").lstrip("#")
            for item in mesh.findall("{*}vertices/{*}input")
            if item.get("semantic") == "POSITION"
        }
        vertices = []
        for source in mesh.findall("{*}source"):
            if source.get("id") not in position_ids:
                continue
            array = source.find("{*}float_array")
            if array is None or not array.text:
                continue
            accessor = source.find("{*}technique_common/{*}accessor")
            stride = int(accessor.get("stride", "3")) if accessor is not None else 3
            values = [float(value) for value in array.text.split()]
            vertices.extend(tuple(values[index:index + 3])
                            for index in range(0, len(values) - stride + 1, stride))
        geometries[geometry.get("id")] = vertices
    return geometries


def _collada_scene_vertices(root, geometries: dict) -> list | None:
    """Transform instanced geometry through the visual scene graph.

    Returns None when the file has no visual scene, so callers can fall back
    to the raw geometry library.
    """
    scenes = {scene.get("id"): scene
              for scene in root.findall(".//{*}library_visual_scenes/{*}visual_scene")}
    if not scenes:
        return None
    instance = root.find("{*}scene/{*}instance_visual_scene")
    scene = scenes.get((instance.get("url") or "").lstrip("#")) if instance is not None else None
    if scene is None:
        scene = next(iter(scenes.values()))
    library_nodes = {node.get("id"): node
                     for node in root.findall(".//{*}library_nodes/{*}node")}
    vertices = []

    def visit(node, parent, depth):
        if depth > _MAX_NODE_DEPTH:
            raise ValueError("COLLADA node hierarchy is too deep")
        matrix = _matmul(parent, _node_transform(node))
        for child in node:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "instance_geometry":
                for x, y, z in geometries.get((child.get("url") or "").lstrip("#"), ()):
                    vertices.append(tuple(
                        matrix[row][0] * x + matrix[row][1] * y + matrix[row][2] * z + matrix[row][3]
                        for row in range(3)))
            elif tag == "node":
                visit(child, matrix, depth + 1)
            elif tag == "instance_node":
                referenced = library_nodes.get((child.get("url") or "").lstrip("#"))
                if referenced is not None:
                    visit(referenced, matrix, depth + 1)

    for node in scene.findall("{*}node"):
        visit(node, _IDENTITY, 0)
    return vertices


@lru_cache(maxsize=512)
def _mesh_dimensions(path: Path, mtime_ns: int, size: int) -> VisualInspection:
    """Read vertices without importing NumPy or a simulator at startup."""
    try:
        suffix = path.suffix.lower()
        if suffix == ".dae":
            root = ET.parse(path).getroot()
            unit = root.find(".//{*}asset/{*}unit")
            scale = float(unit.get("meter", "1")) if unit is not None else 1.0
            geometries = _collada_geometry_vertices(root)
            vertices = _collada_scene_vertices(root, geometries)
            if vertices is None:
                vertices = [vertex for items in geometries.values() for vertex in items]
            vertices = [(x * scale, y * scale, z * scale) for x, y, z in vertices]
            for image in root.findall(".//{*}library_images/{*}image/{*}init_from"):
                reference = (image.text or "").strip()
                if reference and not reference.startswith("data:") and not (path.parent / reference).is_file():
                    return VisualInspection(error=f"missing texture {reference}")
            up_axis = root.findtext(".//{*}asset/{*}up_axis", default="Z_UP")
            if up_axis == "Y_UP":
                vertices = [(x, z, y) for x, y, z in vertices]
            elif up_axis == "X_UP":
                vertices = [(z, y, x) for x, y, z in vertices]
        elif suffix == ".obj":
            vertices = []
            with path.open(errors="replace") as stream:
                for line in stream:
                    if line.startswith("v "):
                        vertices.append(tuple(float(value) for value in line.split()[1:4]))
                    elif line.startswith("mtllib "):
                        for name in line.split()[1:]:
                            if not (path.parent / name).is_file():
                                return VisualInspection(error=f"missing material {name}")
        elif suffix == ".stl":
            content = path.read_bytes()
            vertices = []
            if len(content) >= 84 and 84 + 50 * struct.unpack_from("<I", content, 80)[0] == len(content):
                count = struct.unpack_from("<I", content, 80)[0]
                for index in range(count):
                    values = struct.unpack_from("<12f", content, 84 + 50 * index)
                    vertices.extend((values[3:6], values[6:9], values[9:12]))
            else:
                for line in content.decode("ascii", errors="ignore").splitlines():
                    if line.lstrip().startswith("vertex "):
                        vertices.append(tuple(float(value) for value in line.split()[1:4]))
        else:
            return VisualInspection()
        if not vertices:
            return VisualInspection(error=f"no vertices in {path.name}")
        dimensions = tuple(max(vertex[axis] for vertex in vertices) -
                           min(vertex[axis] for vertex in vertices) for axis in range(3))
        if min(dimensions) <= 0:
            return VisualInspection(error=f"degenerate mesh {path.name}")
        return VisualInspection(dimensions=dimensions)
    except (ET.ParseError, OSError, ValueError, struct.error) as exc:
        return VisualInspection(error=f"unreadable mesh {path.name}: {exc}")


def _inspect_model_visuals(model_dir: Path, search_paths=()) -> VisualInspection:
    """Reject missing visuals or assets; return measured dimensions when possible."""
    model_dir = Path(model_dir)
    try:
        if not (model_dir / "model.config").is_file():
            return VisualInspection(error="missing model.config")
        root = ET.parse(model_dir / "model.sdf").getroot()
        if root.tag != "sdf" or root.find("model") is None:
            return VisualInspection(error="model.sdf has no SDF model")
    except (ET.ParseError, OSError) as exc:
        return VisualInspection(error=f"unreadable model.sdf: {exc}")
    visuals = root.findall(".//visual")
    if not visuals:
        return VisualInspection(error="model has no visual geometry")
    measured = []
    for visual in visuals:
        geometry = visual.find("geometry")
        if geometry is None:
            return VisualInspection(error="visual has no geometry")
        mesh = geometry.find("mesh")
        if mesh is not None:
            uri = (mesh.findtext("uri") or "").strip()
            mesh_file = _mesh_path(uri, model_dir, tuple(Path(p) for p in search_paths))
            if mesh_file is None:
                return VisualInspection(error=f"missing visual mesh {uri}")
            info = _mesh_dimensions(mesh_file, mesh_file.stat().st_mtime_ns, mesh_file.stat().st_size)
            if info.error:
                return info
            if info.dimensions:
                scale = [float(value) for value in (mesh.findtext("scale") or "1 1 1").split()]
                if len(scale) != 3:
                    return VisualInspection(error=f"invalid mesh scale for {uri}")
                measured.append(tuple(size * abs(factor) for size, factor in zip(info.dimensions, scale)))
            continue
        box = geometry.find("box")
        if box is not None:
            measured.append(tuple(float(value) for value in box.findtext("size").split()))
            continue
        cylinder = geometry.find("cylinder")
        if cylinder is not None:
            radius = float(cylinder.findtext("radius"))
            measured.append((2 * radius, 2 * radius, float(cylinder.findtext("length"))))
            continue
        sphere = geometry.find("sphere")
        if sphere is not None:
            diameter = 2 * float(sphere.findtext("radius"))
            measured.append((diameter, diameter, diameter))
            continue
        return VisualInspection(error="unsupported visual geometry")
    dimensions = tuple(max(dimension[axis] for dimension in measured) for axis in range(3)) if measured else None
    if dimensions and min(dimensions) <= 0:
        return VisualInspection(error="degenerate visual geometry")
    return VisualInspection(dimensions=dimensions)


def inspect_model_visuals(model_dir: Path, search_paths=()) -> VisualInspection:
    """Return a validation result even for malformed geometry values."""
    try:
        return _inspect_model_visuals(model_dir, search_paths)
    except (ValueError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        return VisualInspection(error=f"invalid visual geometry: {exc}")
