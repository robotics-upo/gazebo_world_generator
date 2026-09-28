"""Check that a Gazebo model has usable visual geometry and measure its mesh."""

from dataclasses import dataclass
from functools import lru_cache
from glob import escape as glob_escape
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


def _texture_exists(mesh_file: Path, reference: str) -> bool:
    """Mirror Gazebo's texture lookup: beside the mesh, then the model's materials/textures."""
    if (mesh_file.parent / reference).is_file():
        return True
    name = Path(reference.replace("\\", "/")).name
    model_root = next((parent for parent in mesh_file.parents
                       if (parent / "model.config").is_file()), None)
    if model_root is None or not name:
        return False
    if (model_root / "materials" / "textures" / name).is_file():
        return True
    return any(candidate.is_file() for candidate in model_root.rglob(glob_escape(name)))


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


def _collada_geometry_triangles(root) -> dict:
    """Map each geometry id to triangles as index triples into its POSITION vertices."""
    triangles = {}
    for geometry in root.findall(".//{*}library_geometries/{*}geometry"):
        mesh = geometry.find("{*}mesh")
        if mesh is None:
            continue
        found = []
        for primitive in list(mesh):
            tag = primitive.tag.rsplit("}", 1)[-1]
            if tag not in ("triangles", "polylist", "polygons"):
                continue
            inputs = primitive.findall("{*}input")
            if not inputs:
                continue
            stride = max(int(item.get("offset", "0")) for item in inputs) + 1
            offset = next((int(item.get("offset", "0")) for item in inputs
                           if item.get("semantic") == "VERTEX"), 0)
            if tag == "polygons":
                polygons = [[int(v) for v in (element.text or "").split()][offset::stride]
                            for element in primitive.findall("{*}p")]
            else:
                indices = [int(v) for v in (primitive.findtext("{*}p") or "").split()][offset::stride]
                if tag == "triangles":
                    counts = [3] * (len(indices) // 3)
                else:
                    counts = [int(v) for v in (primitive.findtext("{*}vcount") or "").split()]
                polygons, start = [], 0
                for count in counts:
                    polygons.append(indices[start:start + count])
                    start += count
            for polygon in polygons:
                found.extend((polygon[0], polygon[i], polygon[i + 1]) for i in range(1, len(polygon) - 1))
        triangles[geometry.get("id")] = found
    return triangles


def _collada_scene_vertices(root, geometries: dict, triangles: dict = None,
                            scene_triangles: list = None) -> list | None:
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
                geometry_id = (child.get("url") or "").lstrip("#")
                base = len(vertices)
                if scene_triangles is not None and triangles:
                    scene_triangles.extend((base + a, base + b, base + c)
                                           for a, b, c in triangles.get(geometry_id, ()))
                for x, y, z in geometries.get(geometry_id, ()):
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


def _read_mesh(path: Path, faces: list = None):
    """(vertices, error) for a mesh file, without NumPy or a simulator.

    When `faces` is given, it is filled with triangles as index triples into
    the returned vertices.
    """
    try:
        suffix = path.suffix.lower()
        if suffix == ".dae":
            root = ET.parse(path).getroot()
            unit = root.find(".//{*}asset/{*}unit")
            scale = float(unit.get("meter", "1")) if unit is not None else 1.0
            geometries = _collada_geometry_vertices(root)
            triangles = _collada_geometry_triangles(root) if faces is not None else None
            vertices = _collada_scene_vertices(root, geometries, triangles, faces)
            if vertices is None:
                vertices = []
                for geometry_id, items in geometries.items():
                    if faces is not None:
                        base = len(vertices)
                        faces.extend((base + a, base + b, base + c)
                                     for a, b, c in triangles.get(geometry_id, ()))
                    vertices.extend(items)
            vertices = [(x * scale, y * scale, z * scale) for x, y, z in vertices]
            for image in root.findall(".//{*}library_images/{*}image/{*}init_from"):
                reference = (image.text or "").strip()
                if reference and not reference.startswith("data:") and not _texture_exists(path, reference):
                    return None, f"missing texture {reference}"
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
                    elif line.startswith("f ") and faces is not None:
                        corners = []
                        for token in line.split()[1:]:
                            index = int(token.split("/")[0])
                            corners.append(index - 1 if index > 0 else len(vertices) + index)
                        faces.extend((corners[0], corners[i], corners[i + 1])
                                     for i in range(1, len(corners) - 1))
                    elif line.startswith("mtllib "):
                        for name in line.split()[1:]:
                            if not (path.parent / name).is_file():
                                return None, f"missing material {name}"
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
            return None, None
        if suffix == ".stl" and faces is not None:
            faces.extend((i, i + 1, i + 2) for i in range(0, len(vertices) - 2, 3))
        if faces is not None:
            faces[:] = [face for face in faces if max(face) < len(vertices) and min(face) >= 0]
        return vertices, None
    except (ET.ParseError, OSError, ValueError, struct.error) as exc:
        return None, f"unreadable mesh {path.name}: {exc}"


@lru_cache(maxsize=512)
def _mesh_dimensions(path: Path, mtime_ns: int, size: int) -> VisualInspection:
    vertices, error = _read_mesh(path)
    if error:
        return VisualInspection(error=error)
    if vertices is None:
        return VisualInspection()
    if not vertices:
        return VisualInspection(error=f"no vertices in {path.name}")
    dimensions = tuple(max(vertex[axis] for vertex in vertices) -
                       min(vertex[axis] for vertex in vertices) for axis in range(3))
    if min(dimensions) <= 0:
        return VisualInspection(error=f"degenerate mesh {path.name}")
    return VisualInspection(dimensions=dimensions)


@lru_cache(maxsize=64)
def _mesh_triangles(path: Path, mtime_ns: int, size: int, limit: int = 6000) -> tuple:
    """Up to `limit` of a mesh's triangles as point triples (for drawing its shape)."""
    faces = []
    vertices, _ = _read_mesh(path, faces)
    if not vertices or not faces:
        return ()
    stride = max(1, len(faces) // limit)
    return tuple((vertices[a], vertices[b], vertices[c]) for a, b, c in faces[::stride])


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


def _pose_transform(text):
    """(rotation rows, translation) of an SDF pose "x y z roll pitch yaw"."""
    values = [float(value) for value in (text or "").split()] + [0.0] * 6
    x, y, z, roll, pitch, yaw = values[:6]
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    rotation = ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                (-sp, cp * sr, cp * cr))
    return rotation, (x, y, z)


def _apply(transform, point):
    rotation, translation = transform
    return tuple(sum(rotation[row][k] * point[k] for k in range(3)) + translation[row] for row in range(3))


def _primitive_triangles(geometry):
    box = geometry.find("box")
    if box is not None:
        sx, sy, sz = (float(value) / 2 for value in box.findtext("size").split())
        top = [(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]
        return [(top[0], top[1], top[2]), (top[0], top[2], top[3])]
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        radius, half = float(cylinder.findtext("radius")), float(cylinder.findtext("length")) / 2
        ring = [(radius * math.cos(t * math.pi / 12), radius * math.sin(t * math.pi / 12), half)
                for t in range(24)]
        return [((0.0, 0.0, half), ring[i], ring[(i + 1) % 24]) for i in range(24)]
    sphere = geometry.find("sphere")
    if sphere is not None:
        radius = float(sphere.findtext("radius"))
        ring = [(radius * math.cos(t * math.pi / 12), radius * math.sin(t * math.pi / 12), 0.0)
                for t in range(24)]
        return [((0.0, 0.0, radius), ring[i], ring[(i + 1) % 24]) for i in range(24)]
    return []


def model_shape(model_dir: Path, search_paths=(), max_triangles: int = 3000):
    """The model's visual geometry as triangles ((x, y, z) x 3) in the model frame.

    Used to draw the model from above for the LLM, e.g. where a chair's
    backrest is. Returns None when the geometry cannot be read.
    """
    try:
        model = ET.parse(Path(model_dir) / "model.sdf").getroot().find("model")
        if model is None:
            return None
        triangles = []
        for link in model.findall("link"):
            link_pose = _pose_transform(link.findtext("pose"))
            for visual in link.findall("visual"):
                visual_pose = _pose_transform(visual.findtext("pose"))
                geometry = visual.find("geometry")
                if geometry is None:
                    continue
                mesh = geometry.find("mesh")
                if mesh is not None:
                    mesh_file = _mesh_path((mesh.findtext("uri") or "").strip(), Path(model_dir),
                                           tuple(Path(p) for p in search_paths))
                    if mesh_file is None:
                        continue
                    scale = [float(value) for value in (mesh.findtext("scale") or "1 1 1").split()]
                    stat = mesh_file.stat()
                    local = [tuple((x * scale[0], y * scale[1], z * scale[2]) for x, y, z in triangle)
                             for triangle in _mesh_triangles(mesh_file, stat.st_mtime_ns, stat.st_size)]
                else:
                    local = _primitive_triangles(geometry)
                triangles.extend(tuple(_apply(link_pose, _apply(visual_pose, point)) for point in triangle)
                                 for triangle in local)
        if not triangles:
            return None
        stride = max(1, len(triangles) // max_triangles)
        return triangles[::stride]
    except (ET.ParseError, OSError, ValueError, TypeError, AttributeError, IndexError):
        return None
