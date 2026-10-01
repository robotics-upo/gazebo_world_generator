"""Draw a room layout as a labelled top-down PNG for a vision-capable LLM."""

import base64
import io
import math
from typing import Dict, Iterable, List, Sequence

from PIL import Image, ImageDraw

from gazebo_world_generator.src.placement.checks import (
    Violation, doorway_zones, front_direction, inner_half_extents)

PIXELS_PER_METRE = 60
MARGIN = 50
MAX_SIDE = 1400
VIEW_SIZE = 170  # side of one model's top view in the legend panel
SIDE_WIDTH = 130  # one side elevation in the legend panel
VIEW_CELL = VIEW_SIZE + 2 * SIDE_WIDTH + 40  # top view plus four side elevations (2 x 2)
# (label, horizontal axis index, horizontal sign, depth function: smaller = nearer the viewer)
SIDE_VIEWS = (
    ("-y side", 0, 1, lambda p: p[1]),
    ("+y side", 0, -1, lambda p: -p[1]),
    ("-x side", 1, -1, lambda p: p[0]),
    ("+x side", 1, 1, lambda p: -p[0]),
)


def render_layout(room, items: Sequence[Dict], violations: Iterable[Violation] = (),
                  fixed: Sequence[Dict] = (), wall_thickness: float = 0.2,
                  model_views: Sequence[Dict] = ()) -> bytes:
    """PNG bytes: walls, doorway zones, 1 m grid, object shapes with ids and front arrows.

    `model_views` adds a panel on the right showing each model on its own at
    yaw 0 from above (darker = taller), with its local +x/+y axes, so the LLM
    can tell which side is a model's front.
    """
    half_x, half_y = inner_half_extents(room, wall_thickness)
    scale = min(PIXELS_PER_METRE, (MAX_SIDE - 2 * MARGIN) / (2 * max(half_x, half_y)))
    room_width = int(2 * half_x * scale + 2 * MARGIN)
    room_height = height = int(2 * half_y * scale + 2 * MARGIN)
    views = [view for view in model_views if view.get("shape")]
    columns = max(1, (height - 40) // (VIEW_SIZE + 30)) if views else 0
    panel_columns = -(-len(views) // columns) if views else 0
    width = room_width + panel_columns * VIEW_CELL
    if views:
        height = max(height, 40 + min(len(views), columns) * (VIEW_SIZE + 30))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    width = room_width

    def to_px(x: float, y: float):
        return (MARGIN + (x + half_x) * scale, MARGIN + (half_y - y) * scale)

    # 1 m grid with axis labels (metres, room-local).
    for metre in range(math.ceil(-half_x), math.floor(half_x) + 1):
        px, _ = to_px(metre, 0)
        draw.line([(px, MARGIN), (px, room_height - MARGIN)], fill=(225, 225, 225))
        draw.text((px - 6, room_height - MARGIN + 6), str(metre), fill="grey")
    for metre in range(math.ceil(-half_y), math.floor(half_y) + 1):
        _, py = to_px(0, metre)
        draw.line([(MARGIN, py), (width - MARGIN, py)], fill=(225, 225, 225))
        draw.text((MARGIN - 28, py - 6), str(metre), fill="grey")
    draw.text((width - MARGIN - 60, 12), "N (+y) ^", fill="black")
    draw.text((width - MARGIN + 4, room_height - MARGIN + 20), "E (+x) >", fill="black")

    for _, zone in doorway_zones(room):
        draw.rectangle([to_px(zone[0], zone[3]), to_px(zone[1], zone[2])], fill=(210, 235, 210))
        draw.text(to_px(zone[0], zone[3]), "door", fill=(40, 120, 40))
    draw.rectangle([to_px(-half_x, half_y), to_px(half_x, -half_y)], outline="black", width=3)

    flagged = {item_id for violation in violations for item_id in violation.ids}
    for item in list(fixed) + list(items):
        _draw_item(draw, item, to_px, scale, flagged)
    if views:
        _draw_model_views(draw, views, room_width, columns)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_item(draw, item: Dict, to_px, scale: float, flagged) -> None:
    x, y, yaw = item["x"], item["y"], item.get("yaw", 0.0)
    half_w, half_l = item["dims"][0] / 2, item["dims"][1] / 2
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    corners = [to_px(x + cx * cos_yaw - cy * sin_yaw, y + cx * sin_yaw + cy * cos_yaw)
               for cx, cy in ((-half_w, -half_l), (half_w, -half_l), (half_w, half_l), (-half_w, half_l))]
    if item.get("fixed"):
        fill = (200, 200, 200)
    elif item.get("on"):
        fill = (200, 220, 250)
    else:
        fill = (250, 225, 190)
    outline = (220, 0, 0) if item["id"] in flagged else (60, 60, 60)
    shape = item.get("shape")
    draw.polygon(corners, fill=(250, 250, 250) if shape else fill, outline=outline,
                 width=3 if item["id"] in flagged else 1)
    if shape:
        _draw_shape(draw, shape, lambda px, py: to_px(x + px * cos_yaw - py * sin_yaw,
                                                        y + px * sin_yaw + py * cos_yaw))

    heading = front_direction(item)
    reach = max(half_w, half_l) + 0.25
    start = to_px(x, y)
    end = to_px(x + reach * math.cos(heading), y + reach * math.sin(heading))
    draw.line([start, end], fill=(0, 90, 200), width=2)
    draw.ellipse([end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3], fill=(0, 90, 200))
    draw.text((start[0] + 4, start[1] - 12), item["id"], fill="black")


def render_model_views(views: Sequence[Dict]) -> bytes:
    """PNG with just the MODEL VIEWS panel (top + side views of each model)."""
    views = [view for view in views if view.get("shape")]
    columns = max(1, min(len(views), 3))
    rows = -(-len(views) // columns)
    image = Image.new("RGB", (columns * VIEW_CELL + 20, 50 + rows * (VIEW_SIZE + 30)), "white")
    draw = ImageDraw.Draw(image)
    # _draw_model_views fills columns top to bottom; lay out one model per cell, rows first.
    for row in range(rows):
        chunk = views[row * columns:(row + 1) * columns]
        for column, view in enumerate(chunk):
            _draw_model_views(draw, [view], column * VIEW_CELL, 1, top=40 + row * (VIEW_SIZE + 30),
                              title=row == 0 and column == 0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def to_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _draw_shape(draw, triangles, to_px) -> None:
    """A model seen from above: triangles filled lighter for low parts, darker for tall ones."""
    heights = [max(point[2] for point in triangle) for triangle in triangles]
    low = min(heights)
    span = max(max(heights) - low, 1e-6)
    for height, triangle in sorted(zip(heights, triangles), key=lambda entry: entry[0]):
        shade = int(215 - 190 * (height - low) / span)
        draw.polygon([to_px(x, y) for x, y, _ in triangle], fill=(shade, shade, shade))


def _draw_model_views(draw, views, left: int, columns: int, top: int = 40, title: bool = True) -> None:
    """Each model alone at yaw 0: from above with its local axes, plus two side elevations."""
    if title:
        draw.text((left + 10, 10), "MODEL VIEWS at yaw 0: top (darker = taller) and the model "
                                   "seen from each of its four sides (darker = nearer)", fill="black")
    for index, view in enumerate(views):
        column, row = divmod(index, columns)
        ox = left + 10 + column * VIEW_CELL
        oy = top + row * (VIEW_SIZE + 30)
        triangles = view["shape"]
        points = [point for triangle in triangles for point in triangle]
        extent = max(max(abs(p[0]) for p in points), max(abs(p[1]) for p in points), 0.05)
        factor = (VIEW_SIZE / 2 - 22) / extent
        centre = (ox + VIEW_SIZE / 2, oy + VIEW_SIZE / 2)
        draw.rectangle([ox, oy, ox + VIEW_SIZE, oy + VIEW_SIZE], outline=(200, 200, 200))
        _draw_shape(draw, triangles, lambda px, py: (centre[0] + px * factor, centre[1] - py * factor))
        draw.line([centre, (ox + VIEW_SIZE - 6, centre[1])], fill=(220, 0, 0), width=2)
        draw.text((ox + VIEW_SIZE - 20, centre[1] + 4), "+x", fill=(220, 0, 0))
        draw.line([centre, (centre[0], oy + 6)], fill=(0, 150, 0), width=2)
        draw.text((centre[0] + 4, oy + 4), "+y", fill=(0, 150, 0))
        draw.text((ox, oy + VIEW_SIZE + 4), view["type"][:28], fill="black")

        height = VIEW_SIZE / 2 - 6
        top = max(max(p[2] for p in points), 0.05)
        scale = min((SIDE_WIDTH / 2 - 8) / extent, (height - 16) / top)
        # Each elevation shows the model as seen from outside that side (darker = nearer).
        for panel, (label, axis, sign, depth_of) in enumerate(SIDE_VIEWS):
            column_offset, row_offset = divmod(panel, 2)
            px0 = ox + VIEW_SIZE + 10 + column_offset * (SIDE_WIDTH + 10)
            py0 = oy + row_offset * (height + 12)
            draw.rectangle([px0, py0, px0 + SIDE_WIDTH, py0 + height], outline=(200, 200, 200))
            base = (px0 + SIDE_WIDTH / 2, py0 + height - 4)
            depths = [min(depth_of(p) for p in t) for t in triangles]
            low, span = min(depths), max(max(depths) - min(depths), 1e-6)
            for depth, triangle in sorted(zip(depths, triangles), key=lambda e: -e[0]):
                shade = int(40 + 170 * (depth - low) / span)
                draw.polygon([(base[0] + sign * p[axis] * scale, base[1] - p[2] * scale) for p in triangle],
                             fill=(shade, shade, shade))
            draw.text((px0 + 4, py0 + 2), f"{label} (from outside)",
                      fill=(220, 0, 0) if "x" in label else (0, 150, 0))
