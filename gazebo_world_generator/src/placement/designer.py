"""
LLM layout design with measured feedback.

The LLM proposes a layout and the requirements it should meet; ``checks``
measures it and reports concrete violations (with a top-down image when the
model accepts images); the LLM revises. Code never re-plans the room: after
the last round it only nudges apart leftover overlaps, and it falls back to
a seeded sampler when no LLM is available.
"""

import json
import logging
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from gazebo_world_generator.src.placement import checks
from gazebo_world_generator.src.placement.checks import (
    CATALOGUE, FRONT_AXES, Violation, check_layout, doorway_zones, inner_half_extents,
    item_box, penetration)
from gazebo_world_generator.src.utils import llm_utils
from gazebo_world_generator.src.utils.token_estimator import estimate_tokens

logger = logging.getLogger(__name__)

RESPONSE_MARGIN = 256
MAX_RESPONSE_TOKENS = 4096
NUDGE_ITERATIONS = 60
SAMPLER_ATTEMPTS = 400
SAMPLER_GAP = 0.1


class LayoutDesigner:
    """Runs the propose -> check -> revise loop for one room at a time."""

    def __init__(self, llm_interface=None, prompt_manager=None, rng: Optional[random.Random] = None,
                 max_rounds: int = 3, wall_thickness: float = 0.2, chars_per_token: float = 3.0):
        self.llm = llm_interface
        self.prompt_manager = prompt_manager
        self.rng = rng or random.Random()
        self.max_rounds = max_rounds
        self.wall_thickness = wall_thickness
        self.chars_per_token = chars_per_token
        # Switched off for the session if the endpoint rejects image input.
        self.use_images = True

    def design(self, room, items: List[Dict], user_request: str = "",
               fixed: Sequence[Dict] = ()) -> Tuple[List[Dict], List[Violation]]:
        """Return items with poses and the violations that remain."""
        expected = [item["id"] for item in items]
        best: Optional[Tuple[int, List[Dict], List[Dict], List[Violation]]] = None
        if self.llm is not None and self.prompt_manager is not None:
            best = self._llm_rounds(room, items, user_request, fixed, expected)

        if best is None:
            logger.info(f"No LLM layout for '{room.name}'; using seeded placement")
            placed, requirements = [], []
        else:
            _, placed, requirements, _ = best

        placed = self._sample_missing(room, items, placed, fixed)
        violations = check_layout(room, placed, requirements, expected, fixed, self.wall_thickness)
        # An item that does not fit on its support stands on the floor instead.
        unsupported = {v.ids[0] for v in violations if v.check == "no_overlap" and
                       ("overhangs" in v.message or "unknown id" in v.message)}
        for item in placed:
            if item["id"] in unsupported:
                item["on"] = None
        if any(violation.always_on for violation in violations):
            placed = self._nudge(room, placed, fixed)
            violations = check_layout(room, placed, requirements, expected, fixed, self.wall_thickness)
        return placed, violations

    # --- LLM rounds -----------------------------------------------------------------------------

    def _llm_rounds(self, room, items, user_request, fixed, expected):
        by_id = {item["id"]: item for item in items}
        half_x, half_y = inner_half_extents(room, self.wall_thickness)
        model_views = list({item["type"]: {"type": item["type"], "shape": item.get("shape")}
                            for item in items}.values())
        has_views = any(view["shape"] for view in model_views)
        design_text = self.prompt_manager.render(
            "layout_design", user_request=user_request, room_name=room.name, room_type=room.type,
            purpose=getattr(room, "purpose", ""), half_x=round(half_x, 2), half_y=round(half_y, 2),
            doorways=self._doorway_context(room), fixed=list(fixed), items=items,
            catalogue=CATALOGUE, has_image=self.use_images,
            has_views=self.use_images and has_views)

        # best = (score, placed items, requirements, violations, layout JSON)
        best = None
        for round_number in range(1, self.max_rounds + 1):
            if best is None:
                messages = [self._user_message(design_text, room, [], [], fixed, model_views)]
            else:
                _, base, _, base_violations, base_json = best
                problem_ids = sorted({i for v in base_violations for i in v.ids})
                revision_text = self.prompt_manager.render(
                    "layout_revision", round=round_number, max_rounds=self.max_rounds,
                    layout_json=base_json, violations=[str(v) for v in base_violations],
                    problem_ids=problem_ids, has_image=self.use_images)
                messages = [{"role": "user", "content": design_text},
                            {"role": "assistant", "content": base_json},
                            self._user_message(revision_text, room, base, base_violations, fixed)]

            logger.info(f"Layout design round {round_number}/{self.max_rounds} for '{room.name}'...")
            answer = self._ask(messages)
            parsed = llm_utils.extract_json_from_response(answer) if answer else None
            if not isinstance(parsed, dict) or not isinstance(parsed.get("objects"), list):
                logger.warning(f"Layout round {round_number}: response was not a layout JSON object")
                continue

            # Revisions list only the objects that move; everything else stays as in the best round.
            base = best[1] if best else []
            requirements = parsed.get("requirements") if isinstance(parsed.get("requirements"), list) \
                else (best[2] if best else [])
            requirements = [r for r in requirements if isinstance(r, dict)]
            placed = self._apply_answer(parsed, by_id, base)
            violations = check_layout(room, placed, requirements, expected, fixed, self.wall_thickness)
            layout_json = json.dumps({"requirements": requirements,
                                      "fronts": {item["type"]: item.get("front", "+x") for item in placed},
                                      "objects": [
                {key: (round(item[key], 3) if isinstance(item[key], float) else item[key])
                 for key in ("id", "x", "y", "yaw", "on") if item.get(key) is not None}
                for item in placed]})
            current = (checks.score(violations), placed, requirements, violations, layout_json)
            if best is None or current[0] <= best[0]:
                best = current
            requested = ", ".join(r.get("check", "?") for r in requirements) or "none"
            fronts = ", ".join(sorted({f"{item['type']} {item.get('front', '+x')}" for item in placed}))
            logger.info(f"Layout round {round_number}: {len(violations)} problem(s); "
                        f"requirements: {requested}; fronts: {fronts}")
            for violation in violations:
                logger.debug(f"  {violation}")
            if not violations:
                break
        return best[:4] if best else None

    def _apply_answer(self, parsed: Dict, by_id: Dict[str, Dict], base: Sequence[Dict] = ()) -> List[Dict]:
        """Items from the answer, layered over ``base`` (the layout being revised)."""
        fronts = parsed.get("fronts") if isinstance(parsed.get("fronts"), dict) else {}
        placed = {item["id"]: dict(item) for item in base}
        for entry in parsed["objects"]:
            if not isinstance(entry, dict) or entry.get("id") not in by_id:
                continue
            previous = placed.get(entry["id"], {})
            try:
                pose = {key: float(entry.get(key, previous.get(key, 0.0))) for key in ("x", "y", "yaw")}
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in pose.values()):
                continue
            item = dict(previous or by_id[entry["id"]], **pose)
            if "on" in entry or not previous:
                item["on"] = entry.get("on") if isinstance(entry.get("on"), str) else None
            placed[entry["id"]] = item
        for item in placed.values():
            front = fronts.get(item["type"]) or fronts.get(item["id"])
            if front in FRONT_AXES:
                item["front"] = front
        order = {item_id: index for index, item_id in enumerate(by_id)}
        return sorted(placed.values(), key=lambda item: order[item["id"]])

    def _user_message(self, text, room, placed, violations, fixed, model_views=()) -> Dict:
        if not self.use_images:
            return {"role": "user", "content": text}
        from gazebo_world_generator.src.placement.render import render_layout, to_data_url
        png = render_layout(room, placed, violations, fixed, self.wall_thickness, model_views)
        return {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": to_data_url(png)}}]}

    def _ask(self, messages: List[Dict]) -> str:
        answer = self.llm.query(messages, max_tokens=self._response_budget(messages), temperature=0.3)
        if answer or not self.use_images:
            return answer
        # An empty answer to an image request usually means a text-only endpoint.
        logger.warning("LLM gave no answer to a request with an image; continuing without images")
        self.use_images = False
        text_only = [{"role": m["role"], "content": self._text_of(m["content"])} for m in messages]
        return self.llm.query(text_only, max_tokens=self._response_budget(text_only), temperature=0.3)

    def _response_budget(self, messages) -> int:
        window = getattr(self.llm, "context_window", 8192)
        if not isinstance(window, int):
            window = 8192
        available = window - estimate_tokens(messages, self.chars_per_token) - RESPONSE_MARGIN
        return max(256, min(MAX_RESPONSE_TOKENS, available))

    @staticmethod
    def _text_of(content) -> str:
        if isinstance(content, list):
            return "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
        return content

    @staticmethod
    def _doorway_context(room) -> List[Dict]:
        doors = []
        for door, (side, zone) in zip(getattr(room, "doorways", None) or [], doorway_zones(room)):
            doors.append({"side": side,
                          "x": round(door["x"] - room.position["x"], 2),
                          "y": round(door["y"] - room.position["y"], 2),
                          "zone": [round(value, 2) for value in zone]})
        return doors

    # --- deterministic fallbacks ------------------------------------------------------------------

    def _sample_missing(self, room, items, placed, fixed) -> List[Dict]:
        """Seeded placement for items without a pose (no LLM, or ids the LLM skipped)."""
        placed = list(placed)
        done = {item["id"] for item in placed}
        half_x, half_y = inner_half_extents(room, self.wall_thickness)
        zones = [zone for _, zone in doorway_zones(room)]
        occupied = [item_box(item) for item in list(placed) + list(fixed) if not item.get("on")]
        for item in items:
            if item["id"] in done:
                continue
            pose = self._free_pose(item, half_x, half_y, occupied, zones)
            placed.append(dict(item, **pose, on=None))
            occupied.append(item_box(placed[-1]))
        return placed

    def _free_pose(self, item, half_x, half_y, occupied, zones) -> Dict:
        def fits(box):
            if box[0] < -half_x or box[1] > half_x or box[2] < -half_y or box[3] > half_y:
                return False
            if any(min(penetration(box, zone)) > 0 for zone in zones):
                return False
            return all(checks.box_gap(box, other) >= SAMPLER_GAP and min(penetration(box, other)) <= 0
                       for other in occupied)

        for _ in range(SAMPLER_ATTEMPTS):
            yaw = self.rng.choice((0.0, math.pi / 2))
            x, y = self.rng.uniform(-half_x, half_x), self.rng.uniform(-half_y, half_y)
            if fits(checks.footprint(x, y, item["dims"], yaw)):
                return {"x": x, "y": y, "yaw": yaw}
        step = 0.25
        for yaw in (0.0, math.pi / 2):
            y = -half_y
            while y <= half_y:
                x = -half_x
                while x <= half_x:
                    if fits(checks.footprint(x, y, item["dims"], yaw)):
                        return {"x": x, "y": y, "yaw": yaw}
                    x += step
                y += step
        logger.warning(f"No free spot for {item['id']}; placing it at the room centre")
        return {"x": 0.0, "y": 0.0, "yaw": 0.0}

    def _nudge(self, room, placed, fixed) -> List[Dict]:
        """Separate overlaps and clear walls/doorways with the smallest moves; no re-layout."""
        items = [dict(item) for item in placed]
        half_x, half_y = inner_half_extents(room, self.wall_thickness)
        zones = doorway_zones(room)

        def move(item, dx, dy):
            item["x"] += dx
            item["y"] += dy
            for rider in items:
                if rider.get("on") == item["id"]:
                    rider["x"] += dx
                    rider["y"] += dy

        for _ in range(NUDGE_ITERATIONS):
            moved = False
            solid = [item for item in items if not item.get("on")]
            for index, a in enumerate(solid):
                for b in solid[index + 1:] + list(fixed):
                    over_x, over_y = penetration(item_box(a), item_box(b))
                    if over_x <= checks.OVERLAP_TOLERANCE or over_y <= checks.OVERLAP_TOLERANCE:
                        continue
                    share = 1.0 if b.get("fixed") else 0.5
                    if over_x <= over_y:
                        sign = 1 if a["x"] >= b["x"] else -1
                        move(a, sign * (over_x * share + 0.01), 0.0)
                        if not b.get("fixed"):
                            move(b, -sign * (over_x * share + 0.01), 0.0)
                    else:
                        sign = 1 if a["y"] >= b["y"] else -1
                        move(a, 0.0, sign * (over_y * share + 0.01))
                        if not b.get("fixed"):
                            move(b, 0.0, -sign * (over_y * share + 0.01))
                    moved = True
            for item in solid:
                for side, zone in zones:
                    over_x, over_y = penetration(item_box(item), zone)
                    if over_x > checks.OVERLAP_TOLERANCE and over_y > checks.OVERLAP_TOLERANCE:
                        shift = {"north": (0, -over_y), "south": (0, over_y),
                                 "east": (-over_x, 0), "west": (over_x, 0)}[side]
                        move(item, shift[0] + math.copysign(0.01, shift[0] or 1) * bool(shift[0]),
                             shift[1] + math.copysign(0.01, shift[1] or 1) * bool(shift[1]))
                        moved = True
                box = item_box(item)
                dx = max(-half_x - box[0], 0.0) - max(box[1] - half_x, 0.0)
                dy = max(-half_y - box[2], 0.0) - max(box[3] - half_y, 0.0)
                if dx or dy:
                    move(item, dx, dy)
                    moved = True
            if not moved:
                break
        return items
