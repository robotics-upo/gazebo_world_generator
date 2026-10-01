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
from gazebo_world_generator.src.models.visual_quality import side_closedness
from gazebo_world_generator.src.utils import llm_utils
from gazebo_world_generator.src.utils.token_estimator import estimate_tokens

logger = logging.getLogger(__name__)

RESPONSE_MARGIN = 256
MAX_RESPONSE_TOKENS = 4096
NUDGE_ITERATIONS = 60
SAMPLER_ATTEMPTS = 400
SAMPLER_GAP = 0.1
SOLVER_PASSES = 6


class LayoutDesigner:
    """Runs the propose -> check -> revise loop for one room at a time."""

    def __init__(self, llm_interface=None, prompt_manager=None, rng: Optional[random.Random] = None,
                 max_rounds: int = 3, wall_thickness: float = 0.2, chars_per_token: float = 3.0,
                 review_rounds: int = 1):
        self.llm = llm_interface
        self.review_rounds = review_rounds
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
            _, placed, requirements, _ = best[:4]

        # Invented check names were fed back during the rounds; they are not layout problems.
        unknown = [r for r in requirements if not isinstance(r, dict) or r.get("check") not in CATALOGUE]
        if unknown:
            logger.info(f"Ignoring requirements that are not checks: {unknown}")
            requirements = [r for r in requirements if r not in unknown]

        placed = self._sample_missing(room, items, placed, fixed)
        placed = self._solve_relations(room, placed, requirements, expected, fixed)
        placed = self._spread(room, placed, requirements, expected, fixed)
        violations = check_layout(room, placed, requirements, expected, fixed, self.wall_thickness)
        # An item that does not fit on its support stands on the floor instead.
        unsupported = {v.ids[0] for v in violations if v.check == "no_overlap" and
                       ("overhangs" in v.message or "unknown id" in v.message)}
        for item in placed:
            if item["id"] in unsupported:
                item["on"] = None
        if any(violation.always_on for violation in violations):
            placed = self._nudge(room, placed, fixed)
            # Separating overlaps moves objects off their relations (a chair tucked into a table
            # is pushed out, still turned the wrong way): solve them again from the new poses.
            placed = self._solve_relations(room, placed, requirements, expected, fixed)
            violations = check_layout(room, placed, requirements, expected, fixed, self.wall_thickness)
        # Nudging cannot untangle a crowded layout (a row longer than the room): re-place one
        # object of each remaining conflict, plus what stands on it, in free floor.
        stuck = {violation.ids[0] for violation in violations if violation.always_on and violation.ids}
        if stuck:
            logger.info(f"Re-placing {len(stuck)} object(s) the layout could not fit: {sorted(stuck)}")
            keep = [item for item in placed if item["id"] not in stuck and item.get("on") not in stuck]
            placed = self._sample_missing(room, items, keep, fixed)
            placed = self._solve_relations(room, placed, requirements, expected, fixed)
            violations = check_layout(room, placed, requirements, expected, fixed, self.wall_thickness)
        return placed, violations

    # --- LLM rounds -----------------------------------------------------------------------------

    def _llm_rounds(self, room, items, user_request, fixed, expected):
        by_id = {item["id"]: item for item in items}
        self._determine_fronts(items)
        half_x, half_y = inner_half_extents(room, self.wall_thickness)
        model_views = list({item["type"]: {"type": item["type"], "shape": item.get("shape")}
                            for item in items}.values())
        has_views = any(view["shape"] for view in model_views)
        known_fronts = {item["type"]: item["front"] for item in items if item.get("front_known")}
        no_front = sorted({item["type"] for item in items if item.get("front_known") is False})
        design_text = self.prompt_manager.render(
            "layout_design", user_request=user_request, room_name=room.name, room_type=room.type,
            purpose=getattr(room, "purpose", ""), half_x=round(half_x, 2), half_y=round(half_y, 2),
            doorways=self._doorway_context(room), fixed=list(fixed), items=items,
            catalogue=CATALOGUE, has_image=self.use_images,
            has_views=self.use_images and has_views, known_fronts=known_fronts, no_front=no_front)
        context = dict(room=room, by_id=by_id, fixed=fixed, expected=expected, design_text=design_text)

        # best = (score, placed items, requirements, violations, layout JSON)
        best = None
        for round_number in range(1, self.max_rounds + 1):
            if best is None:
                messages = [self._user_message(design_text, room, [], [], fixed, model_views)]
            else:
                messages = self._revision_messages(context, best, round_number, self.max_rounds)
            best = self._run_round(context, messages, best, f"{round_number}/{self.max_rounds}")
            if best is not None and not best[3]:
                break
        if best is not None:
            best = self._review(context, best, user_request)
        return best[:4] if best else None

    def _revision_messages(self, context, best, round_number, max_rounds) -> List[Dict]:
        _, base, _, base_violations, base_json = best
        problem_ids = sorted({i for v in base_violations for i in v.ids})
        revision_text = self.prompt_manager.render(
            "layout_revision", round=round_number, max_rounds=max_rounds,
            layout_json=base_json, violations=[str(v) for v in base_violations],
            problem_ids=problem_ids, has_image=self.use_images,
            relations=checks.relations(context["room"], base, self.wall_thickness))
        return [{"role": "user", "content": context["design_text"]},
                {"role": "assistant", "content": base_json},
                self._user_message(revision_text, context["room"], base, base_violations, context["fixed"])]

    def _run_round(self, context, messages, best, label, extra_requirements=None):
        """Ask, merge the answer over the best layout, check it; return the new best."""
        logger.info(f"Layout design round {label} for '{context['room'].name}'...")
        answer = self._ask(messages)
        parsed = llm_utils.extract_json_from_response(answer) if answer else None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("objects"), list):
            logger.warning(f"Layout round {label}: response was not a layout JSON object")
            return best
        # Revisions list only the objects that move; everything else stays as in the best round.
        base = best[1] if best else []
        requirements = parsed.get("requirements") if isinstance(parsed.get("requirements"), list) \
            else (best[2] if best else [])
        requirements = _merge_requirements(requirements, extra_requirements or [])
        placed = self._apply_answer(parsed, context["by_id"], base)
        placed = self._scatter(context["room"], placed, parsed.get("scatter"), context["fixed"])
        current = self._evaluate(context, placed, requirements)
        if best is None or current[0] <= best[0]:
            best = current
        requested = ", ".join(r.get("check", "?") for r in requirements) or "none"
        fronts = ", ".join(sorted({f"{item['type']} {item.get('front', '+x')}" for item in placed}))
        logger.info(f"Layout round {label}: {len(current[3])} problem(s); "
                    f"requirements: {requested}; fronts: {fronts}")
        for violation in current[3]:
            logger.debug(f"  {violation}")
        return best

    def _evaluate(self, context, placed, requirements):
        placed, requirements = _apply_on_requirements(placed, requirements)
        violations = check_layout(context["room"], placed, requirements, context["expected"],
                                  context["fixed"], self.wall_thickness)
        layout_json = json.dumps({
            "requirements": requirements,
            "fronts": {item["type"]: item.get("front", "+x") for item in placed},
            "objects": [{key: (round(item[key], 3) if isinstance(item[key], float) else item[key])
                         for key in ("id", "x", "y", "yaw", "on") if item.get(key) is not None}
                        for item in placed]})
        return (checks.score(violations), placed, requirements, violations, layout_json)

    def _review(self, context, best, user_request):
        """Ask the LLM to judge the finished layout against the request, image and relations.

        The check loop only verifies what the LLM chose to require; the review
        catches what it forgot (a chair turned away from its desk, everything packed
        into a corner). Added requirements are kept and checked; a review that breaks
        the always-on rules is discarded.
        """
        for review in range(1, self.review_rounds + 1):
            room = context["room"]
            _, placed, requirements, violations, layout_json = best
            review_text = self.prompt_manager.render(
                "layout_review", user_request=user_request, purpose=getattr(room, "purpose", ""),
                layout_json=layout_json, violations=[str(v) for v in violations],
                relations=checks.relations(room, placed, self.wall_thickness),
                space_use=checks.space_use(room, placed, self.wall_thickness),
                catalogue=CATALOGUE, has_image=self.use_images)
            messages = [{"role": "user", "content": context["design_text"]},
                        {"role": "assistant", "content": layout_json},
                        self._user_message(review_text, room, placed, violations, context["fixed"])]
            logger.info(f"Layout review {review}/{self.review_rounds} for '{room.name}'...")
            answer = self._ask(messages)
            logger.debug(f"Layout review answer: {answer!s:.1500}")
            parsed = llm_utils.extract_json_from_response(answer) if answer else None
            if isinstance(parsed, dict) and (parsed.get("looks_like") or parsed.get("expected")):
                logger.info(f"Layout review: looks like: {parsed.get('looks_like')!s:.200}; "
                            f"expected: {parsed.get('expected')!s:.200}")
            if not isinstance(parsed, dict) or parsed.get("ok") is True or not (
                    parsed.get("objects") or parsed.get("requirements") or parsed.get("fronts")
                    or parsed.get("scatter")):
                logger.info(f"Layout review: layout accepted for '{room.name}'")
                return best
            added = [r for r in parsed.get("requirements") or [] if isinstance(r, dict)]
            requirements = _merge_requirements(requirements, added)
            reviewed_items = self._apply_answer(
                {"objects": parsed.get("objects") or [], "fronts": parsed.get("fronts")},
                context["by_id"], placed, correct_fronts=True)
            reviewed_items = self._scatter(room, reviewed_items, parsed.get("scatter"), context["fixed"])
            reviewed = self._evaluate(context, reviewed_items, requirements)
            before = self._evaluate(context, placed, requirements)
            if _always_on(reviewed[3]) > _always_on(before[3]):
                logger.info("Layout review: proposed changes broke basic rules; keeping the layout")
                best = before
            else:
                # The review judges what the checks cannot (use of space, intent), so its moves
                # stand even if they cost soft requirements; the solver and spreading clean those up.
                best = reviewed
            logger.info(f"Layout review: {len(best[3])} problem(s) after changes; "
                        f"requirements: {', '.join(r.get('check', '?') for r in requirements)}")
            if best[3]:
                best = self._run_round(context, self._revision_messages(context, best, 1, 1),
                                       best, "review-fix", requirements)
        return best

    @staticmethod
    def _determine_fronts(items) -> None:
        """Measure each model's front: the open side (seat, shelves, knee space) opposite
        a closed one (backrest, back panel, modesty panel).

        Symmetric models (tables, boxes, cones) get no front, so relations that need
        one are not checked for them. Asking the LLM instead gave inconsistent,
        sometimes invented answers ("the cone's backrest").
        """
        measured = {}
        for item in items:
            shape = item.get("shape")
            if not shape:
                continue  # geometry unknown: keep the default front, and relations checked
            if item["type"] not in measured:
                measured[item["type"]] = measured_front(shape)
            front, closed = measured[item["type"]]
            item["front_known"] = front is not None
            item["front_measured"] = True
            if front:
                item["front"] = front
        for model_type, (front, closed) in measured.items():
            if closed:
                logger.info(f"Model front: {model_type} {front or 'none'} (side closedness "
                            + ", ".join(f"{k} {v:.2f}" for k, v in closed.items()) + ")")

    def _apply_answer(self, parsed: Dict, by_id: Dict[str, Dict], base: Sequence[Dict] = (),
                      correct_fronts: bool = False) -> List[Dict]:
        """Items from the answer, layered over ``base`` (the layout being revised).

        Measured fronts (and measured "no front") are never overridden. Fronts the LLM
        declared are kept unless `correct_fronts` (the review may fix a wrong one).
        """
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
            if (front in FRONT_AXES and not item.get("front_measured")
                    and (correct_fronts or not item.get("front_known"))):
                item["front"] = front
                item["front_known"] = True
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

    def _solve_relations(self, room, placed, requirements, expected, fixed) -> List[Dict]:
        """Satisfy the relations the LLM declared (against_wall, in_front_of, faces) exactly.

        The LLM decides what relates to what and roughly where; it is poor at
        hitting the precise pose, so the checker's computed fix is applied
        directly. A fix is kept only if it does not break the always-on rules.
        """
        placed = [dict(item) for item in placed]
        for _ in range(SOLVER_PASSES):
            violations = check_layout(room, placed, requirements, expected, fixed, self.wall_thickness)
            # Walls first, then what stands at furniture, then turning in place.
            order = ("against_wall", "in_front_of", "faces")
            fixes = [v for v in violations if v.fix and v.check in order]
            if not fixes:
                break
            changed = False
            for violation in sorted(fixes, key=lambda v: order.index(v.check)):
                trial = _apply_fix(placed, violation.fix)
                before = _always_on(check_layout(room, placed, (), expected, fixed, self.wall_thickness))
                after = _always_on(check_layout(room, trial, (), expected, fixed, self.wall_thickness))
                if after > before:
                    # Turning a chair in a tight ring overlaps its neighbour: separate them, and
                    # keep that only if it leaves fewer problems overall.
                    nudged = self._nudge(room, trial, fixed)
                    total = len(check_layout(room, placed, requirements, expected, fixed, self.wall_thickness))
                    nudged_violations = check_layout(room, nudged, requirements, expected, fixed, self.wall_thickness)
                    if _always_on(nudged_violations) <= before and len(nudged_violations) < total:
                        trial, after = nudged, _always_on(nudged_violations)
                if after <= before:
                    logger.info(f"Applied {violation.check} fix: {violation.hint}")
                    placed = trial
                    changed = True
            if not changed:
                break
        return placed

    def _spread(self, room, placed, requirements, expected, fixed) -> List[Dict]:
        """Push objects apart to the min_spacing the LLM asked for, if that leaves fewer problems.

        Spacing is a property of the whole layout: moving one object of a close pair only
        crowds its other neighbour, so the layout is relaxed as a whole and kept only when
        the total number of problems drops without breaking the always-on rules.
        """
        widths = [float(r["width"]) for r in requirements
                  if isinstance(r, dict) and r.get("check") == "min_spacing" and _number(r.get("width"))]
        if not widths:
            return placed
        before = check_layout(room, placed, requirements, expected, fixed, self.wall_thickness)
        if not any(v.check == "min_spacing" for v in before):
            return placed
        tied, walled = checks.tied(requirements)
        trial = self._nudge(room, placed, fixed, clearance=max(widths) + 0.02,
                            tied=frozenset(tied), walled=frozenset(walled))
        trial = self._solve_relations(room, trial, requirements, expected, fixed)
        after = check_layout(room, trial, requirements, expected, fixed, self.wall_thickness)
        if _always_on(after) <= _always_on(before) and len(after) < len(before):
            logger.info(f"Spread objects to the requested spacing: {len(before)} -> {len(after)} problem(s)")
            return trial
        return placed

    def _scatter(self, room, placed, requests, fixed) -> List[Dict]:
        """Place ids at random free spots in an area, for arrangements an LLM cannot make
        irregular itself (clutter, debris, random obstacles): it asks with
        {"scatter": [{"ids": [...], "area": [x0, x1, y0, y1], "gap": 0.5}]}.
        """
        if not isinstance(requests, list):
            return placed
        placed = [dict(item) for item in placed]
        by_id = {item["id"]: item for item in placed}
        half_x, half_y = inner_half_extents(room, self.wall_thickness)
        zones = [zone for _, zone in doorway_zones(room)]
        for request in requests:
            if not isinstance(request, dict):
                continue
            ids = [i for i in request.get("ids") or [] if i in by_id and not by_id[i].get("on")]
            area = request.get("area")
            if not (isinstance(area, list) and len(area) == 4 and all(_number(v) for v in area)):
                area = None
            gap = float(request["gap"]) if _number(request.get("gap")) else 0.5
            occupied = [item_box(item) for item in list(placed) + list(fixed)
                        if item["id"] not in ids and not item.get("on")]
            for item_id in ids:
                pose = self._free_pose(by_id[item_id], half_x, half_y, occupied, zones,
                                       area=[float(v) for v in area] if area else None,
                                       gap=gap, any_yaw=True)
                by_id[item_id].update(pose)
                occupied.append(item_box(by_id[item_id]))
            if ids:
                logger.info(f"Scattered {len(ids)} object(s) in {area or 'the room'}")
        return placed

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

    def _free_pose(self, item, half_x, half_y, occupied, zones, area=None,
                   gap: float = SAMPLER_GAP, any_yaw: bool = False) -> Dict:
        """A random free pose inside the room (or ``area`` = x0, x1, y0, y1), ``gap`` from others."""
        x0, x1, y0, y1 = area or (-half_x, half_x, -half_y, half_y)
        x0, x1 = max(x0, -half_x), min(x1, half_x)
        y0, y1 = max(y0, -half_y), min(y1, half_y)

        def fits(box):
            if box[0] < -half_x or box[1] > half_x or box[2] < -half_y or box[3] > half_y:
                return False
            if any(min(penetration(box, zone)) > 0 for zone in zones):
                return False
            return all(checks.box_gap(box, other) >= gap and min(penetration(box, other)) <= 0
                       for other in occupied)

        for _ in range(SAMPLER_ATTEMPTS):
            yaw = self.rng.uniform(-math.pi, math.pi) if any_yaw else self.rng.choice((0.0, math.pi / 2))
            x, y = self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)
            if fits(checks.footprint(x, y, item["dims"], yaw)):
                return {"x": x, "y": y, "yaw": yaw}
        if area is not None:
            return {}  # the area is full: the caller keeps the LLM's pose
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

    def _nudge(self, room, placed, fixed, clearance: float = 0.0,
               tied: frozenset = frozenset(), walled: frozenset = frozenset()) -> List[Dict]:
        """Separate overlaps and clear walls/doorways with the smallest moves; no re-layout.

        With ``clearance`` (a min_spacing width) objects are also pushed that far apart
        and off the walls, except pairs in ``tied`` and objects in ``walled``.
        """
        items = [dict(item) for item in placed]
        half_x, half_y = inner_half_extents(room, self.wall_thickness)
        zones = doorway_zones(room)

        def grown(box, margin):
            return (box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin)

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
                    margin = 0.0 if frozenset((a["id"], b["id"])) in tied else clearance / 2
                    over_x, over_y = penetration(grown(item_box(a), margin), grown(item_box(b), margin))
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
                box = grown(item_box(item), 0.0 if item["id"] in walled else clearance)
                dx = max(-half_x - box[0], 0.0) - max(box[1] - half_x, 0.0)
                dy = max(-half_y - box[2], 0.0) - max(box[3] - half_y, 0.0)
                if dx or dy:
                    move(item, dx, dy)
                    moved = True
            if not moved:
                break
        return items


def _apply_on_requirements(placed: List[Dict], requirements: List[Dict]):
    """Honour {"check": "on", "a": item, "b": support} as the item's "on" field.

    LLMs often state "the monitor stands on the desk" as a requirement rather than
    the per-object field; either way the item rests on the support's top. An item
    outside its support's footprint is moved onto its centre.
    """
    stands = {r.get("a"): r.get("b") for r in requirements
              if isinstance(r, dict) and r.get("check") == "on"}
    if not stands:
        return placed, requirements
    by_id = {item["id"]: item for item in placed}
    result = []
    for item in placed:
        support = by_id.get(stands.get(item["id"]))
        if support is not None and support is not item:
            item = dict(item, on=support["id"])
            box = checks.item_box(support)
            if not (box[0] <= item["x"] <= box[1] and box[2] <= item["y"] <= box[3]):
                item.update(x=support["x"], y=support["y"])
        result.append(item)
    return result, [r for r in requirements if not (isinstance(r, dict) and r.get("check") == "on")]


def _number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _apply_fix(placed: List[Dict], fix: Dict[str, Dict[str, float]]) -> List[Dict]:
    """Items with the fixed poses applied; what stands on a moved item moves (and turns) with it."""
    by_id = {item["id"]: item for item in placed}
    result = []
    for item in placed:
        if item["id"] in fix:
            result.append(dict(item, **fix[item["id"]]))
            continue
        support = by_id.get(item.get("on"))
        if support is None or support["id"] not in fix:
            result.append(item)
            continue
        moved = dict(support, **fix[support["id"]])
        turn = moved.get("yaw", 0.0) - support.get("yaw", 0.0)
        dx, dy = item["x"] - support["x"], item["y"] - support["y"]
        result.append(dict(item, x=moved["x"] + dx * math.cos(turn) - dy * math.sin(turn),
                           y=moved["y"] + dx * math.sin(turn) + dy * math.cos(turn),
                           yaw=item.get("yaw", 0.0) + turn))
    return result


def _merge_requirements(requirements: List[Dict], added: List[Dict]) -> List[Dict]:
    merged = [r for r in requirements if isinstance(r, dict)]
    seen = {json.dumps(r, sort_keys=True) for r in merged}
    for requirement in added:
        key = json.dumps(requirement, sort_keys=True)
        if key not in seen:
            merged.append(requirement)
            seen.add(key)
    return merged


def _always_on(violations: List[Violation]) -> int:
    return sum(1 for violation in violations if violation.always_on)


def measured_front(triangles, margin: float = 0.3):
    """(front, closedness per side) when one side is clearly more open than its opposite.

    Returns (None, closedness) for models without a clear front (tables, cones, boxes).
    """
    closed = side_closedness(triangles)
    pairs = [("+x", "-x"), ("+y", "-y")]
    first, second = max(pairs, key=lambda pair: abs(closed[pair[0]] - closed[pair[1]]))
    if abs(closed[first] - closed[second]) < margin:
        return None, closed
    return (first if closed[first] < closed[second] else second), closed
