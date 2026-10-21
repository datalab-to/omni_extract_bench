"""Reading what a vendor sent back:

- `parse_model_json`: JSON, with or without a markdown fence;
- `cost_from_response`: a cost under any of several names and units; and
- `as_object`: an answer given as a list."""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def parse_model_json(text):
    """A model's JSON answer, with or without a ```json fence; None when the text isn't JSON."""
    if not text:
        return None
    match = _FENCE.match(text)
    candidate = match.group(1) if match else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def cost_from_response(body, paths=None):
    """The cost in a vendor's response, in dollars, and the field it came from: `(usd, "field=value")`, or
    `(None, None)`. Each known field carries its own unit, since cents read as dollars would overstate
    the cost 100 times."""
    if paths is None:
        paths = (
            (("cost_breakdown", "final_cost_cents"), 0.01),
            (("total_cost",), 0.01),
            (("usage", "cost"), 1.0),
            (("usage_info", "cost"), 1.0),
            (("cost",), 1.0),
        )
    if not isinstance(body, dict):
        return None, None
    for path, multiplier in paths:
        cur = body
        for key in path:
            cur = cur.get(key) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            return round(float(cur) * multiplier, 6), f"{'.'.join(path)}={cur}"
    return None, None


def as_object(value) -> dict | None:
    """An answer as one JSON object, or None. A list of one object is that object; a longer list (one entry
    per extraction pass, as Reducto sends) is merged shallowly, keeping each field's first value."""
    if isinstance(value, dict):
        return value or None
    if isinstance(value, list):
        objects = [x for x in value if isinstance(x, dict)]
        if len(objects) == 1:
            return objects[0] or None
        if len(objects) > 1:
            merged: dict = {}
            for obj in objects:
                for k, v in obj.items():
                    if k not in merged or merged[k] in (None, [], {}):
                        merged[k] = v
            return merged or None
    return None
