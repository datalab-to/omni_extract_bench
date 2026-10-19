"""Reading what a vendor sent back: JSON in a markdown fence (`parse_model_json`), a cost under any of
several names and units (`cost_from_response`), and an answer given as a list (`as_object`).
"""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def parse_model_json(text):
    """A model's JSON answer, read whether or not it is wrapped in a markdown code fence; None when the text
    is not JSON.

    Some models return bare JSON and some wrap it in ```json. A parser that accepts only bare JSON scores
    the others at zero: one run discarded 12 of 24 documents from the most accurate provider that way."""
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
    `(None, None)` when the response states none.

    Vendors report cost under different names and units. A field in cents read as dollars overstates the
    cost 100 times, so each known field carries its own conversion."""
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
    """An adapter's parsed answer as one JSON object, the extraction, or None.

    One vendor answers with a list, one entry per extraction pass. A list of one object is that object. A
    longer list is merged shallowly, the first value of each field kept, so array fields from separate passes
    survive instead of the last pass overwriting them. Without this, every completed Reducto job read as
    "no extraction"."""
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
