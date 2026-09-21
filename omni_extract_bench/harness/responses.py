"""Making sense of what a vendor sent back.

Three different ways a reply arrives wrong, each of which cost documents:

  * a model wrapped its JSON in a markdown fence, and `json.loads` failed on a backtick --
    12 of 24 documents discarded from the most accurate provider in the field
  * a vendor answered with a LIST, one entry per extraction pass, and the adapter threw the
    answer away for not being a dict -- every completed Reducto job read as "no extraction"
  * a cost was reported in CENTS under a name that looked like dollars, overstating by 100x
"""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def parse_model_json(text):
    """Parse a model's JSON answer, tolerating a markdown code fence.

    Some models return bare JSON and some wrap it in ```json. A parser that accepts only the
    first silently scores the second at zero: one benchmark run discarded 12 of 24 documents
    from a provider that was otherwise the most accurate in the field, because ``json.loads``
    failed on a backtick. Returns ``None`` when the text is genuinely not JSON.
    """
    if not text:
        return None
    match = _FENCE.match(text)
    candidate = match.group(1) if match else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def cost_from_response(body, paths=None):
    """Pull a cost figure out of a vendor response, converting to USD.

    Vendors report cost under different names and units, and an adapter that returns only the
    extraction throws it away -- after which the absence gets read as "this vendor does not
    report cost". Units matter: a field named in **cents** reported as dollars overstates by
    100x. Returns ``(usd, "field=value")`` or ``(None, None)``.
    """
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
    """Normalise an adapter's parsed answer to the schema-shaped object, or None.

    One vendor returns a LIST -- one entry per extraction pass. A single-entry list is just the
    extraction; multi-entry lists are merged shallowly, first writer wins, so array fields from
    separate passes survive instead of the last pass overwriting them.

    This existed in the old runner and was lost in the rewrite, which turned every completed
    Reducto job into "completed with no extraction": the work was done and paid for, and the
    adapter threw the answer away for being a list.
    """
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
