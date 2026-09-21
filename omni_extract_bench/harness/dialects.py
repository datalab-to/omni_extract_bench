"""Per-vendor JSON Schema dialects.

Every extraction vendor accepts a slightly different subset of JSON Schema, and the ones that
validate strictly reject schemas the permissive ones accept silently. If the harness sends one
schema shape to everyone, the strict vendors score zero on documents they could have handled --
which is a fact about the harness, reported as a fact about the vendor.

That is not hypothetical. In one benchmark run a vendor came 8th of 9 with 19 of 40 documents
failed; every failure was schema delivery, and once fixed it placed 4th. Another vendor lost
half its documents because the harness rejected JSON wrapped in a markdown fence.

Two categories, kept separate on purpose:

  UNIVERSAL   -- removing metadata the BENCHMARK added. Applied to every vendor, because
                 sending our own grader annotations as part of the task is simply a bug.
  DIALECT     -- reshaping the same schema into the form one vendor accepts. Applied per
                 vendor. Content is never changed: same fields, same types, same descriptions.

The rule for a dialect transform: it may change how a constraint is *encoded*, never what is
*asked for*. Resolving a $ref, collapsing a nullable union, or moving a null from an enum to
the field's optionality are encodings. Removing a field, or telling the model what to extract,
is not -- that belongs in the schema itself, identically for everyone.
"""
from __future__ import annotations

from ..metric import resolve_refs  # noqa: F401

import json
import re


BENCHMARK_ONLY_KEYS = ("evaluation_config", "default")


def strip_benchmark_keys(node, keys=BENCHMARK_ONLY_KEYS):
    """Remove harness-added annotations. Safe for every vendor; changes no field or type."""
    if isinstance(node, list):
        return [strip_benchmark_keys(x, keys) for x in node]
    if not isinstance(node, dict):
        return node
    return {k: strip_benchmark_keys(v, keys) for k, v in node.items() if k not in keys}


def collapse_nullable_union(node):
    """Reduce ``anyOf: [{...}, {"type": "null"}]`` to its non-null branch, merging siblings.

    The nullable idiom declares no type of its own, which several vendors reject. Collapsing it
    is also what a grader does when deciding which branch an answer is judged against, so the
    delivered schema matches the scoring.
    """
    if not isinstance(node, dict):
        return node
    for branch_key in ("anyOf", "oneOf", "allOf"):
        branches = [b for b in (node.get(branch_key) or []) if isinstance(b, dict)]
        if not branches:
            continue
        pick = next((b for b in branches if b.get("type") != "null"), None)
        if pick is not None:
            merged = {k: v for k, v in node.items()
                      if k not in ("anyOf", "oneOf", "allOf")}
            for k, v in pick.items():
                merged.setdefault(k, v)
            return merged
    return node


STRICT_ALLOWED_KEYS = ("type", "enum", "properties", "items", "required", "description")


def to_strict_dialect(node, in_items=False, allowed=STRICT_ALLOWED_KEYS):
    """Reshape for a vendor that validates strictly and requires nullable properties.

    Three rules, and they differ BY POSITION -- the detail that makes this worth writing down,
    because applying nullability everywhere fixes properties and breaks array items at once:

      * keys       an allowlist (see ``STRICT_ALLOWED_KEYS``)
      * properties must be nullable: ``["string", "null"]``, and enums must include ``null``
      * items      must be a BARE type: a ``["string","null"]`` union is rejected here

    Modelled on Extend's validator; useful for any vendor with the same shape of constraints.
    """
    if isinstance(node, list):
        return [to_strict_dialect(x, in_items, allowed) for x in node]
    if not isinstance(node, dict):
        return node

    if "type" not in node and "enum" not in node:
        collapsed = collapse_nullable_union(node)
        if collapsed is not node:
            return to_strict_dialect(collapsed, in_items, allowed)

    out = {}
    for key in allowed:
        if key not in node:
            continue
        value = node[key]
        if key == "properties" and isinstance(value, dict):
            out[key] = {k: to_strict_dialect(v, False, allowed) for k, v in value.items()}
        elif key == "items":
            out[key] = to_strict_dialect(value, True, allowed)
        else:
            out[key] = value

    declared = out.get("type")
    if isinstance(declared, list):
        out["type"] = next((t for t in declared if t != "null"), None)
    if "type" not in out and "enum" not in out and out.get("properties"):
        out["type"] = "object"

    if in_items:
        if isinstance(out.get("type"), str) and out["type"] in (
                "string", "number", "integer", "boolean"):
            return {"type": out["type"]}
        return {k: v for k, v in out.items()
                if k in ("type", "properties", "items", "required")}

    if isinstance(out.get("type"), str) and out["type"] in (
            "string", "number", "integer", "boolean"):
        out["type"] = [out["type"], "null"]
    if isinstance(out.get("enum"), list) and None not in out["enum"]:
        out["enum"] = list(out["enum"]) + [None]
    return out


def to_typed_enum_dialect(node):
    """Reshape for a vendor that requires every enum to declare a matching type.

    ``{"enum": ["MILD", "MODERATE", null]}`` is valid JSON Schema and rejected here twice over:
    once for having no ``type``, and then -- after a type is inferred -- for the ``null`` member
    not matching it. The type is inferred from the enum's own values rather than defaulted, and
    the null is removed, since nullability belongs to the field's optionality, not to the value
    set. Also reduces ``additionalProperties`` from a schema to a boolean, which some validators
    require; the map stays open, only the per-value constraint is lost.
    """
    if isinstance(node, list):
        return [to_typed_enum_dialect(x) for x in node]
    if not isinstance(node, dict):
        return node

    node = collapse_nullable_union(node)
    out = dict(node)

    declared = out.get("type")
    if isinstance(declared, list):
        non_null = [t for t in declared if t != "null"]
        out["type"] = non_null[0] if non_null else "string"

    if "enum" in out and "type" not in out:
        values = [v for v in out["enum"] if v is not None]
        kinds = {type(v) for v in values}
        out["type"] = ({str: "string", bool: "boolean", int: "integer", float: "number"}
                       .get(kinds.pop()) if len(kinds) == 1 else "string")

    if isinstance(out.get("enum"), list) and out.get("type") in (
            "string", "boolean", "integer", "number"):
        out["enum"] = [v for v in out["enum"] if v is not None]

    if isinstance(out.get("additionalProperties"), dict):
        out["additionalProperties"] = True

    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: to_typed_enum_dialect(v) for k, v in out["properties"].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = to_typed_enum_dialect(out["items"])
    return out


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
