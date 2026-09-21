"""Reshaping a schema into the form ONE vendor accepts -- a toolbox `prepare_schema` draws on.

Every extraction vendor accepts a slightly different subset of JSON Schema, and the ones that
validate strictly reject schemas the permissive ones accept silently. If the harness sends one
schema shape to everyone, the strict vendors score zero on documents they could have handled --
which is a fact about the harness, reported as a fact about the vendor. In one benchmark run a
vendor came 8th of 9 with 19 of 40 documents failed; every failure was schema delivery, and
once fixed it placed 4th.

What every vendor is asked is `schema.py`; this is the per-vendor re-encoding of it. The rule:
a transform may change how a constraint is ENCODED, never what is ASKED FOR. Resolving a $ref,
collapsing a nullable union, or moving a null from an enum to the field's optionality are
encodings. Removing a field, or telling the model what to extract, is not.

Only adapters import this. Nothing here runs unless an adapter's `prepare_schema` calls it.
"""
from __future__ import annotations

# Re-exported: an adapter's `prepare_schema` composes it with the transforms below, and the
# scorer's copy is the one that must agree with what a vendor was sent.
from ..metric import resolve_refs  # noqa: F401


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

    collapsed = collapse_nullable_union(node)
    if collapsed is not node:
        # RECURSE on the merged node, as `to_strict_dialect` does: a union whose own branch is
        # a union (`Optional[list[str] | dict]`) otherwise keeps the inner `anyOf`, and a
        # nested union is what the vendor rejected in the first place.
        return to_typed_enum_dialect(collapsed)
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


def drop_schema_metadata(node):
    """Remove `$`-prefixed annotations -- `$schema`, `$id`, `$comment`.

    `resolve_refs` consumes `$ref` and `$defs`; these are what is left, and they describe the
    document rather than the data. Several validators reject them as unknown keys.
    """
    if isinstance(node, list):
        return [drop_schema_metadata(x) for x in node]
    if not isinstance(node, dict):
        return node
    return {k: drop_schema_metadata(v) for k, v in node.items() if not k.startswith("$")}
