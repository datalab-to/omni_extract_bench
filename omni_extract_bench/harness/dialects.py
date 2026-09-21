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

WHAT LIVES HERE: a transform MORE THAN ONE adapter composes. One with a single caller belongs
in that adapter, beside the `prepare_schema` that composes it -- `to_strict_dialect` is
Extend's, `to_typed_enum_dialect` and `drop_schema_metadata` are LlamaExtract's, and each sits
in its own provider module. Only adapters import this, and nothing here runs unless an
adapter's `prepare_schema` calls it.
"""
from __future__ import annotations

# Re-exported: an adapter's `prepare_schema` composes it with the transforms below, and the
# scorer's copy is the one that must agree with what a vendor was sent.
from ..metric import resolve_refs  # noqa: F401


MAX_REF_DEPTH = 200


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
