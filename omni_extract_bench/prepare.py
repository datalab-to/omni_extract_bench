"""Shape a document, and ask a schema the only two questions the scorer has.

This runs BEFORE any value is compared, and it is about the container rather than the contents:
strip the envelope a vendor wrapped an answer in, drop the metadata it hung beside it, and find
the parts of a schema the scorer must not score.

Nothing here alters an extracted value. That is `values.py`'s job, and keeping the two apart is
what makes it possible to say exactly where a value can change.
"""
from __future__ import annotations

from typing import Any

Json = Any  # parsed-JSON value: dict / list / scalar

#: Suffixes a vendor hangs beside a field to carry provenance about it.
_SIDECAR_SUFFIXES = ("_citations", "_meta")


#: Keys that can legitimately keep `value` company inside a METADATA ENVELOPE. An envelope is
#: a wrapper a vendor puts around an answer to attach provenance to it; the answer is
#: `["value"]` and everything else describes it.
_ENVELOPE_METADATA = frozenset({
    "citations", "citation", "confidence", "score", "reasoning", "evidence", "provenance",
    "source", "sources", "page", "pages", "bbox", "span", "spans", "offset", "offsets",
    "extraction_status", "status", "meta",
})


def unwrap(o: Json) -> Json:
    """Strip a {value, metadata} envelope down to the value, recursively.

    An envelope has a "value" key, at least one metadata key, and nothing else. The last
    clause matters: "value" is also an ordinary field name, and rows like
    {"value": 12.4, "unit": "USD"} must survive intact -- 7,130 of them in the corpus.
    Matching on which keys are present rather than how many is what distinguishes
    {"value": x, "citations": [...]} from {"value": x, "citations": [...], "unit": "USD"}.

    Fires zero times on the present corpus; kept because vendors do emit envelopes.
    """
    if isinstance(o, dict):
        others = set(o) - {"value"}
        if "value" in o and others and others <= _ENVELOPE_METADATA:
            return unwrap(o["value"])
        return {k: unwrap(v) for k, v in o.items()}
    if isinstance(o, list):
        return [unwrap(x) for x in o]
    return o


def _drop_field_sidecars(obj: Json) -> Json:
    """Drop per-field provenance siblings so they are not charged as predicted values.

    A key is dropped only if it ends in a sidecar suffix AND its base name is a sibling in the
    same object. That protects a real field that merely ends the same way: `regulatory_citations`
    survives unless `regulatory` sits beside it.

    Applied to every vendor identically; the rule is about shape, not origin.
    """
    if isinstance(obj, dict):
        out: dict[str, Json] = {}
        for k, v in obj.items():
            sidecar = any(k.endswith(suf) and k[: -len(suf)] in obj
                          for suf in _SIDECAR_SUFFIXES)
            if not sidecar:
                out[k] = _drop_field_sidecars(v)
        return out
    if isinstance(obj, list):
        return [_drop_field_sidecars(x) for x in obj]
    return obj


def prep_prediction(obj):
    """Unwrap common response envelopes and drop per-field metadata sidecars.

    Some extractors decorate each field `X` with sibling keys `X_citations` (provenance) and
    `X_meta` (status/reasoning). That metadata is absent from the schema and from ground truth,
    so it is ignored when scoring rather than counted as extra predicted leaves -- otherwise a
    provider would be penalised for returning provenance.
    """
    obj = unwrap(obj)
    if isinstance(obj, dict):
        obj = _drop_field_sidecars(obj)
    return obj if isinstance(obj, dict) else {}


def prep_ground_truth(obj):
    """Unwrap common envelopes around ground truth."""
    obj = unwrap(obj)
    return obj if isinstance(obj, dict) else {}


def is_open_map(node) -> bool:
    """True when a schema node declares an object whose KEYS come from the document.

    ``additionalProperties`` asks the extractor to invent the property names by reading them
    off the page. This benchmark does not evaluate that shape: extraction APIs are built around
    a schema that names its fields, and `dialects.STRICT_ALLOWED_KEYS` does not even forward
    the keyword, so a strict vendor receives a bare ``{"type": "object"}`` and has nothing to
    answer with. Grading such a node would score a request the harness never delivered.

    Detection reads EXPLICIT presence as intent rather than JSON Schema semantics, under which
    ``additionalProperties`` defaults to true and every object would qualify.
    """
    if not isinstance(node, dict):
        return False
    for branch in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(branch, []) or []:
            if is_open_map(sub):
                return True
    extra = node.get("additionalProperties")
    return extra is True or isinstance(extra, dict)


def unwrap_schema(node):
    """Resolve anyOf/oneOf to the non-null branch."""
    if not isinstance(node, dict):
        return {}
    for br in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(br, []) or []:
            if isinstance(sub, dict) and sub.get("type") != "null":
                return sub
    return node


def resolve_refs(schema, root=None, depth=0, max_depth=12):
    """Inline local ``$ref`` so a schema is self-describing.

    Some vendors resolve ``$defs`` themselves; others reject a bare ``$ref`` because it declares
    no type. Inlining changes nothing semantically. Recursive definitions stop at ``max_depth``
    rather than expanding forever.
    """
    if root is None:
        root = schema
    if depth > max_depth or not isinstance(schema, (dict, list)):
        return schema
    if isinstance(schema, list):
        return [resolve_refs(x, root, depth + 1, max_depth) for x in schema]
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        cur = root
        for part in ref[2:].split("/"):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, dict):
            merged = {k: v for k, v in schema.items() if k != "$ref"}
            merged.update({k: v for k, v in cur.items() if k not in merged})
            return resolve_refs(merged, root, depth + 1, max_depth)
    return {k: (resolve_refs(v, root, depth + 1, max_depth) if k != "$defs" else v)
            for k, v in schema.items() if k != "$defs"}
