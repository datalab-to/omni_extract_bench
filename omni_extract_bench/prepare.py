"""Shape a document, and ask a schema the only two questions the scorer has.

This runs BEFORE any value is compared, and it is about the container rather than the contents:
strip the envelope a vendor wrapped an answer in, drop the metadata it hung beside it, and find
the parts of a schema the scorer must not grade.

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
    """Strip a metadata envelope so a cited field becomes its bare value, recursively.

    An object is an envelope when it has a `value` key, at least one recognised metadata key,
    and NOTHING ELSE. That last clause is the whole difficulty, because `value` is also an
    ordinary field name and a row that merely contains one must not be gutted:

        {"value": 12.4, "citations": ["p3"]}                      envelope -- unwrap to 12.4
        {"value": 12.4, "data_period": "FY25", "unit": "USD"}     a row -- 7,130 of these, keep
        {"value": 12.4, "label": "Revenue"}                       a row -- 21,792 of these, keep

    Naming the metadata is what makes that possible. Counting keys instead -- "a `value`, a
    `citations`, and not too many others" -- cannot tell these apart:

        {"value": 12.4, "citations": ["p3"]}                 envelope
        {"value": 12.4, "citations": ["p3"], "unit": "USD"}  a row that cites its source

    and unwrapping the second discards `unit` without trace. What the other keys ARE is the
    question; how many there are never was.

    This fires zero times on the present corpus: no vendor here emits the shape. It is kept
    because vendors do emit it, and a benchmark that silently scored an envelope object as a
    wrong answer would be reporting a fact about the response format as a fact about reading.
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
    """Drop per-field metadata siblings, so provenance is never charged as a predicted value.

    Some extractors decorate each field `X` with siblings `X_citations` (provenance block ids)
    and `X_meta` (extraction status, reasoning, verification). That metadata is in neither the
    schema nor the ground truth, so counting it would penalise a provider for being informative.

    A key is dropped ONLY when it ends in a sidecar suffix AND its base name is also a sibling
    in the SAME object. That pairing is what identifies a sidecar, and it protects a real field
    that happens to end the same way: a genuine `regulatory_citations` whose base `regulatory`
    is absent survives, while `regulatory_citations_citations` (whose base IS present) does not.

    Applied to every vendor's output identically -- the rule is about the SHAPE, not about who
    produced it, even though today only one provider emits it.
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
                merged = dict(sub)
                if "evaluation_config" in node and "evaluation_config" not in merged:
                    merged["evaluation_config"] = node["evaluation_config"]
                return merged
    return node
