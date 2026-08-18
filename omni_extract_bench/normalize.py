"""Value canonicalisation and schema helpers shared by the scorer.

These extend the vendored `longextract_bench` grader (MIT, (c) Micro1 -- see
`vendor/longextract_bench/LICENSE`) rather than replacing it, so the numbers this repo
produces stay identical to the ones the upstream metric produces on the same inputs.

Two extensions, both applied uniformly to every subset and every provider:

1. `canonical()` strips enclosing quote marks. The upstream normaliser folds smart quotes and
   strips `,` `-` `.` and whitespace, but not quote characters. Ground truth sometimes drops
   quotes that the source document actually prints (a field literally named `verbatim_term`,
   for instance). Because row matching keys on such free-text fields, one surviving quote can
   zero an entire document. A quote wrapping a value is cosmetic, exactly like the punctuation
   already folded.

2. Array detection recognises `anyOf` / `oneOf` / `allOf` branches. The upstream helper matched
   only a bare `type: "array"`. Nullable arrays are commonly declared
   `anyOf: [{type: array}, {type: null}]`; those were graded as document-level leaves (content
   still scored) but never counted in recall or precision, so recall came out 0/0 on such
   documents.

Applied via the module global so every call site resolves the patched version.
"""
from __future__ import annotations

from .vendor.longextract_bench import grading as G

_orig_canonical = G.canonical


def canonical(v):
    """Canonical form of a value, with enclosing quotes and bracket punctuation folded.

    Overflow-safe: a model can emit `Infinity`, `NaN`, or a numeric string too large for the
    upstream `int(float(...))`, which raises rather than returning a value.
    """
    try:
        s = _orig_canonical(v)
    except (OverflowError, ValueError):
        return str(v)[:64]
    if isinstance(s, str):
        s = s.strip("\"'").replace("(", "").replace(")", "").replace("/", "")
    return s


def _prop_is_array(v) -> bool:
    """True when a JSON-Schema property is an array, including under anyOf/oneOf/allOf."""
    if not isinstance(v, dict):
        return False
    t = v.get("type")
    if t == "array" or (isinstance(t, list) and "array" in t):
        return True
    for branch in ("anyOf", "oneOf", "allOf"):
        for sub in v.get(branch, []) or []:
            if isinstance(sub, dict):
                st = sub.get("type")
                if st == "array" or (isinstance(st, list) and "array" in st):
                    return True
    return False


def arrays_of(schema) -> list:
    """Top-level property names that hold arrays."""
    return [k for k, v in (schema.get("properties") or {}).items() if _prop_is_array(v)]


# Patch the module globals: the vendored grader resolves these at call time.
G.canonical = canonical
G._arrays = arrays_of


def prep_prediction(obj):
    """Unwrap common response envelopes and drop per-field metadata sidecars.

    Some extractors decorate each field `X` with sibling keys `X_citations` (provenance) and
    `X_meta` (status/reasoning). That metadata is absent from the schema and from ground truth,
    so it is ignored when scoring rather than counted as extra predicted leaves -- otherwise a
    provider would be penalised for returning provenance.
    """
    obj = G.unwrap(obj)
    if isinstance(obj, dict):
        obj = G._drop_datalab_sidecars(obj)
    return obj if isinstance(obj, dict) else {}


def prep_ground_truth(obj):
    """Unwrap common envelopes around ground truth."""
    obj = G.unwrap(obj)
    return obj if isinstance(obj, dict) else {}
