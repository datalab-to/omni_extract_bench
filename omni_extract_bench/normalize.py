"""Value canonicalisation and schema helpers shared by the scorer.

These build on the vendored `longextract_bench` grader (MIT, (c) Micro1 -- see
`vendor/longextract_bench/LICENSE`): its unwrapping, sidecar handling and leaf counting are
used as-is; its string normaliser is replaced (extension 1).

Two extensions, both applied uniformly to every subset and every provider:

1. `canonical()` is THE string fold (see its docstring): format differences are free,
   content differences are not. It replaces the upstream normaliser, which deleted every
   internal period, slash, hyphen and space and so merged `1/2` with `12`.

2. Array detection recognises `anyOf` / `oneOf` / `allOf` branches. The upstream helper matched
   only a bare `type: "array"`. Nullable arrays are commonly declared
   `anyOf: [{type: array}, {type: null}]`; those were graded as document-level leaves (content
   still scored) but never counted in recall or precision, so recall came out 0/0 on such
   documents.

Applied via the module global so every call site resolves the patched version.
"""
from __future__ import annotations

from .vendor.longextract_bench import grading as G

import re
import unicodedata

_PLACEHOLDERS = {'', 'none', 'null', '...', '-', '..', 'na', '--', 'n/a'}


def canonical(v):
    """THE canonical string form. Folds format, keeps content.

    1. None / [] / {} -> "" (absent); booleans -> "true"/"false".
    2. Lowercase; NFKD with combining marks dropped (o-umlaut == o); micro sign == u; smart
       quotes and dashes -> ASCII.
    3. Numbers: `,` `$` `%` removed, then compared numerically (`1,000` == `1000`,
       `100.0` == `100`, `$5` == `5`). Overflow-safe.
    4. Placeholder markers (`n/a`, `none`, `-`, `..`, ...) -> "" -- they assert nothing.
    5. Short footnote/reference markers (`[1]`, `[a]`) removed.
    6. Whitespace collapsed to ONE space; punctuation stripped from the EDGES of the value
       only (`Acme Inc.` == `Acme Inc`, `N.V.,` == `N.V.`, `"quoted"` == `quoted`).
       Punctuation BETWEEN characters is kept: `1/2` != `12`, `Section 2.1` != `Section 21`,
       `v1.2` != `v12`. An earlier version deleted every internal period, slash, hyphen and
       space, which merged those; measured on the reference corpus the narrower rule costs
       every provider 0.3-0.6 points about equally and changes no rank.
    7. Leading zeros inside digit runs dropped (`09. Mai` == `9. Mai`).
    """
    if v is None or v == [] or v == {}:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("\u00b5", "u").replace("\u03bc", "u")
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "-")):
        s = s.replace(a, b)
    num = s.replace(",", "").replace("$", "").replace("%", "")
    try:
        f = float(num)
        if f == f and f not in (float("inf"), float("-inf")):
            return str(int(f)) if f == int(f) else str(f)
        return s[:64]
    except (ValueError, OverflowError):
        pass
    s = re.sub(r"\s*\[\s*(?:\d{1,2}|[a-z])\s*\]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in _PLACEHOLDERS:
        return ""
    s = s.strip(" .,;:!?\"'()[]{}-")
    return re.sub(r"\d+", lambda m: str(int(m.group())), s)


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
