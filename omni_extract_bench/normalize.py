"""Value canonicalisation and schema helpers shared by the scorer.

Three functions here -- `_upstream_canonical`, `unwrap` and `_drop_datalab_sidecars` -- are
taken verbatim from the `longextract_bench` grader (MIT, (c) Micro1; see `NOTICE`), which this
repo vendored in full until they were the only parts still reachable. Keeping them byte-for-byte
is deliberate: `canonical` decides equality for every leaf this benchmark scores, so a
paraphrase would be a silent change to every number, and a diff against upstream should stay
readable.

`canonical()` wraps the upstream form rather than editing it, adding two things:

1. Enclosing quote marks and bracket punctuation are stripped. Upstream folds smart quotes and
   strips `,` `-` `.` and whitespace, but not quote characters. Ground truth sometimes drops
   quotes the source document actually prints (a field literally named `verbatim_term`, for
   instance), and because row matching keys on such free-text fields, one surviving quote can
   zero an entire document. A quote wrapping a value is cosmetic, exactly like the punctuation
   already folded.

2. Overflow safety. A model can emit `Infinity`, `NaN`, or a numeric string too large for
   upstream's `int(float(...))`, which raises rather than returning a value.

What is NOT folded is as load-bearing as what is: the placeholder WORDS ("n/a", "none", "-")
are content, because a page can print them -- see METRIC_SPEC section 5 and
`values.states_nothing` for the empty-string rule that does fold.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

Json = Any  # parsed-JSON value: dict / list / scalar

# ── upstream, verbatim (MIT, (c) Micro1 -- see NOTICE) ───────────────────────────
_PLACEHOLDERS = {"", "..", "...", "-", "--", "n/a", "na", "none", "null"}
_DATALAB_SIDECAR_SUFFIXES = ("_citations", "_meta")


def _upstream_canonical(v: Json) -> str:
    """Canonicalize ONE value so only cosmetic noise is folded — never real content.
    1. None -> "" ; lowercase ; strip.
    2. Typography: smart quotes/apostrophes/dashes -> ascii.
    3. Numbers compared numerically (`1,000`==`1000`, `100.0`==`100`, `$5`==`5`).
    4. Placeholder markers (`..`, `-`, `n/a`, ...) -> "" (treated as empty/null).
    5. Strip ALL whitespace + commas + hyphens + periods
       (`Inst itutional`==`Institutional`, `Table B-1.`==`Table B-1`). Numbers are
       already handled by the numeric path above, so period-stripping here only
       folds cosmetic punctuation on non-numeric strings.
    6. Strip leading zeros inside digit runs (`09. Mai`==`9. Mai`).
    """
    # Empty equivalence: None / [] / {} / blank string all mean "absent" — fold
    # them together so null-vs-empty-list representation differences aren't errors.
    if v is None or v == [] or v == {}:
        return ""
    s = str(v).strip().lower()
    # 1b. Unicode fold: decompose + drop combining accents (ö->o), then map the
    #     micro sign / greek mu to ascii (µg == ug). Pure cosmetic, never content.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("µ", "u").replace("μ", "u")
    # 1c. Strip short footnote / reference markers like "229 [1]", "x [x]" — only
    #     1-2 digit or single-letter brackets so real bracketed content (e.g. years
    #     "[2024]", codes) is preserved.
    s = re.sub(r"\s*\[\s*(?:\d{1,2}|[a-z])\s*\]", "", s)
    for a, b in (
        ("’", "'"),
        ("‘", "'"),
        ("“", '"'),
        ("”", '"'),
        ("–", "-"),
        ("—", "-"),
    ):
        s = s.replace(a, b)
    num = s.replace(",", "").replace("$", "").replace("%", "")
    try:
        f = float(num)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        pass
    if re.sub(r"\s+", " ", s) in _PLACEHOLDERS:
        return ""
    s = re.sub(r"[\s,\-.]", "", s)
    return re.sub(r"\d+", lambda m: str(int(m.group())), s)


def unwrap(o: Json) -> Json:
    """Strip a {value, citations} envelope so a cited field becomes its bare value,
    recursively. No-op for plain output."""
    if isinstance(o, dict):
        if "value" in o and ("citations" in o or "confidence" in o) and len(o) <= 6:
            return unwrap(o["value"])
        return {k: unwrap(v) for k, v in o.items()}
    if isinstance(o, list):
        return [unwrap(x) for x in o]
    return o


def _drop_datalab_sidecars(obj: Json) -> Json:
    """Recursively strip Datalab's per-field metadata sidecars. Datalab decorates each
    extracted field X with sibling keys `X_citations` (provenance block IDs) and `X_meta`
    (extraction_status / reasoning / verification) — metadata no other system emits and
    that is absent from the schema and ground truth. We keep them in the stored output
    but ignore them when scoring so they do not count as extra predicted leaves.

    A key K is dropped ONLY when it ends in a sidecar suffix AND its base name (K minus
    that suffix) is also a sibling key in the SAME object — the signature of a Datalab
    sidecar. This protects a genuine field that merely ends in such a suffix: a real
    `regulatory_citations` field whose base `regulatory` is NOT a sibling is kept, while
    Datalab's own `regulatory_citations_citations`/`_meta` (base `regulatory_citations`
    IS a sibling) are dropped. Pure load-time normalization — no extracted value is
    altered."""
    if isinstance(obj, dict):
        out: dict[str, Json] = {}
        for k, v in obj.items():
            is_sidecar = any(
                k.endswith(s) and k[: -len(s)] in obj for s in _DATALAB_SIDECAR_SUFFIXES
            )
            if not is_sidecar:
                out[k] = _drop_datalab_sidecars(v)
        return out
    if isinstance(obj, list):
        return [_drop_datalab_sidecars(x) for x in obj]
    return obj


# ── this repo's own ──────────────────────────────────────────────────────────────


def canonical(v):
    """Canonical form of a value, with enclosing quotes and bracket punctuation folded.

    Overflow-safe: a model can emit `Infinity`, `NaN`, or a numeric string too large for the
    upstream `int(float(...))`, which raises rather than returning a value.
    """
    try:
        s = _upstream_canonical(v)
    except (OverflowError, ValueError):
        return str(v)[:64]
    if isinstance(s, str):
        s = s.strip("\"'").replace("(", "").replace(")", "").replace("/", "")
    return s


def prep_prediction(obj):
    """Unwrap common response envelopes and drop per-field metadata sidecars.

    Some extractors decorate each field `X` with sibling keys `X_citations` (provenance) and
    `X_meta` (status/reasoning). That metadata is absent from the schema and from ground truth,
    so it is ignored when scoring rather than counted as extra predicted leaves -- otherwise a
    provider would be penalised for returning provenance.
    """
    obj = unwrap(obj)
    if isinstance(obj, dict):
        obj = _drop_datalab_sidecars(obj)
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
