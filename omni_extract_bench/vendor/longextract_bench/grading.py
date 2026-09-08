"""LongExtractBench — deterministic extraction grader.

Scores a model's extraction against a human-reviewed ground-truth answer key for
one document. No third-party dependencies — standard library only.

THE METRICS
-----------
A document is mostly large *arrays* of rows plus a few *document-level* scalar fields.

  * precision / recall  — ARRAY ROWS only.
        recall    = matched_rows / ground_truth_rows   ("did it find the rows?")
        precision = matched_rows / predicted_rows       ("did it return only real
                                                          rows, or pad with junk?")
    Document-level scalars are single values, not rows, so they never move P/R.

  * leaf accuracy        — values on MATCHED array rows + document-level scalars:
        cells in matched array rows + top-level scalars + nested-object fields
        + nested objects inside rows + nested arrays inside rows (scalar arrays as a
        multiset, object arrays recursed element by element).
    Top-level array rows that do NOT match are charged to precision/recall, not here, so
    their leaves are NOT in the pool; but inside a matched row, nested-array elements that
    don't line up DO count as misses.
    "Of the values on the rows it matched, how many are exactly right?"

HOW ROWS ARE PAIRED
-------------------
To compare two row lists we infer a primary KEY from the data (no hand-picking, so no
way to bias the result): a single column that uniquely identifies rows if one exists,
else a unique *combination* of columns, else no key at all. Rows are then bucketed by
key (one bucket when there is no key) and paired BY CONTENT within each bucket —
identical rows lock first, then the best-overlap residual greedily; only when a bucket's
residual is too large to pair greedily does it fall back to positional order to bound
cost. Each ground-truth row can be claimed once, so duplicate rows a model emits stay
unmatched and correctly lower precision.

FAIRNESS
--------
Every value — in keys and in leaf comparisons — passes through one shared `canonical`
normalizer, so purely cosmetic differences never count as errors for ANY model:
case, whitespace (incl. mid-word OCR spaces), thousands separators, `100.0` vs `100`,
`$`/`%`, smart quotes, leading zeros, and placeholder markers (`..`, `-`, `n/a`).

Note: the grader scores against the ground truth as given; it does not judge whether
the ground truth itself is correct.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

# model name -> extraction filename inside each run's per-doc folder
MODEL_FILES: dict[str, str] = {
    "reducto": "reducto.json",
    "extend": "extend.json",
    "openai": "openai.json",
    "gemini": "gemini.json",
    "llamaextract": "llamaextract.json",
    "claude": "claude.json",
    "datalab": "datalab.json",
}

# Per-doc filename resolution (datasets vary in how GT/schema are named).
GT_NAMES = ("ground_truth.json", "ground_truth_pruned_v3.json")
SCHEMA_NAMES = ("schema.json", "final_schema.json", "final_schema_pruned_v3.json")
_PLACEHOLDERS = {"", "..", "...", "-", "--", "n/a", "na", "none", "null"}

Json = Any  # parsed-JSON value: dict / list / scalar


# ── normalization ─────────────────────────────────────────────────────────────


def canonical(v: Json) -> str:
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


# ── envelope unwrapping ─────────────────────────────────────────────────────────


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


# ── key inference (single -> composite -> no key) ───────────────────────────────


def _scalar_field_names(rows: list[dict[str, Json]]) -> set[str]:
    names: set[str] = set()
    for r in rows:
        for k, v in r.items():
            if not isinstance(v, (dict, list)):
                names.add(k)
    return names


def infer_row_key(
    rows_p: list[dict[str, Json]], rows_g: list[dict[str, Json]]
) -> list[str]:
    """Infer the matching key for two row lists, purely from the data.

    1. Single primary key — the best cross-model-matching column that is itself a
       usable identifier (>=80% non-null AND >=50% unique on both sides).
    2. Composite key — if no single column qualifies, greedily combine matchable
       *dimension* columns (non-unique alone, e.g. region+gender) until the
       combination is near-unique on both sides.
    3. [] — neither works; the caller puts all rows in one bucket and content-pairs.
    """
    if not rows_p or not rows_g:
        return []
    # sorted -> deterministic across processes (set iteration is hash-seed dependent,
    # which otherwise makes key selection — and thus grades — vary run to run)
    fields = sorted(_scalar_field_names(rows_p) | _scalar_field_names(rows_g))

    def usable(field: str, rows: list[dict[str, Json]]) -> bool:
        vals = [r.get(field) for r in rows]
        non_null = [v for v in vals if canonical(v) != ""]
        if len(non_null) < 0.8 * len(rows):
            return False
        return len({canonical(v) for v in non_null}) / len(non_null) >= 0.5

    def match_rate(field: str) -> float:
        bvals = {canonical(r.get(field)) for r in rows_g if canonical(r.get(field))}
        hit = sum(1 for r in rows_p if canonical(r.get(field)) in bvals)
        return hit / max(len(rows_p), 1)

    def non_null_rate(field: str, rows: list[dict[str, Json]]) -> float:
        return sum(1 for r in rows if canonical(r.get(field)) != "") / max(len(rows), 1)

    def uniqueness(flds: list[str], rows: list[dict[str, Json]]) -> float:
        combos = {tuple(canonical(r.get(f)) for f in flds) for r in rows}
        return len(combos) / max(len(rows), 1)

    # 1) single primary key — ANCHORED ON GROUND TRUTH. The key is whatever
    # uniquely identifies GT rows; we do NOT require the competitor to also have a
    # usable key. `match_rate` only PREFERS a GT-key the competitor populated (tie-
    # break), never gates it. A competitor that mis-extracts the key just produces
    # unmatched rows (its own precision/recall hit) instead of being force-paired
    # against the wrong GT rows, which would manufacture false per-cell errors.
    # Single key: among GT-usable fields (>=0.8 non-null, >=0.5 unique on GT), pick the
    # MOST-UNIQUE one (match_rate as tiebreak). Wherever a real identifier exists it IS the
    # most-unique field, so this matches rows on the true key — fixing mis-pairs a match_rate
    # selector caused. On the few no-identifier tables the most-unique field is a measurement
    # (a count); keying on it makes leaf accuracy optimistic (scored only on rows whose value
    # already agrees) — but the wrong-value rows then fall into RECALL, so the error is still
    # counted. This is honest ONLY when leaf is always reported WITH recall + precision, which
    # is the contract here.
    gt_usable = [f for f in fields if usable(f, rows_g)]
    if gt_usable:
        return [max(gt_usable, key=lambda f: (uniqueness([f], rows_g), match_rate(f)))]

    # 2) composite of GT dimension fields — greedily add the field that most
    # increases combined uniqueness ON GT (deterministic: `fields` is sorted, and
    # max() keeps the first/alphabetical winner on ties). Still GT-anchored.
    dims = [f for f in fields if non_null_rate(f, rows_g) >= 0.8]
    selected: list[str] = []
    remaining = list(dims)
    while remaining and len(selected) < 4:
        nxt = max(remaining, key=lambda f: uniqueness([*selected, f], rows_g))
        selected.append(nxt)
        remaining.remove(nxt)
        if uniqueness(selected, rows_g) >= 0.95:
            return selected
    return []


def _rowkey(row: dict[str, Json], kf: list[str], idx: int) -> tuple[Any, ...]:
    """Normalized composite key; positional fallback when no key was inferred."""
    if not kf:
        return ("__idx__", idx)
    return tuple(canonical(row.get(f)) for f in kf)


# ── recursive leaf grading ──────────────────────────────────────────────────────


def _count_leaves(v: Json) -> int:
    """Scalar-leaf count of a value — used to penalize an unmatched/extra element."""
    if isinstance(v, dict):
        return sum(_count_leaves(x) for x in v.values())
    if isinstance(v, list):
        return sum(_count_leaves(x) for x in v)
    return 0 if v is None else 1


def _canon_sig(v: Json) -> Any:
    """Hashable, order-insensitive content signature (canonical leaves) for pairing
    rows by exact content rather than by position."""
    if isinstance(v, dict):
        return ("d", tuple(sorted((k, _canon_sig(x)) for k, x in v.items())))
    if isinstance(v, list):
        return ("l", tuple(sorted(repr(_canon_sig(x)) for x in v)))
    return canonical(v)


_GREEDY_CAP = 4000  # residual_p * residual_g above this -> order-pair to bound cost


def _pair_bucket(
    prs: list[dict[str, Json]], grs: list[dict[str, Json]]
) -> tuple[
    list[tuple[dict[str, Json], dict[str, Json]]],
    list[dict[str, Json]],
    list[dict[str, Json]],
]:
    """Pair pred rows to GT rows within ONE key bucket by CONTENT, not order:
    exact-content rows first (O(n)), then greedy best-overlap for the residual.
    Returns (pairs, unmatched_pred, unmatched_gt). The pair COUNT is unchanged from
    order-pairing (always min(len(prs), len(grs))) -> precision/recall never move;
    only WHICH rows pair changes, which can only raise matched leaves, never lower."""
    gsig: dict[Any, list[dict[str, Json]]] = {}
    for r in grs:
        gsig.setdefault(_canon_sig(r), []).append(r)
    pairs: list[tuple[dict[str, Json], dict[str, Json]]] = []
    res_p: list[dict[str, Json]] = []
    for pr in prs:
        bucket = gsig.get(_canon_sig(pr))
        if bucket:
            pairs.append((pr, bucket.pop()))  # identical content -> lock immediately
        else:
            res_p.append(pr)
    res_g = [r for rs in gsig.values() for r in rs]
    if not res_p or not res_g:
        return pairs, res_p, res_g
    if len(res_p) * len(res_g) > _GREEDY_CAP:  # bound worst case -> order-pair residual
        n = min(len(res_p), len(res_g))
        pairs += list(zip(res_p[:n], res_g[:n]))
        return pairs, res_p[n:], res_g[n:]
    gpool = list(res_g)
    up: list[dict[str, Json]] = []
    for pr in res_p:
        if not gpool:
            up.append(pr)
            continue
        j = max(range(len(gpool)), key=lambda k: grade_value(pr, gpool[k])[1])
        pairs.append((pr, gpool.pop(j)))
    return pairs, up, gpool


def _match_rows(
    p: list[dict[str, Json]], g: list[dict[str, Json]], kf: list[str]
) -> tuple[
    list[tuple[dict[str, Json], dict[str, Json]]],
    list[dict[str, Json]],
    list[dict[str, Json]],
]:
    """Match two object-row lists. With a key: bucket by key, then content-pair
    within each bucket (fixes key-collision mispairs on repetitive tables). Without
    a key: one bucket, content-pair all (replaces a positional align that mismatched
    whenever row order/count differed)."""
    if not kf:
        return _pair_bucket(list(p), list(g))
    gb: dict[tuple[Any, ...], list[dict[str, Json]]] = {}
    for i, r in enumerate(g):
        gb.setdefault(_rowkey(r, kf, i), []).append(r)
    pb: dict[tuple[Any, ...], list[dict[str, Json]]] = {}
    for i, r in enumerate(p):
        pb.setdefault(_rowkey(r, kf, i), []).append(r)
    pairs: list[tuple[dict[str, Json], dict[str, Json]]] = []
    up: list[dict[str, Json]] = []
    ug: list[dict[str, Json]] = []
    for k in set(gb) | set(pb):
        kp, kup, kug = _pair_bucket(pb.get(k, []), gb.get(k, []))
        pairs += kp
        up += kup
        ug += kug
    return pairs, up, ug


def grade_value(pv: Json, gv: Json) -> tuple[int, int]:
    """Recursively grade ANY value against ground truth -> (leaf_total, leaf_match).
    * scalar           — 1 leaf; matches if canonical values are equal.
    * object           — recurse over the union of keys.
    * array of objects — key-matched (same inference as top-level); each matched
      pair recursed; extra/missing elements count their leaves as misses.
    * array of scalars — multiset compare (order-insensitive).
    """
    if isinstance(pv, dict) or isinstance(gv, dict):
        pv = pv if isinstance(pv, dict) else {}
        gv = gv if isinstance(gv, dict) else {}
        t = m = 0
        for k in set(pv) | set(gv):
            tt, mm = grade_value(pv.get(k), gv.get(k))
            t += tt
            m += mm
        return t, m
    if isinstance(pv, list) or isinstance(gv, list):
        pv = pv if isinstance(pv, list) else []
        gv = gv if isinstance(gv, list) else []
        pd = [x for x in pv if isinstance(x, dict)]
        gd = [x for x in gv if isinstance(x, dict)]
        if (
            pd or gd
        ):  # array of objects -> content-matched (fixes key-collision mispairs)
            pairs, up, ug = _match_rows(pd, gd, infer_row_key(pd, gd))
            t = m = 0
            for prow, grow in pairs:
                tt, mm = grade_value(prow, grow)
                t += tt
                m += mm
            for prow in up:  # extra pred elements -> their leaves are misses
                t += _count_leaves(prow)
            for grow in ug:  # missing gt elements -> their leaves are misses
                t += _count_leaves(grow)
            return t, m
        cp = Counter(canonical(x) for x in pv)  # array of scalars -> multiset
        cg = Counter(canonical(x) for x in gv)
        return max(sum(cp.values()), sum(cg.values())), sum((cp & cg).values())
    if pv is None and gv is None:
        return 0, 0
    return 1, int(canonical(pv) == canonical(gv))


# ── document grading ─────────────────────────────────────────────────────────────


def _arrays(schema: dict[str, Json]) -> list[str]:
    out: list[str] = []
    for k, v in schema.get("properties", {}).items():
        t = v.get("type")
        if t == "array" or (isinstance(t, list) and "array" in t):
            out.append(k)
    return out


def grade(
    pred: dict[str, Json], gt: dict[str, Json], schema: dict[str, Json]
) -> dict[str, Json]:
    """Score one model's output against the answer key for one document."""
    arr_keys = _arrays(schema)
    gt_rows = pred_rows = matched = 0
    leaf_total = leaf_match = base_total = base_match = 0

    for arr in arr_keys:  # array rows -> precision/recall + in-row leaves
        g: list[dict[str, Json]] = [
            r for r in (gt.get(arr, []) or []) if isinstance(r, dict)
        ]
        p: list[dict[str, Json]] = [
            r for r in (pred.get(arr, []) or []) if isinstance(r, dict)
        ]
        gt_rows += len(g)
        pred_rows += len(p)
        if not g and not p:
            continue
        pairs, _up, _ug = _match_rows(p, g, infer_row_key(p, g))
        for (
            prow,
            grow,
        ) in pairs:  # unmatched pred/gt -> precision/recall hit (no leaves added)
            tt, mm = grade_value(prow, grow)
            matched += 1
            leaf_total += tt
            leaf_match += mm

    for k in (set(gt) | set(pred)) - set(arr_keys):  # document-level leaves (leaf only)
        tt, mm = grade_value(pred.get(k), gt.get(k))
        base_total += tt
        base_match += mm

    tot_leaf = leaf_total + base_total
    tot_match = leaf_match + base_match
    return {
        "gt_rows": gt_rows,
        "pred_rows": pred_rows,
        "matched": matched,
        "precision": matched / pred_rows if pred_rows else 0.0,
        "recall": matched / gt_rows if gt_rows else 0.0,
        "leaf_accuracy": (100 * tot_match / tot_leaf) if tot_leaf else 0.0,
        "leaf_total": tot_leaf,
        "leaf_match": tot_match,
    }


# ── loading ────────────────────────────────────────────────────────────────────


def _collapse(node: Json) -> Json:
    """A single-object list (`[{...}]`) is the same record as the object — common
    in some provider envelopes. Collapse it so callers always get the dict."""
    if isinstance(node, list) and len(node) == 1 and isinstance(node[0], dict):
        return node[0]
    return node


_DATALAB_SIDECAR_SUFFIXES = ("_citations", "_meta")


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


def _load_doc(path: Path, model: str) -> dict[str, Json] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    raw = raw.get("result", raw) if isinstance(raw, dict) else raw
    obj = _collapse(unwrap(raw) if model == "reducto" else raw)
    if model == "datalab" and isinstance(obj, dict):
        obj = _drop_datalab_sidecars(
            obj
        )  # ignore Datalab _citations/_meta sidecars when scoring
    return obj if isinstance(obj, dict) else None


def _load_gt(path: Path) -> dict[str, Json]:
    raw = _collapse(unwrap(json.loads(path.read_text())))
    return raw.get("result", raw) if isinstance(raw, dict) else raw


def find_in(doc_dir: Path, names: tuple[str, ...]) -> Path | None:
    """First existing file from `names` in a doc folder (GT_NAMES / SCHEMA_NAMES)."""
    return next((doc_dir / n for n in names if (doc_dir / n).exists()), None)


def load_prediction(env_path: Path, provider: str) -> dict[str, Json] | None:
    """Load a provider's extraction from its output envelope (None if absent).
    The single source of truth for reading a `<provider>.json` envelope."""
    return _load_doc(env_path, provider)


def load_ground_truth(doc_dir: Path) -> dict[str, Json] | None:
    p = find_in(doc_dir, GT_NAMES)
    return _load_gt(p) if p else None
