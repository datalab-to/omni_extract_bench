#!/usr/bin/env python3
"""Fair typed grader — blends the vendored longextract grader's deterministic row-matching
with Contextual ExtractBench's per-field typed comparators.

Leaf comparison is TYPE-AWARE:
  * Contextual subset: use the field's declared `evaluation_config`
    (string_exact / string_case_insensitive / string_fuzzy / string_semantic / string_url /
     number_exact / number_tolerance / integer_exact / boolean_exact).
  * Other subsets (no declaration): INFER from JSON-Schema type —
     integer/bool -> exact; number -> float-precision tolerance (decimals only, so IDs typed
     as number are NOT loosened); string -> "fair string" = canonical + date-normalize +
     unicode-fraction-normalize. One canonical form, one comparison.

Fully deterministic: no LLM judge, no per-field metric modes, no per-vendor rules. The same
canonical form decides every comparison, so a run is exactly reproducible.

Guards against unfair MERGES (verified in self-test): different phone/account numbers,
distinct floats, and truncated strings all stay MISMATCHED.
"""
from __future__ import annotations
import collections
import json, re
from datetime import datetime
from . import normalize as N              # canonical form + schema helpers
from . import matching as OM               # provably-optimal max-weight row assignment
from .vendor.longextract_bench import grading as G   # MIT (c) Micro1 -- see vendor/LICENSE

# Dimension fields safe to block on (compared EXACTLY). Rows disagreeing here can never
# pair, so per-block optimal == global optimal — see optimal_match.match_rows.
_BLOCK_HINTS = ("segment_type",)


def _row_signature(row):
    """Canonical multiset of the row's scalar leaves, keyed by path, used as the assignment
    weight for row pairing.

    Computing the exact weight would mean a full recursive grade for every candidate pair --
    O(n*m) full grades, infeasible on thousand-row tables. The signature is computed ONCE per
    row (O(n+m) total) and the weight is a dict intersection.

    It covers EVERY scalar leaf of the row, including those inside nested arrays and objects,
    with arrays contributing under an index-free path. The previous version used top-level
    scalars only, so rows whose identity lives in a nested payload -- `{"row_label": "16 to 19
    years old", "values": [...]}` repeated once per section -- tied on weight and were paired by
    position. The score then depended on the ORDER a provider emitted rows in: shuffling a
    correct prediction moved one document from 100.0 to 53.6, and 24 of 68 documents with
    duplicate-label rows moved by up to 46 points. Row order is declared free by this metric,
    so the pairing weight must not see it; an index-free multiset does not.

    For flat rows this equals the exact matched-leaf count, so the assignment is optimal for the
    true objective. For nested rows it is now the matched-leaf count under the same order-free
    treatment scoring applies to scalar arrays. The reported score is always computed by full
    grading afterwards.
    """
    counts = {}

    def walk(o, pre):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.endswith(("_citations", "_meta")):
                    continue
                walk(v, f"{pre}.{k}" if pre else k)
        elif isinstance(o, list):
            for x in o:
                walk(x, f"{pre}[]")
        elif o is not None:
            key = (pre, canon_key(o))
            counts[key] = counts.get(key, 0) + 1

    walk(row, "")
    # one dict key per OCCURRENCE so a plain dict intersection is a multiset intersection
    return {f"{p}#{v}#{i}": True for (p, v), n in counts.items() for i in range(n)}


# Set of grades in which some block exceeded optimal_match.MAX_EXACT and fell back to greedy.
# Greedy is SUBOPTIMAL (measured: it recovers 99.2-99.8% of the optimal assignment weight),
# and the loss lands only on providers that return large tables -- i.e. the ones with the best
# coverage. The fallback cannot be removed (the largest gold array is 6881 rows and the solver
# is O(n^3)), so instead it is RECORDED and surfaced in the result, never silent.
_INEXACT = []


def _pair_rows(pd, gd):
    """Optimal max-weight pairing of predicted rows to GT rows (see optimal_match)."""
    psig = [_row_signature(r) for r in pd]
    gsig = [_row_signature(r) for r in gd]
    pidx = {id(r): i for i, r in enumerate(pd)}
    gidx = {id(r): i for i, r in enumerate(gd)}

    def w(p, g):
        a, b = psig[pidx[id(p)]], gsig[gidx[id(g)]]
        if len(b) < len(a):
            a, b = b, a
        return sum(1 for k, v in a.items() if b.get(k) == v)

    keys = [k for k in _BLOCK_HINTS if any(k in r for r in (pd[:1] + gd[:1]))]
    # sig_fn lets the greedy fallback generate only the pairs that can score above zero,
    # instead of the full product. Same weights, same pairing -- just not O(n*m) to reach it.
    pairs, up, ug, exact = OM.match_rows(pd, gd, w, block_keys=tuple(keys),
                                         key_fn=canon_key,
                                         sig_fn=_row_signature)
    if not exact:
        _INEXACT.append(max(len(pd), len(gd)))
    return pairs, up, ug

# ONE method for ALL subsets and ALL providers: every field is compared the same way.
# Contextual's per-field `evaluation_config` (string_fuzzy / string_semantic / number_tolerance
# / ...) is deliberately NOT honoured. It was previously implemented behind a flag that was
# always False, so it never affected a reported score while adding nine comparison modes, a
# Levenshtein implementation and an LLM judge to the surface that had to be reasoned about.
# Per-vendor comparison rules are also the thing a competitor would most reasonably object to.

# ── string / value normalizers ───────────────────────────────────────────────────
_FRAC = {"½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3", "⅛": "1/8",
         "⅜": "3/8", "⅝": "5/8", "⅞": "7/8", "⅕": "1/5", "⅙": "1/6", "⅐": "1/7"}
def _defrac(s: str) -> str:
    for u, a in _FRAC.items():
        s = s.replace(u, a)
    return s

_DATEFMTS = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y",
             "%d %B %Y", "%d-%b-%Y", "%d%b%Y", "%m/%d/%y", "%Y/%m/%d", "%d.%m.%Y",
             "%m.%d.%Y", "%b %d %Y", "%B %d %Y"]
def _asdate(v):
    s = str(v).strip()
    if not re.search(r"\d", s) or len(s) > 24:
        return None
    for f in _DATEFMTS:
        try:
            return datetime.strptime(s, f).date().isoformat()
        except ValueError:
            pass
    return None

# Sign NOTATION varies by convention; sign SEMANTICS must not. Accounting parentheses,
# the unicode minus, and a trailing minus all mean "negative" — those are spellings of the
# same number and are folded. A disagreement about whether the value IS negative is a real
# error and stays a mismatch: -98.2 never equals +98.2.
_NEG_WRAP = re.compile(r"^\((.*)\)$")


def _sign_normalize(s: str):
    """Return (unsigned_text, sign) with sign in {1,-1}, folding notation variants."""
    s = s.strip()
    sign = 1
    m = _NEG_WRAP.match(s)          # (98.2) -> accounting negative
    if m:
        sign, s = -1, m.group(1).strip()
    s = s.replace("\u2212", "-").replace("\u2013", "-")   # unicode minus / en-dash
    if s.endswith("-"):             # trailing minus (some ERP exports)
        sign, s = -sign, s[:-1].strip()
    while s.startswith(("-", "+")):
        if s[0] == "-":
            sign = -sign
        s = s[1:].strip()
    return s, sign


def _asfloat(v):
    """Parse a float ONLY if it looks decimal (has a '.') — integers/IDs stay exact.

    Sign notation is normalized first, so `(98.2)`, `-98.2` and `\u221298.2` all parse to
    -98.2 — but the resulting SIGN is preserved and compared.
    """
    body, sign = _sign_normalize(str(v))
    body = body.replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    if "." not in body:
        return None
    try:
        return sign * float(body)
    except ValueError:
        return None

# ── leaf comparators -> 1.0 (match) or 0.0 ───────────────────────────────────────
def _canon(v):
    try:
        return N.canonical(v)
    except Exception:
        return str(v).lower()

def canon_key(v):
    """THE canonical form of a value. Two values are equal iff their keys are equal.

    This is the whole comparison rule. There is exactly one of these functions, and both jobs
    that need "are these equal?" -- scoring a leaf, and weighting a candidate row pairing --
    call it. That is not a stylistic preference: the benchmark previously had two independent
    implementations that were required to agree, and they silently diverged twice. Once when
    the pairing weight demanded literal equality while scoring accepted date formats (a correct
    extraction scored 50.0), and again when a numeric fix put integers and decimals in
    different namespaces so `5` and `5.0` stopped pairing (one provider's recall went to 0.000
    across two entire subsets). With a single function, a pairing/scoring disagreement is not a
    bug that testing has to catch -- it is unrepresentable.

    Four cases, in this order:
      1. booleans   -- canonical form, never coerced to a number
      2. numbers    -- rounded to 7 significant digits (matching the previous 1e-6 relative
                       tolerance) and canonicalised, so 5, 5.0 and "5.00" agree. Only decimals
                       parse, so ID-like integers stay exact.
      3. dates      -- ISO form, so 2024-01-15 and 01/15/2024 agree. `_asdate` is strict:
                       "1/2", "Q1" and "2-3-13" are not dates, so this cannot swallow values
                       that mean something else.
      4. otherwise  -- unicode fractions expanded, then canonicalised
    """
    if isinstance(v, bool):
        return _canon(v)
    f = _asfloat(v)
    if f is not None:
        try:
            return _canon(float(f"{f:.7g}"))
        except (ValueError, OverflowError):
            return _canon(v)
    d = _asdate(v)
    if d:
        return f"#d{d}"
    return _canon(_defrac(str(v)))


def cmp_leaf(pred, gold) -> float:
    """1.0 if the two values are equal under `canon_key`, else 0.0."""
    return 1.0 if canon_key(pred) == canon_key(gold) else 0.0


# ── schema helpers ───────────────────────────────────────────────────────────────
def _unwrap_schema(node):
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

def _is_array(node):
    node = _unwrap_schema(node)
    t = node.get("type")
    return t == "array" or (isinstance(t, list) and "array" in t)

# ── the grader — mirrors G.grade/grade_value EXACTLY, but with typed leaf comparison ──
def _count_leaves(node):
    if isinstance(node, dict):
        return sum(_count_leaves(v) for v in node.values())
    if isinstance(node, list):
        return sum(_count_leaves(v) for v in node)
    return 0 if node is None else 1  # None is not a leaf (matches G._count_leaves)


# ── fairness: GT rows that assert nothing ────────────────────────────────────────
# Some ground truth carries placeholder rows whose PAYLOAD is entirely null — e.g.
# `employees: [{data_period: "FY2025 H1", segment_type: "company", value: null}]`. The row
# states no fact; it only says "this metric exists as a concept". An extractor that returns
# `[]` has said exactly the same thing, yet gets charged for every scoping leaf in the row.
# Verified on the 10-Qs: datalab AND gpt-5.6 both return [] there, i.e. no extractor infers
# the convention, so it penalizes everyone for something the schema never states.
# Dropping these rows is uniform (applies to every subset), deterministic, and favors no
# provider. Payload = any field that is not a repeated scoping/dimension field.
_DIMENSION_HINTS = ("period", "segment", "type", "name", "id", "label", "category", "unit",
                    "scale", "date", "quarter", "year")


def _row_asserts_nothing(row):
    if not isinstance(row, dict):
        return False
    payload = [k for k in row
               if not k.endswith(("_citations", "_meta"))
               and not any(h in k.lower() for h in _DIMENSION_HINTS)]
    if not payload:
        return False  # all-dimension row: can't tell, keep it
    return all(row.get(k) is None for k in payload)


def _drop_empty_gt_rows(node):
    """Recursively remove GT array rows whose payload is entirely null."""
    if isinstance(node, dict):
        return {k: _drop_empty_gt_rows(v) for k, v in node.items()}
    if isinstance(node, list):
        kept = [x for x in node if not _row_asserts_nothing(x)]
        return [_drop_empty_gt_rows(x) for x in kept]
    return node

def pair_object_keys(pred: dict, gold: dict):
    """Pair the keys of two objects by CANONICAL form, not by literal string.

    Object keys are values too. Comparing the same string canonically when it sits in a field
    and exactly when it sits in a key is a second definition of "are these equal?" -- the thing
    `canon_key` exists to make unrepresentable -- and it only shows up on objects whose keys
    come from the DOCUMENT rather than from the schema (an open `additionalProperties` map).
    Schema-declared property names are unaffected: both sides spell them the way the schema
    does, so canonical pairing returns exactly what literal pairing returned.

    It matters because ground truth is not reliably verbatim about case. In this corpus a
    document prints a heading in capitals, gold records it title-cased, and an extractor that
    transcribed it faithfully scored zero for the whole group -- penalised for being closer to
    the document than the gold file is. A benchmark cannot ask for verbatim transcription and
    then grade the transcription against a normalised answer.

    Returns a list of ``(pred_key | None, gold_key | None)`` pairs, in gold-then-pred order.

    COLLISION GUARD: if canonicalisation would merge two distinct keys of the SAME object
    (``{"Total", "TOTAL"}``), that object falls back to literal pairing. Merging them would
    silently discard one side's value, which is a worse failure than the one being fixed.
    """
    def index(obj):
        out = {}
        for k in obj:
            ck = canon_key(k) if isinstance(k, str) else k
            if ck in out:
                return None                     # collision -> caller falls back to literal
            out[ck] = k
        return out

    ip, ig = index(pred), index(gold)
    if ip is None or ig is None:
        keys = list(dict.fromkeys(list(gold) + list(pred)))
        return [(k if k in pred else None, k if k in gold else None) for k in keys]

    order = list(dict.fromkeys(list(ig) + list(ip)))
    return [(ip.get(ck), ig.get(ck)) for ck in order]


def fair_grade_value(pv, gv, sch):
    """Recursive leaf scorer, mirroring G.grade_value. Nested arrays penalize
    unmatched rows as leaf misses (exactly like the original). Scalar leaves use the
    typed comparator (evaluation_config or inferred)."""
    sch = _unwrap_schema(sch) if isinstance(sch, dict) else {}
    if isinstance(pv, dict) or isinstance(gv, dict):
        pv = pv if isinstance(pv, dict) else {}
        gv = gv if isinstance(gv, dict) else {}
        props = sch.get("properties") or {}
        t = m = 0
        for pk, gk in pair_object_keys(pv, gv):
            # the schema names properties the way GOLD spells them
            sub = props.get(gk if gk is not None else pk, {})
            tt, mm = fair_grade_value(
                pv.get(pk) if pk is not None else None,
                gv.get(gk) if gk is not None else None,
                sub,
            )
            t += tt; m += mm
        return t, m
    if isinstance(pv, list) or isinstance(gv, list):
        pv = pv if isinstance(pv, list) else []
        gv = gv if isinstance(gv, list) else []
        item = sch.get("items") or {}
        pd = [x for x in pv if isinstance(x, dict)]
        gd = [x for x in gv if isinstance(x, dict)]
        if pd or gd:
            pairs, up, ug = _pair_rows(pd, gd)
            t = m = 0
            for prow, grow in pairs:
                tt, mm = fair_grade_value(prow, grow, item)
                t += tt; m += mm
            for row in up:
                t += _count_leaves(row)   # spurious pred rows -> misses
            for row in ug:
                t += _count_leaves(row)   # missing gt rows -> misses
            return t, m
        # scalar array -> multiset under the same comparator.
        #
        # Counted with a Counter rather than a scan-and-pop. cmp_leaf IS exact equality under
        # canon_key, so a multiset is what the old loop computed -- but it was O(n*m) and
        # recomputed canon_key(x) once per candidate. A survey paper with ~1,100 citations made
        # that ~1.2M canonicalisations of long strings for ONE document-provider pair. Same
        # numbers (fuzzed against the old loop, 3,000 randomized trials, zero mismatches), O(n+m).
        gold_counts = collections.Counter(canon_key(y) for y in gv)
        mm = 0
        for x in pv:
            k = canon_key(x)
            if gold_counts.get(k):
                gold_counts[k] -= 1
                mm += 1
        return max(len(pv), len(gv)), mm
    if pv is None and gv is None:
        return 0, 0
    # one-side None: let cmp_leaf decide via canonical ('none'/'n/a' canonicalize to '' like
    # None, so they match — matching G.grade_value, which never short-circuits on None).
    return 1, int(cmp_leaf(pv, gv) >= 1.0)

def fair_grade(pred, gt, schema):
    del _INEXACT[:]                      # per-grade; see _INEXACT
    """Score one document.

    Top-level array keys are scored by the SAME routine as nested ones
    (`fair_grade_value`); only recall/precision bookkeeping is done here. That unification
    fixes two defects that made top-level arrays behave differently from identical nested
    data:

      1. OMISSION WAS FREE AT THE TOP LEVEL. This function used to score only MATCHED row
         pairs and discard unmatched gold rows, so returning 44 of 349 correct rows scored
         100.0 (denominator 88) while the same data nested one level scored 12.6
         (denominator 698). Coverage lived only in `recall`, but the headline metric is
         leaf_accuracy, so a truncating extractor looked perfect. Omission is a real failure
         and is now charged wherever it occurs.

      2. TOP-LEVEL ARRAYS OF SCALARS WERE NEVER SCORED. Rows were filtered with
         `isinstance(r, dict)`, so `["alpha","beta"]` yielded an empty list, hit the
         `not g and not p` skip, and contributed a denominator of 0 — while being excluded
         from the non-array loop as well. Those fields were silently invisible.

    Both defects inflated scores for extractors that omit data, in the direction of making
    results look better than they were, and applied to every A/B run through this metric.
    """
    pred = N.prep_prediction(pred or {}); gt = N.prep_ground_truth(gt or {})
    # Rows asserting nothing are ignored on BOTH sides. Filtering only ground truth broke
    # identity: a prediction that mirrors ground truth EXACTLY still carried the rows ground
    # truth had just discarded, so they counted as spurious and `grade(gt, gt)` came out at
    # 95.09 on a real 10-Q (11 of 164 rows dropped from one side only). That penalised the
    # most faithful possible extraction, and P1 missed it because generated documents never
    # contain an all-null row.
    gt = _drop_empty_gt_rows(gt)
    pred = _drop_empty_gt_rows(pred)
    schema = schema or {}
    props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
    arr_keys = N.arrays_of(schema) if isinstance(schema, dict) else []
    gt_rows = pred_rows = matched = 0
    leaf_total = leaf_match = base_total = base_match = 0

    for arr in arr_keys:
        gv, pv = gt.get(arr), pred.get(arr)
        # recall/precision are row-level and only defined for arrays of objects
        g_rows = [r for r in (gv or []) if isinstance(r, dict)] if isinstance(gv, list) else []
        p_rows = [r for r in (pv or []) if isinstance(r, dict)] if isinstance(pv, list) else []
        gt_rows += len(g_rows); pred_rows += len(p_rows)
        if g_rows or p_rows:
            pairs, _up, _ug = _pair_rows(p_rows, g_rows)
            matched += len(pairs)
        # scoring itself is delegated, so top-level behaves exactly like nested
        tt, mm = fair_grade_value(pv, gv, _unwrap_schema(props.get(arr, {})))
        leaf_total += tt; leaf_match += mm

    for k in (set(gt) | set(pred)) - set(arr_keys):
        tt, mm = fair_grade_value(pred.get(k), gt.get(k), props.get(k, {}))
        base_total += tt; base_match += mm

    tot = leaf_total + base_total; tm = leaf_match + base_match
    return {"leaf_total": tot, "leaf_match": tm,
            "leaf_accuracy": (100 * tm / tot) if tot else 0.0,
            "gt_rows": gt_rows, "pred_rows": pred_rows, "matched": matched,
            "recall": matched / gt_rows if gt_rows else 0.0,
            "precision": matched / pred_rows if pred_rows else 0.0,
            # False => at least one array was too large to solve exactly and used the greedy
            # fallback, so this score is a slight UNDER-estimate. Never silent.
            "matching_exact": not _INEXACT,
            "greedy_blocks": sorted(_INEXACT, reverse=True)}


# ── self-test: the equivalences hold AND the traps stay mismatched ───────────────
if __name__ == "__main__":
    T = [
        # equivalences the benchmark deliberately grants (format, not content)
        ("10/31/2024", "2024-10-31", 1),    # date format
        ("33.33333333", "33.3333333", 1),   # float precision
        ("1/2", "\u00bd", 1),                   # unicode fraction
        ("5", "5.00", 1),                   # integer vs decimal spelling
        ("(98.2)", "-98.2", 1),             # accounting-negative notation
        ("US Department of Health", "US Department of Health", 1),
        # TRAPS - must NOT merge:
        ("8303911426", "8303511426", 0),    # different phone (integers stay exact)
        ("830-391-1426", "830-351-1426", 0),
        ("800", "800 Ft. from Surface", 0), # under/over-extraction
        ("Ron Bohaty, x@y 402", "Ron Bohaty", 0),
        ("100.5", "100.6", 0),              # distinct floats
        ("1/", "1\u00bc", 0),                   # truncated
        ("2-3-13", "3-2-13", 0),            # swapped date = different day
        ("-98.2", "98.2", 0),               # sign is content, not notation
        ("Acme Inc", "Acme Corp", 0),
    ]
    ok = 0
    for p_, g_, exp in T:
        got = int(cmp_leaf(p_, g_) >= 1.0)
        flag = "OK " if got == exp else "FAIL"
        ok += got == exp
        print(f"  {flag} cmp({p_!r},{g_!r}) = {got} (want {exp})")
    print(f"\n{ok}/{len(T)} self-tests pass")
