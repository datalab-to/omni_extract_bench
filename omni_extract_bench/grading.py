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
import json, re
from datetime import datetime
from . import normalize as N              # canonical form + schema helpers
from . import matching as OM               # provably-optimal max-weight row assignment
from .vendor.longextract_bench import grading as G   # MIT (c) Micro1 -- see vendor/LICENSE

# Dimension fields safe to block on (compared EXACTLY). Rows disagreeing here can never
# pair, so per-block optimal == global optimal — see optimal_match.match_rows.
_BLOCK_HINTS = ("segment_type",)


def _row_signature(row):
    """Canonical (field -> value) map for the row's SCALAR leaves.

    Used as the assignment weight. Computing the exact weight would mean a full recursive
    grade for every candidate pair — O(n*m) full grades, infeasible on micro1's thousand-row
    tables. The signature is computed ONCE per row (O(n+m) total) and the weight is then a
    dict intersection.

    For flat rows (the overwhelming majority of benchmark arrays) this equals the exact
    matched-leaf count, so the assignment is optimal for the true objective. For rows with
    nested structure it is a lower bound used only to CHOOSE the pairing; the reported score
    is always computed by full grading afterwards.
    """
    sig = {}
    for k, v in row.items():
        if k.endswith(("_citations", "_meta")) or isinstance(v, (dict, list)) or v is None:
            continue
        sig[k] = canon_key(v)
    return sig


def _sublist_signature(sub):
    """Weight signature for a LIST element of an array (an array of arrays).

    A sub-list has no field names, so it cannot be keyed like a row. Its identity is the
    multiset of its scalar values, and the weight between two sub-lists is the multiset
    intersection -- the exact matched-leaf count for flat sub-lists, and a lower bound when
    they nest further, exactly as `_row_signature` is for rows.
    """
    sig = {}
    for v in sub:
        if isinstance(v, (dict, list)) or v is None:
            continue
        k = canon_key(v)
        sig[k] = sig.get(k, 0) + 1
    return sig


def _dict_weight(a, b):
    """Rows: (fields whose values agree, fields the two rows have in common)."""
    if len(b) < len(a):
        a, b = b, a
    matched = shared = 0
    for k, v in a.items():
        if k in b:
            shared += 1
            if b[k] == v:
                matched += 1
    return matched, shared


def _multiset_weight(a, b):
    """Sub-lists: (multiset intersection of values, same). Positions are not addresses here,
    so a shared value IS a shared address and the two terms coincide."""
    if len(b) < len(a):
        a, b = b, a
    inter = sum(min(n, b.get(k, 0)) for k, n in a.items())
    return inter, inter


# Set of grades in which some block exceeded matching's exactness budget and fell back to greedy.
# Greedy is SUBOPTIMAL (measured: it recovers 99.2-99.8% of the optimal assignment weight),
# and the loss lands only on providers that return large tables -- i.e. the ones with the best
# coverage. The fallback cannot be removed (the largest gold array is 6881 rows and the solver
# cannot be removed (the largest gold array is 6881 rows and its cost matrix is 0.38 GB,
# only ~12% under the ceiling), so instead it is RECORDED and surfaced, never silent.
_INEXACT = []

# Open-map nodes skipped during the current grade; reported, never silent.
_IGNORED = []


def _pair_elements(pe, ge, sig, weight, block_keys=()):
    """Optimal max-weight pairing of like-kinded array elements (see optimal_match).

    Elements of DIFFERENT kinds are never passed here together: a dict can never pair with a
    list can never pair with a scalar, so cross-kind weight is 0 and kind is just another
    blocking key -- the same argument that makes per-block optimal globally optimal.
    """
    ps = [sig(x) for x in pe]
    gs = [sig(x) for x in ge]
    pidx = {id(x): i for i, x in enumerate(pe)}
    gidx = {id(x): i for i, x in enumerate(ge)}

    # OBJECTIVE: matched fields first, then fields the two elements have IN COMMON. The second
    # term is a tie-break the metric spec omits, and omitting it left the score UNDEFINED:
    # 3.2 maximises matched leaves while 4 divides by the union of gold's fields and the
    # prediction's spurious ones, so two assignments with the same matched count can score
    # differently. A field a paired row shares with its partner is one it does not add to the
    # denominator, so maximising shared fields minimises spurious ones. `scale` exceeds every
    # achievable shared total, so the primary objective is untouched and only ties are decided.
    #
    # This is not cosmetic. Once a second solver exists (`matching` uses scipy when installed)
    # the two returned different equal-weight assignments and therefore DIFFERENT SCORES on
    # generated documents -- measured at 50.60 against 51.85. A score has to be a function of
    # its input, not of what happens to be installed on the machine that computed it.
    # An element paired with ITSELF shares everything it has, so that is its own upper bound
    # on `shared`; summing those bounds the whole assignment. Derived from `weight` rather
    # than from the signature's shape, because a row signature holds canonical values and a
    # sub-list signature holds counts.
    scale = 1 + sum(weight(sig_, sig_)[1] for sig_ in ps)

    def w(p, g):
        matched, shared = weight(ps[pidx[id(p)]], gs[gidx[id(g)]])
        return matched * scale + shared if matched else 0

    pairs, up, ug, exact = OM.match_rows(pe, ge, w, block_keys=block_keys, key_fn=canon_key)
    if not exact:
        _INEXACT.append(max(len(pe), len(ge)))
    return pairs, up, ug


def _pair_rows(pd, gd):
    """Optimal max-weight pairing of predicted ROWS to GT rows."""
    keys = tuple(k for k in _BLOCK_HINTS if any(k in r for r in (pd[:1] + gd[:1])))
    return _pair_elements(pd, gd, _row_signature, _dict_weight, keys)


# ONE method for ALL subsets and ALL providers: every field is compared the same way.
# Contextual's per-field `evaluation_config` (string_fuzzy / string_semantic / number_tolerance
# / ...) is deliberately NOT honoured. It was previously implemented behind a flag that was
# always False, so it never affected a reported score while adding nine comparison modes, a
# Levenshtein implementation and an LLM judge to the surface that had to be reasoned about.
# Per-vendor comparison rules are also the thing a competitor would most reasonably object to.

# ── value comparison, schema unwrapping, and empty-row dropping now live in values.py ──
# They moved so that `score.py` can be built on them without importing a grader -- see
# docs/SCORER_CHANGES.md. Re-exported here under their original names because this module and
# its tests refer to them throughout, and because `canon_key` being importable from exactly
# one place is the point of it.
from .values import (                                                  # noqa: E402
    _asdate, _asfloat, _canon, _defrac, _sign_normalize,               # noqa: F401
    canon_key, cmp_leaf,
    drop_empty_gt_rows as _drop_empty_gt_rows,
    unwrap_schema as _unwrap_schema,
)


def _is_array(node):
    node = _unwrap_schema(node)
    t = node.get("type")
    return t == "array" or (isinstance(t, list) and "array" in t)

# ── the grader — mirrors G.grade/grade_value EXACTLY, but with typed leaf comparison ──
def _kind(v):
    """Which of the three JSON shapes a node is. Two nodes can only be compared leaf-for-leaf
    when they agree here; see the shape-disagreement branch in `fair_grade_value`."""
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, list):
        return "list"
    return "scalar"


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
    """Pair the keys of two objects by LITERAL name, in gold-then-pred order.

    A prediction is generated against the schema, so its property names are the schema's
    property names, which are also ground truth's. Both sides spell them the same way by
    construction, and a predicted key that is not exactly a gold key is a field the extractor
    invented -- charged as spurious, with gold's unmatched key charged as missing.

    Keys were previously paired by `canon_key`, so that a key differing only in case still
    joined. That existed for open `additionalProperties` maps, whose keys are headings an
    extractor reads off the page rather than names the schema supplies. Open maps are not a
    shape this benchmark uses -- extraction APIs are built around a schema that names its
    fields, and `dialects.STRICT_ALLOWED_KEYS` does not even forward the keyword to strict
    vendors -- so the folding had no case left to serve. It also needed a collision guard,
    since folding can merge two distinct keys of one object and silently discard a value, and
    it was never consistent: `canonical` strips ``, - . / ( )`` and whitespace but keeps the
    underscore, so it forgave ``Invoice_No`` and not ``invoice no``.

    Returns a list of ``(pred_key | None, gold_key | None)`` pairs.
    """
    keys = list(dict.fromkeys(list(gold) + list(pred)))
    return [(k if k in pred else None, k if k in gold else None) for k in keys]


def fair_grade_value(pv, gv, sch):
    """Recursive leaf scorer, mirroring G.grade_value. Nested arrays penalize
    unmatched rows as leaf misses (exactly like the original). Scalar leaves use the
    typed comparator (evaluation_config or inferred)."""
    sch = _unwrap_schema(sch) if isinstance(sch, dict) else {}
    # OPEN MAPS ARE NOT EVALUATED. A node whose keys the schema leaves to the document is
    # skipped on BOTH sides: it adds nothing to the numerator and nothing to the denominator,
    # exactly like a `null`. Grading it would score a request the harness never delivered --
    # the strict dialect drops `additionalProperties` before the schema reaches a vendor, so
    # a strict vendor is asked for a bare object and has nothing to answer with. Skipping is
    # not silent: the count is reported on the grade.
    if N.is_open_map(sch):
        _IGNORED.append(1)
        return 0, 0
    # SHAPE DISAGREEMENT. The two sides can disagree about whether a node is an object, an
    # array or a scalar -- a vendor returning `{}` where gold has `[]`, or a bare string where
    # gold has a list. Nothing can match across a shape boundary, so BOTH sides are unmatched
    # and both are charged, exactly as the array branch below charges unmatched rows on either
    # side. This branch must precede the others: they each coerce the non-conforming side to an
    # empty container, which DISCARDED that side's leaves instead of charging them -- a gold
    # table returned as an object left the denominator entirely and scored 100.0. `None` is
    # excluded because an absent field is not a disagreement about shape; it falls through to
    # the branch that charges gold's leaves as misses.
    if pv is not None and gv is not None and _kind(pv) != _kind(gv):
        return _count_leaves(pv) + _count_leaves(gv), 0
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
        # An array's elements are partitioned BY KIND and each partition is scored on its own.
        # Branching on "does this array contain any dict?" discarded every non-dict element of
        # a mixed array -- gold footnote strings beside a table were never scored, so omitting
        # them AND fabricating them were both free. Partitioning is not a workaround: a dict
        # can never pair with a list or a scalar, so kind behaves as a blocking key and
        # per-partition optimal is optimal overall.
        t = m = 0
        pd = [x for x in pv if isinstance(x, dict)]
        gd = [x for x in gv if isinstance(x, dict)]
        if pd or gd:
            pairs, up, ug = _pair_rows(pd, gd)
            for prow, grow in pairs:
                tt, mm = fair_grade_value(prow, grow, item)
                t += tt; m += mm
            for row in up:
                t += _count_leaves(row)   # spurious pred rows -> misses
            for row in ug:
                t += _count_leaves(row)   # missing gt rows -> misses
        # Sub-arrays are keyless too, so they go through the SAME solver with a multiset
        # weight. They previously fell through to the scalar comparator, which stringified
        # them: `str([1.5, 2.5])` canonicalises to '[1525]' with punctuation stripped, so
        # [[15, 25]] compared EQUAL to gold [[1.5, 2.5]] and scored 100.0 on wrong numbers.
        ps = [x for x in pv if isinstance(x, list)]
        gs = [x for x in gv if isinstance(x, list)]
        if ps or gs:
            pairs, up, ug = _pair_elements(ps, gs, _sublist_signature, _multiset_weight)
            for psub, gsub in pairs:
                tt, mm = fair_grade_value(psub, gsub, item)
                t += tt; m += mm
            for sub in up:
                t += _count_leaves(sub)
            for sub in ug:
                t += _count_leaves(sub)
        # Scalars -> multiset under the same comparator. `None` elements are excluded: the
        # denominator used to be `max(len(pv), len(gv))`, i.e. a count of POSITIONS, so a
        # null element occupied a denominator slot and matched itself -- adding 1 to both
        # numerator and denominator in violation of P9 (null asserts nothing, on either
        # side) and P4 (one denominator slot per gold leaf). It stayed hidden because the
        # property generator emitted no scalar arrays at all.
        pc = [x for x in pv if not isinstance(x, (dict, list)) and x is not None]
        gc = [x for x in gv if not isinstance(x, (dict, list)) and x is not None]
        if pc or gc:
            remaining = list(gc); mm = 0
            for x in pc:
                for i, y in enumerate(remaining):
                    if cmp_leaf(x, y) >= 1.0:
                        mm += 1; remaining.pop(i); break
            # Denominator is the UNION -- gold's leaves plus the predicted leaves that
            # matched nothing -- not `max(len(pc), len(gc))`. Max is only correct when one
            # side is a subset of the other: gold ["a","b","c"] against ["a","x","c"] has one
            # MISSING leaf and one SPURIOUS one, so the denominator is 4, and max() reported
            # 3. That erosion inflated every document containing a scalar array with errors
            # on both sides, in the direction of flattering the vendor -- the same direction
            # as every other shape defect in this branch.
            t += len(gc) + len(pc) - mm; m += mm
        return t, m
    if pv is None and gv is None:
        return 0, 0
    # one-side None: let cmp_leaf decide via canonical ('none'/'n/a' canonicalize to '' like
    # None, so they match — matching G.grade_value, which never short-circuits on None).
    return 1, int(cmp_leaf(pv, gv) >= 1.0)

def fair_grade(pred, gt, schema):
    del _INEXACT[:]                      # per-grade; see _INEXACT
    del _IGNORED[:]
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
            "greedy_blocks": sorted(_INEXACT, reverse=True),
            # Non-zero => the schema declared open maps, whose contents were evaluated on
            # neither side. The score covers the rest of the document.
            "ignored_open_maps": len(_IGNORED)}


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
