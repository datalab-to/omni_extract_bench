#!/usr/bin/env python3
"""Airtightness audit for fair_grading — asserts the invariants a consistent/fair grader must
hold, INDEPENDENT of subset. Run: python audit_grader_invariants.py"""
import sys, json
# run from anywhere: `python tests/x.py` puts tests/ on the path, not the repo root
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench import grading as FG

fails = []
def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)

SCH = {"type": "object", "properties": {
    "name": {"type": "string"}, "amt": {"type": "number"}, "n": {"type": "integer"},
    "flag": {"type": "boolean"}, "day": {"type": "string"},
    "rows": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "v": {"type": "number"}}}}}}

def g(pred, gt, sch=SCH):
    return FG.fair_grade(pred, gt, sch)

# 1. IDENTITY: a doc scored against itself = 100% (any values)
d = {"name": "Acme US", "amt": 3.14159, "n": 5, "flag": False, "day": "10/31/2024",
     "rows": [{"id": "a", "v": 1.0}, {"id": "b", "v": 2.5}]}
check("identity: doc vs itself = 100", abs(g(d, d)["leaf_accuracy"] - 100.0) < 1e-9)

# 2. DETERMINISM: same inputs -> identical output across repeated calls
check("determinism: repeated grade identical", g(d, d) == g(d, d) == g(d, d))

# 3. SUBSET-INDEPENDENCE: identical (pred,gt,schema) scores the same no matter the "subset"
#    (fair_grade takes no subset arg -> structurally guaranteed; assert value stable)
p2 = {"name": "Acme US", "amt": 3.14, "n": 5, "flag": False, "day": "2024-10-31",
      "rows": [{"id": "a", "v": 1.0}]}
r1 = g(p2, d); r2 = g(p2, d)
check("subset-independence: no hidden state", r1 == r2)

# 4. EMPTY / NULL handling
check("empty pred vs empty gt = 100 (0 leaves)", g({}, {})["leaf_accuracy"] == 0.0 or g({}, {})["leaf_total"] == 0)
check("all-None gt vs all-None pred = 0 leaves",
      g({"name": None, "amt": None}, {"name": None, "amt": None})["leaf_total"] == 0)

# 5. MISSING is a miss, not skipped (denominator counts it)
r = g({"name": "x"}, {"name": "x", "amt": 5.0})
check("missing field counts in denominator", r["leaf_total"] == 2 and r["leaf_match"] == 1)

# 6. FAIRNESS EQUIVALENCES (uniform, all inferred)
check("date format equivalent", g({"day": "10/31/2024"}, {"day": "2024-10-31"}, SCH)["leaf_match"] == 1)
check("float precision equivalent", g({"amt": 33.33333333}, {"amt": 33.3333333}, SCH)["leaf_match"] == 1)

# 7. TRAPS (must NOT match)
check("distinct floats stay wrong", g({"amt": 100.5}, {"amt": 100.6}, SCH)["leaf_match"] == 0)
check("id-like integers exact", g({"n": 8303911426}, {"n": 8303511426}, SCH)["leaf_match"] == 0)

# 8. ARRAY row coverage.
# THIS INVARIANT PREVIOUSLY ASSERTED A BUG. It required leaf_accuracy == 100.0 when only a
# third of the gold rows were returned, i.e. it codified "omission is free at the top level"
# as intended behaviour and would have blocked the fix. Returning 1 of 3 rows is not a
# perfect answer; recall reports coverage, but the headline metric must not ignore it.
few = {"rows": [{"id": "a", "v": 1.0}]}
many = {"rows": [{"id": "a", "v": 1.0}, {"id": "b", "v": 2.0}, {"id": "c", "v": 3.0}]}
rr = g(few, many, SCH)
check("omission lowers leaf_accuracy AND is visible in recall",
      rr["leaf_accuracy"] < 99.0 and abs(rr["recall"] - 1/3) < 1e-9,
      f"leaf={rr['leaf_accuracy']:.1f} recall={rr['recall']:.2f}")
check("returning a third of the rows scores about a third",
      30.0 <= rr["leaf_accuracy"] <= 40.0, f"leaf={rr['leaf_accuracy']:.1f}")

# 9. OVER-EXTRACTION penalized via precision + extra-row leaves in nested, but top-level extra
#    rows -> precision (leaf_accuracy stays high on matched content)
rr2 = g(many, few, SCH)
check("extra rows hit precision", abs(rr2["precision"] - 1/3) < 1e-9)

# 10. SYMMETRY of leaf count: swapping pred/gt gives same leaf_total for scalar dicts
a = {"name": "x", "amt": 5.0}; b = {"name": "y", "amt": 6.0}
check("scalar-dict leaf_total symmetric", g(a, b)["leaf_total"] == g(b, a)["leaf_total"])

# 11. ONE comparison rule: scoring and row pairing must use the SAME function. Two
#     implementations that had to agree diverged twice; now a disagreement is unrepresentable.
import inspect as _inspect
check("scoring and pairing share one comparator",
      "canon_key(" in _inspect.getsource(FG.cmp_leaf)
      and "canon_key(" in _inspect.getsource(FG._row_signature))
check("no per-field metric modes remain", not hasattr(FG, "HONOR_EVAL_CONFIG"))

# 12. FAST PATHS ARE EXACT. Two optimisations replaced O(n*m) loops with O(n+m) ones. Each is
#     fuzzed here against the loop it replaced, through the real code, because "two
#     implementations of one rule" is exactly how this grader has drifted before. A mismatch
#     is a scoring change and must fail loudly.
import random as _random
from omni_extract_bench import matching as _OM

def _dense_pairs(pred, gt):
    ps = [FG._row_signature(r) for r in pred]; gs = [FG._row_signature(r) for r in gt]
    out = []
    for i, a in enumerate(ps):
        for j, b in enumerate(gs):
            lo, hi = (a, b) if len(a) <= len(b) else (b, a)
            w = sum(1 for k, v in lo.items() if hi.get(k) == v)
            if w > 0: out.append((-w, i, j))
    return sorted(out)

_random.seed(1)
_mm = 0
for _ in range(300):
    n, m = _random.randint(1, 30), _random.randint(1, 30); nc = _random.randint(1, 4)
    voc = _random.choice([2, 3, 40])
    def _row():
        r = {f"c{c}": (_random.choice(["k", "same"]) if voc == 2 else str(_random.randrange(voc)))
             for c in range(nc) if _random.random() < 0.85}
        if _random.random() < 0.1: r["const"] = "X"
        return r
    pred = [_row() for _ in range(n)]; gt = [_row() for _ in range(m)]
    sp, shed = _OM._candidate_pairs(pred, gt, FG._row_signature)
    if shed == 0 and sorted(sp) != _dense_pairs(pred, gt): _mm += 1
check("sparse row-candidate generation == dense loop (300 fuzz)", _mm == 0, f"{_mm} mismatches")

def _scan_and_pop(pv, gv):
    rem = list(gv); mm = 0
    for x in pv:
        for i, y in enumerate(rem):
            if FG.cmp_leaf(x, y) >= 1.0: mm += 1; rem.pop(i); break
    return max(len(pv), len(gv)), mm

_random.seed(2); _mm2 = 0
_voc = ["a", "A ", "b", "5", "5.0", "n/a", None, "", "x-y", "2024-01-15", "01/15/2024"]
for _ in range(1500):
    pv = [_random.choice(_voc) for _ in range(_random.randint(0, 9))]
    gv = [_random.choice(_voc) for _ in range(_random.randint(0, 9))]
    if FG.fair_grade_value(pv, gv, {"type": "array", "items": {"type": "string"}}) != _scan_and_pop(pv, gv): _mm2 += 1
check("scalar-array Counter multiset == scan-and-pop (1500 fuzz)", _mm2 == 0, f"{_mm2} mismatches")

# 14. THE STRING FOLD: edge punctuation, quotes, whitespace and footnote markers are format;
#     punctuation BETWEEN characters is content. The previous fold deleted every internal
#     period, slash, hyphen and space and merged the pairs marked 0 below.
_fold = [("Acme Inc.", "Acme Inc", 1), ("ABN AMRO Bank N.V.", "ABN AMRO BANK N.V.,", 1),
         ('"quoted"', "quoted", 1), ("[1] Y. Bengio", "Y. Bengio", 1), ("Table  B-1.", "Table B-1", 1),
         ("1/2", "12", 0), ("Section 2.1", "Section 21", 0), ("v1.2", "v12", 0), ("1.5M", "15M", 0),
         ("Inst itutional", "Institutional", 0)]
for _a, _b, _want in _fold:
    check(f"fold: {_a!r} vs {_b!r} -> {'equal' if _want else 'distinct'}", int(FG.cmp_leaf(_a, _b)) == _want)

print(f"\n{'ALL INVARIANTS HOLD' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
