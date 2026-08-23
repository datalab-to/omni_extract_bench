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

print(f"\n{'ALL INVARIANTS HOLD' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
