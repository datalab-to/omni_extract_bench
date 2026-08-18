#!/usr/bin/env python3
"""Deep structural audit of the metric — hunts the CLASS of bug that produced the 349-row
corruption, rather than the single instance.

That bug was a structural asymmetry: identical content scored differently depending on where
it sat in the document. The generalisation, and the strongest property this metric can have:

    WRAPPING A DOCUMENT IN AN EXTRA LEVEL MUST NOT CHANGE ITS SCORE.

If that holds for every shape, no depth-dependent scoring path can exist. This file generates
adversarial shapes — scalar arrays, arrays of arrays, empty containers, mixed-type arrays,
over-extraction, type mismatches — and checks the invariant on each, plus targeted checks for
gaps the wrapping test alone would not reveal.

Run: python3 tests/test_metric_structural_audit.py
"""
import copy, json, random, sys

from omni_extract_bench import grading as FG

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if not cond and detail else ""))
    if not cond:
        FAILS.append(name)
    return cond


def score(pred, gt, schema=None):
    return FG.fair_grade(pred, gt, schema or {})["leaf_accuracy"]


def denom(pred, gt, schema=None):
    return FG.fair_grade(pred, gt, schema or {})["leaf_total"]


# ── shape zoo: every structural form the benchmark's documents actually contain ──
SHAPES = {
    "scalar":              ({"a": 1},                      {"a": 2}),
    "string":              ({"a": "x"},                    {"a": "y"}),
    "object":              ({"o": {"a": 1, "b": 2}},       {"o": {"a": 1, "b": 3}}),
    "array_of_objects":    ({"r": [{"i": 1}, {"i": 2}]},   {"r": [{"i": 1}, {"i": 9}]}),
    "array_of_scalars":    ({"t": ["a", "b"]},             {"t": ["a", "c"]}),
    "array_partial":       ({"r": [{"i": 1}]},             {"r": [{"i": 1}, {"i": 2}, {"i": 3}]}),
    "array_extra":         ({"r": [{"i": 1}, {"i": 2}]},   {"r": [{"i": 1}]}),
    "empty_array_pred":    ({"r": []},                     {"r": [{"i": 1}]}),
    "empty_array_gt":      ({"r": [{"i": 1}]},             {"r": []}),
    "array_of_arrays":     ({"m": [[1, 2], [3, 4]]},       {"m": [[1, 2], [3, 9]]}),
    "mixed_array":         ({"x": [{"i": 1}, "loose"]},    {"x": [{"i": 1}, "loose2"]}),
    "nested_array_in_obj": ({"o": {"r": [{"i": 1}]}},      {"o": {"r": [{"i": 1}, {"i": 2}]}}),
    "deep_nesting":        ({"a": {"b": {"c": [{"d": 1}]}}}, {"a": {"b": {"c": [{"d": 2}]}}}),
    "null_vs_value":       ({"a": None},                   {"a": 5}),
    "missing_key":         ({},                            {"a": 5}),
    "extra_key":           ({"a": 5, "z": 9},              {"a": 5}),
    "type_mismatch":       ({"a": [{"i": 1}]},             {"a": {"i": 1}}),
    "large_partial":       ({"r": [{"i": n} for n in range(44)]},
                            {"r": [{"i": n} for n in range(349)]}),
}

print("\n[1] WRAPPING INVARIANCE — depth must never change the score")
for name, (p, g) in SHAPES.items():
    flat = FG.fair_grade(p, g, {})
    wrapped = FG.fair_grade({"w": p}, {"w": g}, {})
    same_score = abs(flat["leaf_accuracy"] - wrapped["leaf_accuracy"]) < 0.05
    same_denom = flat["leaf_total"] == wrapped["leaf_total"]
    check(f"wrap-invariant: {name}", same_score and same_denom,
          f"flat {flat['leaf_accuracy']:.1f}/{flat['leaf_total']} vs "
          f"wrapped {wrapped['leaf_accuracy']:.1f}/{wrapped['leaf_total']}")

print("\n[2] SCHEMA-DECLARED ARRAY must score identically to an undeclared one")
# The top-level path is selected by the SCHEMA (_arrays_plus), so a schema that declares an
# array and one that doesn't must not produce different numbers for the same data.
for name, (p, g) in SHAPES.items():
    key = next(iter(g)) if g else None
    if key is None or not isinstance(g.get(key), list):
        continue
    declared = {"properties": {key: {"type": "array"}}}
    a = FG.fair_grade(p, g, declared)
    b = FG.fair_grade(p, g, {})
    check(f"schema-declaration neutral: {name}",
          abs(a["leaf_accuracy"] - b["leaf_accuracy"]) < 0.05 and a["leaf_total"] == b["leaf_total"],
          f"declared {a['leaf_accuracy']:.1f}/{a['leaf_total']} vs bare {b['leaf_accuracy']:.1f}/{b['leaf_total']}")

print("\n[3] OMISSION AND OVER-EXTRACTION ARE BOTH CHARGED, AT EVERY DEPTH")
gold = {"r": [{"i": n, "v": n} for n in range(10)]}
for depth, wrap in ((0, lambda d: d), (1, lambda d: {"w": d}), (2, lambda d: {"w": {"x": d}})):
    half = wrap({"r": gold["r"][:5]})
    dbl = wrap({"r": gold["r"] + [{"i": 99, "v": 99}] * 5})
    G_ = wrap(gold)
    s_half, s_dbl = score(half, G_), score(dbl, G_)
    check(f"omission charged at depth {depth}", s_half < 99.0, f"{s_half:.1f}")
    check(f"over-extraction charged at depth {depth}", s_dbl < 99.0, f"{s_dbl:.1f}")

print("\n[4] EMPTY / NULL CONTAINERS ARE NOT SILENTLY FREE")
check("empty pred array vs 3 gold rows is not 100",
      score({"r": []}, {"r": [{"i": 1}, {"i": 2}, {"i": 3}]}) < 99.0)
check("empty pred array vs gold rows has non-zero denominator",
      denom({"r": []}, {"r": [{"i": 1}, {"i": 2}]}) > 0)
check("missing key entirely is not 100",
      score({}, {"r": [{"i": 1}, {"i": 2}]}) < 99.0)
check("null pred vs gold rows is not 100",
      score({"r": None}, {"r": [{"i": 1}, {"i": 2}]}) < 99.0)

print("\n[5] MONOTONE IN COVERAGE — more correct rows never scores worse")
gold_rows = [{"i": n, "v": n * 3} for n in range(30)]
prev = -1.0
mono = True
for k in range(0, 31, 5):
    s = score({"r": gold_rows[:k]}, {"r": gold_rows})
    if s < prev - 0.05:
        mono = False
        check("coverage monotonicity", False, f"{k} rows scored {s:.1f} < previous {prev:.1f}")
        break
    prev = s
if mono:
    check("coverage monotonicity: score rises with rows returned", True)
check("full coverage scores 100", abs(score({"r": gold_rows}, {"r": gold_rows}) - 100.0) < 1e-9)

print("\n[6] RANDOMISED WRAPPING FUZZ — arbitrary generated documents")
WORDS = ["a", "bb", "ccc", "2024-01-01", "12.5"]


def rnd_val(rnd, d=0):
    r = rnd.random()
    if d < 2 and r < 0.25:
        return {f"k{i}": rnd_val(rnd, d + 1) for i in range(rnd.randint(1, 3))}
    if d < 2 and r < 0.5:
        if rnd.random() < 0.5:
            return [{f"f{i}": rnd_val(rnd, d + 1) for i in range(rnd.randint(1, 3))}
                    for _ in range(rnd.randint(0, 4))]
        return [rnd.choice(WORDS) for _ in range(rnd.randint(0, 4))]
    return rnd.choice(WORDS + [rnd.randint(0, 99), None, True])


bad = 0
for seed in range(300):
    rnd = random.Random(seed)
    g = {f"t{i}": rnd_val(rnd) for i in range(rnd.randint(1, 4))}
    p = copy.deepcopy(g)
    # perturb so scores are not trivially 100
    ks = list(p)
    if ks:
        p[rnd.choice(ks)] = rnd_val(rnd)
    flat = FG.fair_grade(p, g, {})
    wrapped = FG.fair_grade({"W": p}, {"W": g}, {})
    if abs(flat["leaf_accuracy"] - wrapped["leaf_accuracy"]) > 0.05 or \
       flat["leaf_total"] != wrapped["leaf_total"]:
        bad += 1
        if bad == 1:
            check("fuzz wrap-invariance", False,
                  f"seed {seed}: flat {flat['leaf_accuracy']:.1f}/{flat['leaf_total']} vs "
                  f"wrapped {wrapped['leaf_accuracy']:.1f}/{wrapped['leaf_total']}")
if not bad:
    check("fuzz wrap-invariance: 300 generated documents", True)

print(f"\n{'STRUCTURAL AUDIT CLEAN' if not FAILS else 'DEFECTS FOUND: ' + '; '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
