#!/usr/bin/env python3
"""The vectorised pairing weights, against the scalar function they replace.

`_worth_if_paired` counts two intersections per row pair, and `_best_pairing` calls it once
per cell of an n*m matrix -- 11.8 million times on the corpus's largest exactly-solved block,
which is 74% of that document's scoring time. `_pair_weights` computes the same matrix with
two sparse products instead.

"The same matrix" is the whole claim, so this file tries to break it:

  * every cell, on generated documents, against the scalar `cost` the scorer uses
  * NESTED rows especially -- a row carrying its own array is worth what its sub-pairings are
    worth, which is a recursive solve and not an intersection. Those pairs are not vectorised
    at all, and this checks that the fallback really is taken
  * `shared`, the tie-break term, which the rest of the suite asserts in exactly one place
  * grades are identical with the vectorised path forced on and forced off
  * a value that cannot be hashed declines the path instead of raising

Run: python3 tests/test_pair_weights.py
"""
import os as _os
import random
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
import omni_extract_bench.metric as sc                                       # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


NESTED = {"type": "object", "properties": {"rows": {"type": "array", "items": {
    "type": "object", "properties": {
        "tag": {"type": "string"},
        "n": {"type": "number"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "k": {"type": "string"},
            "vals": {"type": "array", "items": {"type": "number"}},
            "deep": {"type": "array", "items": {"type": "object", "properties": {
                "z": {"type": "string"}}}}}}}}}}}}


def doc(rnd):
    return {"rows": [{
        "tag": rnd.choice(["a", "b", "c", "SAME"]),
        "n": rnd.choice([1, 2, 2.0, "2", None]),
        "items": [{
            "k": rnd.choice(["x", "y", "SAME"]),
            "vals": [rnd.choice([1.5, 2.5, 3.5]) for _ in range(rnd.randint(0, 3))],
            "deep": [{"z": rnd.choice(["p", "q", "SAME"])} for _ in range(rnd.randint(0, 2))],
        } for _ in range(rnd.randint(0, 3))],
    } for _ in range(rnd.randint(1, 5))]}


def shuffled(node, rnd):
    if isinstance(node, list):
        out = [shuffled(x, rnd) for x in node]
        rnd.shuffle(out)
        return out
    if isinstance(node, dict):
        return {k: shuffled(v, rnd) for k, v in node.items()}
    return node


def compare_every_cell(seeds, force_small=True):
    """Grade generated documents, checking each block's matrix cell by cell as it is built."""
    saved = sc.MIN_VECTOR_CELLS
    if force_small:
        sc.MIN_VECTOR_CELLS = 1        # tiny blocks too; the default skips them as not worth it
    seen = {"blocks": 0, "deep": 0, "cells": 0, "bad": 0, "inexact": 0, "shared_used": 0}
    real = sc._best_pairing

    def spy(pred, gold, scale, inexact=None):
        if pred and gold:
            pi = sorted(pred, key=lambda i: pred[i].key)
            gi = sorted(gold, key=lambda j: gold[j].key)
            mine, theirs = [], []
            w = sc._pair_weights(pred, gold, pi, gi, scale, mine)
            if w is not None:
                seen["blocks"] += 1
                if any(pred[k].arrays for k in pi) or any(gold[k].arrays for k in gi):
                    seen["deep"] += 1
                for i, pk in enumerate(pi):
                    for j, gk in enumerate(gi):
                        mt, sh = sc._worth_if_paired(pred[pk], gold[gk], scale, theirs)
                        want = mt * scale + sh if mt else 0
                        seen["cells"] += 1
                        seen["shared_used"] += bool(mt and sh)
                        seen["bad"] += float(w[i, j]) != float(want)
                seen["inexact"] += sorted(set(mine)) != sorted(set(theirs))
        return real(pred, gold, scale, inexact)

    sc._best_pairing = spy
    try:
        for seed in seeds:
            rnd = random.Random(seed)
            gold = doc(rnd)
            pred = shuffled(gold, rnd) if seed % 3 else doc(rnd)
            sc.score(pred, gold, NESTED)
    finally:
        sc._best_pairing = real
        sc.MIN_VECTOR_CELLS = saved
    return seen


print("\nEVERY CELL AGREES WITH THE FUNCTION IT REPLACES")
seen = compare_every_cell(range(120))
report("no cell disagrees with the scalar cost, over generated 3-deep documents",
       seen["bad"] == 0, f"{seen['bad']} of {seen['cells']:,} cells")
report("and enough cells were actually checked for that to mean something",
       seen["cells"] > 20_000, f"{seen['cells']:,} cells in {seen['blocks']:,} blocks")
report("including blocks whose rows carry their own arrays",
       seen["deep"] > 200, f"{seen['deep']:,} nested blocks")
note("a nested row is worth what its sub-pairings are worth; that is a solve, not an "
     "intersection, so those pairs are never read off the product")
report("the tie-break term is exercised, not just the matched term",
       seen["shared_used"] > 1000, f"{seen['shared_used']:,} cells where shared mattered")
note("`shared` is asserted exactly once in the rest of the suite; a weight that dropped it "
     "would still pair correctly most of the time and silently lose tie-breaks")
report("the recursion's `inexact` list is identical either way", seen["inexact"] == 0)

print("\nNESTED ROWS TAKE THE FALLBACK, NOT THE PRODUCT")
rnd = random.Random(1)
gold = {"rows": [{"tag": "a", "items": [{"k": "x", "vals": [1.0]}]},
                 {"tag": "b", "items": [{"k": "y", "vals": [2.0]}]}]}
pred = {"rows": [{"tag": "b", "items": [{"k": "y", "vals": [2.0]}]},
                 {"tag": "a", "items": [{"k": "x", "vals": [1.0]}]}]}
calls = {"n": 0}
real_worth = sc._worth_if_paired
def counted(*a, **k):
    calls["n"] += 1
    return real_worth(*a, **k)
sc._worth_if_paired = counted
saved = sc.MIN_VECTOR_CELLS
sc.MIN_VECTOR_CELLS = 1
try:
    r = sc.score(pred, gold, NESTED)
finally:
    sc._worth_if_paired = real_worth
    sc.MIN_VECTOR_CELLS = saved
report("a document whose every row is nested still scores 1.0 after reordering",
       r["accuracy"] == 1.0, str(r["accuracy"]))
report("and the scalar function was still called for those pairs",
       calls["n"] > 0, f"{calls['n']} calls")

print("\nNESTING ONE SIDE NEVER HAD IS NOT NESTING")
# `_worth_if_paired` walks `set(pred.arrays) | set(gold.arrays)` and descends into
# `_best_pairing`, which returns immediately when either side is empty. So a predicted row
# nesting something the gold never nested adds nothing, and the product already has the
# answer. Asking only "does this row have arrays?" would send those pairs down the slow path
# to add zero -- which on the corpus's three largest arrays is 95%, 99.98% and 76% of
# 704, 379 and 342 million pairs respectively.
addr_tag = (("k", "tag"),)
inner = sc.Row(named={(("k", "z"),): "v"}, arrays={}, key=())
gold_rows = {i: sc.Row(named={addr_tag: f"g{i}"}, arrays={}, key=(f"g{i}",))
             for i in range(6)}
pred_rows = {i: sc.Row(named={addr_tag: f"g{i}"},
                       arrays={(("k", "extra"),): {0: inner}}, key=(f"g{i}",))
             for i in range(6)}
calls = {"n": 0}
real_worth = sc._worth_if_paired
def counted(*a, **k):
    calls["n"] += 1
    return real_worth(*a, **k)
saved = sc.MIN_VECTOR_CELLS
sc.MIN_VECTOR_CELLS = 1
sc._worth_if_paired = counted
try:
    w = sc._pair_weights(pred_rows, gold_rows, list(range(6)), list(range(6)), 97, None)
finally:
    sc._worth_if_paired = real_worth
    sc.MIN_VECTOR_CELLS = saved
report("a block whose predictions nest and whose gold does not is fully vectorised",
       w is not None and calls["n"] == 0, f"{calls['n']} fallback calls")
ok = True
for i in range(6):
    for j in range(6):
        mt, sh = real_worth(pred_rows[i], gold_rows[j], 97)
        ok &= float(w[i, j]) == float(mt * 97 + sh if mt else 0)
report("and every one of its weights is still exactly right", ok)

print("\nTHE SAME GRADE, VECTORISED OR NOT")
same = differing = 0
for seed in range(160):
    rnd = random.Random(seed + 9000)
    gold = doc(rnd)
    pred = shuffled(gold, rnd) if seed % 2 else doc(rnd)
    sc.MIN_VECTOR_CELLS, sc.MAX_VECTOR_CELLS = 1, 64 * 10**6
    a = sc.score(pred, gold, NESTED)
    sc.MAX_VECTOR_CELLS = 0                      # forces the scalar loop everywhere
    b = sc.score(pred, gold, NESTED)
    sc.MIN_VECTOR_CELLS, sc.MAX_VECTOR_CELLS = 16 * 16, 64 * 10**6
    same += a == b
    differing += a != b
report("160 generated documents score identically on every key", differing == 0,
       f"{differing} differ")
note("not just accuracy: matched, total, matching_exact, approximated, all of them")

print("\nIT DECLINES WHAT IT CANNOT DO, RATHER THAN RAISING")
report("a block smaller than the threshold is not vectorised",
       sc._pair_weights({}, {}, [], [], 3, None) is None)
# Past the DENSE ceiling the matrix cannot be held, and the sparse form is offered instead --
# but only when no pair needs the recursion, because corrections write individual cells.
import scipy.sparse as _sp                                                  # noqa: E402

flat_p = {i: sc.Row(named={(("k", "t"),): f"v{i%3}"}, arrays={}, key=(i,)) for i in range(6)}
flat_g = {i: sc.Row(named={(("k", "t"),): f"v{i%3}"}, arrays={}, key=(i,)) for i in range(6)}
nest_p = {i: sc.Row(named={(("k", "t"),): f"v{i%3}"},
                    arrays={(("k", "x"),): {0: sc.Row(named={(("k", "z"),): "1"},
                                                      arrays={}, key=())}}, key=(i,))
          for i in range(6)}
saved_max, saved_min = sc.MAX_VECTOR_CELLS, sc.MIN_VECTOR_CELLS
sc.MAX_VECTOR_CELLS, sc.MIN_VECTOR_CELLS = 0, 1
try:
    got = sc._pair_weights(flat_p, flat_g, list(range(6)), list(range(6)), 97, None)
    report("past the dense ceiling a flat block comes back SPARSE, not None",
           got is not None and _sp.issparse(got), repr(type(got)))
    both = sc._pair_weights(nest_p, nest_p, list(range(6)), list(range(6)), 97, None)
    report("but a block where both sides nest at one path declines instead",
           both is None, repr(type(both)))
    note("a correction writes single cells, which a sparse matrix cannot absorb cheaply and "
         "may need cells the product does not have; declining sends the block where it went "
         "before")
finally:
    sc.MAX_VECTOR_CELLS, sc.MIN_VECTOR_CELLS = saved_max, saved_min

# A vocabulary needs hashable features. Leaves are scalars today, so this is a guard against a
# future where one is not -- and the point is that it declines rather than taking the whole
# run down with a TypeError from inside a matrix build.
addr = (("k", "a"),)
plain = sc.Row(named={addr: "x"}, arrays={}, key=())
odd = sc.Row(named={addr: ["not", "hashable"]}, arrays={}, key=())
saved_min = sc.MIN_VECTOR_CELLS
sc.MIN_VECTOR_CELLS = 1
try:
    rows = {i: (odd if i == 0 else plain) for i in range(4)}
    got = sc._pair_weights(rows, rows, [0, 1, 2, 3], [0, 1, 2, 3], 7, None)
    report("an unhashable value declines the path instead of raising", got is None, repr(got))
except TypeError as exc:
    report("an unhashable value declines the path instead of raising", False,
           f"raised {type(exc).__name__}: {exc}")
finally:
    sc.MIN_VECTOR_CELLS = saved_min
note("returning None is how it declines; the caller then prices pairs one at a time")

print(f"\n{'THE WEIGHTS HOLD' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
