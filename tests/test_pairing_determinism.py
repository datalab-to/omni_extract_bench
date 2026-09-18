#!/usr/bin/env python3
"""Reordering rows must not change a single reported number.

Row order is the thing this benchmark most insists does not matter: a vendor that emits a
table bottom-to-top has not made a mistake, and `align` exists to renumber its rows onto the
ground truth's. The score has always had that property. The DIAGNOSTICS did not.

The cause is that several assignments can be equally optimal. `matched * scale + shared`
pins the optimal VALUE -- so accuracy, precision, recall and f1 are the same whichever
assignment wins -- but it does not pin WHICH rows got paired, and the false-assertion split
reads exactly that. An unpaired predicted row is an invented item; the same row paired is a
handful of invented fields. Before the fix, 35 of 600 random row-documents reported a
different split when the prediction's rows were shuffled, and 11 when the gold's were.

The fix is to sort both sides by a canonical content key before solving, so the solver is
handed the identical problem however the rows arrived. This file is the reason that cannot
quietly stop working. It checks the property at depth 3, because the key is recursive and a
key that accidentally read positions would still pass at depth 1.

Two things here are deliberately NOT invariant, and are asserted to move:
  - an array the caller declared `order_matters`, where the index IS content
  - nothing else

Run: python3 tests/test_pairing_determinism.py
"""
import copy
import itertools
import os as _os
import random
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench import matching as OM                              # noqa: E402
from omni_extract_bench.metric import (                                     # noqa: E402
    KEY, Row, _content_key, _extract_rows_at, flatten, score)

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


# ── schemas ───────────────────────────────────────────────────────────────────────────
def _obj(**props):
    return {"type": "object", "properties": props}


STR = {"type": "string"}
FLAT = {"type": "object", "required": ["r"], "properties": {
    "r": {"type": "array", "items": _obj(**{k: STR for k in "abcde"})}}}
# depth 3: rows -> items -> parts
DEEP = {"type": "object", "required": ["r"], "properties": {
    "r": {"type": "array", "items": _obj(
        k=STR,
        items={"type": "array", "items": _obj(
            n=STR,
            parts={"type": "array", "items": _obj(x=STR, y=STR)})})}}}


def flat_row(rng):
    return {f: str(rng.randint(0, 2)) for f in rng.sample(list("abcde"), rng.randint(1, 4))}


def deep_row(rng):
    """A row three array levels deep, so the content key has to recurse to be right."""
    return {"k": str(rng.randint(0, 2)),
            "items": [{"n": str(rng.randint(0, 2)),
                       "parts": [{"x": str(rng.randint(0, 2)), "y": str(rng.randint(0, 2))}
                                 for _ in range(rng.randint(1, 2))]}
                      for _ in range(rng.randint(1, 3))]}


def shuffled_everywhere(doc, rng):
    """The same document with every array reordered at every depth."""
    doc = copy.deepcopy(doc)
    rng.shuffle(doc)
    for row in doc:
        items = row.get("items") or []
        rng.shuffle(items)
        for item in items:
            rng.shuffle(item.get("parts") or [])
    return doc


def disagreement(gold, pred, variants, schema, **kw):
    """Fields whose value is not the same for every variant of the prediction."""
    base = score({"r": pred}, {"r": gold}, schema, **kw)
    moved = set()
    for variant in variants:
        r = score({"r": variant}, {"r": gold}, schema, **kw)
        moved |= {k for k in base if r[k] != base[k]}
    return moved, base


# ── 1. flat rows, every ordering of both sides ────────────────────────────────────────
print("EVERY REPORTED FIELD SURVIVES REORDERING THE ROWS")
rng = random.Random(17)
moved_pred, moved_gold = set(), set()
for _ in range(120):
    gold = [flat_row(rng) for _ in range(rng.randint(2, 4))]
    pred = [flat_row(rng) for _ in range(rng.randint(2, 4))]
    m, _b = disagreement(gold, pred,
                         [[pred[i] for i in p]
                          for p in itertools.permutations(range(len(pred)))], FLAT)
    moved_pred |= m
    base = score({"r": pred}, {"r": gold}, FLAT)
    for perm in itertools.permutations(range(len(gold))):
        r = score({"r": pred}, {"r": [gold[i] for i in perm]}, FLAT)
        moved_gold |= {k for k in base if r[k] != base[k]}
report("permuting the PREDICTION's rows moves nothing", not moved_pred, str(sorted(moved_pred)))
report("permuting the GOLD's rows moves nothing", not moved_gold, str(sorted(moved_gold)))
note("before the content-key sort: 35/600 and 11/600 documents moved fabricated/invented_item")

# ── 2. depth 3 ────────────────────────────────────────────────────────────────────────
print("\nAND SURVIVES IT AT EVERY DEPTH, NOT JUST THE TOP ONE")
rng = random.Random(23)
moved_deep = set()
for _ in range(60):
    gold = [deep_row(rng) for _ in range(rng.randint(2, 3))]
    pred = [deep_row(rng) for _ in range(rng.randint(2, 3))]
    variants = [shuffled_everywhere(pred, rng) for _ in range(8)]
    m, _b = disagreement(gold, pred, variants, DEEP)
    moved_deep |= m
report("rows, their nested items, and the parts inside those -- all reorderable",
       not moved_deep, str(sorted(moved_deep)))
note("the key is recursive; one that read positions would still have passed at depth 1")

# ── 3. the greedy fallback takes the same path ────────────────────────────────────────
print("\nINCLUDING WHEN THE ARRAY IS TOO BIG TO SOLVE EXACTLY")
rng = random.Random(29)
restore = OM.force_approximate()       # the repo's own hook: shrink the exactness budget
try:
    moved_greedy = set()
    for _ in range(60):
        gold = [flat_row(rng) for _ in range(rng.randint(2, 4))]
        pred = [flat_row(rng) for _ in range(rng.randint(2, 4))]
        m, _b = disagreement(gold, pred,
                             [[pred[i] for i in p]
                              for p in itertools.permutations(range(len(pred)))], FLAT)
        moved_greedy |= m
finally:
    restore()
report("the greedy path is order-independent too", not moved_greedy, str(sorted(moved_greedy)))
note("greedy breaks its own ties on position, which is canonical once the rows are sorted")

# ── 4. the negative case: ordered arrays MUST move ────────────────────────────────────
print("\nBUT AN ORDERED ARRAY IS SUPPOSED TO MOVE -- THERE THE INDEX IS THE ADDRESS")
rng = random.Random(31)
ordered_moved = 0
for _ in range(60):
    gold = [deep_row(rng) for _ in range(2)]
    pred = [deep_row(rng) for _ in range(2)]
    variants = [shuffled_everywhere(pred, rng) for _ in range(6)]
    m, _b = disagreement(gold, pred, variants, DEEP, order_matters=["r[*].items"])
    ordered_moved += bool(m)
report("declaring the inner array ordered makes reordering it a real change",
       ordered_moved > 0, f"{ordered_moved}/60 documents moved")
note("a fix that froze these too would be a bug, not a stronger guarantee")

# ── 5. the key itself ─────────────────────────────────────────────────────────────────
print("\nTHE CONTENT KEY IS READ OFF STRUCTURE, NOT OFF A RENDERED ADDRESS")
one = _extract_rows_at(flatten({"r": [{"a.b": 1}]}), ((KEY, "r"),), frozenset())[0]
two = _extract_rows_at(flatten({"r": [{"a": {"b": 1}}]}), ((KEY, "r"),), frozenset())[0]
report("a field named 'a.b' and a nested a->b do not collide",
       one.key != two.key, f"{one.key} vs {two.key}")
note("`show` renders both as 'a.b', so keying on the rendering handed the tie back to order")

rows = _extract_rows_at(flatten({"r": [{"z": 1, "a": 2}, {"a": 2, "z": 1}]}),
                        ((KEY, "r"),), frozenset())
report("two rows holding the same values key alike however they were written",
       rows[0].key == rows[1].key)
report("and a row's key does not depend on how its nested array is numbered",
       _extract_rows_at(flatten({"r": [{"q": [{"n": 1}, {"n": 2}]}]}),
                        ((KEY, "r"),), frozenset())[0].key
       == _extract_rows_at(flatten({"r": [{"q": [{"n": 2}, {"n": 1}]}]}),
                           ((KEY, "r"),), frozenset())[0].key)
report("every Row is built with its key; the empty default never reaches the sort",
       Row.at(flatten({"a": 1}), (), frozenset()).key != ()
       and _content_key({}, {}) == ((), ()))

# ── 6. the case this was found on ─────────────────────────────────────────────────────
print("\nTHE DOCUMENT THAT EXPOSED IT, PINNED")
gold = [{"c": 2, "a": 0, "d": 1, "e": 1}, {"b": 1}]
pred = [{"e": 1, "a": 0, "c": 1, "b": 1}, {"b": 0, "e": 2, "d": 1}]
a = score({"r": pred}, {"r": gold}, FLAT)
b = score({"r": pred[::-1]}, {"r": gold}, FLAT)
report("two pairings weigh the same; the reported split no longer depends on which",
       (a["fabricated"], a["invented_item"]) == (b["fabricated"], b["invented_item"]),
       f"{a['fabricated']}/{a['invented_item']} vs {b['fabricated']}/{b['invented_item']}")
report("...and the accuracy was never the thing that moved",
       a["accuracy"] == b["accuracy"] == 0.2222222222222222)

print(f"\n{'PAIRING IS ORDER-INDEPENDENT' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
