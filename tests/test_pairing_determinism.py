#!/usr/bin/env python3
"""Reordering rows must not change any reported number, the error split included.

Several pairings can be equally good. They give the same accuracy, but not always the same split of false
assertions into invented items and invented fields, so both sides are sorted by a content key before pairing.
This checks it at depth 3, because the key is recursive. The one thing allowed to move is an array the caller
declared `order_matters`.

Run: python3 tests/test_pairing_determinism.py
"""
import copy
import itertools
import os as _os
import random
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from tests.approximate import approximate                  # noqa: E402
from omni_extract_bench.scorers.field_matching.documents import KEY  # noqa: E402
from omni_extract_bench.scorers.field_matching.compare import score  # noqa: E402
from omni_extract_bench.scorers.field_matching.pairing import Row, _content_key, _extract_rows_at  # noqa: E402
from omni_extract_bench.scorers.field_matching.documents import flatten  # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


def _obj(**props):
    return {"type": "object", "properties": props}


STR = {"type": "string"}
FLAT = {"type": "object", "required": ["r"], "properties": {
    "r": {"type": "array", "items": _obj(**{k: STR for k in "abcde"})}}}
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


print("Every reported field survives reordering the rows")
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

print("\nAnd survives it at every depth, not just the top one")
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

print("\nIncluding when the array is too big to solve exactly")
rng = random.Random(29)
with approximate():
    moved_greedy = set()
    for _ in range(60):
        gold = [flat_row(rng) for _ in range(rng.randint(2, 4))]
        pred = [flat_row(rng) for _ in range(rng.randint(2, 4))]
        m, _b = disagreement(gold, pred,
                             [[pred[i] for i in p]
                              for p in itertools.permutations(range(len(pred)))], FLAT)
        moved_greedy |= m
report("the greedy path is order-independent too", not moved_greedy, str(sorted(moved_greedy)))
note("greedy breaks its own ties on position, which is canonical once the rows are sorted")

print("\nBut an ordered array is supposed to move -- there the index is the address")
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

print("\nThe content key is read off structure, not off a rendered address")
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

print("\nThe document that exposed it, pinned")
gold = [{"c": 2, "a": 0, "d": 1, "e": 1}, {"b": 1}]
pred = [{"e": 1, "a": 0, "c": 1, "b": 1}, {"b": 0, "e": 2, "d": 1}]
a = score({"r": pred}, {"r": gold}, FLAT)
b = score({"r": pred[::-1]}, {"r": gold}, FLAT)
report("two pairings weigh the same; the reported split no longer depends on which",
       (a["fabricated"], a["invented_item"]) == (b["fabricated"], b["invented_item"]),
       f"{a['fabricated']}/{a['invented_item']} vs {b['fabricated']}/{b['invented_item']}")
report("...and the accuracy was never the thing that moved",
       a["accuracy"] == b["accuracy"] == 0.2222222222222222)

print(f"\n{'Pairing is order-independent' if not FAILS else 'Failures:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
