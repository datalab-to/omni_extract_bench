#!/usr/bin/env python3
"""P21, on documents that have arrays.

The exhaustive search in §8 runs over a four-field document with no arrays. A document with
no arrays has a fixed denominator, so `accuracy` there is proportional to the count of correct
values, and the search therefore cannot see the one attack the property is about: emitting
more. Every check here uses an array, so the denominator moves.

P21 has two clauses, and they are monotonicity in two different arguments:

    hold what you emit fixed, make more of it right    the score cannot fall
    hold what is right fixed, emit more                the score cannot rise

Neither implies the other. The first is about editing an assertion, the second about adding
one. The first is not free either: correcting a value can re-pair rows and move the
denominator, which it does in roughly a sixth of cases below.

Also pinned here: the counterexample that retired the old wording of P21, *no prediction
scores above one that produced more correct values*. It is false once an array is present,
and it is false in the direction that matters -- a metric with that property would make
spamming rows free.
"""
import copy
import itertools
import os as _os
import random
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench.metric import score                                 # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


ROW = {"type": "object", "properties": {"k": {"type": "string"},
                                        "v": {"type": "number"},
                                        "w": {"type": "number"}}}
SCHEMA = {"type": "object", "properties": {"rows": {"type": "array", "items": ROW}}}


def gold(n):
    return {"rows": [{"k": f"k{i}", "v": i, "w": i * 2} for i in range(n)]}


print("\nP21 first clause -- correcting one value never lowers the score")

rng = random.Random(9)
drops, moved, tried = [], 0, 0
for _ in range(4000):
    n = rng.randint(2, 4)
    G = gold(n)
    P = copy.deepcopy(G)
    spots = [(i, f) for i in range(n) for f in ("k", "v", "w")]
    corrupt = rng.sample(spots, rng.randint(1, min(6, len(spots))))
    for i, f in corrupt:
        P["rows"][i][f] = f"X{rng.randint(0, 9)}" if f == "k" else 99
    order = list(range(n))
    rng.shuffle(order)                                   # the model returns rows in its order
    P = {"rows": [P["rows"][j] for j in order]}
    where = {j: idx for idx, j in enumerate(order)}

    before = score(copy.deepcopy(P), copy.deepcopy(G), SCHEMA)
    i, f = rng.choice(corrupt)                           # restore exactly one wrong value
    after_doc = copy.deepcopy(P)
    after_doc["rows"][where[i]][f] = G["rows"][i][f]
    after = score(after_doc, copy.deepcopy(G), SCHEMA)

    tried += 1
    if after["total"] != before["total"]:
        moved += 1
    if after["accuracy"] < before["accuracy"] - 1e-9:
        drops.append((G, P, i, f, before["accuracy"], after["accuracy"]))

report("correcting one value never lowers the score", not drops,
       f"{len(drops)} of {tried} corrections lowered it, e.g. {drops[0] if drops else ''}")
note(f"{tried} corrections; the denominator moved in {moved} of them "
     f"({100 * moved / tried:.0f}%), and the score still never fell")


print("\nP21 second clause -- output with no correct value never raises the score")

rng = random.Random(21)
rose, flat_nonzero, fell, kept_matched = 0, 0, 0, True
for _ in range(3000):
    n = rng.randint(1, 4)
    G = gold(n)
    P = {"rows": [{"k": f"k{i}" if rng.random() < .7 else f"X{i}",
                   "v": i if rng.random() < .6 else 900 + i,
                   "w": i * 2 if rng.random() < .6 else 900 + i} for i in range(n)]}
    rng.shuffle(P["rows"])
    before = score(copy.deepcopy(P), copy.deepcopy(G), SCHEMA)
    junk = [{"k": f"ZZ{j}", "v": -1 - j, "w": -100 - j} for j in range(rng.randint(1, 5))]
    after = score({"rows": P["rows"] + junk}, copy.deepcopy(G), SCHEMA)

    kept_matched = kept_matched and after["matched"] == before["matched"]
    if after["accuracy"] > before["accuracy"] + 1e-12:
        rose += 1
    elif abs(after["accuracy"] - before["accuracy"]) <= 1e-12:
        if before["accuracy"] != 0.0:
            flat_nonzero += 1
    else:
        fell += 1

report("appended rows that share nothing never raise the score", rose == 0,
       f"{rose} of 3000 rose")
report("...and strictly lower it whenever the score was above zero", flat_nonzero == 0,
       f"{flat_nonzero} stayed flat while above zero")
report("...while producing no correct value of their own", kept_matched,
       "matched changed, so the appended rows were not inert")
note(f"3000 appends: {fell} fell, {rose} rose, the rest were already at 0.00")


print("\nthe retired wording -- 'no prediction scores above one with more correct values'")

G = gold(3)
honest = {"rows": [{"k": f"k{i}", "v": i, "w": 999} for i in range(3)]}
spam = {"rows": [{"k": f"k{i}", "v": i, "w": i * 2} for i in range(3)]
        + [{"k": f"z{j}", "v": 900 + j, "w": 900 + j} for j in range(10)]}
h = score(honest, copy.deepcopy(G), SCHEMA)
s = score(spam, copy.deepcopy(G), SCHEMA)

report("a prediction with MORE correct values can score lower",
       s["matched"] > h["matched"] and s["accuracy"] < h["accuracy"],
       f"honest {h['matched']}/{h['accuracy']:.2f}  spam {s['matched']}/{s['accuracy']:.2f}")
report("the figures quoted in §8 and §9 are the ones the scorer produces",
       (round(h["accuracy"], 4), round(s["accuracy"], 4)) == (0.6667, 0.2308),
       f"got {h['accuracy']:.4f} and {s['accuracy']:.4f}, want 0.6667 and 0.2308")
note(f"spam recovered every gold value ({s['matched']} of {s['matched']}) and scored "
     f"{s['accuracy']:.2f}; honest recovered {h['matched']} and scored {h['accuracy']:.2f}")
note("a metric forbidding this would make emitting rows free, which is the attack")


print("\nwhat does survive: the best reachable score rises only with correctness")

best = {}
for combo in itertools.product(["omit", "right", "wrong"], repeat=3):
    rows = []
    for i, c in enumerate(combo):
        if c == "right":
            rows.append({"k": f"k{i}", "v": i, "w": i * 2})
        elif c == "wrong":
            rows.append({"k": f"X{i}", "v": 900 + i, "w": 900 + i})
    r = score({"rows": rows}, copy.deepcopy(G), SCHEMA)
    best[r["matched"]] = max(best.get(r["matched"], 0.0), r["accuracy"])

ordered = [best[k] for k in sorted(best)]
report("best reachable accuracy is non-decreasing in correct values produced",
       all(a <= b + 1e-9 for a, b in zip(ordered, ordered[1:])),
       f"{ {k: round(v, 2) for k, v in sorted(best.items())} }")
note(f"over a document WITH an array: { {k: round(v, 2) for k, v in sorted(best.items())} }")

print(f"\n{'VOLUME INCENTIVE HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"  {f}")
_sys.exit(1 if FAILS else 0)
