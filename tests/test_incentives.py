#!/usr/bin/env python3
"""What the metric pays a model to do, checked rather than assumed.

A benchmark is a set of incentives. If omitting a row you could partly read scored better
than reading it, every provider would learn to truncate, and the leaderboard would rank
caution above capability. These tests pin the ordering so that cannot happen quietly.

The short version:

    emit a row with only the fields you can read   strictly dominates omitting it
    omit a row                                     beats emitting one that is wholly wrong
    a paired row beats omission under `accuracy`   even at 25% right
    ...but not under `f1` until about half right    which is why both are reported

Run: python3 tests/test_incentives.py
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench.score import grade                                 # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


# ═══════════════════════════════════════════════════════════════════════════════
print("\nA PAIRED ROW HAS THE SAME DENOMINATOR AS AN OMITTED ONE")
# This is the load-bearing fact. A row's gold leaves are in the denominator whether the
# prediction reaches them or not; pairing only moves them from `unfound` into
# `misread`/`matched`. So attempting a row can only add to the numerator -- there is no
# denominator penalty to weigh against trying.
N, K = 10, 4
F = ["sku", "qty", "price", "code"]
SCH = {"properties": {"lines": {"type": "array", "items": {"properties":
        {f: {"type": "string"} for f in F}}}}}


def row(i, right=K):
    r = {f: f"{f}{i}" for f in F}
    for f in F[right:]:
        r[f] = "WRONG"
    return r


GOLD = {"lines": [row(i) for i in range(N)]}
BASE = [row(i) for i in range(5)]
omit = grade({"lines": BASE}, GOLD, SCH)
partial = [grade({"lines": BASE + [row(i, j) for i in range(5, N)]}, GOLD, SCH)
           for j in range(K + 1)]

report("a row paired on ANY matching field keeps the omitted row's denominator",
       all(p["total"] == omit["total"] for p in partial[1:]),
       f"omit {omit['total']}, attempts {[p['total'] for p in partial[1:]]}")
report("...so under accuracy, attempting beats omitting from one field right",
       all(p["accuracy"] > omit["accuracy"] + 1e-9 for p in partial[1:]),
       f"omit {omit['accuracy']:.1f}, attempts "
       f"{[round(p['accuracy'], 1) for p in partial[1:]]}")
note("if this ever reverses, providers are being paid to truncate")

# METRIC_SPEC section 12 prints these exact figures. Assert them, so the spec cannot drift
# away from the scorer without a test failing.
SPEC_TABLE = {None: (50.0, 66.7), 0: (33.3, 50.0), 1: (62.5, 62.5),
              2: (75.0, 75.0), 4: (100.0, 100.0)}
drift = []
for j, (want_acc, want_f1) in SPEC_TABLE.items():
    r = omit if j is None else partial[j]
    if (abs(round(r["accuracy"], 1) - want_acc) > 1e-9
            or abs(round(r["f1"] * 100, 1) - want_f1) > 1e-9):
        drift.append((j, round(r["accuracy"], 1), round(r["f1"] * 100, 1),
                      want_acc, want_f1))
report("the figures printed in METRIC_SPEC section 12 are the ones the scorer produces",
       not drift, f"drifted: {drift}")

print("\nA ROW WITH NOTHING RIGHT IS WORSE THAN OMITTING IT")
report("a row that pairs with nothing is charged on both sides",
       partial[0]["accuracy"] < omit["accuracy"] - 1e-9
       and partial[0]["total"] > omit["total"],
       f"wholly wrong {partial[0]['accuracy']:.1f} (denominator {partial[0]['total']}) "
       f"vs omit {omit['accuracy']:.1f} ({omit['total']})")
note("so wholesale row fabrication loses, while partially real rows gain")

print("\nf1 AND accuracy DISAGREE ON A MOSTLY-WRONG ROW, ON PURPOSE")
report("accuracy prefers attempting at 25% right; f1 prefers omitting",
       partial[1]["accuracy"] > omit["accuracy"] + 1e-9
       and partial[1]["f1"] < omit["f1"] - 1e-9,
       f"25%: acc {partial[1]['accuracy']:.1f} vs {omit['accuracy']:.1f}, "
       f"f1 {partial[1]['f1']:.3f} vs {omit['f1']:.3f}")
report("both prefer attempting once about half the row is right",
       partial[2]["accuracy"] > omit["accuracy"] + 1e-9
       and partial[2]["f1"] > omit["f1"] + 1e-9)
note("accuracy asks how much you recovered; f1 asks how much of what you said was true")
note("a model padding rows to 25% would climb on accuracy and sink on f1 -- report both")

print("\nABSTAINING PER FIELD DOMINATES BOTH OMITTING AND GUESSING")
# The regime where omitting genuinely wins: a row whose content is mostly scalar-array
# elements the model cannot read, since each wrong element costs two denominator slots.
# Even there, emitting the row with only the readable fields wins on both numbers.
T = 5
ARR_S = {"properties": {"lines": {"type": "array", "items": {"properties": {
    "sku": {"type": "string"},
    "tags": {"type": "array", "items": {"type": "string"}}}}}}}


def arow(i, tags=None):
    return {"sku": f"s{i}", "tags": tags if tags is not None
            else [f"t{i}_{j}" for j in range(T)]}


AG = {"lines": [arow(i) for i in range(N)]}
ABASE = [arow(i) for i in range(5)]
a_omit = grade({"lines": ABASE}, AG, ARR_S)
a_guess = grade({"lines": ABASE + [arow(i, [f"X{j}" for j in range(T)])
                                   for i in range(5, N)]}, AG, ARR_S)
a_honest = grade({"lines": ABASE + [{"sku": f"s{i}"} for i in range(5, N)]}, AG, ARR_S)
report("guessing a row of unreadable array content loses to omitting it",
       a_guess["accuracy"] < a_omit["accuracy"] - 1e-9,
       f"guess {a_guess['accuracy']:.1f} vs omit {a_omit['accuracy']:.1f}")
report("but abstaining per FIELD beats both, on accuracy and on f1",
       a_honest["accuracy"] > a_omit["accuracy"] + 1e-9
       and a_honest["accuracy"] > a_guess["accuracy"] + 1e-9
       and a_honest["f1"] > a_omit["f1"] + 1e-9
       and a_honest["f1"] > a_guess["f1"] + 1e-9,
       f"honest {a_honest['accuracy']:.1f}/{a_honest['f1']:.3f}, "
       f"omit {a_omit['accuracy']:.1f}/{a_omit['f1']:.3f}, "
       f"guess {a_guess['accuracy']:.1f}/{a_guess['f1']:.3f}")
report("abstaining per field costs the omitted row's denominator, and no more",
       a_honest["total"] == a_omit["total"],
       f"honest {a_honest['total']} vs omit {a_omit['total']}")

print("\nAND A STRICT SCHEMA CANNOT TAKE ABSTENTION AWAY")
# Strict structured outputs require every declared property to be present, so a model may
# not omit `tags`. It can still decline: null and [] carry no addresses.
same = [grade({"lines": ABASE + [{"sku": f"s{i}", "tags": t} for i in range(5, N)]},
              AG, ARR_S) for t in (None, [])]
report("`tags: null` and `tags: []` score exactly as omitting the key does",
       all(abs(s["accuracy"] - a_honest["accuracy"]) < 1e-9
           and s["total"] == a_honest["total"] for s in same),
       f"{[round(s['accuracy'], 1) for s in same]} vs {a_honest['accuracy']:.1f}")
note("so a vendor forced to emit every property is not forced to fabricate")

print(f"\n{'INCENTIVES POINT THE RIGHT WAY' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
