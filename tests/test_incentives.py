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
from omni_extract_bench.metric import score                                 # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


print("\nA PAIRED ROW HAS THE SAME DENOMINATOR AS AN OMITTED ONE")
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
omit = score({"lines": BASE}, GOLD, SCH)
partial = [score({"lines": BASE + [row(i, j) for i in range(5, N)]}, GOLD, SCH)
           for j in range(K + 1)]

report("a row of NAMED fields keeps the omitted row's denominator once it pairs",
       all(p["total"] == omit["total"] for p in partial[1:]),
       f"omit {omit['total']}, attempts {[p['total'] for p in partial[1:]]}")
note("named fields only -- a wrong scalar-array element DOES grow it, see the last section")
report("...so under accuracy, attempting beats omitting from one field right",
       all(p["accuracy"] > omit["accuracy"] + 1e-9 for p in partial[1:]),
       f"omit {omit['accuracy']:.1f}, attempts "
       f"{[round(p['accuracy'], 1) for p in partial[1:]]}")
note("if this ever reverses, providers are being paid to truncate")

SPEC_TABLE = {None: (0.5, 0.667), 0: (0.333, 0.5), 1: (0.625, 0.625),
              2: (0.75, 0.75), 4: (1.0, 1.0)}
drift = []
for j, (want_acc, want_f1) in SPEC_TABLE.items():
    r = omit if j is None else partial[j]
    if (abs(round(r["accuracy"], 3) - want_acc) > 1e-9
            or abs(round(r["f1"], 3) - want_f1) > 1e-9):
        drift.append((j, round(r["accuracy"], 3), round(r["f1"], 3),
                      want_acc, want_f1))
report("the figures printed in METRIC_SPEC section 8 are the ones the scorer produces",
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
T = 5
ARR_S = {"properties": {"lines": {"type": "array", "items": {"properties": {
    "sku": {"type": "string"},
    "tags": {"type": "array", "items": {"type": "string"}}}}}}}


def arow(i, tags=None):
    return {"sku": f"s{i}", "tags": tags if tags is not None
            else [f"t{i}_{j}" for j in range(T)]}


AG = {"lines": [arow(i) for i in range(N)]}
ABASE = [arow(i) for i in range(5)]
a_omit = score({"lines": ABASE}, AG, ARR_S)
a_guess = score({"lines": ABASE + [arow(i, [f"X{j}" for j in range(T)])
                                   for i in range(5, N)]}, AG, ARR_S)
a_honest = score({"lines": ABASE + [{"sku": f"s{i}"} for i in range(5, N)]}, AG, ARR_S)
report("guessing a row of unreadable array content loses to omitting it",
       a_guess["accuracy"] < a_omit["accuracy"] - 1e-9,
       f"guess {a_guess['accuracy']:.1f} vs omit {a_omit['accuracy']:.1f}")
report("a row can PAIR and still score below omitting, so 'paired' is not the test",
       a_guess["matched_rows"] == N and a_guess["accuracy"] < a_omit["accuracy"] - 1e-9,
       f"matched_rows {a_guess['matched_rows']}/{N}, "
       f"acc {a_guess['accuracy']:.1f} vs omit {a_omit['accuracy']:.1f}")
note("this is why P17 is about values read CORRECTLY, not about the row pairing")
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

print("\nP17 AS STATED IN THE SPEC: ONLY VALUES READ CORRECTLY")
for lbl, gold_doc, omit_pred, honest_pred, sch in (
    ("named fields", {"lines": [row(i) for i in range(N)]},
     {"lines": BASE}, {"lines": BASE + [{"sku": f"sku{i}"} for i in range(5, N)]}, SCH),
    ("scalar arrays", AG, {"lines": ABASE},
     {"lines": ABASE + [{"sku": f"s{i}"} for i in range(5, N)]}, ARR_S),
):
    o, h = score(omit_pred, gold_doc, sch), score(honest_pred, gold_doc, sch)
    report(f"P17 holds for {lbl}: same denominator, higher numerator",
           h["total"] == o["total"] and h["matched"] > o["matched"]
           and h["accuracy"] > o["accuracy"] + 1e-9,
           f"omit {o['matched']}/{o['total']}, honest {h['matched']}/{h['total']}")

print("\nRULE 2: WHAT ABSTAINING COSTS, IN BOTH DIRECTIONS")
R2_S = {"properties": {"a": {"type": "string"}, "note": {"type": "string"},
                       "tags": {"type": "array", "items": {"type": "string"}}}}
R2_G = {"a": "keep", "note": "N", "tags": ["t1", "t2"]}
spellings = [score(p, R2_G, R2_S) for p in (
    {"a": "keep", "tags": ["t1", "t2"]},
    {"a": "keep", "note": None, "tags": ["t1", "t2"]},
    {"a": "keep", "note": [], "tags": ["t1", "t2"]})]
report("null, [] and omitting the key cost exactly the same",
       len({(round(r["accuracy"], 6), round(r["f1"], 6), r["total"]) for r in spellings}) == 1,
       f"{[round(r['accuracy'], 2) for r in spellings]}")

abstain = spellings[0]
wrong = score({"a": "keep", "note": "WRONG", "tags": ["t1", "t2"]}, R2_G, R2_S)
right = score({"a": "keep", "note": "N", "tags": ["t1", "t2"]}, R2_G, R2_S)
arr_abstain = score({"a": "keep", "note": "N", "tags": ["t1"]}, R2_G, R2_S)
arr_wrong = score({"a": "keep", "note": "N", "tags": ["t1", "WRONG"]}, R2_G, R2_S)
report("abstaining never costs MORE than a wrong value",
       abstain["accuracy"] >= wrong["accuracy"] - 1e-9
       and abstain["f1"] >= wrong["f1"] - 1e-9,
       f"abstain {abstain['accuracy']:.2f}/{abstain['f1']:.3f}, "
       f"wrong {wrong['accuracy']:.2f}/{wrong['f1']:.3f}")
report("...and inside an array it costs strictly less",
       arr_abstain["accuracy"] > arr_wrong["accuracy"] + 1e-9,
       f"abstain {arr_abstain['accuracy']:.2f} vs wrong {arr_wrong['accuracy']:.2f}")
report("but a value you get RIGHT always beats silence",
       right["accuracy"] > abstain["accuracy"] + 1e-9
       and right["f1"] > abstain["f1"] + 1e-9,
       f"right {right['accuracy']:.2f}/{right['f1']:.3f} vs "
       f"abstain {abstain['accuracy']:.2f}/{abstain['f1']:.3f}")
note("so rule 2 is about the hit-rate, not about caution -- see the f1 threshold below")

print("\nAND A STRICT SCHEMA CANNOT TAKE ABSTENTION AWAY")
same = [score({"lines": ABASE + [{"sku": f"s{i}", "tags": t} for i in range(5, N)]},
              AG, ARR_S) for t in (None, [])]
report("`tags: null` and `tags: []` score exactly as omitting the key does",
       all(abs(s["accuracy"] - a_honest["accuracy"]) < 1e-9
           and s["total"] == a_honest["total"] for s in same),
       f"{[round(s['accuracy'], 1) for s in same]} vs {a_honest['accuracy']:.1f}")
note("so a vendor forced to emit every property is not forced to fabricate")

print("\nORDER-FREEDOM IS A TRADE, AND SECTION 8 PRINTS ITS PRICE")
TAGS_S = {"properties": {"t": {"type": "array", "items": {"type": "string"}}}}
five = {"t": [f"v{i}" for i in range(5)]}
wrong_in_place = {"t": [f"v{i}" for i in range(4)] + ["WRONG"]}
one_dropped = {"t": [f"v{i}" for i in range(1, 5)]}

got = {
    ("wrong in place", "free"):    score(wrong_in_place, five, TAGS_S)["accuracy"],
    ("wrong in place", "ordered"): score(wrong_in_place, five, TAGS_S, ("t",))["accuracy"],
    ("one dropped", "free"):       score(one_dropped, five, TAGS_S)["accuracy"],
    ("one dropped", "ordered"):    score(one_dropped, five, TAGS_S, ("t",))["accuracy"],
}
want = {("wrong in place", "free"): 0.667, ("wrong in place", "ordered"): 0.8,
        ("one dropped", "free"): 0.8, ("one dropped", "ordered"): 0.0}
drift2 = [(k, round(v, 3), want[k]) for k, v in got.items() if abs(round(v, 3) - want[k]) > 1e-9]
report("the four order_matters figures in section 8 are the ones the scorer produces",
       not drift2, f"drifted: {drift2}")
report("`order_matters` helps a substitution and ruins an omission",
       got[("wrong in place", "ordered")] > got[("wrong in place", "free")]
       and got[("one dropped", "ordered")] < got[("one dropped", "free")],
       f"{got}")
note("omission is the commoner extraction failure, so the trade usually runs the wrong way")

print("\nWRAPPING A SCALAR IN A ONE-FIELD OBJECT CHANGES NOTHING")
BARE_S = {"properties": {"t": {"type": "array", "items": {"type": "string"}}}}
WRAP_S = {"properties": {"t": {"type": "array", "items": {"properties":
          {"tag": {"type": "string"}}}}}}
bare = score({"t": ["a", "b", "c", "X"]}, {"t": list("abcd")}, BARE_S)
wrap = score({"t": [{"tag": c} for c in ["a", "b", "c", "X"]]},
             {"t": [{"tag": c} for c in "abcd"]}, WRAP_S)
report("a bare scalar list and a list of one-field objects score identically",
       abs(bare["accuracy"] - wrap["accuracy"]) < 1e-9
       and bare["total"] == wrap["total"]
       and bare["invented_item"] == wrap["invented_item"],
       f"bare {bare['accuracy']:.1f}/{bare['total']} vs wrap "
       f"{wrap['accuracy']:.1f}/{wrap['total']}")
note("so the fix for a double-charged list is a second field to pair on, not a wrapper")

print("\nINVENTED ROWS ARE CHARGED AT SCALE, NOT JUST IN THE 27-ROW EXAMPLE")
ROWS_S = {"properties": {"l": {"type": "array", "items": {"properties":
          {f"f{k}": {"type": "string"} for k in range(4)}}}}}
ten = {"l": [{f"f{k}": f"r{i}v{k}" for k in range(4)} for i in range(10)]}
flood = {"l": ten["l"] + [{f"f{k}": f"J{j}_{k}" for k in range(4)} for j in range(1000)]}
perfect_acc, flood_acc = score(ten, ten, ROWS_S)["accuracy"], score(flood, ten, ROWS_S)["accuracy"]
report("ten perfect rows score 1.0; the same ten plus a thousand invented score 0.0099",
       abs(perfect_acc - 1.0) < 1e-9 and abs(round(flood_acc, 4) - 0.0099) < 1e-9,
       f"{perfect_acc:.2f} and {flood_acc:.2f}")
note("accuracy <= matched/|gold|, so inventing can only move a score down")

print(f"\n{'INCENTIVES POINT THE RIGHT WAY' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
