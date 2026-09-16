"""Every number in METRIC_SPEC section 5, checked against the scorer."""
import sys
sys.path.insert(0, "/Users/paulscemama/omni_extract_bench")
from omni_extract_bench.score import grade

# ── the scorer now requires a schema ──────────────────────────────────────────────────
# These tests are about scoring, not schema plumbing, so derive one from the ground truth.
# That is the realistic case anyway: gold conforms to the schema that was sent. Deriving it
# from gold alone is deliberate -- a key the PREDICTION invented genuinely is not a slot the
# model was offered, which is what tells `invented field` from `fabricated`.
from omni_extract_bench.score import grade as _grade_impl          # noqa: E402
from omni_extract_bench.score import explain as _explain_impl      # noqa: E402


def _schema_from(doc):
    if isinstance(doc, dict):
        return {"type": "object", "properties": {k: _schema_from(v) for k, v in doc.items()}}
    if isinstance(doc, list):
        merged = {}
        for e in doc:
            if isinstance(e, dict):
                merged.update(e)
        return {"type": "array", "items": _schema_from(merged) if merged else {}}
    return {}


def grade(pred, gt, schema=None, *a, **kw):
    return _grade_impl(pred, gt, schema or _schema_from(gt), *a, **kw)


def explain(pred, gt, schema=None, *a, **kw):
    return _explain_impl(pred, gt, schema or _schema_from(gt), *a, **kw)
# ──────────────────────────────────────────────────────────────────────────────────────


def acc(pred, gold, **kw): 
    r = grade(pred, gold, None, **kw); return round(r["accuracy"], 1), r["total"]

rows = [
  ("{a:1,b:null} / {a:1,b:null}", ({"a":1,"b":None}, {"a":1,"b":None}), (100.0, 1)),
  ("{a:1,b:null} / {a:1}",        ({"a":1},          {"a":1,"b":None}), (100.0, 1)),
  ("{a:1,b:null} / {a:1,b:5}",    ({"a":1,"b":5},    {"a":1,"b":None}), (50.0, 2)),
  ("{a:1,b:5} / {a:1,b:null}",    ({"a":1,"b":None}, {"a":1,"b":5}),    (50.0, 2)),
  ("{a:1,b:5} / {a:1}",           ({"a":1},          {"a":1,"b":5}),    (50.0, 2)),
]
ok = True
for lbl, (p, g), want in rows:
    got = acc(p, g)
    ok &= got == want
    print(f"  {lbl:32} {got}  want {want}  {'ok' if got == want else 'MISMATCH'}")

print()
lazy_gold = dict({f"f{i}": None for i in range(17)}, name="A", total=1.0, date="2024-01-01")
lazy_pred = {f"f{i}": None for i in range(17)}
g0 = acc(lazy_pred, lazy_gold)
ok &= g0 == (0.0, 3)
print(f"  {'lazy: nulls only':32} {g0}  want (0.0, 3)  {'ok' if g0 == (0.0,3) else 'MISMATCH'}")
print(f"  {'  (85.0 = 17/20 if nulls matched)':32}")

print()
ORD = ["xs"]
ords = [
  ("[a,null,c] / [a,null,c]", (["a",None,"c"], ["a",None,"c"]), 100.0),
  ("[a,null,c] / [a,b,c]",    (["a","b","c"],  ["a",None,"c"]),  66.7),
  ("[a,null,c] / [a,c]",      (["a","c"],      ["a",None,"c"]),  33.3),
]
for lbl, (p, g), want in ords:
    got = round(grade({"xs": p}, {"xs": g}, None, order_matters=ORD)["accuracy"], 1)
    ok &= got == want
    print(f"  ordered {lbl:26} {got}  want {want}  {'ok' if got == want else 'MISMATCH'}")
free = round(grade({"xs": ["a","c"]}, {"xs": ["a",None,"c"]})["accuracy"], 1)
ok &= free == 100.0
print(f"  order-free [a,null,c] / [a,c]            {free}  want 100.0")

print()
drop_g = {"r": [{"k": None, "v": None}, {"k": "x"}]}
print(f"  gold all-null row omitted      {round(grade({'r':[{'k':'x'}]}, drop_g)['accuracy'],1)}  want 100.0")
print(f"  pred all-null row invented     "
      f"{round(grade({'r':[{'k':'x'},{'k':None,'v':None}]}, {'r':[{'k':'x'}]})['accuracy'],1)}  want 100.0")

# ── section 5: abstention IS measurable ───────────────────────────────────────────────
# The spec used to claim otherwise. It does not, because abstaining is producing no value at
# an address -- null and an absent key are two spellings of one behaviour, and `precision`
# separates a model that declines from one that guesses. The numbers printed in section 5 are
# these.
print("\nABSTENTION IS MEASURABLE (section 5)")
NN = 20
AB_S = {"properties": {"lines": {"type": "array", "items": {"properties": {
    "sku": {"type": "string"}, "note": {"type": "string"}}}}}}
ab_gold = {"lines": [{"sku": f"s{i}", "note": f"n{i}"} for i in range(NN)]}
declines = {"lines": [{"sku": f"s{i}", "note": (None if i % 4 == 1 else f"n{i}")}
                      for i in range(NN)]}
omits = {"lines": [dict({"sku": f"s{i}"}, **({} if i % 4 == 1 else {"note": f"n{i}"}))
                   for i in range(NN)]}
guesses = {"lines": [{"sku": f"s{i}", "note": ("n0" if i % 4 == 1 else f"n{i}")}
                     for i in range(NN)]}
truncated = {"lines": [{"sku": f"s{i}", "note": f"n{i}"} for i in range(NN - 5)]}
D, O, G, T = (grade(x, ab_gold, AB_S) for x in (declines, omits, guesses, truncated))

checks = [
    ("declining with null scores exactly as omitting the key",
     (round(D["accuracy"], 2), round(D["f1"], 4), round(D["precision"], 4))
     == (round(O["accuracy"], 2), round(O["f1"], 4), round(O["precision"], 4))),
    ("...and both give precision 1.00: nothing said was untrue",
     D["precision"] == 1.0 and O["precision"] == 1.0),
    ("guessing instead of declining is charged by precision and f1",
     G["precision"] < D["precision"] - 1e-9 and G["f1"] < D["f1"] - 1e-9),
    ("...while accuracy cannot tell them apart",
     abs(G["accuracy"] - D["accuracy"]) < 1e-9),
    ("the section 5 figures are the ones the scorer produces",
     (round(D["accuracy"], 2), round(D["f1"] * 100, 2), round(D["precision"] * 100, 2)) == (87.50, 93.33, 100.00)
     and (round(G["accuracy"], 2), round(G["f1"] * 100, 2), round(G["precision"] * 100, 2)) == (87.50, 87.50, 87.50)),
    ("truncation is distinguishable from declining, by how much is unfound",
     T["unfound"] > D["unfound"]),
]
for name, passed in checks:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
print("          declining scatters across hard fields; truncation leaves a contiguous tail")
if not all(c[1] for c in checks):
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════════════
# THE EMPTY STRING IS A THIRD SPELLING OF ABSENCE (section 5)
# ---------------------------------------------------------------------------------------
# A document can print "N/A"; it cannot print emptiness. `""` is what a blank cell becomes
# on the way into JSON, so it means what `null` means and must score the same -- otherwise
# the number moves with a vendor's serialization habit rather than with what it read. It
# did: one provider's `""` convention cost it 7.92 points on `longarray` before this rule.
#
# The rule stops at the empty string. "N/A", "None" and "-" are ink on the page and the
# corpus uses them as real gold values 24,980 times, so folding those would delete answers.
print("\nTHE EMPTY STRING SCORES AS AN ABSENCE (section 5)")
ES_S = {"properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
es_gold = {"a": "keep", "b": None}
spellings = {spelling: grade(pred, es_gold, ES_S) for spelling, pred in (
    ("null", {"a": "keep", "b": None}),
    ("omitted", {"a": "keep"}),
    ("empty string", {"a": "keep", "b": ""}),
    ("whitespace", {"a": "keep", "b": "   "}),
)}
placeholder = grade({"a": "keep", "b": "n/a"}, es_gold, ES_S)
gold_blank = grade({"a": "keep", "b": "x"}, {"a": "keep", "b": ""}, ES_S)
gold_na = grade({"a": "keep", "b": "n/a"}, {"a": "keep", "b": "n/a"}, ES_S)
row_S = {"properties": {"r": {"type": "array", "items": {"properties": {
    "v": {"type": "string"}}}}}}
blank_row = grade({"r": [{"v": "x"}]}, {"r": [{"v": "x"}, {"v": ""}]}, row_S)

es_checks = [
    ("all four spellings of absence score identically",
     len({(round(r["accuracy"], 6), round(r["f1"], 6), r["total"]) for r in spellings.values()}) == 1),
    ("...and none of them is charged as a fabrication",
     all(r["fabricated"] == 0 for r in spellings.values())),
    ("a predicted '' where gold is silent adds no address",
     spellings["empty string"]["total"] == spellings["null"]["total"]),
    ("'n/a' is still a value, and still charged",
     placeholder["fabricated"] == 1 and placeholder["accuracy"] < spellings["null"]["accuracy"]),
    ("...and matches a gold 'n/a', which is content the page carries",
     gold_na["accuracy"] == 100.0),
    ("a gold '' asks for nothing, so asserting there is charged",
     gold_blank["fabricated"] == 1),
    ("a gold row whose payload is only '' is dropped like an all-null row",
     blank_row["accuracy"] == 100.0 and blank_row["gt_rows"] == 1),
]
for name, passed in es_checks:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
if not all(c[1] for c in es_checks):
    sys.exit(1)

sys.exit(0 if ok else 1)
