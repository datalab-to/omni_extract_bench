"""Every number in METRIC_SPEC section 5, checked against the scorer."""
import os as _os
import sys
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench.metric import score

from omni_extract_bench.metric import score as _score_impl          # noqa: E402


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


def score(pred, gt, schema=None, *a, **kw):
    return _score_impl(pred, gt, schema or _schema_from(gt), *a, **kw)


def explain(pred, gt, schema=None, *a, **kw):
    """The per-address view. `score` is the only entry point; verdicts come off it."""
    return _score_impl(pred, gt, schema or _schema_from(gt), *a, verdicts=True, **kw)["verdicts"]


def acc(pred, gold, **kw): 
    r = score(pred, gold, None, **kw); return round(r["accuracy"], 3), r["total"]

rows = [
  ("{a:1,b:null} / {a:1,b:null}", ({"a":1,"b":None}, {"a":1,"b":None}), (1.0, 1)),
  ("{a:1,b:null} / {a:1}",        ({"a":1},          {"a":1,"b":None}), (1.0, 1)),
  ("{a:1,b:null} / {a:1,b:5}",    ({"a":1,"b":5},    {"a":1,"b":None}), (0.5, 2)),
  ("{a:1,b:5} / {a:1,b:null}",    ({"a":1,"b":None}, {"a":1,"b":5}),    (0.5, 2)),
  ("{a:1,b:5} / {a:1}",           ({"a":1},          {"a":1,"b":5}),    (0.5, 2)),
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
print(f"  {'  (0.85 = 17/20 if nulls matched)':32}")

print()
ORD = ["xs"]
ords = [
  ("[a,null,c] / [a,null,c]", (["a",None,"c"], ["a",None,"c"]), 1.0),
  ("[a,null,c] / [a,b,c]",    (["a","b","c"],  ["a",None,"c"]), 0.667),
  ("[a,null,c] / [a,c]",      (["a","c"],      ["a",None,"c"]), 0.333),
]
for lbl, (p, g), want in ords:
    got = round(score({"xs": p}, {"xs": g}, None, order_matters=ORD)["accuracy"], 3)
    ok &= got == want
    print(f"  ordered {lbl:26} {got}  want {want}  {'ok' if got == want else 'MISMATCH'}")
free = round(score({"xs": ["a","c"]}, {"xs": ["a",None,"c"]})["accuracy"], 3)
ok &= free == 1.0
print(f"  order-free [a,null,c] / [a,c]            {free}  want 1.0")

print()
drop_g = {"r": [{"k": None, "v": None}, {"k": "x"}]}
print(f"  gold all-null row omitted      {round(score({'r':[{'k':'x'}]}, drop_g)['accuracy'],3)}  want 1.0")
print(f"  pred all-null row invented     "
      f"{round(score({'r':[{'k':'x'},{'k':None,'v':None}]}, {'r':[{'k':'x'}]})['accuracy'],3)}  want 1.0")

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
D, O, G, T = (score(x, ab_gold, AB_S) for x in (declines, omits, guesses, truncated))

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
     (round(D["accuracy"], 4), round(D["f1"], 4), round(D["precision"], 4)) == (0.875, 0.9333, 1.0)
     and (round(G["accuracy"], 4), round(G["f1"], 4), round(G["precision"], 4)) == (0.875, 0.875, 0.875)),
    ("truncation is distinguishable from declining, by how much is unfound",
     T["unfound"] > D["unfound"]),
]
for name, passed in checks:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
print("          declining scatters across hard fields; truncation leaves a contiguous tail")
if not all(c[1] for c in checks):
    sys.exit(1)


print("\nTHE EMPTY STRING SCORES AS AN ABSENCE (section 5)")
ES_S = {"properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
es_gold = {"a": "keep", "b": None}
spellings = {spelling: score(pred, es_gold, ES_S) for spelling, pred in (
    ("null", {"a": "keep", "b": None}),
    ("omitted", {"a": "keep"}),
    ("empty string", {"a": "keep", "b": ""}),
    ("whitespace", {"a": "keep", "b": "   "}),
)}
placeholder = score({"a": "keep", "b": "n/a"}, es_gold, ES_S)
gold_blank = score({"a": "keep", "b": "x"}, {"a": "keep", "b": ""}, ES_S)
gold_na = score({"a": "keep", "b": "n/a"}, {"a": "keep", "b": "n/a"}, ES_S)
row_S = {"properties": {"r": {"type": "array", "items": {"properties": {
    "v": {"type": "string"}}}}}}
blank_row = score({"r": [{"v": "x"}]}, {"r": [{"v": "x"}, {"v": ""}]}, row_S)

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
     gold_na["accuracy"] == 1.0),
    ("a gold '' asks for nothing, so asserting there is charged",
     gold_blank["fabricated"] == 1),
    ("a gold row whose payload is only '' is dropped like an all-null row",
     blank_row["accuracy"] == 1.0 and blank_row["gt_rows"] == 1),
]
for name, passed in es_checks:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
if not all(c[1] for c in es_checks):
    sys.exit(1)

sys.exit(0 if ok else 1)
