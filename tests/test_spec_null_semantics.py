"""Every number in METRIC_SPEC section 10, checked against the scorer."""
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
sys.exit(0 if ok else 1)
