"""Every number in METRIC_SPEC section 10, checked against the scorer."""
import sys
sys.path.insert(0, ".")
from omni_extract_bench.score import grade

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
