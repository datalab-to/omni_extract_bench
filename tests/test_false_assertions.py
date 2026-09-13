#!/usr/bin/env python3
"""The three kinds of false assertion, and the identities that tie them together.

`1 - precision` is the rate at which the model asserts something untrue. That single number
hides three different bugs, so `grade` splits it:

    misread          the document has this value; the model read it wrongly
    fabricated       the schema offered the slot, the document is silent, the model filled it
    invented item    a value under an array element that paired with nothing
    invented field   a name the schema never declared

The split keys off the SCHEMA, not gold's nulls. A gold field written `null` and a gold field
left out mean the same thing (METRIC_SPEC section 5), so keying off gold would sort two
identical documents into different buckets.

Run: python3 tests/test_false_assertions.py
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench.score import (                                     # noqa: E402
    INDEX, KEY, _classify_extra, _schema_leaves, explain, format_node, grade, show)

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


SCH = {"type": "object", "properties": {
    "invoice_no": {"type": "string"},
    "total":      {"type": "number"},
    "discount":   {"type": ["number", "null"]},
    "lines": {"type": "array", "items": {"type": "object", "properties": {
        "sku":  {"type": "string"},
        "note": {"type": ["string", "null"]}}}},
    "bag": {"type": "object", "additionalProperties": {"type": "string"}}}}
GT = {"invoice_no": "INV-1", "total": 100.0, "discount": None,
      "lines": [{"sku": "a", "note": None}, {"sku": "b", "note": None}]}
PRED = {"invoice_no": "INV-1",          # match
        "total": 999.0,                 # misread
        "discount": 5.0,                # fabricated: schema slot, gold silent
        "vendor": "Acme",               # invented field: not in the schema
        "lines": [{"sku": "a", "note": "n/a"},   # fabricated cell in a real row
                  {"sku": "b"},
                  {"sku": "zz"}]}       # invented item: row pairs with nothing

# ═══════════════════════════════════════════════════════════════════════════════
print("\nWHAT THE SCHEMA OFFERS IS WHAT CAN BE FABRICATED")
slots = _schema_leaves(SCH)
report("the schema's leaf slots are found, at every depth",
       {format_node(k) for k in slots} == {"invoice_no", "total", "discount",
                                           "lines[*].sku", "lines[*].note"},
       str(sorted(format_node(k) for k in slots)))
report("an open map offers no slot, so nothing inside it can be fabricated",
       not any(format_node(k).startswith("bag") for k in slots))
note("that subtree is not graded at all, so a value there was never asked for")

print("\nGOLD'S NULLS ARE NOT THE AUTHORITY -- THE SCHEMA IS")
# These two ground truths are semantically identical (section 5). Keying off gold's nulls
# would call the first `fabricated` and the second `invented`, which is the bug this avoids.
NULL_GT = {"lines": [{"sku": "a", "note": None}]}
GONE_GT = {"lines": [{"sku": "a"}]}
SAME_P = {"lines": [{"sku": "a", "note": "x"}]}
both = [grade(SAME_P, g, SCH)["fabricated"] for g in (NULL_GT, GONE_GT)]
report("gold `note: null` and gold with no `note` classify identically",
       both == [1, 1], f"got {both}, want [1, 1]")

print("\nTHE THREE KINDS ARE TOLD APART")
cases = {
    "discount":      "fabricated",
    "lines[0].note": "fabricated",
    "lines[p2].sku": "invented item",
    "vendor":        "invented field",
    "bag.H":         "invented field",
}
addrs = {"discount": ((KEY, "discount"),),
         "lines[0].note": ((KEY, "lines"), (INDEX, 0), (KEY, "note")),
         "lines[p2].sku": ((KEY, "lines"), (INDEX, "p2"), (KEY, "sku")),
         "vendor": ((KEY, "vendor"),),
         "bag.H": ((KEY, "bag"), (KEY, "H"))}
for name, want in cases.items():
    got = _classify_extra(addrs[name], slots)
    report(f"{name} is {want}", got == want, f"got {got!r}")

print("\nTHE `pN` LABEL IS LOAD-BEARING, SO PIN IT")
# `_align_one` labels an unpaired predicted row `p<index>`. That internal convention is now
# what tells an extra array element from a filled field, so it gets its own test: if the
# labelling ever changes to a plain integer, a whole invented row would be reported as a
# handful of separately fabricated fields.
labels = [v.address for v in explain(PRED, GT, SCH) if v.verdict == "invented item"]
report("an unpaired predicted row is labelled with a string index, not an integer",
       labels and all(any(k == INDEX and isinstance(x, str) for k, x in a) for a in labels),
       f"got {[show(a) for a in labels]}")
report("...and that label never collides with a gold index",
       all(not any(k == INDEX and isinstance(x, int) and show(a).count("[p") == 0
                   for k, x in a) for a in labels),
       f"got {[show(a) for a in labels]}")
report("an integer index in an extra address would be a filled slot, not an extra item",
       _classify_extra(((KEY, "lines"), (INDEX, 3), (KEY, "note")), slots) == "fabricated")
note("so the two are distinguished by the label alone -- hence this test")

print("\nTHE IDENTITIES HOLD")
r = grade(PRED, GT, SCH)
parts = (r["matched"], r["misread"], r["unfound"], r["fabricated"],
         r["invented_item"], r["invented_field"])
report("asserted = matched + misread + fabricated + invented item + invented field",
       r["asserted"] == r["matched"] + r["misread"] + r["fabricated"]
       + r["invented_item"] + r["invented_field"],
       f"asserted {r['asserted']} vs parts {parts}")
report("total = matched + misread + unfound + fabricated + invented item + invented field",
       r["total"] == sum(parts), f"total {r['total']} vs sum {sum(parts)}")
report("precision = matched / asserted",
       abs(r["precision"] - r["matched"] / r["asserted"]) < 1e-12)
report("recall = matched / (matched + misread + unfound)",
       abs(r["recall"] - r["matched"] / (r["matched"] + r["misread"] + r["unfound"])) < 1e-12)
report("1 - precision = every false assertion over everything asserted",
       abs((1 - r["precision"])
           - (r["misread"] + r["fabricated"] + r["invented_item"] + r["invented_field"])
           / r["asserted"]) < 1e-12)
note("misread is counted in `asserted` and in gold, but only ONCE in `total` --")
note("which is the whole reason accuracy and f1 differ")

print("\nTHE COUNTS ARE THE VERDICT HISTOGRAM")
import collections                                                          # noqa: E402
hist = collections.Counter(v.verdict for v in explain(PRED, GT, SCH))
report("grade's counts equal explain's labels, so the two surfaces cannot drift",
       (r["matched"], r["misread"], r["unfound"], r["fabricated"],
        r["invented_item"], r["invented_field"])
       == (hist["match"], hist["wrong value"], hist["missing"], hist["fabricated"],
           hist["invented item"], hist["invented field"]),
       f"grade {parts} vs explain {dict(hist)}")

print("\nA SCHEMA IS REQUIRED")
for bad in (None, {}, "not a schema", []):
    try:
        grade(PRED, GT, bad)
        report(f"{bad!r} is refused", False, "no TypeError raised")
    except TypeError as exc:
        report(f"{bad!r} is refused", "schema is required" in str(exc), str(exc)[:70])
note("without it an open map cannot be found, and a fabricated value cannot be told")
note("from an invented one -- reporting fabricated=0 would be a false claim")

print("\nEXISTING NUMBERS ARE UNCHANGED BY THE SPLIT")
report("accuracy, precision, recall and f1 still come from matched and the address sets",
       abs(r["accuracy"] - 100 * r["matched"] / r["total"]) < 1e-9
       and abs(r["f1"] - (2 * r["precision"] * r["recall"]
                          / (r["precision"] + r["recall"]))) < 1e-12)

print("\nA SCALAR ARRAY HAS NO CELLS TO MISREAD")
# Elements of a scalar array compare as a multiset, so they have no identity. A value read
# wrongly there is not one misreading: it is one gold element nobody produced plus one
# element the model produced that is not there. Charged on both sides, which is harsher than
# the same error in a named field -- worth knowing, and pinned so it cannot drift silently.
ARR_S = {"properties": {"q": {"type": "array", "items": {"type": "number"}},
                        "name": {"type": "string"}}}
ARR_G = {"name": "Cloud", "q": [10.5, 12.0]}
in_array = grade({"name": "Cloud", "q": [10.5, 99.9]}, ARR_G, ARR_S)
in_field = grade({"name": "Nope", "q": [10.5, 12.0]}, ARR_G, ARR_S)
report("a wrong value in a scalar array is unfound + invented item, not misread",
       (in_array["misread"], in_array["unfound"], in_array["invented_item"]) == (0, 1, 1),
       f"got {(in_array['misread'], in_array['unfound'], in_array['invented_item'])}")
report("a wrong value in a named field is misread",
       (in_field["misread"], in_field["unfound"], in_field["invented_item"]) == (1, 0, 0),
       f"got {(in_field['misread'], in_field['unfound'], in_field['invented_item'])}")
report("so the array error costs more, because it is charged on both sides",
       in_array["accuracy"] < in_field["accuracy"] - 1e-9,
       f"array {in_array['accuracy']:.1f} vs field {in_field['accuracy']:.1f}")
note(f"array {in_array['accuracy']:.1f} (denominator {in_array['total']}), "
     f"field {in_field['accuracy']:.1f} (denominator {in_field['total']})")

# What order-freedom buys, and what it costs. Both directions are pinned, because the cost
# looks like a defect on its own and only reads correctly next to the compensation.
ARR_S2 = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
FLD_S2 = {"properties": {k: {"type": "string"} for k in ("t1", "t2", "t3")}}
A_G, F_G = {"tags": ["a", "x", "c"]}, {"t1": "a", "t2": "x", "t3": "c"}
report("reordering is free for the array and fatal for named fields",
       abs(grade({"tags": ["c", "a", "x"]}, A_G, ARR_S2)["accuracy"] - 100) < 1e-9
       and grade({"t1": "c", "t2": "a", "t3": "x"}, F_G, FLD_S2)["accuracy"] < 1e-9)
report("...which is what the harsher value error pays for",
       grade({"tags": ["a", "W", "c"]}, A_G, ARR_S2)["accuracy"]
       < grade({"t1": "a", "t2": "W", "t3": "c"}, F_G, FLD_S2)["accuracy"] - 1e-9)
report("order_matters buys cell identity back",
       abs(grade({"tags": ["a", "W", "c"]}, A_G, ARR_S2,
                 order_matters=["tags"])["accuracy"] - 200 / 3) < 1e-6)
report("omission and invention cost the same either way",
       abs(grade({"tags": ["a", "c"]}, A_G, ARR_S2)["accuracy"]
           - grade({"t1": "a", "t3": "c"}, F_G, FLD_S2)["accuracy"]) < 1e-9
       and abs(grade({"tags": ["a", "x", "c", "Z"]}, A_G, ARR_S2)["accuracy"]
               - grade({"t1": "a", "t2": "x", "t3": "c", "t4": "Z"},
                       F_G, FLD_S2)["accuracy"]) < 1e-9)
note("so the two shapes differ ONLY on order and on localising a wrong value")

print("\nTHE IDENTITIES HOLD OVER GENERATED DOCUMENTS, NOT JUST THE EXAMPLE")
import copy                                                                 # noqa: E402
import random                                                               # noqa: E402


def _schema_from(doc):
    """The schema gold conforms to. Derived from GOLD only, so a key the prediction
    invented genuinely is not a slot the model was offered."""
    if isinstance(doc, dict):
        return {"type": "object", "properties": {k: _schema_from(v) for k, v in doc.items()}}
    if isinstance(doc, list):
        merged = {}
        for e in doc:
            if isinstance(e, dict):
                merged.update(e)
        return {"type": "array", "items": _schema_from(merged) if merged else {}}
    return {}


rnd = random.Random(77)


def scalar():
    return rnd.choice([1, 2.5, "a", "B", None, "2024-10-31", "", True])


def rand_doc(d=0):
    o = {}
    for i in range(rnd.randint(1, 4)):
        r = rnd.random()
        if d < 2 and r < .35:  o[f"f{i}"] = [rand_doc(d + 1) for _ in range(rnd.randint(0, 3))]
        elif r < .5:           o[f"f{i}"] = [scalar() for _ in range(rnd.randint(0, 3))]
        elif d < 2 and r < .6: o[f"f{i}"] = rand_doc(d + 1)
        else:                  o[f"f{i}"] = scalar()
    return o


def perturb(node):
    node = copy.deepcopy(node)

    def walk(x):
        if isinstance(x, dict):
            for k in list(x):
                if rnd.random() < .2:    x[k] = scalar()
                elif rnd.random() < .1:  del x[k]
                else:                    walk(x[k])
            if rnd.random() < .2:        x[f"z{rnd.randint(0, 9)}"] = scalar()
        elif isinstance(x, list):
            if rnd.random() < .4: rnd.shuffle(x)
            if rnd.random() < .2: x.append(scalar() if not x or not isinstance(x[0], dict)
                                           else rand_doc(2))
            for e in x: walk(e)
    walk(node)
    return node


bad = seen = fired = 0
kinds = collections.Counter()
for _ in range(600):
    g = rand_doc()
    sch = _schema_from(g)
    if not isinstance(sch, dict) or not sch:
        continue
    for pr in (copy.deepcopy(g), perturb(g), rand_doc(), {}):
        try:
            r = grade(pr, g, sch)
        except TypeError:
            continue
        seen += 1
        parts = (r["matched"], r["misread"], r["unfound"], r["fabricated"],
                 r["invented_item"], r["invented_field"])
        for k in ("fabricated", "invented_item", "invented_field", "misread", "unfound"):
            kinds[k] += r[k]
        ok = (r["asserted"] == r["matched"] + r["misread"] + r["fabricated"]
              + r["invented_item"] + r["invented_field"]
              and r["total"] == sum(parts)
              and (not r["asserted"]
                   or abs(r["precision"] - r["matched"] / r["asserted"]) < 1e-12)
              and (not (r["matched"] + r["misread"] + r["unfound"])
                   or abs(r["recall"] - r["matched"]
                          / (r["matched"] + r["misread"] + r["unfound"])) < 1e-12))
        hist = collections.Counter(v.verdict for v in explain(pr, g, sch))
        ok &= parts == (hist["match"], hist["wrong value"], hist["missing"],
                        hist["fabricated"], hist["invented item"], hist["invented field"])
        bad += not ok
        fired += any(parts[3:])
report(f"both identities and the histogram hold over {seen} generated gradings",
       bad == 0, f"{bad} violations")
report("and the generator actually produced false assertions to check",
       fired > seen // 10 and all(kinds[k] for k in
                                  ("fabricated", "invented_field", "misread", "unfound")),
       f"{fired} gradings with an extra address; totals {dict(kinds)}")

print(f"\n{'FALSE ASSERTION SPLIT HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
