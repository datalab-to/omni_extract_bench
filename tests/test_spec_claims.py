#!/usr/bin/env python3
"""Claims METRIC_SPEC.md makes that no other test covers.

Sections 5 and 8 have their own files (`test_spec_null_semantics`, `test_incentives`), and
section 4's buckets are covered by `test_false_assertions`. This file takes the rest: the
budget figures in section 3, and the formulas in section 4.

The point is that the spec should not be able to drift away from the scorer silently.

Run: python3 tests/test_spec_claims.py
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench import matching as OM                              # noqa: E402
from omni_extract_bench.score import explain, grade                        # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


SCH = {"properties": {"n": {"type": "string"}, "t": {"type": "number"},
                      "d": {"type": ["number", "null"]},
                      "lines": {"type": "array", "items": {"properties": {
                          "sku": {"type": "string"}, "qty": {"type": "number"}}}}}}
GT = {"n": "INV", "t": 100.0, "d": None,
      "lines": [{"sku": "a", "qty": 1}, {"sku": "b", "qty": 2}]}
PRED = {"n": "INV", "t": 999.0, "d": 5.0, "z": "x",
        "lines": [{"sku": "a", "qty": 1}, {"sku": "zz", "qty": 9}]}
R = grade(PRED, GT, SCH)

# ═══════════════════════════════════════════════════════════════════════════════
print("\nSECTION 3 -- THE EXACTNESS BUDGET")
report("MAX_CELLS is 250 million, as the spec states",
       OM.MAX_CELLS == 250 * 10**6, f"got {OM.MAX_CELLS}")
report("MAX_EXACT is 20 000 on the smaller dimension",
       OM.MAX_EXACT == 20000, f"got {OM.MAX_EXACT}")
pct = 100 * 6881 ** 2 / OM.MAX_CELLS
report("the corpus's largest array (6881 rows) is ~19% of the cap, so it solves exactly",
       18.5 <= pct <= 19.5 and 6881 ** 2 <= OM.MAX_CELLS and 6881 <= OM.MAX_EXACT,
       f"6881^2 = {6881**2} = {pct:.1f}% of {OM.MAX_CELLS}")

print("\nSECTION 4 -- EVERY KEY THE SPEC NAMES EXISTS")
named = ["matched", "misread", "unfound", "fabricated", "invented_item", "invented_field",
         "asserted", "total", "accuracy", "precision", "recall", "f1", "found",
         "read_right", "gt_rows", "pred_rows", "matched_rows", "matching_exact",
         "approximated", "skipped_open_maps"]
report("all 20 reported keys are present, and nothing else is",
       sorted(R) == sorted(named),
       f"missing {sorted(set(named) - set(R))}, extra {sorted(set(R) - set(named))}")

print("\nSECTION 4 -- THE FORMULAS")
report("asserted = matched + misread + fabricated + invented_item + invented_field",
       R["asserted"] == R["matched"] + R["misread"] + R["fabricated"]
       + R["invented_item"] + R["invented_field"])
report("total = the six buckets summed",
       R["total"] == R["matched"] + R["misread"] + R["unfound"] + R["fabricated"]
       + R["invented_item"] + R["invented_field"])
report("accuracy = 100 * matched / total",
       abs(R["accuracy"] - 100 * R["matched"] / R["total"]) < 1e-12)
report("precision = matched / asserted",
       abs(R["precision"] - R["matched"] / R["asserted"]) < 1e-12)
report("recall = matched / (matched + misread + unfound)",
       abs(R["recall"] - R["matched"] / (R["matched"] + R["misread"] + R["unfound"])) < 1e-12)
report("f1 = harmonic mean of precision and recall",
       abs(R["f1"] - 2 * R["precision"] * R["recall"]
           / (R["precision"] + R["recall"])) < 1e-12)
report("found = (matched + misread) / total",
       abs(R["found"] - (R["matched"] + R["misread"]) / R["total"]) < 1e-12)
report("read_right = matched / (matched + misread)",
       abs(R["read_right"] - R["matched"] / (R["matched"] + R["misread"])) < 1e-12)
report("found * read_right = accuracy / 100",
       abs(R["found"] * R["read_right"] - R["accuracy"] / 100) < 1e-12)

print("\nSECTION 4 -- WHEN accuracy AND f1 AGREE")
# The section used to say misread appearing twice was "the entire reason accuracy and f1
# differ". It is not: they AGREE on a misread and diverge with misread = 0. What is counted
# twice in `gold + asserted` and once in `total` is matched + misread -- every address both
# documents use -- so they agree exactly when the documents use the same set of addresses.
AF_S = {"properties": {k: {"type": "number"} for k in "abc"}}
rows = {
    "perfect":            ({"a": 1, "b": 2},         {"a": 1, "b": 2}, 100.00, 100.00),
    "one value misread":  ({"a": 1, "b": 99},        {"a": 1, "b": 2},  50.00,  50.00),
    "one value missing":  ({"a": 1},                 {"a": 1, "b": 2},  50.00,  66.67),
    "one value invented": ({"a": 1, "b": 2, "c": 9}, {"a": 1, "b": 2},  66.67,  80.00),
}
for lbl, (pred, gold, want_acc, want_f1) in rows.items():
    r = grade(pred, gold, AF_S)
    report(f"{lbl}: accuracy {want_acc}, f1 {want_f1}",
           abs(round(r["accuracy"], 2) - want_acc) < 1e-9
           and abs(round(r["f1"] * 100, 2) - want_f1) < 1e-9,
           f"got acc {r['accuracy']:.2f}, f1 {r['f1']*100:.2f}")

agree = []
for lbl, (pred, gold, _a, _f) in rows.items():
    r = grade(pred, gold, AF_S)
    same_addresses = r["unfound"] == 0 and r["fabricated"] == 0 \
        and r["invented_item"] == 0 and r["invented_field"] == 0
    matches = abs(r["accuracy"] / 100 - r["f1"]) < 1e-12
    agree.append((lbl, same_addresses == matches))
report("they agree exactly when both documents use the same set of addresses",
       all(ok for _l, ok in agree), str(agree))
report("...and a misread is NOT what splits them",
       abs(grade({"a": 1, "b": 99}, {"a": 1, "b": 2}, AF_S)["accuracy"] / 100
           - grade({"a": 1, "b": 99}, {"a": 1, "b": 2}, AF_S)["f1"]) < 1e-12)

print("\nSECTION 3 -- CLEARING THE PAIRING BAR DOES NOT MAKE A ROW FREE")
BAR_S = {"properties": {"lines": {"type": "array", "items": {"properties": {
    "sku": {"type": "string"},
    "tags": {"type": "array", "items": {"type": "string"}}}}}}}
BAR_G = {"lines": [{"sku": "a", "tags": ["t1", "t2"]}]}
omitted = grade({"lines": []}, BAR_G, BAR_S)
cleared = grade({"lines": [{"sku": "a", "tags": ["X", "Y"]}]}, BAR_G, BAR_S)
report("a row that clears the bar can still enlarge the denominator",
       cleared["total"] > omitted["total"],
       f"omitted {omitted['total']}, cleared {cleared['total']}")
report("...so 'charged once' would be wrong: it is charged for what it invents too",
       cleared["invented_item"] > 0, f"invented_item {cleared['invented_item']}")

print("\nSECTION 4 -- THE BUCKETS PARTITION EVERY ADDRESS (P16)")
verdicts = [v for v in explain(PRED, GT, SCH) if not v.verdict.startswith("skipped")]
report("every address gets exactly one of the six verdicts",
       len(verdicts) == R["total"]
       and {v.verdict for v in verdicts} <= {"match", "wrong value", "missing", "fabricated",
                                             "invented item", "invented field"},
       f"{len(verdicts)} verdicts vs total {R['total']}: "
       f"{sorted({v.verdict for v in verdicts})}")

print("\nSECTION 4 -- matched_rows IS NOT ROW CORRECTNESS")
CUR_S = {"properties": {"lines": {"type": "array", "items": {"properties": {
    "cur": {"type": "string"}, "sku": {"type": "string"}}}}}}
cur = grade({"lines": [{"cur": "USD", "sku": "q"}, {"cur": "USD", "sku": "r"}]},
            {"lines": [{"cur": "USD", "sku": "x"}, {"cur": "USD", "sku": "y"}]}, CUR_S)
report("a value repeated on every row pairs rows whose content is entirely wrong",
       cur["matched_rows"] == 2 and cur["accuracy"] == 50.0,
       f"matched_rows {cur['matched_rows']}, accuracy {cur['accuracy']}")

print("\nSECTION 9 -- P18 AND P19")
try:
    grade(PRED, GT, SCH, order_matters=["nope"])
    report("P18: an order_matters name fitting no array raises", False, "no error")
except ValueError:
    report("P18: an order_matters name fitting no array raises", True)
try:
    grade(PRED, GT, None)
    report("P19: a grade without a schema raises", False, "no error")
except TypeError:
    report("P19: a grade without a schema raises", True)

print(f"\n{'SPEC CLAIMS HOLD' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
