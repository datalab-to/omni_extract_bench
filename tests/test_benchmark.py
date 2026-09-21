#!/usr/bin/env python3
"""The benchmark runner's decisions, checked without a network or a vendor.

Three of them are worth pinning, because each is a rule you cannot see is wrong by reading a
result: WHICH documents get re-attempted on a resume, WHAT happens to a document the vendor
could not answer, and HOW the corpus number is aggregated. A mistake in the first spends money
or loses a document silently; a mistake in the last is a wrong headline.

`fetch_manifest` and `fetch_documents` are not covered: they are a HuggingFace download, and a
test that asserts one is testing `huggingface_hub`. Everything downstream of them is here.

Run: python3 tests/test_benchmark.py
"""
import json
import sys
import tempfile
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from omni_extract_bench import score                            # noqa: E402
from omni_extract_bench.benchmark import (                        # noqa: E402
    Doc, Run, needs_run, summarise)

FAILS = []


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


TMP = Path(tempfile.mkdtemp())

print("WHAT A RESUME SKIPS")
# A record means the vendor was called and answered. Whether a failure deserved another try is
# decided inside `predict`, with the status in hand; by the time a record exists the attempts
# are spent, and re-running would pay twice for an answer already bought.
for label, exists in (("no record at all", False), ("a record from a previous run", True)):
    path = TMP / f"{abs(hash(label))}.json"
    if exists:
        path.write_text(json.dumps({"result": {"a": 1}, "error": None}))
    report(f"re-attempt {label}: {not exists}", needs_run(path) is (not exists))
_failed = TMP / "failed.json"
_failed.write_text(json.dumps({"result": {"__error__": "x"},
                               "error": {"type": "VendorError", "transient": True}}))
report("a recorded failure is not re-attempted either, transient or not",
       needs_run(_failed) is False)

print("\nEVERY DOCUMENT COMES BACK, FAILURES INCLUDED")
# Coverage is only visible if a failure occupies a row. A failed row carries NULL metrics, not
# zero: a zero claims the model tried and missed every field, which is a different fact.
root, out = TMP / "corpus", TMP / "run"
(out / "predictions").mkdir(parents=True)
root.mkdir()
SCH = {"type": "object", "properties": {
    "id": {"type": "string"},
    "rows": {"type": "array", "items": {"type": "object", "properties": {
        "sku": {"type": "string"}, "qty": {"type": "integer"}}}}}}
GOLD = {"id": "A-1", "rows": [{"sku": "X", "qty": 1}, {"sku": "Y", "qty": 2}]}
WRITTEN = {
    "perfect":     ("micro1",   GOLD),
    "one-misread": ("micro1",   {"id": "A-1", "rows": [{"sku": "X", "qty": 1},
                                                       {"sku": "Z", "qty": 2}]}),
    "refused":     ("micro1",   {"__error__": "HTTP 400: schema too deep"}),
    "nothing":     ("internal", {}),
    "good":        ("internal", GOLD),
}
docs = []
for doc_id, (suite, pred) in WRITTEN.items():
    (root / f"{doc_id}.json").write_text(json.dumps(GOLD))
    (out / "predictions" / f"{doc_id}.json").write_text(json.dumps(pred))
    docs.append(Doc(doc_id, suite, root / f"{doc_id}.pdf", root / f"{doc_id}.json", SCH))
docs.append(Doc("never-ran", "internal", root / "x.pdf", root / "perfect.json", SCH))

rows = Run("vendor", "vendor", {}, out).score(docs)
by_id = {r["doc_id"]: r for r in rows}
report("one row per document, including the one never predicted", len(rows) == 6, str(len(rows)))
report("a correct prediction scores 1.0", by_id["perfect"]["accuracy"] == 1.0)
report("a misread row is charged, not fatal", by_id["one-misread"]["accuracy"] == 0.8,
       str(by_id["one-misread"]["accuracy"]))
for doc_id in ("refused", "nothing", "never-ran"):
    row = by_id[doc_id]
    report(f"{doc_id}: status=error with NULL metrics, and a reason",
           row["status"] == "error" and "accuracy" not in row and row.get("error"), str(row))
report("scores.jsonl has a line per document",
       sum(1 for _ in (out / "scores.jsonl").open()) == 6)

print("\nA SCHEMA THE SCORER CAN READ")
# 257 of the 620 benchmark documents declare their row type in `$defs` and reference it with
# `$ref`. `score` REFUSES such a schema -- an additionalProperties object behind a ref would be
# graded while the same object written inline is skipped, so the two spellings would disagree.
# Without resolving first, 41% of the corpus scored `status=error` while predicting perfectly:
# the run log said `ok`, and coverage said 0.
REF_SCHEMA = {
    "$defs": {"Row": {"type": "object", "properties": {"sku": {"type": "string"},
                                                       "qty": {"type": "integer"}}}},
    "type": "object",
    "properties": {"id": {"type": "string", "evaluation_config": {"tol": 1}},
                   "rows": {"type": "array", "items": {"$ref": "#/$defs/Row"}}},
}
REF_GOLD = {"id": "A-1", "rows": [{"sku": "X", "qty": 1}, {"sku": "Y", "qty": 2}]}
ref_root, ref_out = TMP / "refcorpus", TMP / "refrun"
(ref_out / "predictions").mkdir(parents=True)
ref_root.mkdir()
(ref_root / "r.json").write_text(json.dumps(REF_GOLD))
(ref_out / "predictions" / "r.json").write_text(json.dumps(REF_GOLD))
ref_rows = Run("v", "v", {}, ref_out).score(
    [Doc("r", "extractbench", ref_root / "r.pdf", ref_root / "r.json", REF_SCHEMA)])
report("a $ref schema scores instead of erroring", ref_rows[0]["status"] == "scored",
       str(ref_rows[0].get("error")))
report("...and a perfect prediction scores 1.0", ref_rows[0].get("accuracy") == 1.0)
report("benchmark-only annotations are not gradable slots either way",
       score({"id": "A-1"}, {"id": "A-1"}, REF_SCHEMA)["accuracy"] == 1.0)
report("the caller's schema is not mutated by preparing it",
       "$defs" in REF_SCHEMA and "$ref" in json.dumps(REF_SCHEMA))

print("\nEVERY DOCUMENT COUNTS ONCE, WHATEVER IT WEIGHS")
# A MEAN OVER DOCUMENTS, not a ratio of sums. The corpus runs from 26 addresses to 35,239, so
# summing the counts and dividing lets one document decide the number for all of them --
# measured on a real six-document run, 0.8076 against 0.9733.
def scored(suite, matched, total, **rest):
    misread = rest.pop("misread", total - matched)
    r = {"suite": suite, "status": "scored", "total": total, "matched": matched,
         "misread": misread, "unfound": 0, "fabricated": 0, "invented_item": 0,
         "invented_field": 0}
    r.update(rest)
    r.setdefault("accuracy", matched / total)
    r.setdefault("precision", matched / total)
    r.setdefault("recall", matched / total)
    return r


lopsided = [scored("s", 9, 10), scored("s", 500, 1000)]
o = summarise(lopsided)
report("a small document is worth as much as a huge one",
       abs(o["accuracy"] - 0.7) < 1e-9, f'{o["accuracy"]} -- a ratio of sums gives 0.504')

print("\nAND NO CORPUS NUMBER IS PICKED FOR YOU")
# Weighting the suites equally rather than by size is a real choice (METRIC_SPEC section 7),
# and `summarise` does not make it. `per_suite` carries the SAME block, so the suite-weighted
# figure is a mean the caller takes -- of any of these, not just accuracy.
skew = ([scored("big", 9, 10) for _ in range(90)]
        + [scored("small", 1, 10) for _ in range(10)])
s = summarise(skew)
report("the top-level figure follows the large suite", abs(s["accuracy"] - 0.82) < 1e-9,
       f'{s["accuracy"]}')
unified = sum(x["accuracy"] for x in s["per_suite"].values()) / len(s["per_suite"])
report("...and one line over per_suite weights the suites equally instead",
       abs(unified - 0.5) < 1e-9, f"{unified}")
report("every suite carries the whole block, so that works for any of them",
       set(s["per_suite"]["big"]) == set(s) - {"per_suite"}, str(sorted(s["per_suite"]["big"])))

print("\nWHERE IT WENT WRONG, NOT ONLY HOW MUCH")
# Two vendors at the same accuracy are not the same vendor when one is missing fields and the
# other is inventing them. Rates, and named `_rate`: scores.jsonl spells these same five words
# as counts.
kinds = summarise([scored("s", 5, 10, misread=0, unfound=5),
                   scored("s", 5, 10, misread=0, fabricated=5)])
report("the error rates are means of per-document rates",
       (kinds["unfound_rate"], kinds["fabricated_rate"]) == (0.25, 0.25), str(kinds))
report("...and the ones that did not happen read zero",
       kinds["misread_rate"] == 0.0 and kinds["invented_field_rate"] == 0.0)

print("\nA FAILURE COSTS COVERAGE, NOT A FALSE PRECISION")
# A document with no usable prediction asserted nothing, so scoring its `matched / asserted`
# zero would say everything it claimed was wrong. The block is over what scored; `coverage` is
# what the failures cost, and `accuracy * coverage` is the mean over every document.
half = ([scored("s", 10, 10) for _ in range(5)]
        + [{"suite": "s", "status": "error", "error": "timeout"} for _ in range(5)])
h = summarise(half)
report("a failed document does not drag precision down to nothing", h["precision"] == 1.0)
report("...it shows up as coverage", h["coverage"] == 0.5)
report("...and the flat mean over every document is still one multiplication away",
       abs(h["accuracy"] * h["coverage"] - 0.5) < 1e-9)
report("an empty run does not divide by zero",
       summarise([])["accuracy"] == 0.0 and summarise([])["coverage"] == 0.0
       and summarise([])["unfound_rate"] == 0.0)

print("\nA MISSING SDK STOPS THE PROVIDER AND STORES NOTHING")
# The failure mode this prevents: every document gets `{"__error__": "ModuleNotFoundError..."}`
# written, no resume re-attempts any of them because that is not transient, and the provider
# reports 0% coverage forever over one missing install.
from omni_extract_bench import benchmark as _bench                         # noqa: E402
from omni_extract_bench.harness import vendor as _vendor                   # noqa: E402
from omni_extract_bench.harness.extraction import MissingDependency        # noqa: E402

_pdf = TMP / "doc.pdf"
_pdf.write_bytes(b"%PDF-1.4\n")
_out = TMP / "sdk-run"
_docs = [Doc(f"d{i}", "s", _pdf, root / "perfect.json", SCH) for i in range(6)]
_saved = _vendor.adapter
_vendor.adapter = lambda provider: (_ for _ in ()).throw(
    MissingDependency("the datalab adapter could not import what it needs"))
try:
    _bench.ProviderRun("datalab",
                       [_bench.Run("datalab", "datalab", {}, _out)], 3
                       ).predict(_docs, timeout=30)
    report("a missing SDK stops the run", False, "it returned normally")
except MissingDependency:
    report("a missing SDK stops the run", True)
finally:
    _vendor.adapter = _saved
report("nothing is written for it",
       not list((_out / "predictions").glob("*.json"))
       and not list((_out / "records").glob("*.json")),
       "a prediction was stored, so a resume would skip it forever")
report("...so every document stays resumable",
       all(needs_run(_out / "records" / f"{d.doc_id}.json") for d in _docs))

print("\nA MISSING EXTRA SAYS WHAT TO INSTALL")
# `[benchmark]` is an extra, so the common first run is one where it is absent. That must read
# as something to install and not as a traceback -- `cli.py` lets ImportError through its error
# boundary for exactly this. Setting a module to None in sys.modules makes importing it raise,
# so this holds whether or not the extra is actually installed here.
import unittest.mock                                                       # noqa: E402

from omni_extract_bench import benchmark                                   # noqa: E402

for missing, call, wanted in [
    ("huggingface_hub", lambda: benchmark.fetch(TMP / "corpus"), "huggingface_hub"),
    ("pyarrow.parquet", lambda: benchmark.read_manifest(TMP / "m.parquet", TMP), "pyarrow"),
]:
    with unittest.mock.patch.dict(sys.modules, {missing: None}):
        try:
            call()
            report(f"{missing} missing is reported", False, "it did not raise")
        except ImportError as exc:
            text = str(exc)
            report(f"{missing} missing raises ImportError", True)
            report(f"...naming {wanted} and the install command",
                   wanted in text and "omni-extract-bench[benchmark]" in text, text)
        except Exception as exc:                                    # noqa: BLE001
            report(f"{missing} missing raises ImportError", False,
                   f"raised {type(exc).__name__} instead: {exc}")

print(f"\n{'THE RUNNER HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
sys.exit(1 if FAILS else 0)
