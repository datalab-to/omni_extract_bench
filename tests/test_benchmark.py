#!/usr/bin/env python3
"""The benchmark runner's decisions, checked without a network or a vendor.

Three of them are worth pinning, because each is a rule you cannot see is wrong by reading a
result: WHICH documents get re-attempted on a resume, WHAT happens to a document the vendor
could not answer, and HOW the corpus number is aggregated. A mistake in the first spends money
or loses a document silently; a mistake in the last is a wrong headline.

`fetch`'s download is not covered: it is a HuggingFace download, and a test that asserts one is
testing `huggingface_hub`. What it is asked for, and everything downstream of it, is here.

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
    Run, grade, needs, record)
from omni_extract_bench.scorers.field_matching import summarize                   # noqa: E402
from omni_extract_bench.data import Doc                            # noqa: E402

FAILS = []


def graded(run, docs, **kw):
    """`grade`, then the lines of `docs` it wrote, in their order: grade returns the summary."""
    from omni_extract_bench.benchmark import read_scores

    grade(run, docs, **kw)
    got = {x["doc_id"]: x for x in read_scores(run)}
    return [got[d.doc_id] for d in docs if d.doc_id in got]


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


TMP = Path(tempfile.mkdtemp())

print("WHAT A RESUME SKIPS")
_resume = Run("datalab", {}, TMP / "resume")
(_resume.records).mkdir(parents=True)
for label, exists in (("no record at all", False), ("a record from a previous run", True)):
    doc = Doc(str(abs(hash(label))), "s", TMP / "x.pdf", TMP / "g.json", {})
    if exists:
        (_resume.records / f"{doc.doc_id}.json").write_text(
            json.dumps({"result": {"a": 1}, "error": None}))
    report(f"re-attempt {label}: {not exists}", needs(_resume, doc) is (not exists))
(_resume.records / "failed.json").write_text(json.dumps(
    {"result": None, "error": {"type": "VendorError", "transient": True}}))
report("a recorded failure is not re-attempted either, transient or not",
       needs(_resume, Doc("failed", "s", TMP / "x.pdf", TMP / "g.json", {})) is False)
report("...unless retry_failed asks for one whose failure may not recur",
       needs(_resume, Doc("failed", "s", TMP / "x.pdf", TMP / "g.json", {}), retry_failed=True) is True)
(_resume.records / "refused.json").write_text(json.dumps(
    {"result": None, "error": {"type": "VendorError", "status": 400, "transient": False}}))
report("...and never one that was an answer, a 400",
       needs(_resume, Doc("refused", "s", TMP / "x.pdf", TMP / "g.json", {}), retry_failed=True) is False)

print("\nEVERY DOCUMENT COMES BACK, FAILURES INCLUDED")
root, out = TMP / "corpus", TMP / "run"
(out / "predictions").mkdir(parents=True)
(out / "records").mkdir()
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
    "nothing":     ("internal", {}),
    "good":        ("internal", GOLD),
}
docs = []
for doc_id, (suite, pred) in WRITTEN.items():
    (root / f"{doc_id}.json").write_text(json.dumps(GOLD))
    (out / "predictions" / f"{doc_id}.json").write_text(json.dumps(pred))
    docs.append(Doc(doc_id, suite, root / f"{doc_id}.pdf", root / f"{doc_id}.json", SCH))
(root / "refused.json").write_text(json.dumps(GOLD))
docs.append(Doc("refused", "micro1", root / "refused.pdf", root / "refused.json", SCH))
record(Run("datalab", {}, out), "refused", {"result": None, "error": {
    "type": "VendorError", "status": 400, "transient": False, "message": "schema too deep"}})
docs.append(Doc("never-ran", "internal", root / "x.pdf", root / "perfect.json", SCH))

summary_of_run = grade(Run("datalab", {}, out), docs)
rows = graded(Run("datalab", {}, out), docs)
by_id = {r["doc_id"]: r for r in rows}
report("one row per attempted document: the one never attempted is pending, not a row",
       len(rows) == 5 and "never-ran" not in by_id and summary_of_run["pending"] == 1, str(len(rows)))
report("a correct prediction scores 1.0", by_id["perfect"]["accuracy"] == 1.0)
report("a misread row is charged, not fatal", by_id["one-misread"]["accuracy"] == 0.8,
       str(by_id["one-misread"]["accuracy"]))
for doc_id in ("refused", "nothing"):
    row = by_id[doc_id]
    report(f"{doc_id}: status=error with NULL metrics, and a reason",
           row["status"] == "error" and "accuracy" not in row and row.get("error"), str(row))
report("a failed call writes no prediction: predictions/ holds only answers",
       not (out / "predictions" / "refused.json").exists())
report("...and its row carries the record's error",
       by_id["refused"]["error"] == "VendorError: schema too deep", str(by_id["refused"]))
report("scores.jsonl has a line per attempted document",
       sum(1 for _ in (out / "scores.jsonl").open()) == 5)
import shutil as _shutil                                                       # noqa: E402
brought = TMP / "brought"
_shutil.copytree(out / "predictions", brought / "predictions")
_theirs = {r["doc_id"]: r for r in graded(Run("datalab", {}, brought), docs)}
_mine = graded(Run("my-own-model", {}, brought), docs)
report("a folder of your own may name a provider there is no adapter for",
       len(_mine) == len(_theirs) and json.loads((brought / "summary.json").read_text())["settings"] == {})
report("a folder you bring keeps no records, so a document missing from it is failed, not pending",
       _theirs["never-ran"]["status"] == "error" and len(_theirs) == 6, str(_theirs.get("never-ran")))

print("\nA SCHEMA THE SCORER CAN READ")
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
ref_rows = graded(Run("datalab", {}, ref_out),
    [Doc("r", "extractbench", ref_root / "r.pdf", ref_root / "r.json", REF_SCHEMA)])
report("a $ref schema scores instead of erroring", ref_rows[0]["status"] == "scored",
       str(ref_rows[0].get("error")))
report("...and a perfect prediction scores 1.0", ref_rows[0].get("accuracy") == 1.0)
report("benchmark-only annotations are not gradable slots either way",
       score({"id": "A-1"}, {"id": "A-1"}, REF_SCHEMA)["accuracy"] == 1.0)
report("the caller's schema is not mutated by preparing it",
       "$defs" in REF_SCHEMA and "$ref" in json.dumps(REF_SCHEMA))

print("\nWHERE A DATASET IS: ON DISK, OR ON HUGGINGFACE")
from omni_extract_bench.data import DEFAULT_DATASET, _hf_dataset as hf_dataset, dataset_id  # noqa: E402

_SUBSET = "manifests/sample40.parquet"
report("ours is an address like any other",
       hf_dataset(DEFAULT_DATASET) == ("datalab-to/omni_extract_bench", None))
report("@revision pins the dataset", hf_dataset("hf://datasets/o/n@v1.2") == ("o/n", "v1.2"))
report("a dataset is itself whatever its revision",
       dataset_id("hf://datasets/o/n@v1.2") == dataset_id("hf://datasets/o/n") == "hf://datasets/o/n")
for _local in ("corpus", "runs/../corpus", "/abs/corpus", Path("rel/corpus"), Path("/abs/c")):
    report(f"a path on disk is not an address: {_local!r}", hf_dataset(_local) is None)
for _bad in ("hf://models/o/n", "hf://spaces/o/n", "hf://datasets/o", "hf://datasets/o/",
             "hf://datasets//n", "hf://datasets/o/n@", "hf://datasets/o/n/manifest.parquet"):
    try:
        hf_dataset(_bad)
        report(f"a malformed address is refused: {_bad}", False, "it was accepted")
    except ValueError as exc:
        report(f"a malformed address is refused, and named: {_bad}",
               _bad in str(exc) and DEFAULT_DATASET in str(exc), str(exc))
try:
    hf_dataset(Path(DEFAULT_DATASET))
    report("an address made a Path is refused", False, "it was accepted")
except ValueError as exc:
    report("an address made a Path is refused, with the fix", "as a str" in str(exc), str(exc))

print("\nA MANIFEST OF YOUR OWN, INSTEAD OF OURS")
import omni_extract_bench.benchmark as _bench                                  # noqa: E402
import omni_extract_bench.data as DS                                           # noqa: E402
import pyarrow as _pa                                                          # noqa: E402
import pyarrow.parquet as _pq                                                  # noqa: E402

BYO_SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def byo(tmp, how):
    """A dataset folder: two documents and a manifest, with doc_path/gt_path written however
    `how` says."""
    corpus = tmp / "corpus"
    (corpus / "pdfs").mkdir(parents=True)
    rows = []
    for i in range(2):
        pdf = corpus / "pdfs" / f"d{i}.pdf"
        pdf.write_bytes(b"%PDF")
        gt = corpus / f"d{i}.gt.json"
        gt.write_text(json.dumps({"a": "x"}))
        rows.append({"doc_id": f"d{i}", "suite": "mine", "doc_path": how(pdf, corpus),
                     "gt_path": how(gt, corpus), "schema": json.dumps(BYO_SCHEMA)})
    _pq.write_table(_pa.Table.from_pylist(rows), corpus / "manifest.parquet")
    return corpus


_relative = lambda p, root: str(p.relative_to(root))                           # noqa: E731
_absolute = lambda p, root: str(p)                                             # noqa: E731

# A dataset of your own must need no HuggingFace access at all, so the hub is made unimportable.
import unittest.mock as _mock                                                  # noqa: E402

with _mock.patch.dict(sys.modules, {"huggingface_hub": None}):
    tmp = Path(tempfile.mkdtemp())
    _, docs = DS.read_manifest(byo(tmp, _relative))
    report("a relative path resolves against the dataset's folder",
           len(docs) == 2 and all(d.doc_path.exists() and d.gt_path.exists() for d in docs),
           str([str(d.doc_path) for d in docs]))
    report("...and the schema comes through parsed, per document",
           docs[0].schema == BYO_SCHEMA and docs[0].suite == "mine")

    tmp = Path(tempfile.mkdtemp())
    _, docs = DS.read_manifest(byo(tmp, _absolute))
    report("an absolute path is used as it is, whatever the root",
           all(d.doc_path.exists() for d in docs), str([str(d.doc_path) for d in docs]))

    tmp = Path(tempfile.mkdtemp())
    _two = byo(tmp, _relative)
    (_two / "manifests").mkdir()
    (_two / "manifest.parquet").rename(_two / "manifests" / "sub.parquet")
    _, docs = DS.read_manifest(_two, "manifests/sub.parquet")
    report("...and a manifest below it still resolves against the dataset's folder",
           len(docs) == 2 and all(d.doc_path.exists() for d in docs))

_exact = Path(tempfile.mkdtemp()) / "m.parquet"
# "{}" is a valid doc_id, path and JSON, so the row is built from COLUMNS itself.
_pq.write_table(_pa.Table.from_pylist([{c: "{}" for c in DS.COLUMNS}]), _exact)
try:
    _, _only = DS.read_manifest(_exact.parent, _exact.name)
    report("a manifest of exactly COLUMNS is enough to read",
           len(_only) == 1 and _only[0].schema == {})
except Exception as exc:                                                       # noqa: BLE001
    report("a manifest of exactly COLUMNS is enough to read", False,
           f"{type(exc).__name__}: {exc}")

_extra = Path(tempfile.mkdtemp()) / "m.parquet"
_pq.write_table(_pa.Table.from_pylist([{**{c: "{}" for c in DS.COLUMNS}, "language": "tha"}]), _extra)
_, (_one,) = DS.read_manifest(_extra.parent, _extra.name)
report("a column of the dataset's own is carried on the document, not refused",
       dict(_one.extra) == {"language": "tha"}, str(dict(_one.extra)))
import pickle as _pickle                                                       # noqa: E402
report("...and it pickles, as grading's process pool needs", _pickle.loads(_pickle.dumps(_one)) == _one)

_short = Path(tempfile.mkdtemp()) / "m.parquet"
_pq.write_table(_pa.Table.from_pylist(
    [{"doc_id": "a", "doc_path": "x.pdf", "gt_path": "x.json", "schema": "{}"}]), _short)
try:
    DS.read_manifest(_short.parent, _short.name)
    report("a manifest missing a column is refused", False, "it was accepted")
except ValueError as exc:
    report("a manifest missing a column names the column and the file",
           "suite" in str(exc) and str(_short) in str(exc), str(exc))

_bad_schema = Path(tempfile.mkdtemp()) / "m.parquet"
_pq.write_table(_pa.Table.from_pylist(
    [{"doc_id": "a", "suite": "s", "doc_path": "x.pdf", "gt_path": "x.json",
      "schema": "{not json"}]), _bad_schema)
try:
    DS.read_manifest(_bad_schema.parent, _bad_schema.name)
    report("a schema column that is not JSON is refused", False, "it was accepted")
except ValueError as exc:
    report("...and an unreadable schema names the row it is in",
           "row 0" in str(exc) and "a" in str(exc), str(exc))

import huggingface_hub as _hf                                                  # noqa: E402

_real_exists, _real_snapshot = _hf.file_exists, _hf.snapshot_download
_hf.file_exists = lambda *a, **k: False
_hf.snapshot_download = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("a gigabyte was downloaded before the manifest was asked for"))
_nowhere = "hf://datasets/someone/corpus@v2"
try:
    DS.read_manifest(_nowhere, "manifests/nope.parquet")
    report("a dataset without the manifest is refused", False, "it was accepted")
except ValueError as exc:
    report("a dataset without the manifest is refused, before anything is downloaded",
           _nowhere in str(exc) and "manifests/nope.parquet" in str(exc), str(exc))
_asked = []
_hf.file_exists = lambda *a, **k: _asked.append((a, k)) or True
_hf.snapshot_download = lambda *a, **k: _asked.append((a, k)) or "/cached/corpus"
try:
    try:
        DS.read_manifest(_nowhere, "manifests/nope.parquet")
    except ValueError:
        pass                     # Note: /cached/corpus holds no manifest; only what was asked counts.
    report("the check asks for the manifest, and the download for the whole dataset, at the revision",
           _asked == [(("someone/corpus", "manifests/nope.parquet"),
                       {"repo_type": "dataset", "revision": "v2"}),
                      (("someone/corpus",), {"repo_type": "dataset", "revision": "v2"})],
           str(_asked))
    _hf.file_exists = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    DS._on_hub(DEFAULT_DATASET, DS.MANIFEST)
    report("...but a hub that cannot be reached does not stop a cached corpus",
           DS.download_dataset(DEFAULT_DATASET) == Path("/cached/corpus"))
finally:
    _hf.file_exists, _hf.snapshot_download = _real_exists, _real_snapshot

_snap = Path(tempfile.mkdtemp()) / "0123abcd"
(_snap / "manifests").mkdir(parents=True)
_rows = [{"doc_id": "d0", "suite": "s", "doc_path": "pdfs/d0.pdf", "gt_path": "gold/d0.json",
          "schema": json.dumps(BYO_SCHEMA)}]
_pq.write_table(_pa.Table.from_pylist(_rows), _snap / "manifests" / "sub.parquet")
_real_fetch, DS.download_dataset = DS.download_dataset, lambda *_: _snap
_real_hub, DS._on_hub = DS._on_hub, lambda *_: None
try:
    _sub = _bench.plan(["datalab"], out=Path(tempfile.mkdtemp()), dataset="hf://datasets/o/n",
                       manifest="manifests/sub.parquet")
    report("a manifest below a dataset's root reads paths from the dataset's root",
           _sub.docs[0].doc_path == _snap / "pdfs" / "d0.pdf", str(_sub.docs[0].doc_path))
    _bench.prepare(_sub.runs[0], _sub.dataset, _sub.timeout, _sub.manifest)
    _written = json.loads((_sub.runs[0].out / "settings.json").read_text())
    report("...and settings.json records it beside the dataset and manifest",
           _written["dataset"] == "hf://datasets/o/n" and _written["commit"] == "0123abcd"
           and _written["manifest"] == "manifests/sub.parquet", str(_written))
finally:
    DS.download_dataset, DS._on_hub = _real_fetch, _real_hub
try:
    _bench.plan(["datalab"], dataset="hf://datasets/o")
    report("a malformed address is refused when the plan is built", False, "it was accepted")
except ValueError:
    report("a malformed address is refused when the plan is built, before anything is bought",
           True)

_gone = Path(tempfile.mkdtemp()) / "nope.parquet"
try:
    DS.read_manifest(_gone.parent, _gone.name)
    report("a manifest that does not exist is refused", False, "it was accepted")
except ValueError as exc:
    report("a manifest that does not exist is named in the error",
           str(_gone) in str(exc) and "doc_id" in str(exc), str(exc))

two = Path(tempfile.mkdtemp())
_a, _b = byo(Path(tempfile.mkdtemp()), _relative), byo(Path(tempfile.mkdtemp()), _relative)
first = _bench.plan(["datalab"], out=two, dataset=_a)
_bench.prepare(first.runs[0], first.dataset, first.timeout)
report("settings.json records which dataset the run measured",
       json.loads((first.runs[0].out / "settings.json").read_text())["dataset"]
       == str(_a.resolve()))
try:
    _bench.plan(["datalab"], out=two, dataset=_b)
    report("a second dataset in the same directory is refused", False, "it was accepted")
except ValueError as exc:
    report("a second dataset in the same directory is refused, before anything is read",
           str(_a.resolve()) in str(exc) and str(_b.resolve()) in str(exc), str(exc))
_bench.plan(["datalab"], out=two, dataset=_a)
report("...and the same dataset again is not", True)

bad = Path(tempfile.mkdtemp()) / "m.parquet"
_pq.write_table(_pa.Table.from_pylist(
    [{"doc_id": "a/b", "suite": "s", "doc_path": "x.pdf", "gt_path": "x.json",
      "schema": "{}"}]), bad)
try:
    DS.read_manifest(bad.parent, bad.name)
    report("a doc_id that is not a filename is refused", False, "it was accepted")
except ValueError as exc:
    report("a doc_id that is not a filename is refused",
           "row 0" in str(exc) and "'a/b'" in str(exc), str(exc))

print("\nEVERY DOCUMENT COUNTS ONCE, WHATEVER IT WEIGHS")
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
o = summarize(lopsided)
report("a small document is worth as much as a huge one",
       abs(o["accuracy"] - 0.7) < 1e-9, f'{o["accuracy"]} -- a ratio of sums gives 0.504')

print("\nAND NO CORPUS NUMBER IS PICKED FOR YOU")
skew = ([scored("big", 9, 10) for _ in range(90)]
        + [scored("small", 1, 10) for _ in range(10)])
s = summarize(skew)
report("the top-level figure follows the large suite", abs(s["accuracy"] - 0.82) < 1e-9,
       f'{s["accuracy"]}')
unified = sum(x["accuracy"] for x in s["per_suite"].values()) / len(s["per_suite"])
report("...and one line over per_suite weights the suites equally instead",
       abs(unified - 0.5) < 1e-9, f"{unified}")
report("every suite carries the whole block, so that works for any of them",
       set(s["per_suite"]["big"]) == set(s) - {"per_suite"}, str(sorted(s["per_suite"]["big"])))

print("\nWHERE IT WENT WRONG, NOT ONLY HOW MUCH")
kinds = summarize([scored("s", 5, 10, misread=0, unfound=5),
                   scored("s", 5, 10, misread=0, fabricated=5)])
report("the error rates are means of per-document rates",
       (kinds["unfound_rate"], kinds["fabricated_rate"]) == (0.25, 0.25), str(kinds))
report("...and the ones that did not happen read zero",
       kinds["misread_rate"] == 0.0 and kinds["invented_field_rate"] == 0.0)

print("\nA FAILURE COSTS COVERAGE, NOT A FALSE PRECISION")
half = ([scored("s", 10, 10) for _ in range(5)]
        + [{"suite": "s", "status": "error", "error": "timeout"} for _ in range(5)])
h = summarize(half)
report("a failed document does not drag precision down to nothing", h["precision"] == 1.0)
report("...it shows up as coverage", h["coverage"] == 0.5)
report("...and the flat mean over every document is still one multiplication away",
       abs(h["accuracy"] * h["coverage"] - 0.5) < 1e-9)
report("an empty run does not divide by zero",
       summarize([])["accuracy"] == 0.0 and summarize([])["coverage"] == 0.0
       and summarize([])["unfound_rate"] == 0.0)

print("\nA MISSING SDK STOPS THE PROVIDER AND STORES NOTHING")
from omni_extract_bench import benchmark as _bench                         # noqa: E402
from omni_extract_bench.harness import registry as _vendor                   # noqa: E402
from omni_extract_bench.harness.errors import MissingDependency  # noqa: E402

_pdf = TMP / "doc.pdf"
_pdf.write_bytes(b"%PDF-1.4\n")
_out = TMP / "sdk-run"
_docs = [Doc(f"d{i}", "s", _pdf, root / "perfect.json", SCH) for i in range(6)]
_saved = _vendor.adapter
_vendor.adapter = lambda provider: (_ for _ in ()).throw(
    MissingDependency("the datalab adapter could not import what it needs"))
try:
    _bench.predict_all(_bench.Plan(
        runs=(_bench.Run("datalab", {}, _out),), docs=tuple(_docs), dataset=str(TMP),
        manifest="manifest.parquet", timeout=30, workers=3, score_workers=1,
        verdicts=True, score_only=False))
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
       all(needs(_bench.Run("datalab", {}, _out), d) for d in _docs))

print("\nA RUN NEVER MIXES SCORERS OR GOLDS")
_mix_out = Path(tempfile.mkdtemp())
_mix_set = byo(Path(tempfile.mkdtemp()), _relative)
_mix = _bench.plan(["datalab"], out=_mix_out, dataset=_mix_set, score_workers=1)
_run = _mix.runs[0]
(_run.predictions).mkdir(parents=True)
for d in _mix.docs:
    (_run.predictions / f"{d.doc_id}.json").write_text(json.dumps({"a": "x"}))
_docs = list(_mix.docs)
_rows = graded(_run, _docs, workers=1)
report("every line says which scorer graded it", all(r["scorer"] == _bench.SCORER for r in _rows), str(_rows))


def _regraded(**kw):
    """The documents grading scores again, of `_docs`."""
    seen = []
    real = _bench._score_one
    _bench._score_one = lambda doc, **k: seen.append(doc.doc_id) or real(doc, **k)
    try:
        grade(_run, _docs, workers=1, **kw)
    finally:
        _bench._score_one = real
    return seen


report("every grading grades every document again, with no vendor call: grading is fresh",
       _regraded() == [d.doc_id for d in _docs] and _regraded() == [d.doc_id for d in _docs])
_docs[0].gt_path.write_text(json.dumps({"a": "y"}))
report("so a changed gold reaches the next grading",
       graded(_run, _docs, workers=1)[0]["accuracy"] == 0.0)

print("\nA MISSING EXTRA SAYS WHAT TO INSTALL")
import unittest.mock                                                       # noqa: E402

from omni_extract_bench import benchmark                                   # noqa: E402

for missing, call, wanted in [
    ("huggingface_hub", lambda: DS.read_manifest(DS.DEFAULT_DATASET), "huggingface_hub"),
    ("pyarrow.parquet", lambda: DS.read_manifest(TMP, "m.parquet"), "pyarrow"),
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
