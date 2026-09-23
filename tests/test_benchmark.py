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
    Doc, Run, needs_run, summarise)

FAILS = []


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


TMP = Path(tempfile.mkdtemp())

print("WHAT A RESUME SKIPS")
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

print("\nWHERE A MANIFEST IS: ON DISK, OR IN A HUGGINGFACE DATASET")
from omni_extract_bench.benchmark import (                                     # noqa: E402
    DEFAULT_MANIFEST, HfPath, hf_path)

_SUBSET = "hf://datasets/datalab-to/omni_extract_bench/manifests/sample40.parquet"
report("ours is an address like any other",
       hf_path(DEFAULT_MANIFEST)
       == HfPath("datalab-to/omni_extract_bench", None, "manifest.parquet"))
report("a manifest below the root keeps its whole path",
       hf_path(_SUBSET) == HfPath("datalab-to/omni_extract_bench", None,
                                  "manifests/sample40.parquet"))
report("@revision pins the dataset",
       hf_path("hf://datasets/o/n@v1.2/m.parquet") == HfPath("o/n", "v1.2", "m.parquet"))
for _addr in (DEFAULT_MANIFEST, _SUBSET, "hf://datasets/o/n@abc123/a/b.parquet"):
    report(f"an address prints back as written: {_addr}", str(hf_path(_addr)) == _addr)
for _local in ("manifest.parquet", "runs/../m.parquet", "/abs/m.parquet",
               Path("rel/m.parquet"), Path("/abs/m.parquet")):
    report(f"a path on disk is not an address: {_local!r}", hf_path(_local) is None)
for _bad in ("hf://models/o/n/m.parquet", "hf://spaces/o/n/m.parquet", "hf://datasets/o",
             "hf://datasets/o/n", "hf://datasets/o/n/", "hf://datasets//n/m.parquet",
             "hf://datasets/o/n@/m.parquet"):
    try:
        hf_path(_bad)
        report(f"a malformed address is refused: {_bad}", False, "it was accepted")
    except ValueError as exc:
        report(f"a malformed address is refused, and named: {_bad}",
               _bad in str(exc) and DEFAULT_MANIFEST in str(exc), str(exc))
# `Path("hf://...")` is `hf:/...`: a relative file that does not exist. Said at the address,
# not as "there is no manifest at hf:/datasets/..." after the fact.
try:
    hf_path(Path(_SUBSET))
    report("an address made a Path is refused", False, "it was accepted")
except ValueError as exc:
    report("an address made a Path is refused, with the fix", "as a str" in str(exc), str(exc))

print("\nA MANIFEST OF YOUR OWN, INSTEAD OF OURS")
import omni_extract_bench.benchmark as _bench                                  # noqa: E402
import pyarrow as _pa                                                          # noqa: E402
import pyarrow.parquet as _pq                                                  # noqa: E402

BYO_SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def byo(tmp, how):
    """Two documents and a manifest, with doc_path/gt_path written however `how` says."""
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
    path = corpus / "manifest.parquet"
    _pq.write_table(_pa.Table.from_pylist(rows), path)
    return path


_relative = lambda p, root: str(p.relative_to(root))                           # noqa: E731
_absolute = lambda p, root: str(p)                                             # noqa: E731

# A manifest of your own must need no HuggingFace access at all, so `fetch` may not run.
_real_fetch = _bench.fetch
_bench.fetch = lambda *_: (_ for _ in ()).throw(AssertionError("fetch ran"))
try:
    tmp = Path(tempfile.mkdtemp())
    docs = _bench.BenchmarkRun(["datalab"], manifest=byo(tmp, _relative)).corpus()
    report("a relative path resolves against the manifest's own directory",
           len(docs) == 2 and all(d.doc_path.exists() and d.gt_path.exists() for d in docs),
           str([str(d.doc_path) for d in docs]))
    report("...and the schema comes through parsed, per document",
           docs[0].schema == BYO_SCHEMA and docs[0].suite == "mine")

    tmp = Path(tempfile.mkdtemp())
    docs = _bench.BenchmarkRun(["datalab"], manifest=byo(tmp, _absolute)).corpus()
    report("an absolute path is used as it is, whatever the root",
           all(d.doc_path.exists() for d in docs), str([str(d.doc_path) for d in docs]))

    # The corpus moves after the manifest is written: relative paths need to be told where.
    tmp = Path(tempfile.mkdtemp())
    byo(tmp, _relative)
    moved = tmp / "moved"
    (tmp / "corpus").rename(moved)
    docs = _bench.BenchmarkRun(["datalab"], manifest=moved / "manifest.parquet",
                               data_root=moved).corpus()
    report("...and --data-root overrides what it is relative to",
           all(d.doc_path.exists() for d in docs))
finally:
    _bench.fetch = _real_fetch

# COLUMNS is written out rather than read off Doc._fields, so the two are free to drift. What
# may NOT drift is the direction a corpus on disk feels: a manifest carrying exactly COLUMNS
# and nothing else has to be enough to read, or the reader has grown a column nobody was told
# about and the KeyError is back.
_exact = Path(tempfile.mkdtemp()) / "m.parquet"
# "{}" is at once a valid doc_id, a usable path and valid JSON, so the row is built from
# COLUMNS itself: a column added to COLUMNS is covered here without editing this line.
_pq.write_table(_pa.Table.from_pylist([{c: "{}" for c in _bench.COLUMNS}]), _exact)
try:
    _only = _bench.read_manifest(_exact, _exact.parent)
    report("a manifest of exactly COLUMNS is enough to read",
           len(_only) == 1 and _only[0].schema == {})
except Exception as exc:                                                       # noqa: BLE001
    report("a manifest of exactly COLUMNS is enough to read", False,
           f"{type(exc).__name__}: {exc}")

# A manifest that IS there but is not one: named at the file, not as a KeyError with a column
# name in it and no file, and not as a bare JSONDecodeError over a corpus of thousands.
_short = Path(tempfile.mkdtemp()) / "m.parquet"
_pq.write_table(_pa.Table.from_pylist(
    [{"doc_id": "a", "doc_path": "x.pdf", "gt_path": "x.json", "schema": "{}"}]), _short)
try:
    _bench.read_manifest(_short, _short.parent)
    report("a manifest missing a column is refused", False, "it was accepted")
except ValueError as exc:
    report("a manifest missing a column names the column and the file",
           "suite" in str(exc) and str(_short) in str(exc), str(exc))

_bad_schema = Path(tempfile.mkdtemp()) / "m.parquet"
_pq.write_table(_pa.Table.from_pylist(
    [{"doc_id": "a", "suite": "s", "doc_path": "x.pdf", "gt_path": "x.json",
      "schema": "{not json"}]), _bad_schema)
try:
    _bench.read_manifest(_bad_schema, _bad_schema.parent)
    report("a schema column that is not JSON is refused", False, "it was accepted")
except ValueError as exc:
    report("...and an unreadable schema names the row it is in",
           "row 0" in str(exc) and "a" in str(exc), str(exc))

# A dataset with no manifest in it is not a corpus, and the corpus is a gigabyte: say so
# before the download, not after it.
import huggingface_hub as _hf                                                  # noqa: E402

_real_exists, _real_snapshot = _hf.file_exists, _hf.snapshot_download
_hf.file_exists = lambda *a, **k: False
_hf.snapshot_download = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("a gigabyte was downloaded before the manifest was asked for"))
_nowhere = "hf://datasets/someone/corpus@v2/manifests/nope.parquet"
try:
    _bench.fetch(hf_path(_nowhere))
    report("a dataset without the manifest is refused", False, "it was accepted")
except ValueError as exc:
    report("a dataset without the manifest is refused, before anything is downloaded",
           _nowhere in str(exc), str(exc))
# The check asks for the manifest that was addressed, at the revision that was addressed.
_asked = []
_hf.file_exists = lambda *a, **k: _asked.append((a, k)) or True
_hf.snapshot_download = lambda *a, **k: _asked.append((a, k)) or "/cached/corpus"
try:
    _bench.fetch(hf_path(_nowhere))
    report("the check and the download both ask for the addressed file and revision",
           _asked == [(("someone/corpus", "manifests/nope.parquet"),
                       {"repo_type": "dataset", "revision": "v2"}),
                      (("someone/corpus",), {"repo_type": "dataset", "revision": "v2"})],
           str(_asked))
    # A hub that will not answer -- offline, with the corpus already in the cache -- is not
    # an answer of no. The download decides, as it did before there was a check at all.
    _hf.file_exists = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    report("...but a hub that cannot be reached does not stop a cached corpus",
           _bench.fetch(hf_path(DEFAULT_MANIFEST)) == Path("/cached/corpus"))
finally:
    _hf.file_exists, _hf.snapshot_download = _real_exists, _real_snapshot

# A manifest in a dataset resolves against the dataset, wherever in it the manifest sits, and
# the commit it was read at is what `settings.json` records beside the address.
_snap = Path(tempfile.mkdtemp()) / "0123abcd"
(_snap / "manifests").mkdir(parents=True)
_rows = [{"doc_id": "d0", "suite": "s", "doc_path": "pdfs/d0.pdf", "gt_path": "gold/d0.json",
          "schema": json.dumps(BYO_SCHEMA)}]
_pq.write_table(_pa.Table.from_pylist(_rows), _snap / "manifests" / "sub.parquet")
_real_fetch, _bench.fetch = _bench.fetch, lambda *_: _snap
try:
    _sub = _bench.BenchmarkRun(
        ["datalab"], out=Path(tempfile.mkdtemp()),
        manifest="hf://datasets/o/n/manifests/sub.parquet")
    _docs = _sub.corpus()
    report("a manifest below a dataset's root reads paths from the dataset's root",
           _docs[0].doc_path == _snap / "pdfs" / "d0.pdf", str(_docs[0].doc_path))
    report("...and the commit it was read at is kept", _sub.corpus_revision == "0123abcd")
    _sub.prepare()
    _written = json.loads((_sub.runs[0].out / "settings.json").read_text())
    report("...and settings.json records it beside the address",
           _written["corpus"] == "hf://datasets/o/n/manifests/sub.parquet"
           and _written["corpus_revision"] == "0123abcd", str(_written))
finally:
    _bench.fetch = _real_fetch
try:
    _bench.BenchmarkRun(["datalab"], manifest=_SUBSET, data_root=Path("/elsewhere"))
    report("--data-root with a manifest in a dataset is refused", False, "it was accepted")
except ValueError as exc:
    report("--data-root with a manifest in a dataset is refused, naming both",
           "/elsewhere" in str(exc) and _SUBSET in str(exc), str(exc))
try:
    _bench.BenchmarkRun(["datalab"], manifest="hf://datasets/o/n")
    report("a malformed address is refused when the run is built", False, "it was accepted")
except ValueError:
    report("a malformed address is refused when the run is built, before anything is bought",
           True)

# A manifest that is not there is named, whether it is ours or one that was pointed at.
_gone = Path(tempfile.mkdtemp()) / "nope.parquet"
try:
    _bench.read_manifest(_gone, _gone.parent)
    report("a manifest that does not exist is refused", False, "it was accepted")
except ValueError as exc:
    report("a manifest that does not exist is named in the error",
           str(_gone) in str(exc) and "doc_id" in str(exc), str(exc))

# A directory is named for the provider and its options and never for the corpus, so nothing
# in the NAME keeps two corpora apart. `settings.json` records which one, and `prepare` reads
# it back -- otherwise `scores.jsonl` merges both by doc_id and `summary.json` averages them.
two = Path(tempfile.mkdtemp())
first = _bench.BenchmarkRun(["datalab"], out=two, manifest=Path("/corpora/a.parquet"))
first.prepare()
report("settings.json records which corpus the run measured",
       json.loads((first.runs[0].out / "settings.json").read_text())["corpus"]
       == "/corpora/a.parquet")
try:
    _bench.BenchmarkRun(["datalab"], out=two, manifest=Path("/corpora/b.parquet")).prepare()
    report("a second corpus in the same directory is refused", False, "it was accepted")
except ValueError as exc:
    report("a second corpus in the same directory is refused",
           "a.parquet" in str(exc) and "b.parquet" in str(exc), str(exc))
first.prepare()
report("...and the same corpus again is not", True)
report("our corpus names itself too, by its address",
       _bench.BenchmarkRun(["datalab"]).corpus_id == DEFAULT_MANIFEST)
report("...and a subset of it is a different corpus",
       _bench.BenchmarkRun(["datalab"], manifest=_SUBSET).corpus_id == _SUBSET)

# A doc_id IS a filename: `predictions/<doc_id>.json`, and the key a resume reads. Caught at
# the manifest, where the row can be named, not as a FileNotFoundError in a worker hours in.
bad = Path(tempfile.mkdtemp()) / "m.parquet"
_pq.write_table(_pa.Table.from_pylist(
    [{"doc_id": "a/b", "suite": "s", "doc_path": "x.pdf", "gt_path": "x.json",
      "schema": "{}"}]), bad)
try:
    _bench.read_manifest(bad, bad.parent)
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
o = summarise(lopsided)
report("a small document is worth as much as a huge one",
       abs(o["accuracy"] - 0.7) < 1e-9, f'{o["accuracy"]} -- a ratio of sums gives 0.504')

print("\nAND NO CORPUS NUMBER IS PICKED FOR YOU")
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
kinds = summarise([scored("s", 5, 10, misread=0, unfound=5),
                   scored("s", 5, 10, misread=0, fabricated=5)])
report("the error rates are means of per-document rates",
       (kinds["unfound_rate"], kinds["fabricated_rate"]) == (0.25, 0.25), str(kinds))
report("...and the ones that did not happen read zero",
       kinds["misread_rate"] == 0.0 and kinds["invented_field_rate"] == 0.0)

print("\nA FAILURE COSTS COVERAGE, NOT A FALSE PRECISION")
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
import unittest.mock                                                       # noqa: E402

from omni_extract_bench import benchmark                                   # noqa: E402

for missing, call, wanted in [
    ("huggingface_hub", lambda: benchmark.fetch(benchmark.hf_path(benchmark.DEFAULT_MANIFEST)), "huggingface_hub"),
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
