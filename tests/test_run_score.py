#!/usr/bin/env python3
"""The manifest contract: what a row may say, and what comes back when it says it wrong.

These are about the table layer, not the metric -- `test_score_*` own the metric. What is
worth pinning here is everything that would otherwise be discovered halfway through a run:
that a failed row still occupies a row, that a carried column survives, that two parts of a
fan-out concatenate into one dataset, and that a manifest which cannot mean what it says is
refused before any work is done.

Run: python3 tests/test_run_score.py
"""
import json
import sys
import tempfile
from pathlib import Path

# run from anywhere: `python tests/x.py` puts tests/ on the path, not the repo root
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from omni_extract_bench import run_score as rs  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


def raises(name, fragment, fn):
    """A refused manifest raises a plain ValueError whose message is the explanation."""
    try:
        fn()
    except ValueError as exc:
        check(name, fragment in str(exc), f"wrong message: {exc}")
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"raised {type(exc).__name__}: {exc}")
    else:
        check(name, False, "no error raised")


SCHEMA = {"type": "object", "properties": {"n": {"type": "string"},
                                           "rows": {"type": "array", "items": {
                                               "type": "object",
                                               "properties": {"a": {"type": "string"}}}}}}
GT = {"n": "one", "rows": [{"a": "x"}, {"a": "y"}]}
TMP = Path(tempfile.mkdtemp(prefix="oeb-manifest-"))
SEQ = iter(range(10_000))


def written(obj, stem):
    """A JSON file, since gt and pred are always files."""
    path = TMP / f"{stem}-{next(SEQ)}.json"
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return str(path)


def row(doc_id, pred, gt=GT, schema=SCHEMA, **extra):
    """One manifest row: two paths and an inline schema, which is the whole contract."""
    return {"doc_id": doc_id, "gt_path": written(gt, "gt"), "pred_path": written(pred, "pred"),
            "schema": json.dumps(schema).encode(), **extra}


def manifest(rows, name):
    """Write a manifest. Every row is padded to the union of keys, because `from_pylist`
    takes its columns from the first record and a real manifest is a table."""
    keys = {k: None for r in rows for k in r}
    path = TMP / name
    pq.write_table(pa.Table.from_pylist([{**keys, **r} for r in rows]), path)
    return str(path)


def score(rows, name, **kw):
    """Run one manifest into its own output directory; return the scores rows."""
    out = str(TMP / f"out-{name}")
    tally = rs.run(manifest(rows, f"{name}.parquet"), out, **kw)
    return pq.read_table(f"{out}/scores.parquet").to_pylist(), tally, out


print("[1] the schema rides in the row, the documents are read from their paths")
rows, tally, _ = score([row("d1", GT)], "basic")
check("it scores", rows[0]["accuracy"] == 100.0 and tally["scored"] == 1, str(tally))
check("the paths stay on the row, so a score can say what it graded",
      rows[0]["gt_path"].endswith(".json") and rows[0]["pred_path"].endswith(".json"),
      str(rows[0]))
check("and so does the schema, so a run can be audited without its manifest",
      "schema" in rows[0], str(list(rows[0])))

print("\n[2] unknown columns ride through")
rows, _, _ = score([row("d1", GT, vendor="acme", suite="invoices", cost_usd=0.5)], "carry")
check("vendor/suite/cost carried",
      (rows[0]["vendor"], rows[0]["suite"], rows[0]["cost_usd"]) == ("acme", "invoices", 0.5),
      str(rows[0]))

print("\n[3] a failed row is still a row -- coverage is only visible if failures occupy rows")
rows, tally, _ = score([
    row("empty_dict", {}),
    row("error_blob", {"__error__": "context length exceeded"}),
    {"doc_id": "bad_gt", "gt_path": written("{not json", "gt"),
     "pred_path": written(GT, "pred"), "schema": json.dumps(SCHEMA).encode()},
    {"doc_id": "no_such_pred", "gt_path": written(GT, "gt"),
     "pred_path": str(TMP / "nope.json"), "schema": json.dumps(SCHEMA).encode()},
    row("fine", GT)], "failures")
by_id = {r["doc_id"]: r for r in rows}
check("nothing dropped", len(rows) == 5, str(len(rows)))
check("{} is an error, not a zero",
      by_id["empty_dict"]["status"] == "error" and by_id["empty_dict"]["accuracy"] is None,
      str(by_id["empty_dict"]))
check("an error blob keeps its reason",
      by_id["error_blob"]["error"] == "context length exceeded", str(by_id["error_blob"]))
check("unreadable ground truth is ours, not theirs", by_id["bad_gt"]["status"] == "error")
check("...and an error carries the whole traceback: row 400 of 620 cannot be reproduced by hand",
      "Traceback (most recent call last)" in (by_id["bad_gt"]["error"] or "")
      and "JSONDecodeError" in (by_id["bad_gt"]["error"] or ""),
      str(by_id["bad_gt"]["error"])[:80])
check("a missing prediction is an error too",
      by_id["no_such_pred"]["status"] == "error", str(by_id["no_such_pred"]))
check("tally", tally == {"scored": 1, "error": 4}, str(tally))

print("\n[4] verdicts join back on doc_id, and carry both raw sides")
# Two fields per row, so the second row still pairs on `a` when `b` is misread. With one
# field a wrong value shares nothing, and the row is unpairable rather than misread.
schema2 = {"type": "object", "properties": {"rows": {"type": "array", "items": {
    "type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}}}}
gt2 = {"rows": [{"a": "x", "b": "1"}, {"a": "y", "b": "2"}]}
pred2 = {"rows": [{"a": "x", "b": "1"}, {"a": "y", "b": "WRONG"}]}
rows, _, out = score([row("d1", pred2, gt=gt2, schema=schema2)], "verdicts.parquet")
verdicts = pq.read_table(f"{out}/verdicts.parquet").to_pylist()
misread = [v for v in verdicts if v["verdict"] == "misread"]
check("one misread at the right address",
      len(misread) == 1 and misread[0]["address"] == "rows[1].b", str(misread))
check("raw values are JSON-encoded, so None and \"None\" cannot collapse",
      (misread[0]["gold_raw"], misread[0]["pred_raw"]) == ('"2"', '"WRONG"'), str(misread[0]))
check("verdict rows agree with the summary count",
      sum(1 for v in verdicts if v["verdict"] == "matched") == rows[0]["matched"] == 3)
check("every verdict names its document", {v["doc_id"] for v in verdicts} == {"d1"})

print("\n[5] parts of a fan-out concatenate, including a batch with nothing to infer from")
rows, _, out = score([row("a", {}), row("b", {}), row("c", GT), row("d", GT)],
                     "parts", batch_size=2)
check("one part per batch", len(list(Path(out, "scores.parquet").iterdir())) == 2)
check("input order preserved", [r["doc_id"] for r in rows] == ["a", "b", "c", "d"], str(rows))
check("an all-failed part still concatenates with a scored one",
      [r["status"] for r in rows] == ["error", "error", "scored", "scored"])

print("\n[6] a manifest that cannot mean what it says is refused before any work")
raises("a missing column", "no pred_path column",
       lambda: score([{"doc_id": "d", "gt_path": written(GT, "gt"), "schema": b"{}"}], "e1"))
raises("a column colliding with ours", "collide",
       lambda: score([row("d", GT, accuracy=1.0)], "e2"))
raises("a repeated doc_id", "duplicate doc_id",
       lambda: score([row("same", GT), row("same", GT)], "e3"))

print("\n[7] a CSV manifest, where the schema cell is JSON text rather than bytes")
csv = TMP / "m.csv"
gt_path, pred_path = written(GT, "gt"), written(GT, "pred")
csv.write_text("doc_id,gt_path,pred_path,schema\n"
               f'd1,{gt_path},{pred_path},"{json.dumps(SCHEMA).replace(chr(34), chr(34) * 2)}"\n')
rs.run(str(csv), str(TMP / "out-csv"))
check("a CSV scores", pq.read_table(f"{TMP / 'out-csv'}/scores.parquet").to_pylist()[0]["accuracy"] == 100.0)

print("\n[8] an envelope is NOT unwrapped: a bare extraction is the contract")
rows, _, _ = score([row("d1", {"result": GT})], "envelope")
check("a {\"result\": ...} prediction is graded as written",
      rows[0]["accuracy"] == 0.0, str(rows[0]))

print("\n[9] a DIRECTORY of parts is a manifest too, which is what a predict run writes")
parts = TMP / "parts"
parts.mkdir()
for i, doc in enumerate(("p1", "p2")):
    pq.write_table(pa.Table.from_pylist([row(doc, GT)]), parts / f"part-{i:05d}.parquet")
rs.run(str(parts), str(TMP / "out-dir"))
rows = pq.read_table(f"{TMP / 'out-dir'}/scores.parquet").to_pylist()
check("both parts were read as one manifest",
      sorted(r["doc_id"] for r in rows) == ["p1", "p2"], str(rows))

print("\n[10] nothing is held that does not have to be")
# Verdicts are handed to pyarrow's dataset writer one document at a time rather than gathered
# per batch: a batch that happens to hold a few huge tables is otherwise gigabytes, on exactly
# the documents worth keeping. What is checked here is that streaming them loses nothing.
rows, _, out = score([row("a", GT), row("b", GT), row("c", GT), row("d", GT)],
                     "streamed", batch_size=2)
verdicts = pq.read_table(f"{out}/verdicts.parquet").to_pylist()
check("every document's addresses are there, across batches",
      sorted({v["doc_id"] for v in verdicts}) == ["a", "b", "c", "d"], str(verdicts[:2]))
check("and the counts agree with the scores table",
      len(verdicts) == sum(r["total"] for r in rows), f"{len(verdicts)} vs {[r['total'] for r in rows]}")

# `Executor.map` submits everything and holds finished results until the slowest earlier one
# catches up, which puts the memory ceiling back. The window is what keeps it flat.
live, peak = 0, 0


class Fake:
    def __init__(self, value):
        global live, peak
        live += 1
        peak = max(peak, live)
        self.value = value

    def result(self):
        global live
        live -= 1
        return self.value


got = list(rs.in_order(Fake, range(20), window=4))
check("in_order yields input order", got == list(range(20)), str(got))
check("with at most `window` in flight", peak <= 4, f"peak {peak}")

print("\n[11] a relative path is owned by the benchmark; an absolute one is external")
# The rule Delta Lake draws, and the one COCO and HuggingFace both assume: a manifest holds
# relative paths, the caller names the root. Inferring the root from where the manifest sits
# would break the moment predict's output becomes score's input.
check("relative is taken from the root", rs.resolve("gold/x.json", "/data/bench")
      == "/data/bench/gold/x.json")
check("absolute is left alone", rs.resolve("/elsewhere/x.json", "/data/bench")
      == "/elsewhere/x.json")
check("a URI is left alone", rs.resolve("s3://b/x.json", "/data/bench") == "s3://b/x.json")
check("no root means no change", rs.resolve("gold/x.json", ".") == "gold/x.json")

# and end to end: a manifest of relative paths, scored from somewhere else entirely
home = TMP / "rooted"
(home / "gold").mkdir(parents=True)
(home / "preds").mkdir()
(home / "gold" / "d.json").write_text(json.dumps(GT))
(home / "preds" / "d.json").write_text(json.dumps(GT))
pq.write_table(pa.Table.from_pylist([{"doc_id": "d", "gt_path": "gold/d.json",
                                      "pred_path": "preds/d.json",
                                      "schema": json.dumps(SCHEMA).encode()}]),
               home / "manifest.parquet")
# the manifest and the output are ordinary command-line paths; only what is INSIDE the
# manifest is taken from the root
tally = rs.run(str(home / "manifest.parquet"), str(TMP / "rooted-run"), root=str(home))
check("scored without being run from the dataset directory", tally["scored"] == 1, str(tally))
check("the output went where it was asked, not under the root",
      (TMP / "rooted-run" / "scores.parquet").exists() and not (home / "rooted-run").exists())

print("[12] an output directory holds one run")
# Parts are named by batch, so a second, smaller run into the same directory would overwrite
# the parts it reaches and leave the rest -- a table that is two runs interleaved and says so
# nowhere. This is the check that makes that impossible rather than merely unlikely.
_, _, once = score([row(f"m{i}", GT) for i in range(6)], "reuse", batch_size=2)
raises("a second run into the same --out is refused",
       "already holds a run", lambda: rs.run(manifest([row("m0", GT)], "reuse2.parquet"),
                                             once, batch_size=2))
rs.run(manifest([row("m0", GT)], "reuse3.parquet"), once, batch_size=2, overwrite=True)
after = pq.read_table(f"{once}/scores.parquet")
check("--overwrite replaces the run rather than merging with it", after.num_rows == 1,
      f"{after.num_rows} rows, so parts of the first run survived")
check("verdicts are replaced too",
      pq.read_table(f"{once}/verdicts.parquet").num_rows
      == len(set(pq.read_table(f"{once}/verdicts.parquet")["doc_id"].to_pylist())) * 3,
      "stale verdict parts left behind")
raises("jobs below 1 is refused by name, not by ZeroDivisionError",
       "at least 1", lambda: score([row("j", GT)], "jobs0", jobs=0))
raises("batch_size below 1 likewise", "at least 1",
       lambda: score([row("b", GT)], "bs0", batch_size=0))

print(f"\n{'ALL MANIFEST TESTS PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
