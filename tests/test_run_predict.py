#!/usr/bin/env python3
"""The predict runner, without calling a vendor.

Only `extract_one` is stubbed -- the dispatch to a vendor adapter, which is the one part that
needs an API key and costs money. Everything around it is the real thing: the manifest checks,
the per-row accounting, what gets written where, and the claim the module makes about itself,
that its output table is a manifest the scorer can read.

That last one is the test worth having. `predict` writes `pred_status` and `pred_error` rather
than `status` and `error` for exactly this reason, and nothing but an end-to-end run would
notice if that stopped being true.

Run: python3 tests/test_run_predict.py
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

try:
    from omni_extract_bench.harness import run_provider
except ImportError as exc:  # the vendor adapters' HTTP clients are the `harness` extra
    print(f"  SKIP  needs the harness extra ({exc})")
    sys.exit(0)

from omni_extract_bench import run_predict as rp  # noqa: E402
from omni_extract_bench import run_score as rs  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


SCHEMA = {"type": "object", "properties": {"n": {"type": "string"}}}
GT = {"n": "one"}
TMP = Path(tempfile.mkdtemp(prefix="oeb-predict-"))
PROVIDER = sorted(run_provider.PROVIDERS)[0]


def stub(result):
    """Stand in for the vendor call. `predict_row` imports `extract_one` when it runs, so
    replacing the module attribute is enough -- no import machinery to fight."""
    def extract_one(provider, pdf, schema, args):
        if isinstance(result, Exception):
            raise result
        return result
    run_provider.extract_one = extract_one


def manifest(rows, name):
    keys = {k: None for row in rows for k in row}
    path = TMP / name
    pq.write_table(pa.Table.from_pylist([{**keys, **row} for row in rows]), path)
    return str(path)


PDF = TMP / "doc.pdf"
PDF.write_bytes(b"%PDF-1.4 fake")


def row(doc_id, **extra):
    return {"doc_id": doc_id, "doc_path": str(PDF),
            "schema": json.dumps(SCHEMA).encode(), **extra}


print("[1] a prediction is written bare, and the row records where")
stub(GT)
out = str(TMP / "run1")
tally = rp.run(manifest([row("d1", suite="invoices")], "m1.parquet"), out, PROVIDER, jobs=1)
table = pq.read_table(f"{out}/manifest").to_pylist()
check("tally", tally == {"ok": 1, "error": 0}, str(tally))
check("the file holds the bare extraction, no envelope",
      json.loads(Path(table[0]["pred_path"]).read_text()) == GT)
check("status and latency are on the row",
      table[0]["pred_status"] == "ok" and table[0]["pred_latency_s"] is not None, str(table[0]))
check("unknown columns ride through", table[0]["suite"] == "invoices")
check("and so does the schema, because the scorer needs it", "schema" in table[0])
check("the provider is written onto the row, since the manifest no longer carries it",
      table[0]["provider"] == PROVIDER, str(table[0]))

print("\n[2] a vendor failure is a row with a reason, not a hole")
stub(RuntimeError("vendor exploded"))
out = str(TMP / "run2")
tally = rp.run(manifest([row("d1")], "m2.parquet"), out, PROVIDER, jobs=1)
table = pq.read_table(f"{out}/manifest").to_pylist()
check("tally", tally == {"ok": 0, "error": 1}, str(tally))
check("the reason is on the row", "vendor exploded" in (table[0]["pred_error"] or ""),
      str(table[0]))
check("and an __error__ blob is still written, so the document stays scoreable",
      "__error__" in json.loads(Path(table[0]["pred_path"]).read_text()))

print("\n[3] an unknown provider is refused up front, not per document")
stub(GT)
out = str(TMP / "run3")
try:
    rp.run(manifest([row("d1")], "m3.parquet"), out, "nope", jobs=1)
except ValueError as exc:
    check("refused before any document is sent", "unknown provider" in str(exc), str(exc))
else:
    check("refused before any document is sent", False, "no error raised")

print("\n[4] the manifest checks are the scorer's, so they behave the same way")
for name, rows, fragment in (
        ("no document column", [{"doc_id": "d", "schema": b"{}"}], "no doc_path column"),
        ("a column colliding with ours", [row("d", pred_status="ok")], "collide"),
        ("a repeated doc_id", [row("same"), row("same")], "duplicate doc_id")):
    try:
        rp.run(manifest(rows, f"bad-{len(FAILS)}-{name[:6]}.parquet"), str(TMP / "bad"), PROVIDER)
    except ValueError as exc:
        check(name, fragment in str(exc), f"wrong message: {exc}")
    else:
        check(name, False, "no error raised")

print("\n[5] the output table is a manifest the scorer reads -- the whole point of the shape")
# `gt_path` rides through predict untouched, so the table comes out carrying the ground
# truth, the prediction's path and the schema: everything `score` needs, nothing joined on.
stub(GT)
(TMP / "gt.json").write_text(json.dumps(GT))
out = str(TMP / "run5")
rp.run(manifest([row("d1", gt_path=str(TMP / "gt.json"))], "m5.parquet"), out,
       PROVIDER, jobs=1)
scored = str(TMP / "scored5")
tally = rs.run(f"{out}/manifest", scored)
rows = pq.read_table(f"{scored}/scores").to_pylist()
check("the predict output scores with no join", tally == {"scored": 1, "error": 0},
      str(tally))
check("perfectly, since the stub returned the ground truth", rows[0]["accuracy"] == 100.0)
check("and the prediction's own accounting sits beside the score",
      rows[0]["pred_status"] == "ok" and rows[0]["status"] == "scored", str(rows[0]))

print(f"\n{'ALL PREDICT TESTS PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
