#!/usr/bin/env python3
"""The site's filenames.

A doc_id is whatever the manifest said. The site names files after it, so an id that is not a
filename -- one with a slash, or with `..` in it -- decides where those files land. That is the
one thing in `ui` worth pinning: everything else it does is read off the two tables.

Run: python3 tests/test_ui.py
"""
import json
import sys
import tempfile
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from omni_extract_bench import run_score as rs, ui  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


SCHEMA = {"type": "object", "properties": {"n": {"type": "string"}}}
TMP = Path(tempfile.mkdtemp(prefix="oeb-ui-"))
(TMP / "gold").mkdir()
(TMP / "gold" / "d.json").write_text(json.dumps({"n": "one"}))
(TMP / "gold" / "p.json").write_text(json.dumps({"n": "two"}))

print("[1] a doc_id that is not a filename still lands inside the site")
for n, doc_id in enumerate(["../escape", "a/b", "..", "", "café", "plain-id"]):
    run, site = TMP / f"run{n}", TMP / f"site{n}"
    pq.write_table(pa.Table.from_pylist([{"doc_id": doc_id,
                                          "gt_path": str(TMP / "gold/d.json"),
                                          "pred_path": str(TMP / "gold/p.json"),
                                          "schema": json.dumps(SCHEMA).encode()}]),
                   TMP / f"m{n}.parquet")
    rs.run(str(TMP / f"m{n}.parquet"), str(run))
    ui.build(ui.runs_from([str(run)]), site)

    made = [p.relative_to(site) for p in site.rglob("*") if p.is_file()]
    inside = all(not str(p).startswith("..") for p in made)
    named = {p.parent.name for p in made if p.suffix == ".json" and p.name != "index.js"}
    check(f"{doc_id!r} writes only under the site, in doc/ and schema/",
          inside and named <= {"doc", "schema"}, str(sorted(map(str, made))))

    # The viewer fetches `file` when there is one; without it the URL would name a file that
    # is not there, which is a document that silently never loads.
    index = json.loads((site / "index.js").read_text().split("= ", 1)[1].rstrip(";\n"))
    entry = index["docs"][0]
    stem = entry.get("file", entry["doc_id"])
    check(f"{doc_id!r}: the index names the file that was written",
          (site / "doc" / f"{stem}.json").is_file(), f"index says {stem!r}")

check("a plain id is left alone, so an ordinary index carries no extra key",
      "file" not in json.loads(
          (TMP / "site5" / "index.js").read_text().split("= ", 1)[1].rstrip(";\n"))["docs"][0])
check("two ids that reduce to the same name stay apart",
      ui.slug("a/b") != ui.slug("a-b"), f"{ui.slug('a/b')} == {ui.slug('a-b')}")

print(f"\n{'ALL UI TESTS PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
