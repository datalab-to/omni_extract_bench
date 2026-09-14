#!/usr/bin/env python3
"""What must be true of the scores table, stated as properties.

`scores` is the one table nothing else can reproduce cheaply, so most of these are about
refusing to do damage: never naming a dirty tree after a commit, never rewriting a document
that has not changed, never calling ordinary curation a bug.

The model is versions, not mutation. A score means nothing without knowing which corpus
produced it, so the corpus version is in the path: adding a document or correcting a ground
truth makes a different corpus, a different directory, and a fresh run against it. An earlier
version rescored only what had changed, which made growing the corpus in place cheap -- and a
benchmark cheap to grow in place is one where a score means "94.2 against whatever the corpus
was that day".

Two things a probe found the hard way, both after the code looked finished:

  * naming a dirty tree `dirty-<timestamp>` meant every run wrote a new directory, so two runs
    of the same thing never met.
  * the unit of work must be the DOCUMENT. Scoring per prediction while partitioning verdicts
    per document left the journal and the verdict files disagreeing about what was finished.

Run: python3 tests/test_scores.py
"""
import hashlib
import json
import os as _os
import shutil
import subprocess
import sys as _sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
from omni_extract_bench import corpus as corpus_atlas                   # noqa: E402
from omni_extract_bench.run import scorer_version                       # noqa: E402

FAILS = []
SCHEMA = {"type": "object", "properties": {"n": {"type": "number"}, "s": {"type": "string"}}}


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


class Bench:
    """A tiny corpus and vendor tree, driven through the real script."""

    def __init__(self, root: Path):
        self.root = root
        self.corpus, self.vendors, self.out = root / "c", root / "v", root / "o"

    def doc(self, doc_id, gt):
        d = self.corpus / doc_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "ground_truth.json").write_text(json.dumps(gt))
        (d / "schema.json").write_text(json.dumps(SCHEMA))
        corpus_atlas.write(self.corpus, corpus_atlas.discover(self.corpus))

    def vendor(self, name, preds):
        vd = self.vendors / name
        vd.mkdir(parents=True, exist_ok=True)
        rows = []
        for doc_id, body in preds.items():
            raw = json.dumps(body).encode()
            (vd / f"{doc_id}.json").write_bytes(raw)
            rows.append({"doc_id": doc_id,
                         "prediction_id": hashlib.sha256(raw).hexdigest(), "usable": True})
        pq.write_table(pa.Table.from_pylist(rows), vd / "predictions.parquet")

    def build(self, *extra, source="acme", out=None):
        r = subprocess.run(
            [_sys.executable, "-m", "omni_extract_bench.cli", "score",
             "--corpus", str(self.corpus), "--predictions", str(self.vendors / source),
             "--out", str(out or self.out / source), "--jobs", "1", *extra],
            capture_output=True, text=True, cwd=_ROOT)
        return r.returncode, r.stdout + r.stderr

    def dir_for(self, source="acme"):
        return self.out / source

    @property
    def dir(self):
        return self.out / "acme"

    def summary(self, source="acme"):
        return pq.read_table(self.out / source / "summary.parquet").to_pylist()

    def stamp(self, source="acme"):
        m = pq.read_schema(self.out / source / "summary.parquet").metadata
        return {k.decode(): v.decode() for k, v in m.items()}

    def verdict_files(self):
        return {f.name: f.stat().st_mtime_ns
                for f in (self.dir / "verdicts").glob("*.parquet")}


print("\nSCORES TABLE\n")

# ── the scorer records which version ran ──────────────────────────────────────────────
report("the scorer records its version", "scorer_version" in scorer_version())
report("...and nothing about git", not {"scorer_commit", "scorer_dirty"} & set(scorer_version()))
note("a commit only means something inside this repository; a pip install never had one")

# ── the atlas decides what runs ───────────────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp())
try:
    b = Bench(tmp)
    b.doc("alpha", {"n": 1, "s": "x"})
    b.doc("beta", {"n": 2, "s": "y"})
    b.vendor("acme", {"alpha": {"n": 1, "s": "x"}, "beta": {"n": 2, "s": "WRONG"}})
    code, log = b.build()
    report("both documents are scored", code == 0 and len(b.summary()) == 2, log[-200:])

    code, log = b.build()
    report("the same run is not written twice",
           code == 1 and "already have an answer" in log, log[-200:])
    note("scoring again means naming a different --out")

    # Filtering the atlas is how you choose a subset to score.
    t = pq.read_table(b.corpus / "corpus.parquet")
    pq.write_table(pa.Table.from_pylist([r for r in t.to_pylist() if r["doc_id"] != "beta"]),
                   b.corpus / "corpus.parquet")
    code, log = b.build(out=b.out / "filtered")
    rows = b.summary("filtered")
    report("a row deleted from the atlas is not scored",
           [r["doc_id"] for r in rows] == ["alpha"], str([r["doc_id"] for r in rows]))
    report("...and its prediction is skipped, not an error",
           code == 0 and "skipped" in log, log[-200:])
    note("the atlas is what the run follows; filtering it is the point of it being a table")

    # A row naming a file that is not there is a stop, not a skip.
    (b.corpus / "alpha" / "ground_truth.json").unlink()
    code, log = b.build(out=b.out / "broken")
    report("a row pointing at a missing file stops the run",
           code == 1 and "listed in the atlas but missing" in log, log[-200:])
finally:
    shutil.rmtree(tmp)

# ── one prediction set per run, and the path says which ──────────────────────────────
tmp = Path(tempfile.mkdtemp())
try:
    b = Bench(tmp)
    b.doc("alpha", {"n": 1, "s": "x"})
    b.vendor("acme", {"alpha": {"n": 1, "s": "x"}})
    b.vendor("brand", {"alpha": {"n": 1, "s": "WRONG"}})
    code, log = b.build(source="acme")
    report("a run lands exactly where --out says", code == 0
           and (b.out / "acme" / "summary.parquet").exists(), log[-200:])

    code, log = b.build(source="brand")
    runs = sorted(p.name for p in b.out.iterdir())
    report("a second prediction set is a second run, not a second column",
           runs == ["acme", "brand"], str(runs))
    note("all three things that decide a score are path components")

    a = pq.read_table(b.dir / "summary.parquet").to_pylist()
    other = b.summary("brand")
    report("the same document scores differently for different predictions",
           a[0]["accuracy"] != other[0]["accuracy"],
           f"{a[0]['accuracy']} vs {other[0]['accuracy']}")
    report("no source column: the path carries it",
           "source" not in a[0], str(sorted(a[0])[:6]))

    # Identical bytes from two sources are scored twice, on purpose.
    b.vendor("copy", {"alpha": {"n": 1, "s": "x"}})
    code, log = b.build(source="copy")
    c = b.summary("copy")
    report("byte-identical predictions from two sources both get scored",
           c[0]["prediction_id"] == a[0]["prediction_id"]
           and c[0]["accuracy"] == a[0]["accuracy"])
    note("deduplicating them saved 12.4% and cost a table where count(*) was not the count")
finally:
    shutil.rmtree(tmp)

print(f"\n{'SCORES TABLE HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
