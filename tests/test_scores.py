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
from omni_extract_bench.corpus import version as corpus_version         # noqa: E402
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

# ── the scorer name ───────────────────────────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp())
try:
    def git(*a):
        return subprocess.run(["git", "-C", str(tmp), *a], capture_output=True,
                              check=True, text=True)

    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (tmp / "a.py").write_text("x = 1\n")
    git("add", "."); git("commit", "-qm", "one")
    stamp = scorer_version(tmp)
    report("a clean tree records its commit", len(stamp["scorer_commit"]) == 40)

    report("a clean tree is recorded as clean", stamp["scorer_dirty"] == "False")

    (tmp / "a.py").write_text("x = 2\n")
    report("an edited tree is recorded as dirty, not given a different name",
           scorer_version(tmp)["scorer_dirty"] == "True")
    note("a commit alone would be a claim the working tree cannot support")

    git("checkout", "--", "a.py")
    (tmp / "b.py").write_text("shadow = True\n")
    report("an untracked file counts as dirty",
           scorer_version(tmp)["scorer_dirty"] == "True")
    note("git diff cannot see it, but a stray module changes what gets imported")

    # Someone who vendors this into their own project, or commits their virtualenv, must not
    # have every edit anywhere in their tree reported as a change to the scorer -- nor have
    # merely importing it, which writes __pycache__, do the same.
    from omni_extract_bench.run import _is_artifact
    report("a .pyc is not a source change",
           _is_artifact("?? vendor/omni_extract_bench/__pycache__/x.cpython-311.pyc")
           and _is_artifact("?? pkg/x.pyc"))
    report("...but a real file is",
           not _is_artifact(" M vendor/omni_extract_bench/values.py"))
    note("importing the package would otherwise mark it modified in any project "
         "that has not ignored __pycache__")
finally:
    shutil.rmtree(tmp)

# ── the corpus is versioned, and a version is answered once ───────────────────────────
tmp = Path(tempfile.mkdtemp())
try:
    b = Bench(tmp)
    b.doc("alpha", {"n": 1, "s": "x"})
    b.vendor("acme", {"alpha": {"n": 1, "s": "x"}})
    b.vendor("brand", {"alpha": {"n": 1, "s": "WRONG"}})
    code, log = b.build()
    report("a first run scores this source's predictions",
           code == 0 and len(b.summary()) == 1, log[-200:])
    v1 = b.stamp()["corpus_version"]
    report("the corpus version is stamped on the table", len(v1) == 16, v1)
    note("in the metadata, not the path: --out is exactly what you asked for")

    code, log = b.build()
    report("the same corpus, scorer and source are not scored twice",
           code == 1 and "already have an answer" in log, log[-200:])
    note("one rule, no exception for dirty: iterating means pointing --out elsewhere")

    # Scoring the same thing again is a second run and a join, not a verb on the tool.
    code, log = b.build(out=b.out / "again")
    first, again = b.summary(), b.summary("again")
    report("scoring the same inputs twice gives the same numbers",
           code == 0 and [r["accuracy"] for r in first] == [r["accuracy"] for r in again],
           log[-200:])
    note("comparing two runs is a join on (doc_id, prediction_id); there is no --recheck")

    b.doc("beta", {"n": 2, "s": "y"})            # no predictions for it yet
    code, log = b.build(out=b.out / "v2")
    v2 = b.stamp("v2")["corpus_version"] if (b.out / "v2" / "summary.parquet").exists() else None
    report("adding a document makes a different corpus version", v2 not in (None, v1),
           f"{v1} -> {v2}")
    report("...and the previous run is still there, untouched",
           b.stamp()["corpus_version"] == v1)
    note("versions are comparable within one and visibly incomparable across two")

    b.doc("alpha", {"n": 99, "s": "x"})          # the gold was wrong; fix it
    code, log = b.build(out=b.out / "v3")
    report("correcting a ground truth makes a different corpus version too",
           b.stamp("v3")["corpus_version"] not in (v1, v2))

    b.doc("alpha", {"n": 1, "s": "x"})           # put it back
    code, log = b.build(out=b.out / "v4")
    report("restoring the corpus returns to its own version",
           b.stamp("v4")["corpus_version"] == v2, b.stamp("v4")["corpus_version"])
    note("the version is a hash of the contents, so it is a fact rather than a counter")

    # Non-determinism itself is guarded where it can be guarded properly:
    # tests/test_pairing_determinism.py shuffles both sides of 600 generated documents. A
    # rescore-and-diff verb would only ever re-check what a run happened to cover.
finally:
    shutil.rmtree(tmp)

# ── the corpus version is a fact about the corpus ─────────────────────────────────────
def E(doc_id, gt, schema):
    g, s_ = corpus_atlas.expected_paths(doc_id)
    return corpus_atlas.Entry(doc_id, g, s_, gt, schema)


a = [E("d1", "gt1", "s1"), E("d2", "gt2", "s2")]
report("the same corpus hashes the same, whatever the row order",
       corpus_version(a) == corpus_version(list(reversed(a))))
report("a changed gold changes the version",
       corpus_version(a) != corpus_version([E("d1", "gt1-fixed", "s1"), a[1]]))
report("a changed schema changes the version",
       corpus_version(a) != corpus_version([E("d1", "gt1", "s1-fixed"), a[1]]))
report("an added document changes the version, even with no predictions for it",
       corpus_version(a) != corpus_version(a + [E("d3", "gt3", "s3")]))
report("CURATING A DOCUMENT OUT changes the version",
       corpus_version(a) != corpus_version(a[:1]))
note("a filtered corpus is a different benchmark, and its scores belong somewhere else")

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
