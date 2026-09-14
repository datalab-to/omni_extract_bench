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
_sys.path.insert(0, _os.path.join(_ROOT, "scripts"))
from omni_extract_bench import corpus as corpus_atlas                   # noqa: E402
from omni_extract_bench.corpus import version as corpus_version         # noqa: E402
from build_scores import KEY, check_dedup, scorer_version, survey       # noqa: E402

FAILS = []
BUILD = _os.path.join(_ROOT, "scripts", "build_scores.py")
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

    def build(self, *extra):
        r = subprocess.run(
            [_sys.executable, BUILD, "--corpus", str(self.corpus), "--vendors",
             str(self.vendors), "--out", str(self.out), "--jobs", "1", *extra],
            capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    @property
    def dir(self):
        return next(p for p in (self.out / "scores").glob("*/dirty"))

    def summary(self):
        return pq.read_table(self.dir / "summary.parquet").to_pylist()

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
    name, stamp = scorer_version(tmp)
    report("a clean tree is named by its commit",
           name == stamp["scorer_commit"] and len(name) == 40)

    (tmp / "a.py").write_text("x = 2\n")
    dirty_name, _ = scorer_version(tmp)
    report("a modified tree is named `dirty`, not a commit", dirty_name == "dirty")

    git("checkout", "--", "a.py")
    (tmp / "b.py").write_text("shadow = True\n")
    report("an untracked file counts as dirty", scorer_version(tmp)[0] == "dirty")
    note("git diff cannot see it, but a stray module changes what gets imported")

    again, _ = scorer_version(tmp)
    report("the dirty name is stable across runs", again == "dirty")
    note("a timestamp here meant nothing was ever current and incremental never engaged")
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
    report("a first run scores everything", code == 0 and len(b.summary()) == 2, log[-200:])
    v1 = b.dir.parent.name
    report("the corpus version is in the path", len(v1) == 16, v1)

    code, log = b.build()
    report("the same corpus and scorer are not scored twice",
           code == 1 and "already have an answer" in log, log[-200:])
    note("a finished table is the answer; rescoring can only agree or reveal a bug")

    code, log = b.build("--recheck")
    report("--recheck reproduces it", code == 0 and "0 differ" in log, log[-200:])

    b.doc("beta", {"n": 2, "s": "y"})            # no predictions for it yet
    code, log = b.build()
    v2 = sorted(p.name for p in (b.out / "scores").iterdir())
    report("adding a document makes a different corpus version",
           len(v2) == 2 and v1 in v2, str(v2))
    report("...and the previous version is still there, untouched", (b.out / "scores" / v1).exists())
    note("versions are comparable within one and visibly incomparable across two")

    b.doc("alpha", {"n": 99, "s": "x"})          # the gold was wrong; fix it
    code, log = b.build()
    report("correcting a ground truth makes a different corpus version too",
           len(list((b.out / "scores").iterdir())) == 3)

    b.doc("alpha", {"n": 1, "s": "x"})           # put it back
    b.doc("beta", {"n": 2, "s": "y"})
    code, log = b.build()
    report("restoring the corpus returns to its own version, already answered",
           code == 1 and "already have an answer" in log, log[-160:])
    note("the version is a hash of the contents, so it is a fact rather than a counter")

    # ── genuine non-determinism IS caught ─────────────────────────────────────────────
    d = b.out / "scores" / v1 / "dirty"
    rows = pq.read_table(d / "summary.parquet").to_pylist()
    rows[0]["accuracy"] = 12.5
    pq.write_table(pa.Table.from_pylist(rows), d / "summary.parquet")
    b.doc("alpha", {"n": 1, "s": "x"})
    for extra in (b.corpus / "beta",):
        shutil.rmtree(extra)
    b.doc("alpha", {"n": 1, "s": "x"})           # back to exactly v1
    code, log = b.build("--recheck")
    report("--recheck catches a score that changed on identical inputs",
           code == 1 and "NON-DETERMINISM" in log, log[-300:])
    note("this is the check that would have caught the greedy order-sensitivity")
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

# ── dedup, and the assumption under it ────────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp())
try:
    b = Bench(tmp)
    b.doc("big", {"n": 1, "s": "x"})
    b.doc("small", {"n": 2, "s": "y"})
    for v in ("alpha", "beta"):
        b.vendor(v, {"big": {"n": 1, "s": "x"}, "small": {"n": 9, "s": v}})
    work, paths, _awaiting, _v, _s = survey(b.corpus, b.vendors)
    total = sum(len(p) for p in paths.values())
    distinct = sum(len(w.preds) for w in work)
    report("two vendors emitting identical bytes are scored once",
           total == 4 and distinct == 3, f"{distinct} of {total}")
    report("work is grouped by document, not by prediction",
           len(work) == 2 and all(isinstance(w.preds, tuple) for w in work))
    note("the document is the unit: one parse of the gold, one verdict file, one journal batch")
    report("byte-identical predictions under one key pass the dedup check",
           check_dedup(paths) == 0)

    (b.vendors / "beta" / "big.json").write_bytes(b'{"n": 1, "s":  "x"}')
    _w, paths, _a, _v, _s = survey(b.corpus, b.vendors)
    report("a key whose files are NOT identical is caught", check_dedup(paths) == 1)
    note("without this, one vendor's score is attributed to another's prediction")

    pq.write_table(pa.Table.from_pylist([
        {"doc_id": "ghost", "prediction_id": "z" * 64, "usable": True}]),
        b.vendors / "alpha" / "predictions.parquet")
    _w, _p, _a, _v, skipped = survey(b.corpus, b.vendors)
    report("a prediction whose document is not in the atlas is skipped, not fatal",
           skipped == ["ghost"], str(skipped))
    note("curating a document out leaves its predictions behind; that is normal")
finally:
    shutil.rmtree(tmp)

report("the row key carries the gold and the schema, not just the prediction",
       KEY == ("doc_id", "prediction_id", "gt_sha256", "schema_sha256"), str(KEY))

print(f"\n{'SCORES TABLE HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
