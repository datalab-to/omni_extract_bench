#!/usr/bin/env python3
"""What must be true of the scores table, stated as properties.

`scores` is the one table nothing else can reproduce cheaply, so most of these are about
refusing to do damage: never naming a dirty tree after a commit, never rewriting a document
that has not changed, never calling ordinary curation a bug.

Three that a probe found the hard way, each after the code looked finished:

  * `--force` conflated "what to rescore" with "what to compare against", so the determinism
    check silently never ran on the one command that most wanted it. It is now two modes.
  * naming a dirty tree `dirty-<timestamp>` meant every run wrote a new directory, so nothing
    was ever current and incremental scoring never engaged at all.
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
        pq.write_table(pa.Table.from_pylist(
            [{"doc_id": p.name, "max_array_rows": 1}
             for p in sorted(self.corpus.iterdir()) if p.is_dir()]),
            self.corpus / "corpus.parquet")

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
        return self.out / "scores" / "dirty"

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

# ── the incremental contract ──────────────────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp())
try:
    b = Bench(tmp)
    b.doc("alpha", {"n": 1, "s": "x"})
    b.vendor("acme", {"alpha": {"n": 1, "s": "x"}})
    b.vendor("brand", {"alpha": {"n": 1, "s": "WRONG"}})
    code, log = b.build()
    report("a first run scores everything", code == 0 and len(b.summary()) == 2, log[-200:])
    first = b.verdict_files()

    code, log = b.build()
    report("a second run with nothing changed does nothing",
           "nothing to do" in log and b.verdict_files() == first, log[-200:])
    note("the table is a cache of work that cost hours; re-deriving it is the failure")

    b.doc("beta", {"n": 2, "s": "y"})
    b.vendor("acme", {"alpha": {"n": 1, "s": "x"}, "beta": {"n": 2, "s": "y"}})
    b.vendor("brand", {"alpha": {"n": 1, "s": "WRONG"}, "beta": {"n": 2, "s": "y"}})
    code, log = b.build()
    after = b.verdict_files()
    report("adding a document adds its rows and its verdict file",
           len(b.summary()) == 3 and "beta.parquet" in after, log[-200:])
    report("...and does not touch any other document's verdicts",
           after["alpha.parquet"] == first["alpha.parquet"])
    note("adding one document must not cost 9.8 CPU-hours of rescoring")

    b.doc("alpha", {"n": 99, "s": "x"})          # the gold was wrong; fix it
    code, log = b.build()
    third = b.verdict_files()
    report("correcting a ground truth is not reported as non-determinism",
           code == 0 and "NON-DETERMINISM" not in log, log[-300:])
    note("a check that fires on ordinary curation is a check nobody will keep")
    report("...and rescores only that document",
           third["alpha.parquet"] != after["alpha.parquet"]
           and third["beta.parquet"] == after["beta.parquet"])
    report("the corrected row records the gold it was scored against",
           len({r["gt_sha256"] for r in b.summary() if r["doc_id"] == "alpha"}) == 1)

    # ── genuine non-determinism IS caught ─────────────────────────────────────────────
    rows = b.summary()
    for r in rows:
        if r["doc_id"] == "beta":
            r["accuracy"] = 12.5
    pq.write_table(pa.Table.from_pylist(rows), b.dir / "summary.parquet")
    code, log = b.build("--recheck")
    report("--recheck catches a score that changed on identical inputs",
           code == 1 and "NON-DETERMINISM" in log, log[-300:])
    note("this is the check that would have caught the greedy order-sensitivity")
    report("--recheck writes nothing",
           any(r["accuracy"] == 12.5 for r in b.summary()))
finally:
    shutil.rmtree(tmp)

# ── dedup, and the assumption under it ────────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp())
try:
    b = Bench(tmp)
    b.doc("big", {"n": 1, "s": "x"})
    b.doc("small", {"n": 2, "s": "y"})
    for v in ("alpha", "beta"):
        b.vendor(v, {"big": {"n": 1, "s": "x"}, "small": {"n": 9, "s": v}})
    work, paths, _awaiting = survey(b.corpus, b.vendors)
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
    _w, paths, _a = survey(b.corpus, b.vendors)
    report("a key whose files are NOT identical is caught", check_dedup(paths) == 1)
    note("without this, one vendor's score is attributed to another's prediction")

    pq.write_table(pa.Table.from_pylist([
        {"doc_id": "ghost", "prediction_id": "z" * 64, "usable": True}]),
        b.vendors / "alpha" / "predictions.parquet")
    try:
        survey(b.corpus, b.vendors)
        report("a prediction for an unknown document stops the build", False, "accepted")
    except ValueError as exc:
        report("a prediction for an unknown document stops the build",
               "not in the corpus" in str(exc))
finally:
    shutil.rmtree(tmp)

report("the row key carries the gold and the schema, not just the prediction",
       KEY == ("doc_id", "prediction_id", "gt_sha256", "schema_sha256"), str(KEY))

print(f"\n{'SCORES TABLE HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
