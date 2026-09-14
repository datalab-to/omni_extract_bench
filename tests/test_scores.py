#!/usr/bin/env python3
"""What must be true of the scores table, stated as properties.

`scores` is the one table nothing else can reproduce cheaply -- lose a file and it is hours of
compute back -- so the properties here are mostly about refusing to do damage: never
overwriting a stored table, never naming a dirty tree after a commit, never inventing a row
for a prediction that does not exist.

The three that a probe found the hard way:

  * a payload that still has its envelope must stop the run, not be unwrapped. Unwrap rules
    guessing wrong caused two bugs; this replaces the rule with a check.
  * the metrics on a non-graded row must be null, not zero. Zero is an interpretation, and one
    that a later `mean()` would silently adopt.
  * scoring is deduplicated by `(doc_id, prediction_id)`, which is only sound because files
    sharing that key are byte-identical.

Run: python3 tests/test_scores.py
"""
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
from build_scores import (                                               # noqa: E402
    METRICS, check_dedup, no_envelope, run, schema_for, scorer_version, tasks, write_table)

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


print("\nSCORES TABLE\n")

# ── a stored payload must be bare ─────────────────────────────────────────────────────
# The old scorer unwrapped {"result": ...} here. In this layout the builders store the bare
# extraction, so an envelope arriving means a builder is wrong -- and unwrapping it would hide
# that while scoring under a prediction_id that hashed the wrapper.
try:
    no_envelope({"result": {"a": 1}, "_secs": 3.0}, {}, "somewhere.json")
    report("an enveloped payload stops the run", False, "it was accepted")
except ValueError as exc:
    report("an enveloped payload stops the run", "envelope" in str(exc))

report("a bare payload passes through unchanged",
       no_envelope({"a": 1}, {}, "x") == {"a": 1})

# The one case where "result" is not an envelope: a schema that declares it. No corpus schema
# does today, which is exactly why the case needs a test rather than an assumption.
report("a schema that declares 'result' makes it a real field, not an envelope",
       no_envelope({"result": 1}, {"result": {"type": "integer"}}, "x") == {"result": 1})

# ── non-graded rows carry nulls, not zeros ────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp())
try:
    rows = [{"doc_id": "a", "prediction_id": "h" * 64, "kind": "unusable"},
            {"doc_id": "b", "prediction_id": "g" * 64, "kind": "graded", "secs": 0.1,
             **{m: (99.5 if m in ("accuracy",) else 1) for m in METRICS},
             "matching_exact": True, "approximated": [], "skipped_open_maps": []}]
    path = write_table(rows, tmp, "deadbeef", {"scorer_commit": "deadbeef"})
    back = pq.read_table(path).to_pylist()
    unusable = next(r for r in back if r["kind"] == "unusable")
    report("an unusable prediction records null metrics, never 0.0",
           all(unusable[m] is None for m in METRICS),
           f"accuracy={unusable['accuracy']!r}")
    note("0.0 would be an interpretation; a later mean() over the column would adopt it")

    # ── the table is immutable ────────────────────────────────────────────────────────
    # A file named for a commit is that commit's answer. If it can be rewritten, the directory
    # listing stops being a version history.
    try:
        write_table(rows, tmp, "deadbeef", {"scorer_commit": "deadbeef"})
        report("writing over a stored table is refused", False, "it was overwritten")
    except SystemExit as exc:
        report("writing over a stored table is refused", "immutable" in str(exc))

    # ── the schema does not depend on which rows were scored ──────────────────────────
    only_unusable = write_table([rows[0]], tmp, "cafe", {})
    report("a run that graded nothing still types accuracy as a float",
           pq.read_schema(only_unusable).field("accuracy").type == pa.float64())

    meta = pq.read_schema(path).metadata
    report("the table stamps what ran, not just what the filename says",
           b"scorer_commit" in meta and b"kinds" in meta,
           str(sorted(meta)))
finally:
    shutil.rmtree(tmp)

# ── the journal resumes rather than rescoring ─────────────────────────────────────────
# Hours of compute behind one crash. The journal is the only thing between a crash and
# starting over.
tmp = Path(tempfile.mkdtemp())
try:
    corpus = tmp / "corpus"
    (corpus / "d").mkdir(parents=True)
    (corpus / "d" / "ground_truth.json").write_text('{"a": 1}')
    (corpus / "d" / "schema.json").write_text('{"type": "object", "properties": {"a": {}}}')
    (tmp / "d.json").write_text('{"a": 1}')
    todo = [("d", "p" * 64, str(tmp / "d.json"), True)]

    journal = tmp / "j.jsonl"
    first = run(todo, corpus, 1, journal)
    report("a scored row is journalled as it lands, not at the end",
           journal.exists() and len(journal.read_text().strip().splitlines()) == 1)

    # Make rescoring impossible. If the second run produces a row anyway, it came from the
    # journal -- which is the property.
    (corpus / "d" / "ground_truth.json").unlink()
    second = run(todo, corpus, 1, journal)
    report("a second run reuses the journal instead of rescoring",
           len(second) == 1 and second[0]["kind"] == "graded"
           and second[0]["accuracy"] == first[0]["accuracy"])
finally:
    shutil.rmtree(tmp)

# ── the dirty-tree rule ───────────────────────────────────────────────────────────────
# A working tree with edits cannot be named by a commit without the name lying, and the lie
# breaks the non-determinism check: two runs from two different dirty trees would look like
# the same scorer disagreeing with itself.
tmp = Path(tempfile.mkdtemp())
try:
    run = lambda *a: subprocess.run(["git", "-C", str(tmp), *a], capture_output=True,
                                    check=True, text=True)
    run("init", "-q")
    run("config", "user.email", "t@t"); run("config", "user.name", "t")
    (tmp / "a.py").write_text("x = 1\n")
    run("add", "."); run("commit", "-qm", "one")
    clean_name, clean_stamp = scorer_version(tmp)
    report("a clean tree is named by its commit",
           clean_name == clean_stamp["scorer_commit"] and len(clean_name) == 40)
    report("a clean tree is stamped not-dirty", clean_stamp["scorer_dirty"] == "False")

    (tmp / "a.py").write_text("x = 2\n")
    dirty_name, dirty_stamp = scorer_version(tmp)
    report("a modified tree is not named by a commit",
           dirty_name.startswith("dirty-") and dirty_name != dirty_stamp["scorer_commit"])

    run("checkout", "--", "a.py")
    (tmp / "b.py").write_text("shadow = True\n")
    untracked_name, _ = scorer_version(tmp)
    report("an untracked file counts as dirty",
           untracked_name.startswith("dirty-"),
           f"got {untracked_name[:12]}")
    note("git diff cannot see it, but a stray module changes what gets imported")
finally:
    shutil.rmtree(tmp)

# ── dedup, and the assumption under it ────────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp())
try:
    corpus, vendors = tmp / "corpus", tmp / "vendors"
    corpus.mkdir()
    for doc, rowcount in (("big", 900), ("small", 2)):
        (corpus / doc).mkdir()
        (corpus / doc / "ground_truth.json").write_text('{"a": 1}')
        (corpus / doc / "schema.json").write_text('{"type": "object"}')
    pq.write_table(pa.Table.from_pylist(
        [{"doc_id": "big", "max_array_rows": 900},
         {"doc_id": "small", "max_array_rows": 2}]), corpus / "corpus.parquet")

    shared_body = b'{"same": 1}'
    shared_id = "s" * 64
    for vendor in ("alpha", "beta"):
        vd = vendors / vendor
        vd.mkdir(parents=True)
        (vd / "big.json").write_bytes(shared_body)
        (vd / "small.json").write_bytes(f'{{"{vendor}": 1}}'.encode())
        pq.write_table(pa.Table.from_pylist([
            {"doc_id": "big", "prediction_id": shared_id, "usable": True},
            {"doc_id": "small", "prediction_id": vendor * 16, "usable": True},
        ]), vd / "predictions.parquet")

    todo, paths = tasks(corpus, vendors)
    report("two vendors emitting identical bytes are scored once",
           len(todo) == 3 and sum(len(v) for v in paths.values()) == 4,
           f"{len(todo)} tasks from {sum(len(v) for v in paths.values())} predictions")
    report("the most expensive document is scheduled first",
           todo[0][0] == "big")
    note("the cost spread is p50 = 6 array rows against a p100 of 26,725")

    report("byte-identical predictions under one key pass the dedup check",
           check_dedup(paths) == 0)

    # Break the assumption the dedup rests on and the check must see it.
    (vendors / "beta" / "big.json").write_bytes(b'{"same":  1}')
    _, paths = tasks(corpus, vendors)
    report("a key whose files are NOT identical is caught",
           check_dedup(paths) == 1)
    note("without this, one vendor's score would be attributed to another's prediction")

    # A prediction for a document the corpus does not have is a build error, not a null row.
    pq.write_table(pa.Table.from_pylist([
        {"doc_id": "ghost", "prediction_id": "z" * 64, "usable": True}]),
        vendors / "alpha" / "predictions.parquet")
    try:
        tasks(corpus, vendors)
        report("a prediction for an unknown document stops the build", False, "accepted")
    except ValueError as exc:
        report("a prediction for an unknown document stops the build",
               "not in the corpus" in str(exc))
finally:
    shutil.rmtree(tmp)

print(f"\n{'SCORES TABLE HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
