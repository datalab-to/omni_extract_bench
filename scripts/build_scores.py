#!/usr/bin/env python3
"""Score the benchmark's predictions into one table per scorer version.

    <out>/scores/<scorer_commit>/summary.parquet              one row per scored prediction
    <out>/scores/<scorer_commit>/verdicts/<doc_id>.parquet    every address, gold and pred

Scores through `omni_extract_bench.bench`, the same path an outside user gets, so there is one
set of rules whether the predictions came from our run or someone's laptop. This script only
adds what is specific to the published benchmark: where the predictions live, which of them are
duplicates, and how to avoid re-deriving work that has not changed.

**A document is the unit of work.** One document is scored against every prediction for it,
writes one verdict file, and contributes one batch of summary rows. Scoring per prediction
instead means the journal and the verdict files disagree about what is finished, and the parsed
ground truth -- up to 13 MB -- is re-read once per vendor.

**A row is identified by what determined it:**

    (doc_id, prediction_id, gt_sha256, schema_sha256)    within a scorer_commit

Carrying the gold and schema hashes is what lets a corrected ground truth be a different row
rather than the same row disagreeing with itself. Without them, ordinary curation reported
`NON-DETERMINISM` when nothing was non-deterministic, and a check that cries wolf is not a
check.

**Two modes, each with one job.** By default a prediction is scored unless the table already
holds a row for exactly these inputs, so adding a document costs one document. `--recheck`
scores everything, compares against the stored table and writes nothing -- the determinism
audit. They are separate because skipping identical inputs and verifying identical inputs are
opposites: an earlier version folded them into one `--force` flag, which disabled the check on
the one run that most wanted it.

**One row per distinct prediction, not per vendor.** Two vendors that emitted byte-identical
extractions have the same score by construction -- 738 of 5,936 (12.4%), 360 of those shared
across vendors. Which vendors a row belongs to is recovered by joining the vendor atlases on
`(doc_id, prediction_id)`. `--check-dedup` asserts the assumption underneath.

`prediction_id` alone is not a key: 48 ids appear under more than one document, because a
vendor's error payloads are identical whatever the input. Predictions that are absent get no
row -- no file, no bytes, nothing to key them on.

Usage:
    uv run --with pyarrow --with scipy --with numpy python scripts/build_scores.py \
        --corpus build/ --vendors build/vendors --out build/
    uv run ... python scripts/build_scores.py ... --jobs 8
    uv run ... python scripts/build_scores.py ... --recheck
    uv run ... python scripts/build_scores.py ... --check-dedup
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from omni_extract_bench.bench import (                                   # noqa: E402
    GROUND_TRUTH, SCHEMA, Case, document, score, summary_row, verdict_rows)
from omni_extract_bench.layout import PREDICTION_ID_VERSION              # noqa: E402

SCORES_DIR, SUMMARY, VERDICTS = "scores", "summary.parquet", "verdicts"

#: What identifies a row. Everything else on it is a measurement.
KEY = ("doc_id", "prediction_id", "gt_sha256", "schema_sha256")

_CFG: dict = {}


class Work(NamedTuple):
    """One document and every distinct prediction for it."""

    doc_id: str
    gt_sha256: str
    schema_sha256: str
    preds: tuple           # ((prediction_id, payload path), ...)
    cost: int              # biggest array in the gold, for scheduling


def scorer_version(repo: Path) -> tuple[str, dict]:
    """The name for this table, and the provenance the name does not carry.

    A working tree with edits cannot be named by a commit without the name lying, and the lie
    is not cosmetic: the table's whole value is that the same inputs under the same name must
    give the same number. So a dirty tree is named `dirty`, which cannot be mistaken for a
    commit.

    Just `dirty`, not `dirty-<timestamp>`. The timestamp looked safer -- two different working
    trees could not collide -- but it meant every run wrote a new directory, so nothing was
    ever current, incremental scoring never engaged while iterating, and the run accumulated a
    directory per invocation. `scores/dirty/` is scratch: freely overwritten, never published,
    and whatever you last had in the tree.

    Untracked files count as dirty -- a stray module changes what gets imported, and
    `git diff` cannot see it.
    """
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, check=True).stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    name = "dirty" if dirty else commit
    return name, {"scorer_commit": commit, "scorer_dirty": str(dirty)}


def environment() -> dict:
    """What ran, beyond the code. The same commit on x86 and ARM may not agree, and nobody has
    checked -- so record enough to tell those runs apart afterwards rather than assuming.
    """
    import numpy
    import scipy
    return {"python": platform.python_version(),
            "platform": f"{platform.system()}-{platform.machine()}",
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "prediction_id_version": PREDICTION_ID_VERSION}


def survey(corpus: Path, vendors: Path):
    """Every document with its predictions, most expensive first, plus the dedup evidence.

    The hashes come from the files, not from parsing them: this decides what to score, and
    parsing 660 ground truths to find out costs more than the decision is worth.

    Most expensive first because the cost spread is extreme -- a median document has 6 array
    rows against a largest of 26,725 -- so a pool fed any other way finishes its cheap work
    and waits on one straggler.
    """
    atlas = {r["doc_id"]: r for r in pq.read_table(corpus / "corpus.parquet").to_pylist()}
    paths = defaultdict(list)
    for vd in sorted(p for p in vendors.iterdir() if p.is_dir()):
        for r in pq.read_table(vd / "predictions.parquet").to_pylist():
            if r["doc_id"] not in atlas:
                raise ValueError(f"{vd.name}: {r['doc_id']} is not in the corpus")
            paths[(r["doc_id"], r["prediction_id"])].append(vd / f"{r['doc_id']}.json")

    by_doc = defaultdict(list)
    for (doc_id, pid), files in paths.items():
        by_doc[doc_id].append((pid, str(files[0])))

    def sha(doc_id, name):
        return hashlib.sha256((corpus / doc_id / name).read_bytes()).hexdigest()

    work = [Work(doc_id, sha(doc_id, GROUND_TRUTH), sha(doc_id, SCHEMA),
                 tuple(sorted(preds)), atlas[doc_id]["max_array_rows"])
            for doc_id, preds in by_doc.items()]
    return sorted(work, key=lambda w: -w.cost), paths


def stored(out: Path) -> dict:
    """Rows already written for this scorer, keyed by identity."""
    path = out / SUMMARY
    if not path.exists():
        return {}
    return {tuple(r[k] for k in KEY): r for r in pq.read_table(path).to_pylist()}


def _init(corpus: str, verdicts: bool):
    _CFG.update(corpus=Path(corpus), verdicts=verdicts)


def score_document(work: Work):
    """Score one document against all of its predictions. Returns (summary rows, verdicts).

    The ground truth and schema are parsed once here and reused across every prediction for
    this document, which is the other reason the document is the unit: there are up to nine.
    """
    doc = document(_CFG["corpus"], work.doc_id)
    rows, verds = [], []
    for pid, payload in work.preds:
        case = Case(doc=doc, prediction_id=pid, raw=Path(payload).read_bytes())
        outcome = score(case, verdicts=_CFG["verdicts"])
        rows.append(summary_row(case, outcome))
        verds.extend(verdict_rows(case, outcome))
    return rows, verds


def results(work: list, corpus: Path, jobs: int, verdicts: bool):
    """Scored documents, from a pool or from this process.

    One worker runs inline rather than spawning a pool of one: a spawned worker re-imports the
    main module, and tracebacks arrive from another process with their frames flattened.
    `--jobs 1` is what you reach for when something is wrong.
    """
    if jobs == 1:
        _init(str(corpus), verdicts)
        yield from ((w, *score_document(w)) for w in work)
        return
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init,
                             initargs=(str(corpus), verdicts)) as pool:
        for w, (rows, verds) in zip(work, pool.map(score_document, work, chunksize=1)):
            yield w, rows, verds


def write_verdicts(out: Path, doc_id: str, verds: list) -> None:
    """One document's verdict file, replaced whole.

    Written to a temporary name and renamed, so a crash leaves the previous file rather than a
    truncated one. Writing it BEFORE the document is journalled is deliberate: a crash in
    between rescores the document next time and overwrites this, which is harmless. The other
    order would leave a journalled document with no verdicts.
    """
    target = out / VERDICTS / f"{doc_id}.parquet"
    if not verds:
        target.unlink(missing_ok=True)
        return
    tmp = target.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(verds), tmp, compression="zstd")
    os.replace(tmp, target)


def write_summary(out: Path, rows: list, stamp: dict) -> None:
    """The summary table, whole. Small enough that rewriting it is free, and parquet has no
    append, so there is no other option.
    """
    fields = list(dict.fromkeys(k for r in rows for k in r))
    blank = {k: None for k in fields}
    kinds = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    meta = {**stamp, "rows": str(len(rows)),
            "kinds": json.dumps(dict(sorted(kinds.items()))),
            "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    table = pa.Table.from_pylist([{**blank, **r} for r in rows]).replace_schema_metadata(meta)
    tmp = out / (SUMMARY + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out / SUMMARY)
    print(f"  {SUMMARY}: {(out / SUMMARY).stat().st_size / 1024:.0f} KB, {len(rows)} rows, "
          f"{dict(sorted(kinds.items()))}")


def check_dedup(paths) -> int:
    """Assert what the dedup rests on: one key, one set of bytes."""
    bad = 0
    for (doc_id, pid), files in paths.items():
        if len(files) > 1 and len({f.read_bytes() for f in files}) != 1:
            bad += 1
            print(f"  DIFFER {doc_id[:46]} {pid[:12]}: {[f.parent.name for f in files]}")
    shared = sum(1 for v in paths.values() if len(v) > 1)
    print(f"  check-dedup: {shared} keys held by more than one vendor, {bad} not identical")
    return bad


def recheck(out: Path, work: list, corpus: Path, jobs: int) -> int:
    """Score everything and compare against the stored table. Writes nothing.

    A row reaching the comparison was produced from identical prediction bytes, gold, schema
    and code, so a different number means the scorer is not a function of its inputs. That is
    how the greedy row-pairing behaved: one document recording 99.2316, then 99.2301, then
    99.2328 in a single session with nothing to point at.
    """
    old = stored(out)
    if not old:
        print(f"  nothing stored at {out / SUMMARY} to check against", file=sys.stderr)
        return 1
    same = differ = absent = 0
    for _w, rows, _v in results(work, corpus, jobs, verdicts=False):
        for row in rows:
            was = old.get(tuple(row[k] for k in KEY))
            if was is None:
                absent += 1
            elif was.get("kind") != row.get("kind") or (
                    row.get("kind") == "graded"
                    and abs((was.get("accuracy") or 0) - (row.get("accuracy") or 0)) > 1e-12):
                differ += 1
                print(f"  NON-DETERMINISM {row['doc_id'][:46]}: "
                      f"{was.get('accuracy')} -> {row.get('accuracy')}", file=sys.stderr)
            else:
                same += 1
    print(f"  recheck: {same} identical, {differ} differ, {absent} not in the stored table")
    return 1 if differ else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--vendors", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--recheck", action="store_true",
                    help="score everything and compare against the stored table; writes nothing")
    ap.add_argument("--no-verdicts", action="store_true")
    ap.add_argument("--check-dedup", action="store_true",
                    help="assert that predictions sharing a key are byte-identical, and stop")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    work, paths = survey(args.corpus, args.vendors)
    if args.check_dedup:
        return 1 if check_dedup(paths) else 0

    name, stamp = scorer_version(repo)
    stamp.update(environment())
    out = args.out / SCORES_DIR / name
    total = sum(len(w.preds) for w in work)
    print(f"  {sum(len(v) for v in paths.values())} predictions -> {total} distinct "
          f"over {len(work)} documents, scorer {name[:16]}")

    if args.recheck:
        return recheck(out, work, args.corpus, args.jobs)

    # A document needs scoring unless the table already holds a row for every one of its
    # predictions, against exactly this gold and schema.
    have = stored(out)
    todo = [w for w in work
            if not all((w.doc_id, pid, w.gt_sha256, w.schema_sha256) in have
                       for pid, _path in w.preds)]
    keep = [r for k, r in have.items()
            if k[0] not in {w.doc_id for w in todo}]
    if have:
        print(f"  {len(keep)} rows still current, {len(todo)} documents to score")
    if not todo:
        print("  nothing to do")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / VERDICTS).mkdir(exist_ok=True)
    journal = out / f"{SUMMARY}.partial.jsonl"
    done = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.setdefault(r["doc_id"], []).append(r)
        print(f"  resuming: {len(done)} documents already scored")
        todo = [w for w in todo if w.doc_id not in done]

    fresh = [r for rows in done.values() for r in rows]
    t0 = time.monotonic()
    with journal.open("a") as log:
        for n, (w, rows, verds) in enumerate(
                results(todo, args.corpus, args.jobs, not args.no_verdicts), 1):
            write_verdicts(out, w.doc_id, verds)
            for r in rows:
                log.write(json.dumps(r) + "\n")
            log.flush()
            fresh.extend(rows)
            if n % 50 == 0 or n == len(todo):
                rate = n / (time.monotonic() - t0)
                print(f"  {n}/{len(todo)} documents  {rate:.2f}/s  "
                      f"eta {(len(todo) - n) / rate / 60:.0f}m", flush=True)

    write_summary(out, keep + fresh, stamp)
    journal.unlink(missing_ok=True)
    nverd = len(list((out / VERDICTS).glob("*.parquet")))
    vsize = sum(f.stat().st_size for f in (out / VERDICTS).glob("*.parquet")) / 1e6
    print(f"  {VERDICTS}/: {nverd} files, {vsize:.1f} MB")

    failed = [r for r in fresh if r["kind"] == "failed"]
    for r in failed[:10]:
        print(f"  NOT SCORED {r['doc_id'][:46]}: {r['error'][:110]}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
