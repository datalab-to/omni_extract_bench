#!/usr/bin/env python3
"""Score every distinct prediction against the corpus, into one immutable table per scorer.

    <out>/scores/<scorer_commit>.parquet

Reads the two trees the other builders produce -- `corpus.parquet` beside the document
directories, `vendors/<vendor>/predictions.parquet` beside the payloads -- and writes one row
per distinct `(doc_id, prediction_id)`.

**Not one row per (vendor, doc_id, prediction_id).** A score is a function of the prediction's
bytes and the scorer's code, so two vendors that emitted byte-identical extractions have the
same score by construction, and scoring both would be computing a known answer twice. In the
current run that is 738 of 5,936 (12.4%), 360 of them shared across vendors rather than within
one. Which vendors a row belongs to is recovered by joining the vendor atlases on
`(doc_id, prediction_id)`, where that fact already lives.

The dedup rests on an assumption worth stating, because it is the kind that rots quietly:
every file sharing a `(doc_id, prediction_id)` is byte-identical. That is what
`prediction_id` means, and `--check-dedup` asserts it rather than trusting it.

`prediction_id` alone is *not* the key: 48 ids appear under more than one document, because a
vendor's error payloads are identical whatever the input.

**Predictions that are absent get no row.** The run produced four, and they have no file, so
no bytes, so no `prediction_id` -- there is nothing to key them on. Absence is a fact about a
vendor's coverage and it already shows as a missing row in that vendor's atlas. Inventing a
scores row for a prediction that does not exist would put it in the one table that is supposed
to describe predictions that do.

Usage:
    uv run --with pyarrow --with scipy --with numpy python scripts/build_scores.py \
        --corpus build/ --vendors build/vendors --out build/
    uv run ... python scripts/build_scores.py ... --jobs 4
    uv run ... python scripts/build_scores.py ... --recheck build/scores/<commit>.parquet
    uv run ... python scripts/build_scores.py ... --check-dedup
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from omni_extract_bench.layout import PREDICTION_ID_VERSION          # noqa: E402

#: One file per scorer version, never overwritten. The directory listing is then the version
#: history, in a store that has none.
SCORES_DIR = "scores"

#: The scorer's scalar outputs, in the table in this order. All of them, not the twelve the
#: first run happened to keep: a column costs about eight bytes and re-running to recover one
#: costs hours, so the asymmetry only points one way.
METRICS = ("accuracy", "f1", "precision", "recall", "found", "read_right",
           "matched", "total", "asserted", "misread", "unfound",
           "fabricated", "invented_item", "invented_field",
           "gt_rows", "pred_rows", "matched_rows")

#: Where the metric stopped being exact. `matching_exact` says *that* it happened;
#: `approximated` and `skipped_open_maps` say *where*, which is what you need when a number
#: looks wrong. Both are empty on almost every row and cost nothing when they are.
WITNESSES = ("matching_exact", "approximated", "skipped_open_maps")

_CFG: dict = {}


def scorer_version(repo: Path) -> tuple[str, dict]:
    """The name for this table, and the provenance that the name does not carry.

    The filename is a commit, so it has to *be* one. A working tree with edits in it cannot be
    named by a commit without the name lying, and the lie is not cosmetic: the whole value of
    the key is that `same prediction_id AND same scorer_commit, different score` means a bug.
    Two runs from two different dirty trees under one commit would trip that check forever
    while nothing was wrong.

    So a dirty tree gets a timestamp instead, which cannot be mistaken for a commit and does
    not claim to be reproducible.

    Untracked files count as dirty. A stray module in `omni_extract_bench/` changes what gets
    imported, and `git diff` does not see it.
    """
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, check=True).stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    name = (f"dirty-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}" if dirty else commit)
    return name, {"scorer_commit": commit, "scorer_dirty": str(dirty)}


def environment() -> dict:
    """What ran, beyond the code. Same commit on x86 and ARM may not agree, and nobody has
    checked -- so record enough to tell those runs apart after the fact instead of assuming.
    """
    import numpy
    import scipy
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "prediction_id_version": PREDICTION_ID_VERSION,
    }


def tasks(corpus: Path, vendors: Path):
    """Every distinct `(doc_id, prediction_id)`, most expensive first.

    Longest-first because the cost distribution is extreme -- the median document has 6 array
    rows and the largest has 26,725 -- so a pool fed in any other order finishes its cheap work
    early and then waits on one straggler.

    Returns the tasks and, for `--check-dedup`, every payload path behind each key.
    """
    atlas = {r["doc_id"]: r for r in pq.read_table(corpus / "corpus.parquet").to_pylist()}
    paths = defaultdict(list)
    usable = {}
    for vd in sorted(p for p in vendors.iterdir() if p.is_dir()):
        for r in pq.read_table(vd / "predictions.parquet").to_pylist():
            key = (r["doc_id"], r["prediction_id"])
            if key[0] not in atlas:
                raise ValueError(f"{vd.name}: {key[0]} is not in the corpus")
            paths[key].append(vd / f"{r['doc_id']}.json")
            usable[key] = r["usable"]
    ordered = sorted(paths, key=lambda k: -atlas[k[0]]["max_array_rows"])
    return [(d, p, str(paths[(d, p)][0]), usable[(d, p)]) for d, p in ordered], paths


def _init(corpus: str):
    _CFG["corpus"] = Path(corpus)


def score_one(task):
    """Grade one prediction. Returns a row; never raises out of the worker."""
    from omni_extract_bench.dialects import resolve_refs, strip_benchmark_keys
    from omni_extract_bench.score import grade

    doc_id, pid, payload, usable = task
    row = {"doc_id": doc_id, "prediction_id": pid}
    if not usable:
        return {**row, "kind": "unusable"}

    d = _CFG["corpus"] / doc_id
    t0 = time.monotonic()
    try:
        schema = resolve_refs(strip_benchmark_keys(
            json.loads((d / "schema.json").read_text())))
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        gt = no_envelope(json.loads((d / "ground_truth.json").read_text()), props,
                         f"{doc_id}/ground_truth.json")
        pred = no_envelope(json.loads(Path(payload).read_text()), props, payload)
        r = grade(pred, gt, schema)
    except Exception as exc:
        return {**row, "kind": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}
    return {**row, "kind": "graded", "secs": round(time.monotonic() - t0, 3),
            **{k: r[k] for k in METRICS},
            "matching_exact": r["matching_exact"],
            "approximated": [str(a) for a in r["approximated"]],
            "skipped_open_maps": [str(a) for a in r["skipped_open_maps"]]}


def no_envelope(obj, schema_props, where: str):
    """Assert that a stored payload is bare, instead of unwrapping it.

    The old scorer peeled a `{"result": ...}` envelope here, guided by the schema. In this
    layout both builders store the bare extraction, so an envelope reaching this point is a
    bug in the builder that wrote it -- and unwrapping it silently would hide that while
    producing a score under a `prediction_id` that hashed the wrapper.

    Two bugs came from unwrap rules guessing wrong. A rule that can guess wrong is replaced
    here by a check that cannot.
    """
    if isinstance(obj, dict) and "result" in obj and "result" not in schema_props:
        raise ValueError(f"{where}: still wrapped in a 'result' envelope; "
                         "the builder should have stored the bare extraction")
    return obj


def run(todo, corpus: Path, jobs: int, journal: Path):
    """Score, appending each row to a journal as it lands.

    Hours of compute with the table written only at the end is hours to lose to one crash, and
    `scores` is the one thing in this system nothing else can reproduce cheaply. The journal is
    the crash log; the parquet is the deliverable, written once the set is complete.

    The journal is named for the scorer, so only a clean tree resumes: a dirty one gets a fresh
    timestamp each run and starts over. That is the intended behaviour rather than a gap --
    resuming a dirty run after an edit would merge rows produced by two different codebases
    into a single table, which is the exact corruption the rest of this file exists to prevent.
    It does mean an abandoned dirty run leaves its journal behind to be deleted by hand.
    """
    done = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["doc_id"], r["prediction_id"])] = r
        print(f"  resuming: {len(done)} already scored")
    todo = [t for t in todo if (t[0], t[1]) not in done]

    rows = list(done.values())
    t0 = time.monotonic()
    with journal.open("a") as log:
        for i, row in enumerate(_results(todo, corpus, jobs), 1):
            log.write(json.dumps(row) + "\n")
            log.flush()
            rows.append(row)
            if i % 100 == 0 or i == len(todo):
                rate = i / (time.monotonic() - t0)
                print(f"  {i}/{len(todo)}  {rate:.1f}/s  "
                      f"eta {(len(todo) - i) / rate / 60:.0f}m", flush=True)
    return rows


def _results(todo, corpus: Path, jobs: int):
    """Rows in order, from a pool or from this process.

    One worker runs inline rather than spawning a pool of one. That is not a micro-optimisation
    -- a spawned worker re-imports the main module, so a caller that is not guarded by
    `if __name__ == "__main__"` re-runs itself, and any traceback arrives from another process
    with its frames flattened. `--jobs 1` is what you reach for when something is wrong, which
    is exactly when both of those matter.
    """
    if jobs == 1:
        _init(str(corpus))
        yield from (score_one(t) for t in todo)
        return
    pool = ProcessPoolExecutor(max_workers=jobs, initializer=_init, initargs=(str(corpus),))
    with pool:
        yield from pool.map(score_one, todo, chunksize=1)


def schema_for(rows):
    """An explicit schema, so a column's type does not depend on which rows were scored.

    Inferring from the data would give `accuracy` a different type in a run that graded
    nothing, and would make the nulls on unusable rows unrepresentable at all.
    """
    f = [pa.field("doc_id", pa.string()), pa.field("prediction_id", pa.string()),
         pa.field("kind", pa.string()), pa.field("error", pa.string()),
         pa.field("secs", pa.float64())]
    ints = {"matched", "total", "asserted", "misread", "unfound", "fabricated",
            "invented_item", "invented_field", "gt_rows", "pred_rows", "matched_rows"}
    f += [pa.field(m, pa.int64() if m in ints else pa.float64()) for m in METRICS]
    f += [pa.field("matching_exact", pa.bool_()),
          pa.field("approximated", pa.list_(pa.string())),
          pa.field("skipped_open_maps", pa.list_(pa.string()))]
    return pa.schema(f)


def write_table(rows, out: Path, name: str, stamp: dict) -> Path:
    """Write the table, refusing to overwrite one that exists.

    Immutability is the point: a file named for a commit is that commit's answer, and if it can
    be rewritten then the version history the directory listing gives you is worthless. Use
    `--recheck` to compare a rerun against a stored table rather than replacing it.
    """
    path = out / SCORES_DIR / f"{name}.parquet"
    if path.exists():
        raise SystemExit(f"{path} exists; scores are immutable. "
                         f"To compare a rerun against it: --recheck {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = schema_for(rows)
    blank = {f.name: None for f in schema}
    table = pa.Table.from_pylist([{**blank, **r} for r in rows], schema=schema)
    kinds = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    meta = {**stamp, "rows": str(len(rows)),
            "kinds": json.dumps(dict(sorted(kinds.items()))),
            "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pq.write_table(table.replace_schema_metadata(meta), path, compression="zstd")
    print(f"  {path.name}: {path.stat().st_size / 1024:.0f} KB, {len(rows)} rows, "
          f"{dict(sorted(kinds.items()))}")
    return path


def recheck(stored: Path, rows):
    """Compare a rerun against a stored table: the non-determinism test, run deliberately.

    Same prediction and same scorer must give the same number. When they do not, the scorer is
    not a function of its inputs -- which is how the greedy row-pairing order-sensitivity
    behaved, one document quietly recording 99.2316, then 99.2301, then 99.2328 in a single
    session with nothing to point at.
    """
    old = {(r["doc_id"], r["prediction_id"]): r
           for r in pq.read_table(stored).to_pylist()}
    differ = absent = 0
    for r in rows:
        k = (r["doc_id"], r["prediction_id"])
        if k not in old:
            absent += 1
            continue
        if old[k]["kind"] != r["kind"] or (
                r["kind"] == "graded"
                and abs((old[k]["accuracy"] or 0) - (r["accuracy"] or 0)) > 1e-12):
            differ += 1
            print(f"  NON-DETERMINISM {r['doc_id'][:46]} {r['prediction_id'][:12]}: "
                  f"{old[k].get('accuracy')} -> {r.get('accuracy')}")
    print(f"  recheck: {len(rows) - differ - absent} identical, {differ} differ, "
          f"{absent} not in the stored table")
    return differ


def check_dedup(paths) -> int:
    """Assert what the dedup rests on: one key, one set of bytes.

    Cheap, and the alternative to checking is finding out by attributing one vendor's score to
    another's prediction.
    """
    bad = 0
    for (doc_id, pid), files in paths.items():
        if len(files) == 1:
            continue
        if len({f.read_bytes() for f in files}) != 1:
            bad += 1
            print(f"  DIFFER {doc_id[:46]} {pid[:12]}: "
                  f"{[f.parent.name for f in files]}")
    shared = sum(1 for v in paths.values() if len(v) > 1)
    print(f"  check-dedup: {shared} keys held by more than one vendor, {bad} not identical")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--vendors", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--recheck", type=Path,
                    help="rescore and compare against this table instead of writing one")
    ap.add_argument("--check-dedup", action="store_true",
                    help="assert that predictions sharing a key are byte-identical, and stop")
    ap.add_argument("--limit", type=int,
                    help="score only the N cheapest documents, for a smoke test")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    todo, paths = tasks(args.corpus, args.vendors)
    if args.check_dedup:
        return 1 if check_dedup(paths) else 0

    name, stamp = scorer_version(repo)
    stamp.update(environment())
    if args.limit:
        todo = todo[-args.limit:]        # `tasks` orders most expensive first
    print(f"  {sum(len(v) for v in paths.values())} predictions -> {len(todo)} to score, "
          f"{args.jobs} workers, scorer {name}")

    journal = args.out / SCORES_DIR / f"{name}.partial.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    rows = run(todo, args.corpus, args.jobs, journal)

    failed = [r for r in rows if r["kind"] == "error"]
    for r in failed[:10]:
        print(f"  ERROR {r['doc_id'][:46]}: {r['error'][:120]}")

    if args.recheck:
        return 1 if recheck(args.recheck, rows) else 0
    write_table(rows, args.out, name, stamp)
    journal.unlink()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
