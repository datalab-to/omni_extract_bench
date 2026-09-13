#!/usr/bin/env python3
"""Score every vendor's R2 predictions against the Hub ground truth, exact matching.

Reads predictions from `s3://datalab-training-pipelines/omni-extract-bench/runs/full/
baselines/<vendor>/<suite>/<doc_id>.json` and grades them against the corpus on the Hub.

Accounting follows `cli.py` deliberately. A document a vendor did not return, or returned
an error for, scores 0 and STAYS in the mean -- dropping it would reward failing on hard
documents. A document this harness could not score is excluded and reported separately,
because that is a harness fault, not a vendor result.

Two things this does that `cli.py` cannot yet:

  * It unwraps the run envelope. `cli.py::_unwrap` only unwraps an exact
    {"result": ...}, and every file in this run carries at least "_secs" beside it, so
    through the CLI every document of every vendor scores 0.00 while `usable()` still
    returns True -- it reads as total provider failure. See TO_LOOK_AT.md item 1; the
    rule lives here rather than in the library until that lands.
  * It walks the Hub snapshot directly, because the corpus is a tree of per-document
    folders and the CLI wants three flat directories.

Resumable: results append to <out>/<vendor>.jsonl as they finish, and a rerun skips
documents already there. Safe to Ctrl-C and restart.

Usage:
    uv run --python 3.11 --with boto3 --with huggingface_hub python scripts/score_r2.py
    uv run ... python scripts/score_r2.py --vendors datalab_full reducto_full --jobs 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from huggingface_hub import snapshot_download

from omni_extract_bench.dialects import resolve_refs, strip_benchmark_keys
from omni_extract_bench.prediction_io import usable
from omni_extract_bench.score import grade

BUCKET = "datalab-training-pipelines"
BASE = "omni-extract-bench/runs/full/baselines"
REPO = "datalab-to/omni_extract_bench"
ENV = Path.home() / "datalab" / "gke_pipelines" / ".env"

VENDORS = ["datalab_full", "reducto_full", "extend_full", "llamaextract_full",
           "azure-cu_full", "claude_full", "gemini_full", "gpt_full", "mistral_full"]

#: Resident memory to budget per worker, in GB. The largest document in this corpus peaks
#: near 1.9 GB on the greedy path and ~6 GB if the exact caps are ever raised; 4 leaves
#: room without leaving cores idle. Longest-first scheduling starts the expensive
#: documents together, so peak memory arrives early in the run, not late.
GB_PER_WORKER = 4

_local = threading.local()


def _creds():
    cfg = {}
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
               if k not in cfg]
    if missing:
        sys.exit(f"{ENV} is missing {', '.join(missing)}")
    return cfg


CFG = None


def r2():
    if not hasattr(_local, "c"):
        _local.c = boto3.client(
            "s3", endpoint_url=CFG["R2_ENDPOINT_URL"],
            aws_access_key_id=CFG["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=CFG["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(retries={"max_attempts": 5, "mode": "standard"},
                          max_pool_connections=32))
    return _local.c


def default_jobs():
    """Budget by memory as well as cores, and take whichever is tighter."""
    cpus = os.cpu_count() or 1
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (AttributeError, ValueError, OSError):
        return min(cpus, 4)
    return max(1, min(cpus, int(ram // GB_PER_WORKER)))


def unwrap(obj, schema_props=()):
    """Peel the run envelope: {"result": ..., "_secs": ..., ...}.

    Decided from the schema, not from the metadata key names -- an earlier attempt keyed
    on a leading underscore and still missed "recovered_after_timeout". If the schema
    declares no top-level "result" property, a "result" key is the envelope.
    """
    if isinstance(obj, dict) and "result" in obj and "result" not in schema_props:
        return obj["result"]
    return obj


def fetch(task):
    """Download one prediction to the cache. Returns (vendor, suite, doc_id, path|None)."""
    vendor, suite, doc_id, cache = task
    dest = Path(cache) / vendor / suite / f"{doc_id}.json"
    if dest.exists():
        return vendor, suite, doc_id, dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        body = r2().get_object(Bucket=BUCKET,
                               Key=f"{BASE}/{vendor}/{suite}/{doc_id}.json")["Body"]
        dest.write_bytes(body.read())
        return vendor, suite, doc_id, dest
    except Exception:
        return vendor, suite, doc_id, None      # nothing returned for this document


def score(task):
    """Grade one document in a worker process. Exact matching -- no similarity."""
    vendor, suite, doc_id, pred_path, root = task
    d = Path(root) / "data" / suite / doc_id
    try:
        schema = resolve_refs(strip_benchmark_keys(
            json.loads((d / "schema.json").read_text())))
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        gt = unwrap(json.loads((d / "ground_truth.json").read_text()), props)
    except Exception as exc:
        return dict(vendor=vendor, suite=suite, doc=doc_id, kind="failed",
                    why=f"ground truth/schema: {exc}")

    if pred_path is None:
        return dict(vendor=vendor, suite=suite, doc=doc_id, kind="absent")
    try:
        pred = unwrap(json.loads(Path(pred_path).read_text()), props)
    except Exception as exc:
        return dict(vendor=vendor, suite=suite, doc=doc_id, kind="failed",
                    why=f"unreadable prediction: {exc}")
    if not usable(pred):
        err = pred.get("__error__") if isinstance(pred, dict) else None
        return dict(vendor=vendor, suite=suite, doc=doc_id, kind="empty", why=err)
    try:
        t0 = time.monotonic()
        r = grade(pred, gt, schema)
        return dict(vendor=vendor, suite=suite, doc=doc_id, kind="graded",
                    secs=round(time.monotonic() - t0, 2),
                    accuracy=r["accuracy"], f1=r["f1"], precision=r["precision"],
                    recall=r["recall"], found=r["found"], read_right=r["read_right"],
                    matched=r["matched"], total=r["total"],
                    fabricated=r["fabricated"], invented_item=r["invented_item"],
                    invented_field=r["invented_field"],
                    matching_exact=r["matching_exact"])
    except Exception as exc:
        return dict(vendor=vendor, suite=suite, doc=doc_id, kind="failed",
                    why=f"{type(exc).__name__}: {exc}")


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def report(vendor, rows, out):
    graded = [r for r in rows if r["kind"] == "graded"]
    empty = [r for r in rows if r["kind"] in ("empty", "absent")]
    failed = [r for r in rows if r["kind"] == "failed"]
    scores = [r["accuracy"] for r in graded] + [0.0] * len(empty)
    measured = len(graded) + len(empty)

    lines = [f"\n{'=' * 74}", f"{vendor}", "=" * 74,
             f"  documents   {len(rows)}",
             f"  scored      {measured}   ({len(graded)} returned, "
             f"{len(empty)} empty/absent -> 0)"]
    if failed:
        lines.append(f"  NOT SCORED  {len(failed)}   excluded -- harness, not vendor")
    lines += [f"  score       {mean(scores):.2f}   over the {measured} scored",
              f"  on returned {mean([r['accuracy'] for r in graded]):.2f}",
              f"  f1          {mean([r['f1'] for r in graded]) * 100:.2f}",
              ""]
    by = {}
    for r in graded:
        by.setdefault(r["suite"], []).append(r["accuracy"])
    for s in sorted(by):
        lines.append(f"    {s:<14} {mean(by[s]):6.2f}   n={len(by[s])}")

    inexact = [r for r in graded if not r.get("matching_exact", True)]
    if inexact:
        lines.append(f"\n    approximately matched (greedy fallback): {len(inexact)}")
        for r in inexact:
            lines.append(f"      {r['suite']}/{r['doc'][:56]}")
    for r in failed[:20]:
        lines.append(f"\n    FAILED {r['suite']}/{r['doc'][:50]}\n      {r.get('why')}")
    text = "\n".join(lines)
    print(text, flush=True)
    (out / f"{vendor}.summary.txt").write_text(text)
    return mean(scores), measured, len(graded)


def main():
    global CFG
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendors", nargs="+", default=VENDORS,
                    help=f"default: all {len(VENDORS)}")
    ap.add_argument("--jobs", "-j", type=int, default=default_jobs())
    ap.add_argument("--limit", type=int, help="first N documents per vendor, for a smoke test")
    ap.add_argument("--out", type=Path, default=Path("r2_scores"))
    ap.add_argument("--cache", type=Path,
                    default=Path.home() / ".cache" / "omni_extract_bench_preds")
    args = ap.parse_args()

    CFG = _creds()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(snapshot_download(REPO, repo_type="dataset"))
    man = json.loads((root / "manifest.json").read_text())

    docs = [(s, e["doc_id"]) for s in sorted(man) for e in man[s]]
    # Limit first, in manifest order, so --limit stays a quick smoke test rather than
    # "the N most expensive documents in the corpus".
    if args.limit:
        docs = docs[:args.limit]
    # Then longest first, which is scheduling only and does not change what runs: cost
    # spans three orders of magnitude, and with a pool this wide the makespan is set by
    # when the biggest document STARTS, not by the total.
    docs.sort(key=lambda sd: (root / "data" / sd[0] / sd[1]
                              / "ground_truth.json").stat().st_size, reverse=True)

    print(f"{len(docs)} documents x {len(args.vendors)} vendors, exact matching, "
          f"-j {args.jobs}", flush=True)
    print(f"cache {args.cache}\nout   {args.out.resolve()}\n", flush=True)

    board = []
    for vendor in args.vendors:
        jsonl = args.out / f"{vendor}.jsonl"
        done = {}
        if jsonl.exists():
            for line in jsonl.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    done[(r["suite"], r["doc"])] = r
        todo = [sd for sd in docs if sd not in done]
        print(f"\n--- {vendor}: {len(done)} already done, {len(todo)} to go ---",
              flush=True)

        if todo:
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=16) as pool:
                fetched = list(pool.map(
                    fetch, [(vendor, s, d, str(args.cache)) for s, d in todo]))
            miss = sum(1 for *_, p in fetched if p is None)
            print(f"  downloaded {len(fetched) - miss}/{len(fetched)} in "
                  f"{time.monotonic() - t0:.0f}s" + (f", {miss} absent" if miss else ""),
                  flush=True)

            tasks = [(v, s, d, str(p) if p else None, str(root)) for v, s, d, p in fetched]
            t0 = time.monotonic()
            with jsonl.open("a") as fh, ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futs = [pool.submit(score, t) for t in tasks]
                for i, f in enumerate(as_completed(futs), 1):
                    r = f.result()
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()                      # so Ctrl-C never loses finished work
                    done[(r["suite"], r["doc"])] = r
                    if i % 20 == 0 or i == len(futs):
                        el = time.monotonic() - t0
                        eta = el / i * (len(futs) - i)
                        print(f"  [{i:>4}/{len(futs)}] {el / 60:5.1f}m elapsed, "
                              f"~{eta / 60:4.1f}m left", flush=True)

        board.append((vendor, *report(vendor, list(done.values()), args.out)))

    print(f"\n{'=' * 74}\nLEADERBOARD  (exact matching, {len(docs)} documents)\n{'=' * 74}")
    print(f"{'vendor':<22}{'score':>9}{'returned':>11}{'scored':>9}")
    for v, sc, measured, ret in sorted(board, key=lambda t: -t[1]):
        print(f"{v:<22}{sc:>9.2f}{f'{ret}/{measured}':>11}{measured:>9}")
    (args.out / "leaderboard.txt").write_text(
        "\n".join(f"{v:<22}{sc:>9.2f}  {ret}/{measured}"
                  for v, sc, measured, ret in sorted(board, key=lambda t: -t[1])))
    print(f"\nwrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
