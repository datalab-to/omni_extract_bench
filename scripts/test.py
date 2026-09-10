"""Load the benchmark from the Hub and self-grade it as a smoke test.

The dataset is a tree of per-document folders, not a table, so `load_dataset`
cannot read it: pointed at the repo it globs the loose JSON files and tries to
parse them as JSON-lines, which fails. Pull the snapshot instead and walk
`manifest.json`.

Grading is CPU-bound Python, so documents are graded in worker processes.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import snapshot_download

from omni_extract_bench.dialects import resolve_refs
from omni_extract_bench.score import grade, explain

REPO = "datalab-to/omni_extract_bench"


def documents(root: Path):
    """Yield (suite, doc_id, doc_dir) for every document in the manifest."""
    manifest = json.loads((root / "manifest.json").read_text())
    for suite, entries in manifest.items():
        for entry in entries:
            yield suite, entry["doc_id"], root / "data" / suite / entry["doc_id"]


def read(doc: Path):
    """Load one document's ground truth and schema.

    Schemas ship with local `$ref`; `grade` refuses one, so inline them here.
    """
    gt = json.loads((doc / "ground_truth.json").read_text())
    schema = resolve_refs(json.loads((doc / "schema.json").read_text()))
    return gt, schema


def self_grade(task):
    """Grade one document against itself. Runs in a worker process.

    Takes a path rather than the parsed document: the biggest ground truth here
    is 2.3 MB of JSON, and pickling that through the pool would cost more than
    the grading it saves. Returns only the summary, for the same reason.
    """
    suite, doc_id, doc = task
    gt, schema = read(Path(doc))
    start = time.monotonic()
    result = grade(gt, gt, schema)
    return suite, doc_id, result, time.monotonic() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="stop after N documents")
    ap.add_argument("--suite", help="only this suite (extractbench, micro1, ...)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                    help="worker processes; 1 grades in-process, for debugging")
    args = ap.parse_args()

    root = Path(snapshot_download(REPO, repo_type="dataset"))

    tasks = [(s, d, doc) for s, d, doc in documents(root)
             if not args.suite or s == args.suite]

    # Limit first, in manifest order, so --limit stays a quick sample rather than
    # "the N most expensive documents in the corpus".
    if args.limit:
        tasks = tasks[:args.limit]

    # Then longest first, which is scheduling only and does not change what runs.
    # Cost per document spans three orders of magnitude, and with a pool this wide
    # the makespan is set by when the biggest document STARTS: pick it up last and
    # every other worker idles waiting on it. Ground-truth size is a good enough
    # stand-in for cost to get the ordering right.
    tasks.sort(key=lambda t: (t[2] / "ground_truth.json").stat().st_size, reverse=True)

    # Paths do not survive spawn as Path objects on every platform; send strings.
    tasks = [(s, d, str(doc)) for s, d, doc in tasks]

    print(f"{len(tasks)} documents, {args.jobs} workers")
    started = time.monotonic()
    done = 0
    imperfect = []

    def record(suite, doc_id, result, elapsed):
        nonlocal done
        done += 1
        print(f"[{done:>3}/{len(tasks)}] {result['accuracy']:6.2f} {elapsed:6.1f}s  "
              f"{suite}/{doc_id}", flush=True)
        if result["accuracy"] != 100:
            imperfect.append((suite, doc_id, result))

    if args.jobs == 1:
        for task in tasks:
            record(*self_grade(task))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(self_grade, t) for t in tasks]
            for future in as_completed(futures):
                record(*future.result())

    wall = time.monotonic() - started
    print(f"\n{len(tasks)} documents in {wall:.1f}s")

    for suite, doc_id, result in imperfect:
        print(f"\n{suite}/{doc_id}: {result['accuracy']:.2f} "
              f"({result['matched']}/{result['total']})")
        # A self-grade below 100 is a scorer bug, so show the addresses it got
        # wrong rather than just the number. Re-read here: the worker sent back
        # the summary only, and there should be few enough of these to not care.
        gt, schema = read(root / "data" / suite / doc_id)
        for v in explain(gt, gt, schema):
            if v.verdict != "match":
                print(f"    {v.verdict:15} {v.address}")

    if imperfect:
        print(f"\n{len(imperfect)} documents do not self-grade to 100")
        sys.exit(1)
    print("all self-grade to 100")


if __name__ == "__main__":
    main()
