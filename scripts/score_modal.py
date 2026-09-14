#!/usr/bin/env python3
"""Fan a scoring run out across Modal containers, producing the same tables as a local run.

    modal run scripts/score_modal.py --corpus build/ --vendors build/vendors --shards 40

The output is byte-for-byte what `build_scores.py` writes, in the same place:

    <out>/scores/<corpus_version>/<scorer_commit>/summary.parquet
    <out>/scores/<corpus_version>/<scorer_commit>/verdicts/<doc_id>.parquet

so a run done here and a run done on a laptop are the same artifact, and `--recheck` can
compare one against the other. That is the whole point of doing it this way rather than
inventing a remote-specific format.

**A shard is a slice of documents.** One container per (vendor, document) would be 5,940
containers, almost all cold start. A document is already the unit of work -- it is scored
against every prediction for it, parses its gold once, and writes one verdict file -- so
sharding needs no new concept: pack documents into shards of roughly equal cost, longest
first.

**Verdicts go to the volume; summaries come back as values.** The two tables differ by two
orders of magnitude (~600 KB of summary against ~0.2 GB of verdicts for the nine-vendor run),
and verdicts are already one file per document, so a container writes its own and nothing has
to be merged. Summary rows are small enough to return, so the driver assembles one table
rather than stitching fragments.

**Cost is not uniform and the containers should not be.** The three most expensive documents
are 61% of the corpus's CPU, and one of them takes 1,126s alone. Packing by estimated cost
keeps a shard's makespan from being set by scheduling luck; a shard holding a giant simply
holds fewer documents.

Staging: the corpus and predictions are uploaded to a Modal Volume once and mounted
read-only. Re-running against the same data skips the upload.

Usage:
    modal run scripts/score_modal.py --corpus build/ --vendors build/vendors --out build/
    modal run scripts/score_modal.py ... --shards 60
    modal run scripts/score_modal.py ... --stage        # force a re-upload
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: Where staged inputs and produced verdicts live between the driver and the containers.
VOLUME = "omni-extract-bench-runs"
MOUNT = Path("/data")

#: Everything a container needs. The repository is added as local source rather than
#: installed, so the scorer that runs remotely is the working tree you launched from -- not a
#: release that may be older, which is the kind of gap a `scorer_commit` column cannot see.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("scipy>=1.13.1", "numpy", "pyarrow>=15.0.0")
    .add_local_python_source("omni_extract_bench")
)

app = modal.App("omni-extract-bench-score", image=image)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)


def pack(work, shards: int):
    """Group documents into shards of roughly equal estimated cost, longest first.

    Greedy longest-processing-time: repeatedly give the next most expensive document to the
    shard holding the least. It is the standard makespan heuristic and it is within 4/3 of
    optimal, which is far better than this needs to be -- the point is only that one shard
    does not end up with all three of the documents that are 61% of the corpus.
    """
    buckets = [[] for _ in range(shards)]
    totals = [0] * shards
    for w in sorted(work, key=lambda w: -w.cost):
        i = totals.index(min(totals))
        buckets[i].append(w)
        totals[i] += max(w.cost, 1)
    return [(b, t) for b, t in zip(buckets, totals) if b]


@app.function(volumes={str(MOUNT): volume}, timeout=60 * 60 * 4, memory=8192,
              max_containers=64)
def score_shard(shard: list, run_dir: str, verdicts: bool):
    """Score one shard. Writes its verdict files; returns its summary rows.

    Returns the rows rather than writing a fragment table because the driver has to produce a
    single `summary.parquet` anyway, and stitching fragments is a second way for a run to be
    incomplete.
    """
    sys.path.insert(0, "/root")
    from omni_extract_bench.bench import Case, document, score, summary_row, verdict_rows

    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus = MOUNT / "corpus"
    out = MOUNT / run_dir
    (out / "verdicts").mkdir(parents=True, exist_ok=True)

    rows = []
    for entry, preds in shard:
        doc = document(corpus, _Entry(*entry))
        verds = []
        for pid, rel in preds:
            case = Case(doc=doc, prediction_id=pid,
                        raw=(MOUNT / "vendors" / rel).read_bytes())
            outcome = score(case, verdicts=verdicts)
            rows.append(summary_row(case, outcome))
            verds.extend(verdict_rows(case, outcome))
        if verds:
            pq.write_table(pa.Table.from_pylist(verds),
                           out / "verdicts" / f"{doc.doc_id}.parquet", compression="zstd")
    volume.commit()
    return rows


class _Entry(tuple):
    """The atlas row, rebuilt inside the container.

    A NamedTuple does not survive the trip unless the class is importable on both sides, and
    `corpus.Entry` is -- but sending plain tuples keeps the remote function's signature free of
    the layout module, so a change to the atlas's columns cannot silently mean something else
    in a container built from an older image.
    """

    doc_id = property(lambda self: self[0])
    ground_truth_path = property(lambda self: self[1])
    schema_path = property(lambda self: self[2])
    gt_sha256 = property(lambda self: self[3])
    schema_sha256 = property(lambda self: self[4])


def stage(corpus: Path, vendors: Path, force: bool) -> None:
    """Upload the corpus and the predictions the run needs, once.

    Only the two files a document is, plus the prediction payloads actually referenced. PDFs,
    page images and provenance stay home: the scorer never opens one, and they are 812 MB of
    the published corpus.
    """
    from omni_extract_bench import corpus as corpus_atlas

    entries = corpus_atlas.read(corpus)
    with volume.batch_upload(force=force) as up:
        up.put_file(corpus / corpus_atlas.ATLAS, f"corpus/{corpus_atlas.ATLAS}")
        for e in entries:
            up.put_file(corpus / e.ground_truth_path, f"corpus/{e.ground_truth_path}")
            up.put_file(corpus / e.schema_path, f"corpus/{e.schema_path}")
        for vd in sorted(p for p in vendors.iterdir() if p.is_dir()):
            for f in sorted(vd.glob("*.json")):
                up.put_file(f, f"vendors/{vd.name}/{f.name}")
    print(f"  staged {len(entries)} documents and their predictions")


@app.local_entrypoint()
def main(corpus: str, vendors: str, out: str = "build", shards: int = 40,
         stage_data: bool = False, no_verdicts: bool = False):
    """Shard the work, run it, and write the same tables a local run writes."""
    sys.path.insert(0, str(REPO))
    from build_scores import (SCORES_DIR, SUMMARY, VERDICTS, environment,
                              scorer_version, survey, write_summary)

    corpus_p, vendors_p, out_p = Path(corpus), Path(vendors), Path(out)
    work, paths, awaiting, corpus_v, skipped = survey(corpus_p, vendors_p)
    name, stamp = scorer_version(REPO)
    stamp.update({"corpus_version": corpus_v, **environment()})
    run_dir = f"{SCORES_DIR}/{corpus_v}/{name}"
    local = out_p / run_dir

    total = sum(len(w.preds) for w in work)
    print(f"  corpus {corpus_v}, scorer {name[:16]}")
    print(f"  {total} distinct predictions over {len(work)} documents")
    if (local / SUMMARY).exists():
        print(f"  {local / SUMMARY} exists; this corpus and this scorer already have an "
              f"answer.\n  To verify it reproduces: build_scores.py --recheck", file=sys.stderr)
        return 1

    stage(corpus_p, vendors_p, force=stage_data)

    # Paths are made relative here so a container never has to know where the driver's files
    # live -- only where the volume mounted them.
    packed = pack(work, shards)
    payload = [[[tuple(w.entry), [(pid, str(Path(p).relative_to(vendors_p))) for pid, p in w.preds]]
                for w in bucket] for bucket, _cost in packed]
    print(f"  {len(packed)} shards, cost {min(c for _, c in packed):,}-"
          f"{max(c for _, c in packed):,} array rows each")

    t0 = time.monotonic()
    rows = []
    for shard_rows in score_shard.starmap(
            [(s, run_dir, not no_verdicts) for s in payload]):
        rows.extend(shard_rows)
        print(f"  {len(rows)}/{total} predictions scored", flush=True)
    print(f"  {time.monotonic() - t0:.0f}s wall across {len(packed)} containers")

    local.mkdir(parents=True, exist_ok=True)
    (local / VERDICTS).mkdir(exist_ok=True)
    write_summary(local, rows, stamp)
    n = 0
    for f in volume.iterdir(f"{run_dir}/{VERDICTS}"):
        data = b"".join(volume.read_file(f.path))
        (local / VERDICTS / Path(f.path).name).write_bytes(data)
        n += 1
    print(f"  {VERDICTS}/: {n} files fetched to {local / VERDICTS}")

    failed = [r for r in rows if r["kind"] == "failed"]
    for r in failed[:10]:
        print(f"  NOT SCORED {r['doc_id'][:46]}: {r['error'][:110]}", file=sys.stderr)
    return 1 if failed else 0
