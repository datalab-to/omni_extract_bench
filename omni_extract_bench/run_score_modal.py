from __future__ import annotations

import os

import modal

app = modal.App("omni-extract-bench-score")

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("pyarrow>=15.0.0", "fsspec>=2024.2.0", "s3fs>=2024.2.0", "scipy>=1.13.1")
         # The local package rather than a published wheel: a scorer change should be one
         # `modal run` away, not a release.
         .add_local_python_source("omni_extract_bench"))

#: The named Modal secret holding the bucket credentials -- the way Modal recommends for code
#: that other people run, since it lives outside the source and is shared across a team. Set
#: `OEB_MODAL_SECRET` to point at one you already have:
#:
#:     modal secret create oeb-s3 \
#:         AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... FSSPEC_S3_ENDPOINT_URL=...
#:
#: The endpoint is only needed for R2 or another S3-compatible store; on AWS, drop it.
SECRET = os.environ.get("OEB_MODAL_SECRET", "oeb-s3")


@app.function(image=image, secrets=[modal.Secret.from_name(SECRET)],
              cpu=1, memory=4096, timeout=3600, retries=1)
def score_part(manifest: str, out: str, offset: int, length: int, name: str) -> dict:
    """Score rows `[offset, offset+length)` of the manifest and write them as one part.

    Retried once, which is safe because a part is written under a name only this batch uses:
    a retry overwrites its own output rather than adding to someone else's.
    """
    from omni_extract_bench.run_score import open_manifest, score_batch

    rows = open_manifest(manifest).to_table().slice(offset, length).combine_chunks()
    # combine_chunks leaves one chunk per column, so this is exactly one record batch.
    return score_batch(rows.to_batches()[0], out, name)


@app.function(image=image, secrets=[modal.Secret.from_name(SECRET)], timeout=24 * 3600)
def orchestrate(manifest: str, out: str, rows: int) -> dict:
    """Fan the manifest out and add up what comes back. Runs in a container, not on a laptop.

    The driver lives here rather than in the local entrypoint for one reason: a run of a real
    corpus takes hours, and nobody should have to leave a laptop open for it. Everything --
    submitting the work, collecting the tallies -- happens server-side.
    """
    from omni_extract_bench.run_score import REQUIRED, RESERVED, check_manifest, open_manifest

    data = open_manifest(manifest)
    # Before a single container starts: a missing column or a repeated doc_id should cost one
    # read, not a fleet.
    check_manifest(data, REQUIRED, RESERVED)
    total = data.count_rows()
    print(f"{total} documents, {rows} rows per container -> {out}", flush=True)

    work = [(manifest, out, offset, min(rows, total - offset), f"part-{i:05d}")
            for i, offset in enumerate(range(0, total, rows))]

    tally: dict[str, int] = {"scored": 0, "error": 0}
    broke = 0
    # `order_outputs=False` because nothing here needs them in order -- each container has
    # already written its own part -- and ordering would hold every finished tally behind the
    # slowest early batch, which is precisely when you want to see progress.
    for result in score_part.starmap(work, order_outputs=False, return_exceptions=True):
        # A container that died -- OOM, timeout, a bug -- is not a scored row and must not be
        # counted as one. Its part is simply absent, which is what makes a rerun a rerun.
        if isinstance(result, Exception):
            broke += 1
            print(f"  container failed: {type(result).__name__}: {result}", flush=True)
            continue
        for status, count in result.items():
            tally[status] += count
        print(f"  {sum(tally.values())}/{total} documents  "
              + "  ".join(f"{k}={v}" for k, v in tally.items()), flush=True)
    return {**tally, "containers_died": broke}


@app.local_entrypoint()
def main(manifest: str, out: str, rows: int = 16):
    """Start the run and wait for it.

    Waiting rather than spawning, even though the point of this file is a run that outlives the
    laptop. A `spawn` without `--detach` is silently discarded -- Modal stops an ephemeral App
    when the calling program exits, so the call never runs while the command still prints a
    tally and exits zero. That is the worst failure a runner can have.

    Blocking is correct in both modes. Without `--detach` you watch it finish. With `--detach`,
    Modal keeps the last triggered function alive when this process goes away, and the last
    triggered function is exactly this one -- so the laptop can still close mid-run, which is
    what this file is for.
    """
    print(f"scoring {manifest} -> {out}")
    print(f"  logs:  modal app logs {app.app_id}")
    print("  started with --detach? then this laptop can close and the run carries on")
    tally = orchestrate.remote(manifest, out, rows)
    print(f"{out}: " + "  ".join(f"{k}={v}" for k, v in tally.items()))
