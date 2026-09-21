#!/usr/bin/env python3
"""Run the benchmark on Modal, detached, and copy what it produced to R2.

    modal secret create oeb-vendor-keys DATALAB_API_KEY=... REDUCTO_API_KEY=... \
                                        OPENROUTER_API_KEY=...
    modal secret create oeb-r2 AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
                               R2_BUCKET=oeb-results \
                               R2_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com

    modal run --detach modal_bench.py --providers "datalab reducto"
    modal run --detach modal_bench.py --providers datalab \
        --options '{"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]}'

`--detach` is the point: the run survives closing the terminal, and a benchmark takes hours.

WHY THIS IS SHORT. Modal's advice for a long job is to checkpoint to a Volume, retry on
interruption, and resume from what is on disk. The benchmark already does the third part --
a document is predicted again only when it has no record, graded again only when it has no row
in `scores.jsonl` -- so the whole wrapper is a Volume, a timeout and a retry count.

THE RUN DIRECTORY IS A VOLUME, NOT AN R2 MOUNT. Every durable write here is a temp file and
`os.replace`, and a `CloudBucketMount` does not support rename: "Certain file operations, such
as renaming files, are not supported". Pointed at R2 the first prediction fails with
`PermissionError`. R2 is where the results are copied afterwards, over the S3 API.

ONE CONTAINER, NOT ONE PER PROVIDER. The concurrency cap is per adapter and shared across the
runs of it -- two datalab tiers split one pool of ten -- and that only holds inside one process.
Fanning out per provider would put ten in flight per container, which is the thing the shared
pool exists to stop.
"""
import json
import logging
import os
import pathlib

import modal

APP_NAME = "omni-extract-bench"
STATE = pathlib.Path("/state")
DAY_S = 24 * 60 * 60

app = modal.App(APP_NAME)

#: The package's own dependencies, installed rather than the package, so the working tree is
#: what runs. `boto3` is this wrapper's, for the copy at the end.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "scipy>=1.13.1",                                        # the scorer
        "httpx>=0.27", "requests>=2.31", "openai>=1.0",         # [harness]
        "huggingface_hub>=0.23", "pyarrow>=15.0.0", "rich>=13.0",   # [benchmark]
        "boto3>=1.34",
    )
    .add_local_python_source("omni_extract_bench")
)

#: The corpus and the runs. A Volume commits every few seconds and on shutdown, so a container
#: killed at the timeout leaves the same resumable state a Ctrl-C would, and the retry does not
#: re-download 846MB of corpus either.
state = modal.Volume.from_name("oeb-state", create_if_missing=True)


def upload(out: pathlib.Path, prefix: str, full: bool = False) -> int:
    """Copy a finished run tree to R2 over the S3 API. Returns the number of objects written.

    BOTO3 RATHER THAN A BUCKET MOUNT, so no FUSE is in the path of the one thing worth
    keeping. A mount would also refuse `copy2`, which calls `os.utime`.

    Overwrites: the same logical run is uploaded again on every resume, and the latest state
    is the one worth having. `region_name="auto"` is R2's.
    """
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["R2_BUCKET"]
    # The small files are the result; `predictions/` and `records/` are 1240 files per run and
    # only worth carrying when somebody asks, which is what `--full` is.
    wanted = ["settings.json", "summary.json", "scores.jsonl"]
    sent = 0
    for run_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        paths = [run_dir / name for name in wanted]
        if full:
            paths += sorted(run_dir.rglob("*.json")) + sorted(run_dir.rglob("*.jsonl"))
        # `rglob` picks the three above up again, and an object uploaded twice is a second
        # PUT for the same bytes. Deduplicated in order, so the result is stable.
        for path in dict.fromkeys(paths):
            if not path.is_file():
                continue
            key = f"{prefix}/{path.relative_to(out)}"
            client.upload_file(str(path), bucket, key)
            sent += 1
    logging.getLogger(__name__).info("uploaded %d objects to r2://%s/%s", sent, bucket, prefix)
    return sent


@app.function(
    image=image,
    volumes={str(STATE): state},
    secrets=[modal.Secret.from_name("oeb-vendor-keys"),
             modal.Secret.from_name("oeb-r2")],
    timeout=DAY_S,                                  # the most Modal allows
    retries=modal.Retries(max_retries=10),          # each retry resumes, so ~10 days
    cpu=8,                                          # grading is the CPU-bound half
)
def bench(providers: str, options: str = "", limit: int = 0, suites: str = "",
          prefix: str = "runs", full: bool = False) -> dict:
    """One invocation of the benchmark, then the copy to R2.

    Arguments are the strings `modal run` can parse, in the shapes `oeb benchmark` takes.
    """
    from omni_extract_bench.benchmark import BenchmarkRun

    # The root stays at WARNING and only our namespace is turned up, as `oeb` does it: httpx
    # logs a line per request, and `openai` logs its own copy of the same line.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logging.getLogger("omni_extract_bench").setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)

    out = STATE / prefix
    run = BenchmarkRun(
        providers.split(),
        out=out,
        data_root=STATE / "benchmark",
        suites=suites.split() or None,
        limit=limit,
        options=json.loads(options) if options else None,
    )
    # No `confirm`: a detached run has no terminal to ask, so `go` logs the plan instead and
    # `modal app logs` shows what it is about to spend before it spends it.
    summary = run.go()
    state.commit()          # before the upload reads it back, rather than waiting for a tick
    upload(out, prefix, full=full)
    return summary


@app.local_entrypoint()
def main(providers: str, options: str = "", limit: int = 0, suites: str = "",
         prefix: str = "runs", full: bool = False):
    """What `modal run` calls. `modal run` parses str/int/float/bool, so providers is a
    space-separated string and options is JSON -- the same text `oeb benchmark` takes."""
    summary = bench.remote(providers, options, limit, suites, prefix, full)
    for label, row in summary.items():
        print(f"{label:<36} accuracy {row['accuracy']:.4f}  "
              f"coverage {row['scored']}/{row['documents']}")
