"""Score the benchmark on Modal, fanned out over (vendor, document).

    modal run scripts/score_modal.py                      # every vendor
    modal run scripts/score_modal.py --vendors datalab,reducto
    modal run scripts/score_modal.py --limit 20           # a smoke test

Corpus and predictions are read from R2; each container writes its own verdict table there and
returns only a summary row, because one document's verdicts can be 410,012 rows and the map's
results all come back through the driver.

Shape, from what a probe actually measured rather than from guessing:

* **`.map()` over (vendor, document), not hand-packed shards.** `REMOTE_RUNS.md` argued that
  5,940 inputs would be 5,940 cold starts. It is not how `.map()` works -- the probe saw five
  containers take [2, 2, 2, 3, 11] inputs, because Modal feeds a container many inputs.
* **`@modal.enter()` for per-container setup**, which is what makes that reuse pay. Boot was
  0.5-0.9s.
* **No corpus Volume.** Fetching a document's three objects costs ~0.55s, about a tenth of
  scoring it. A Volume is an optimisation, not a prerequisite.
* **No `@modal.concurrent`.** Scoring is CPU-bound and Modal's own guidance is that input
  concurrency is counterproductive for that. One input at a time, scale with containers.
* **A named Secret.** Decorator arguments are evaluated when the module is imported, and the
  module is imported IN the container too -- reading a local `.env` there crash-looped every
  container for 178 seconds before the probe gave up.
"""
import json
import pathlib
import sys
import time

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
BUCKET = "datalab-training-pipelines"
ROOT = "omni-extract-bench"
VENDORS = ("azure-cu", "claude", "datalab", "extend", "gemini", "gpt", "llamaextract",
           "mistral", "reducto")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("scipy==1.17.1", "numpy==2.4.6", "pyarrow==25.0.1", "boto3>=1.34")
    .add_local_dir(str(REPO / "omni_extract_bench"), "/root/omni_extract_bench")
)
app = modal.App("oeb-score", image=image)


def client():
    import os

    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], region_name="auto",
                        config=Config(retries={"max_attempts": 5, "mode": "standard"},
                                      max_pool_connections=16))


@app.cls(secrets=[modal.Secret.from_name("oeb-r2")], cpu=2.0, memory=8192,
         max_containers=40, timeout=3600, retries=2)
class Scorer:
    @modal.enter()
    def setup(self):
        sys.path.insert(0, "/root")
        self.s3 = client()

    @modal.method()
    def one(self, unit):
        vendor, doc_id, run = unit
        from omni_extract_bench.bench import (Case, Document, prediction_id, score,
                                              summary_row, verdict_rows)
        from omni_extract_bench.harness.dialects import resolve_refs, strip_benchmark_keys

        started = time.time()
        get = lambda key: self.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        try:
            gt = json.loads(get(f"{ROOT}/corpus/{doc_id}/ground_truth.json"))
            schema = resolve_refs(strip_benchmark_keys(
                json.loads(get(f"{ROOT}/corpus/{doc_id}/schema.json"))))
            raw = get(f"{ROOT}/vendors/{vendor}/{doc_id}.json")
        except Exception as exc:                                        # noqa: BLE001
            return {"doc_id": doc_id, "vendor": vendor, "kind": "failed",
                    "error": f"fetch: {type(exc).__name__}: {exc}"}

        case = Case(doc=Document(doc_id=doc_id, gt=gt, schema=schema),
                    prediction_id=prediction_id(raw), raw=raw)
        outcome = score(case, verdicts=True)       # `score` never raises; see bench.Outcome
        row = summary_row(case, outcome)
        row["vendor"] = vendor
        row["secs"] = round(time.time() - started, 3)

        verds = verdict_rows(case, outcome)
        if verds:
            import pyarrow as pa
            import pyarrow.parquet as pq
            tmp = pathlib.Path(f"/tmp/{doc_id}.parquet")
            pq.write_table(pa.Table.from_pylist(verds), tmp, compression="zstd")
            self.s3.upload_file(str(tmp), BUCKET,
                                f"{run}/{vendor}/verdicts/{doc_id}.parquet")
            tmp.unlink()

        # The row goes to the bucket as well as back through the map. A run that dies -- and
        # the first one did, when the laptop holding its heartbeat went to sleep -- then loses
        # nothing: every finished unit is on disk somewhere durable, `collect` assembles the
        # summaries afterwards, and starting again skips what is already there.
        self.s3.put_object(Bucket=BUCKET, Key=f"{run}/{vendor}/rows/{doc_id}.json",
                           Body=json.dumps(row).encode())
        return row


@app.function(secrets=[modal.Secret.from_name("oeb-r2")], cpu=0.25, memory=2048,
              timeout=7200, retries=0)
def orchestrate(picked: list, limit: int, run: str) -> dict:
    """Choose the work, run it, write the summaries -- all of it inside Modal.

    This is the whole point of the module's shape. `modal run` ties an ephemeral app to the
    local client's heartbeat, so a laptop going to sleep stopped the app mid-run once already;
    and while the orchestration lived in the local entrypoint, closing the lid also meant no
    summaries however much scoring had finished. Spawned, none of that is true: the laptop
    hands over three arguments and can go away.

    A quarter of a CPU, because this container spends its life waiting on `.map()`.
    """
    import sys as _sys
    _sys.path.insert(0, "/root")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from omni_extract_bench import s3 as S3

    local = pathlib.Path("/tmp/oeb")
    local.mkdir(parents=True, exist_ok=True)
    S3.fetch_one(f"s3://{BUCKET}/{ROOT}/corpus/corpus.parquet", local / "corpus.parquet")
    doc_ids = [r["doc_id"] for r in pq.read_table(local / "corpus.parquet").to_pylist()]
    if limit:
        doc_ids = doc_ids[:limit]

    units, resumed = [], 0
    for v in picked:
        held = {n[:-5] for n in S3.names(f"s3://{BUCKET}/{ROOT}/vendors/{v}")
                if n.endswith(".json") and "/" not in n}
        done = {n[len("rows/"):-5] for n in S3.names(f"s3://{BUCKET}/{run}/{v}")
                if n.startswith("rows/") and n.endswith(".json")}
        resumed += len(done)
        units += [(v, d, run) for d in doc_ids if d in held and d not in done]
    print(f"  {len(doc_ids)} documents, {len(picked)} vendors, {len(units)} units"
          + (f", resuming over {resumed}" if resumed else ""), flush=True)

    started = time.time()
    rows, raised, seen = [], [], 0
    if units:
        for result in Scorer().one.map(units, order_outputs=False, return_exceptions=True):
            seen += 1
            if isinstance(result, dict):
                rows.append(result)
            else:
                raised.append(repr(result)[:200])
            if seen % 500 == 0 or seen == len(units):
                print(f"    {seen}/{len(units)}  {time.time() - started:.0f}s", flush=True)

    # Summaries from the rows the map returned, not read back out of the bucket. Reading them
    # back cost ~350s of one-round-trip-per-object, and lost the units that failed before they
    # could write anything -- a vendor came out one document short with no failure reported.
    #
    # A resumed run has to read, though: rows from an earlier attempt are only in the bucket.
    by_vendor = {}
    for r in rows:
        by_vendor.setdefault(r["vendor"], []).append(r)
    report = {}
    for v in sorted(picked):
        rs = by_vendor.get(v, [])
        if resumed:
            have = {r["doc_id"] for r in rs}
            names = [n for n in S3.names(f"s3://{BUCKET}/{run}/{v}")
                     if n.startswith("rows/") and n.endswith(".json")
                     and n[len("rows/"):-5] not in have]
            if names:
                into = local / "earlier" / v
                S3.fetch(f"s3://{BUCKET}/{run}/{v}", into, names)
                rs = rs + [json.loads((into / n).read_bytes()) for n in names]
        if not rs:
            continue
        _write_vendor(pa, pq, S3, run, v, rs, local)
        graded = [r for r in rs if r.get("kind") == "graded"]
        report[v] = {"rows": len(rs), "graded": len(graded),
                     "mean": round(sum(r["accuracy"] for r in graded) / len(graded), 2)
                             if graded else None,
                     "failed": [f"{r['doc_id']}: {r.get('error')}"
                                for r in rs if r.get("kind") == "failed"]}
    return {"run": run, "units": len(units), "secs": round(time.time() - started),
            "raised": raised, "vendors": report}


@app.local_entrypoint()
def main(vendors: str = ",".join(VENDORS), limit: int = 0, run: str = "", wait: bool = True):
    """Spawn the run. Everything after this happens in Modal.

        modal run --detach scripts/score_modal.py::main --no-wait

    **`--detach` is not optional with `--no-wait`.** `modal run` builds an EPHEMERAL app and
    stops it when the entrypoint returns, which kills the thing that was just spawned -- the
    collected result is a `RemoteError` with no message, because the container was killed
    rather than raising. `--detach` keeps the app alive; then this machine really is finished
    once it prints a call id.
    """
    run = run or f"{ROOT}/scores/{time.strftime('%Y-%m-%dT%H-%M-%SZ', time.gmtime())}"
    picked = [v.strip() for v in vendors.split(",") if v.strip()]
    call = orchestrate.spawn(picked, limit, run)
    print(f"  run prefix : s3://{BUCKET}/{run}")
    print(f"  call id    : {call.object_id}")
    if not wait:
        print("\n  running in Modal; this machine is no longer involved"
              " (as long as you passed --detach).\n  Collect it with:\n"
              f"    modal run scripts/score_modal.py::status "
              f"--call {call.object_id}")
        return
    show(call.get())


@app.local_entrypoint()
def status(call: str):
    """Collect a spawned run, whenever. Blocks until it finishes."""
    show(modal.FunctionCall.from_id(call).get())


def show(out: dict):
    print(f"\n  {out['units']} units, {out['secs']}s, {len(out['raised'])} raised")
    for v, r in sorted(out["vendors"].items()):
        mean = f"{r['mean']:>6.2f}" if r["mean"] is not None else "     -"
        print(f"  {v:<14} {r['rows']:>4} rows   mean {mean} over {r['graded']}"
              + (f"   {len(r['failed'])} FAILED" if r["failed"] else ""))
    for v, r in sorted(out["vendors"].items()):
        for f in r["failed"][:3]:
            print(f"      {v}/{f}")
    print(f"  s3://{BUCKET}/{out['run']}")


def write_summaries(S3, run, picked, local):
    """One summary.parquet per vendor, from that vendor's rows in the bucket."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    for v in sorted(picked):
        names = [n for n in S3.names(f"s3://{BUCKET}/{run}/{v}")
                 if n.startswith("rows/") and n.endswith(".json")]
        if not names:
            continue
        into = local / run.replace("/", "_") / v
        S3.fetch(f"s3://{BUCKET}/{run}/{v}", into, names)
        rs = [json.loads((into / n).read_bytes()) for n in names]
        _write_vendor(pa, pq, S3, run, v, rs, local)


def _write_vendor(pa, pq, S3, run, v, rs, local):
    fields = list(dict.fromkeys(k for r in rs for k in r))
    blank = {k: None for k in fields}
    table = pa.Table.from_pylist([{**blank, **r} for r in rs])
    out = local / f"{v}.parquet"
    pq.write_table(table, out, compression="zstd")
    S3.upload(out, f"s3://{BUCKET}/{run}/{v}/summary.parquet")
    graded = [r for r in rs if r.get("kind") == "graded"]
    mean = sum(r["accuracy"] for r in graded) / len(graded) if graded else 0.0
    kinds = {}
    for r in rs:
        kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
    print(f"  {v:<14} {len(rs):>4} rows   mean {mean:>6.2f} over {len(graded)}   {kinds}")
