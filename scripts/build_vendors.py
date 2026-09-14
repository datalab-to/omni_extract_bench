#!/usr/bin/env python3
"""Build the prediction trees and their atlases, one per vendor.

Reads predictions in the shape the run produced -- `<vendor>/<suite>/<doc_id>.json`, each a
`{"result": ..., "_secs": ...}` envelope -- and writes the shape `docs/DATA_LAYOUT.md`
settles on:

    <out>/vendors/<vendor>.parquet
    <out>/predictions/<vendor>/<doc_id>.json      the bare extraction, nothing else

Storing the bare result is not a space optimisation: the envelope is a few dozen bytes on
files averaging 98 KB, and stripping it measured **0.0% smaller** across all 5,936. It is an
identity decision. With no envelope there is nothing to unwrap, so `prediction_id` is a plain
hash of the file, reproducible in any language with no rule to agree on.

Run metadata does not go in the rows. `timeout_s`, `model` and the rest live in `_raw/`, and
sampling showed they are run- and vendor-level constants rather than per-document facts:
`timeout_s` is 1800 for all 5,940, `model` is fixed per vendor, and `wall_s` duplicates the
envelope's `_secs`. So the builder fetches ONE `_raw` object per vendor for the stamp instead
of 5,940 for columns -- 1,084 MB of traffic that buys nothing.

`tier` is deliberately not carried. It is hand-written prose, not a fact: gpt's predictions
hold two different descriptions of the identical model `openai/gpt-5.6-sol`.

Usage:
    uv run --with pyarrow --with boto3 python scripts/build_vendors.py --out build/
    uv run ... python scripts/build_vendors.py --out build/ --vendors datalab_full
    uv run ... python scripts/build_vendors.py --out build/ --verify-only
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from omni_extract_bench.layout import (                                   # noqa: E402
    PREDICTION_ID_VERSION, extract_result, prediction_id, verify)

BUCKET = "datalab-training-pipelines"
R2_ROOT = "omni-extract-bench/runs/full/baselines"
ENV_FILE = Path.home() / "datalab" / "gke_pipelines" / ".env"

#: Where the run's predictions are mirrored locally. This is what `scripts/score_r2.py`
#: populates, and it holds `<vendor>/<suite>/<doc_id>.json` exactly as R2 does.
DEFAULT_SOURCE = Path.home() / ".cache" / "omni_extract_bench_preds"


def r2_client():
    """A client for the run bucket, or None if the credentials are not present.

    Only needed for the per-vendor stamp, so a build without credentials still produces
    correct tables -- it just cannot say which model made the predictions.
    """
    try:
        import boto3
    except ImportError:
        return None
    if not ENV_FILE.exists():
        return None
    cfg = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            cfg[key.strip()] = value.strip().strip('"').strip("'")
    needed = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    if any(k not in cfg for k in needed):
        return None
    return boto3.client("s3", endpoint_url=cfg["R2_ENDPOINT_URL"],
                        aws_access_key_id=cfg["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=cfg["R2_SECRET_ACCESS_KEY"],
                        region_name="auto")


def run_stamp(s3, vendor: str, sample: Path) -> dict:
    """Run-level provenance for one vendor, from a single `_raw` record.

    One object, not 660: `timeout_s` and `model` are constants of the run rather than facts
    about a document, so fetching them per document would be 1,084 MB for a repeated value.
    """
    stamp = {"vendor": vendor}
    if s3 is None:
        return stamp
    key = f"{R2_ROOT}/{vendor}/_raw/{sample.parent.name}/{sample.stem}.json"
    try:
        raw = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except Exception as exc:                                # pragma: no cover - network
        stamp["stamp_error"] = f"{type(exc).__name__}"
        return stamp
    manifest = raw.get("run_manifest") or {}
    for field in ("timeout_s", "model", "max_output_tokens", "captured_at"):
        if manifest.get(field) is not None:
            stamp[field] = str(manifest[field])
    return stamp


def classify(result):
    """What the stored extraction is, for the `usable` column.

    `prediction_io.usable` is the scorer's definition and this mirrors it: an object, not
    empty, no `__error__`. Across the whole run only two outcomes occur -- 5,683 usable and
    253 error payloads -- but the shape check stays, because a vendor returning something
    else should be visible rather than assumed away.
    """
    usable = bool(isinstance(result, dict) and result and "__error__" not in result)
    error = result.get("__error__") if isinstance(result, dict) else None
    return usable, error


def build_vendor(vendor: str, source: Path, out: Path, s3):
    files = sorted((source / vendor).rglob("*.json"))
    if not files:
        raise ValueError(f"no predictions under {source / vendor}")

    preds = out / "predictions" / vendor
    preds.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        envelope_text = f.read_text()
        body = extract_result(envelope_text).encode()
        (preds / f"{f.stem}.json").write_bytes(body)

        envelope = json.loads(envelope_text)
        try:
            result = json.loads(body)
        except ValueError:
            result = None
        usable, error = classify(result)
        rows.append({
            "doc_id": f.stem,
            "prediction_id": prediction_id(body),
            "bytes": len(body),
            "secs": envelope.get("_secs"),
            "recovered_after_timeout": bool(envelope.get("recovered_after_timeout")),
            "usable": usable,
            "error": error,
        })

    stamp = run_stamp(s3, vendor, files[0])
    stamp.update({"prediction_id_version": PREDICTION_ID_VERSION,
                  "rows": str(len(rows)),
                  "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    table = pa.Table.from_pylist(rows).replace_schema_metadata(stamp)
    (out / "vendors").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "vendors" / f"{vendor}.parquet", compression="zstd")
    return rows, stamp


def expected_payloads(rows):
    """(path relative to the vendor's prediction directory, sha256) for `verify`.

    `prediction_id` IS the payload's hash, so the atlas needs no separate checksum column.
    """
    for r in rows:
        yield f"{r['doc_id']}.json", r["prediction_id"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--vendors", nargs="+", help="default: every directory under --source")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    vendors = args.vendors or sorted(p.name for p in args.source.iterdir() if p.is_dir())

    if args.verify_only:
        failed = 0
        for vendor in vendors:
            rows = pq.read_table(args.out / "vendors" / f"{vendor}.parquet").to_pylist()
            problems = verify(expected_payloads(rows),
                              args.out / "predictions" / vendor)
            failed += len(problems)
            print(f"  {vendor:<20}{len(rows):>5} rows, {len(problems)} problems")
            for p in problems[:5]:
                print(f"      {p}")
        return 1 if failed else 0

    s3 = r2_client()
    if s3 is None:
        print("no R2 credentials; tables will be built without the run stamp")

    total, t0 = 0, time.monotonic()
    for vendor in vendors:
        rows, stamp = build_vendor(vendor, args.source, args.out, s3)
        problems = verify(expected_payloads(rows), args.out / "predictions" / vendor)
        total += len(problems)
        model = stamp.get("model", "-")
        print(f"  {vendor:<20}{len(rows):>5} predictions  "
              f"{len(problems)} problems  model={model}")
        for p in problems[:5]:
            print(f"      {p}")
    print(f"  {len(vendors)} vendors in {time.monotonic() - t0:.1f}s")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
