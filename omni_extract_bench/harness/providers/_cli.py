"""The command line every adapter shares.

    python -m omni_extract_bench.harness.providers.<vendor> --pdf X.pdf --schema S.json --out O.json

Reproducing one document by hand is the reason these are runnable at all -- the benchmark calls
`extract()` directly and never comes through here. Seven copies of the same argparse block is
how `--timeout` came to be defined in four adapters and missing from three, which let a vendor
run to a different limit than the others without anything noticing.

An adapter names only the flags that are ITS OWN; everything a flag parses is passed through to
`extract` as a keyword, so adding an option to an adapter needs no edit here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..extraction import MissingCredential, VendorError

SHARED = ("pdf", "schema", "out", "timeout")


def write_output(out_path: Path, *, provider: str, result, latency_s: float, usage) -> None:
    """The `{result, _meta}` shape every adapter writes when run by hand."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"result": result,
         "_meta": {"provider": provider, "latency_s": round(latency_s, 2), "usage": usage}},
        indent=2, default=str))


def run_cli(extract, provider: str, *extra, description: str | None = None) -> None:
    """Parse, call `extract`, write the envelope. `extra` is `(flag, kwargs)` per own option."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    # The harness always passes this; the default is for running the adapter by hand.
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="seconds this document may take, end to end")
    for flag, kwargs in extra:
        ap.add_argument(flag, **kwargs)
    args = ap.parse_args()

    options = {k: v for k, v in vars(args).items() if k not in SHARED and v is not None}
    started = time.time()
    try:
        got = extract(args.pdf, json.loads(args.schema.read_text()),
                      timeout=args.timeout, **options)
    except (VendorError, MissingCredential) as exc:
        # The adapter raised where the failure happened, so the message already says what went
        # wrong. Print it, not a traceback.
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None

    write_output(args.out, provider=provider, result=got.result,
                 latency_s=round(time.time() - started, 2),
                 usage={**got.cost._asdict(), "job_id": got.job_id, **(got.raw or {})})
    print(f"saved -> {args.out}", file=sys.stderr)
