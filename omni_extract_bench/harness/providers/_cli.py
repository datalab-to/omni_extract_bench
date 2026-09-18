"""The command line every adapter shares.

    python -m omni_extract_bench.harness.providers.<vendor> --pdf X.pdf --schema S.json --out O.json

Reproducing one document by hand is the reason these are runnable at all -- the benchmark calls
`extract()` directly and never comes through here. Seven copies of the same argparse block is
how `--timeout` came to be defined in four adapters and missing from three, which let a vendor
run to a different limit than the others without anything noticing.

An adapter names no flags at all: they are generated from its `Config`, so adding an option is
one field and nothing else.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from ..extraction import MissingCredential, VendorError

def write_output(out_path: Path, *, provider: str, result, latency_s: float, usage) -> None:
    """The `{result, _meta}` shape every adapter writes when run by hand."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"result": result,
         "_meta": {"provider": provider, "latency_s": round(latency_s, 2), "usage": usage}},
        indent=2, default=str))


def run_cli(extract, config_type, provider: str, *, description: str | None = None) -> None:
    """Parse, call `extract`, write the envelope.

    THE FLAGS ARE BUILT FROM THE CONFIG, so this command line and `oeb benchmark --options`
    offer the same options with the same defaults. They were written out twice before -- once
    as the adapter's keyword arguments and once as an argparse block here -- so a default could
    be changed in one and not the other, and a hand-run would answer a different question from
    the benchmark it was meant to explain.
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    # The harness always passes this; the default is for running the adapter by hand.
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="seconds this document may take, end to end")
    for f in dataclasses.fields(config_type):
        needed = f.default is dataclasses.MISSING
        ap.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, required=needed,
                        type={"int": int, "float": float}.get(str(f.type).split()[0], str),
                        default=None if needed else f.default,
                        choices=f.metadata.get("choices"),
                        help=f.metadata.get("help"))
    args = ap.parse_args()
    config = config_type(**{f.name: getattr(args, f.name)
                            for f in dataclasses.fields(config_type)})

    started = time.time()
    try:
        got = extract(args.pdf, json.loads(args.schema.read_text()),
                      timeout=args.timeout, config=config)
    except (VendorError, MissingCredential) as exc:
        # The adapter raised where the failure happened, so the message already says what went
        # wrong. Print it, not a traceback.
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None

    write_output(args.out, provider=provider, result=got.result,
                 latency_s=round(time.time() - started, 2),
                 usage={**got.cost._asdict(), "job_id": got.job_id, **(got.raw or {})})
    print(f"saved -> {args.out}", file=sys.stderr)
