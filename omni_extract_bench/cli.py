"""Command-line interface.

    omni-extract-bench bench   --predictions preds/ [--corpus DIR] [--out run/]
    omni-extract-bench explain --predictions preds/ --doc <doc_id>
    omni-extract-bench verify  --corpus DIR
    omni-extract-bench score   --pred p.json --gt g.json --schema s.json

`bench` is the one to reach for: a corpus is a directory of document directories, each holding
`ground_truth.json` and `schema.json`, and predictions are `<doc_id>.json`. `--corpus` defaults
to the published benchmark, and pointing it elsewhere is how you score against your own ground
truth. See `docs/USING.md`.

`score` is the single-pair escape hatch, for when there is no benchmark involved at all.

A schema is required, and is passed through `strip_benchmark_keys` and `resolve_refs` first --
the scorer refuses a schema it cannot see through.

Two earlier commands, `score-dir` and `leaderboard`, took `--gt-dir` and `--schema-dir` and
paired files by basename across flat per-type directories. They could not read the benchmark
layout at all, so they were a second and incompatible way in. `bench` replaces both.

NOTE: aggregation here is a flat mean over documents. METRIC_SPEC section 7 defines the
published number as an equal-weight mean over SUBSETS, which needs the subset each document
belongs to and is not implemented yet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dialects import resolve_refs, strip_benchmark_keys
from .prediction_io import usable
from .score import grade


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _unwrap(obj):
    """Accept either a bare extraction or a `{"result": ...}` envelope."""
    if isinstance(obj, dict) and set(obj) == {"result"}:
        return obj["result"]
    return obj


class Failed(Exception):
    """One document could not be scored. Never fatal to a run, never silent either."""


def _schema_for(schema_path):
    """Load a schema and make it gradeable.

    Two transforms, both required rather than tidy. `strip_benchmark_keys` removes our own
    annotations, which are not part of the schema the model answered. `resolve_refs` inlines
    `$ref`, because the scorer refuses a schema it cannot see through -- an
    `additionalProperties` object behind a pointer would be graded while the same object
    written inline is skipped.
    """
    if not schema_path or not Path(schema_path).exists():
        raise Failed(f"no schema at {schema_path}. The scorer needs one: without it an "
                     f"additionalProperties object cannot be found, and a fabricated value "
                     f"cannot be told from an invented one")
    return resolve_refs(strip_benchmark_keys(_load(schema_path)))


def _score_one(pred_path, gt_path, schema_path):
    """Score one document. Returns the grade, or None if the provider returned nothing.

    Raises `Failed` for anything else that goes wrong, so a caller scoring a whole corpus can
    record which documents failed and carry on. A run of nine providers over a hundred and
    sixty documents must not be lost to one malformed file.
    """
    schema = _schema_for(schema_path)
    try:
        gt = _unwrap(_load(gt_path))
    except (OSError, ValueError) as exc:
        raise Failed(f"unreadable ground truth: {exc}") from exc
    try:
        pred = _unwrap(_load(pred_path)) if pred_path and Path(pred_path).exists() else None
    except (OSError, ValueError) as exc:
        raise Failed(f"unreadable prediction: {exc}") from exc
    if not usable(pred):
        return None
    try:
        return grade(pred, gt, schema)
    except Failed:
        raise
    except Exception as exc:                                            # noqa: BLE001
        raise Failed(f"{type(exc).__name__}: {exc}") from exc


def cmd_score(args):
    try:
        r = _score_one(args.pred, args.gt, args.schema)
    except Failed as exc:
        print(f"could not score: {exc}", file=sys.stderr)
        return 1
    if r is None:
        print("prediction is empty or errored -> scores 0")
        return 0
    print(json.dumps(r, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omni-extract-bench",
                                 description="Score document-extraction predictions.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score one prediction")
    s.add_argument("--pred", required=True)
    s.add_argument("--gt", required=True)
    s.add_argument("--schema", required=True)
    s.set_defaults(fn=cmd_score)


    # The corpus-level commands. These read the benchmark layout -- a directory per document
    # holding ground_truth.json and schema.json -- rather than the flat per-type directories
    # the three commands above take.
    from . import run as _run

    n = sub.add_parser("bench", help="score a directory of predictions against a corpus")
    n.add_argument("--predictions", required=True, help="directory of <doc_id>.json")
    n.add_argument("--corpus", help="default: download the published benchmark")
    n.add_argument("--out", help="write summary.parquet and verdicts/ here")
    n.add_argument("--no-verdicts", action="store_true",
                   help="skip the per-address table; saves memory, not much time")
    n.set_defaults(fn=_run.cmd_bench)

    e = sub.add_parser("explain", help="show every address for one document")
    e.add_argument("--predictions", required=True)
    e.add_argument("--doc", required=True)
    e.add_argument("--corpus")
    e.add_argument("--all", action="store_true", help="include addresses that matched")
    e.set_defaults(fn=_run.cmd_explain)

    v = sub.add_parser("verify", help="check a corpus directory against the contract")
    v.add_argument("--corpus", required=True)
    v.set_defaults(fn=_run.cmd_verify)


    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
