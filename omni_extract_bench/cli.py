"""Command-line interface.

    oeb score     --manifest jobs.parquet --out run/   [--jobs N] [--batch-size N]
    oeb predict   --manifest jobs.parquet --out preds/ [--jobs N] [--batch-size N]
    oeb score-one --pred p.json --gt g.json --schema s.json

Two verbs and an escape hatch. `score` and `predict` each take one tabular manifest and write
a directory of parquet parts; `score-one` grades a single triple and prints it, for when there
is no table involved at all.

A manifest is any parquet or CSV whose rows name their inputs -- `run_score` and
`run_predict` document the columns. Paths go through fsspec, so `s3://`, `gs://` and a local
path are the same thing here and there is no separate remote runner.

`score-one` builds a one-row manifest and calls the same `score_row` a run does, rather than
repeating the resolve-and-grade path. Two definitions of what scoring a row means is one more
than there should be.

A manifest may also be a DIRECTORY of parquet parts, because a dataset is what gets opened --
so `predict --out preds/` is followed by `score --manifest preds/manifest.parquet` with nothing in
between.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .run_score import open_uri, score_row


def read_bytes(uri):
    with open_uri(uri) as fh:
        return fh.read()


def _ui():
    """Imported when the verb runs, not when the module loads: the viewer needs pyarrow, and
    `score-one` should work on a machine that has only the scorer."""
    from . import ui

    return ui


def cmd_score(args) -> int:
    from . import run_score

    tally = run_score.run(args.manifest, args.out, jobs=args.jobs,
                          batch_size=args.batch_size, root=args.root,
                          overwrite=args.overwrite)
    print(f"{args.out}: " + "  ".join(f"{k}={v}" for k, v in tally.items()))
    # Zero: the run did its job. Documents the system under test could not answer are the
    # benchmark's findings, not this command's failure, and they are all in the table with
    # their reasons. A manifest that cannot be read still exits non-zero, from main().
    return 0


def cmd_predict(args) -> int:
    from . import run_predict

    tally = run_predict.run(args.manifest, args.out, args.provider, jobs=args.jobs,
                            batch_size=args.batch_size, timeout=args.timeout, mode=args.mode,
                            completion_model=args.completion_model, root=args.root,
                            overwrite=args.overwrite)
    print(f"{args.out}: " + "  ".join(f"{k}={v}" for k, v in tally.items()))
    return 1 if tally["error"] else 0


def cmd_score_one(args) -> int:
    # The schema is inline in a manifest; here it is a path like the other two, because a
    # path is what someone at a terminal has.
    row, _ = score_row({"doc_id": "one", "gt_path": args.gt, "pred_path": args.pred,
                        "schema": read_bytes(args.schema)})
    if row["status"] != "scored":
        print(f"{row['status']}: {row['error']}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in row.items() if k not in ("status", "error")}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="omni-extract-bench",
                                 description="Score document-extraction predictions.")
    ap.add_argument("-q", "--quiet", action="store_true", help="progress off; results only")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score a manifest of documents")
    s.add_argument("--manifest", required=True, help="parquet or CSV naming the work")
    s.add_argument("--out", required=True,
                   help="written as <out>/scores.parquet and <out>/verdicts.parquet")
    s.add_argument("--jobs", type=int, default=1, help="worker processes")
    s.add_argument("--batch-size", type=int,
                   help="manifest rows per part, and the unit one worker takes; "
                        "default: about four batches per job")
    s.add_argument("--root", default=".",
                   help="what a relative path in a manifest is relative to; absolute paths\n"
                        "and URIs are used as written")
    s.add_argument("--overwrite", action="store_true",
                   help="replace a run already in --out, instead of refusing")
    s.set_defaults(fn=cmd_score)

    p = sub.add_parser("predict", help="run a manifest of documents through their providers")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True,
                   help="written as <out>/predictions and <out>/manifest.parquet")
    p.add_argument("--jobs", type=int, default=4, help="concurrent requests")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--root", default=".",
                   help="what a relative path in a manifest is relative to; absolute paths\n"
                        "and URIs are used as written")
    p.add_argument("--overwrite", action="store_true",
                   help="replace a run already in --out, instead of refusing")
    p.set_defaults(fn=cmd_predict)

    u = sub.add_parser("ui", help="write a browsable site for one or more runs")
    u.add_argument("--run", nargs="+", required=True, metavar="RUN",
                   help="a run directory, or a glob over several; its name labels the column")
    u.add_argument("--out", required=True, help="the site directory")
    u.add_argument("--root", default=".",
                   help="what a relative path in a manifest is relative to; absolute paths\n"
                        "and URIs are used as written")
    u.set_defaults(fn=lambda a: _ui().cmd_ui(a))

    o = sub.add_parser("score-one", help="score a single prediction/gt/schema triple")
    o.add_argument("--pred", required=True)
    o.add_argument("--gt", required=True)
    o.add_argument("--schema", required=True)
    o.set_defaults(fn=cmd_score_one)

    args = ap.parse_args(argv)
    # Progress on stderr, results on stdout, so `oeb score-one | jq` stays a pipe.
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError) as exc:
        # One boundary for everything a manifest can get wrong -- a missing column, a name
        # that collides with ours, a repeated doc_id, a path that is not there. These are
        # already written to be read, so print the message and not a traceback.
        print(f"  {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
