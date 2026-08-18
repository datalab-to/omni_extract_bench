"""Command-line interface.

    omni-extract-bench score --pred p.json --gt g.json --schema s.json
    omni-extract-bench score-dir --pred-dir preds/ --gt-dir gt/ --schema-dir schemas/
    omni-extract-bench leaderboard --pred-root baselines/ --gt-dir gt/ --schema-dir schemas/

`score-dir` pairs files by basename. `leaderboard` expects one sub-directory per provider
under `--pred-root` and scores them all over the same document list, which is what makes the
comparison fair: every provider is scored over an identical denominator, and a document a
provider failed to return scores 0 rather than being dropped from its mean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .grading import fair_grade
from .prediction_io import usable


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _unwrap(obj):
    """Accept either a bare extraction or a `{"result": ...}` envelope."""
    if isinstance(obj, dict) and set(obj) == {"result"}:
        return obj["result"]
    return obj


def _score_one(pred_path, gt_path, schema_path):
    pred = _unwrap(_load(pred_path)) if pred_path and Path(pred_path).exists() else None
    gt = _unwrap(_load(gt_path))
    schema = _load(schema_path) if schema_path and Path(schema_path).exists() else {}
    if not usable(pred):
        return None
    return fair_grade(pred, gt, schema)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def cmd_score(args):
    r = _score_one(args.pred, args.gt, args.schema)
    if r is None:
        print("prediction is empty or errored -> scores 0")
        return 0
    print(json.dumps(r, indent=2))
    return 0


def _pairs(gt_dir, pred_dir, schema_dir):
    for gt_file in sorted(Path(gt_dir).glob("*.json")):
        stem = gt_file.stem
        schema = Path(schema_dir) / f"{stem}.json" if schema_dir else None
        yield stem, Path(pred_dir) / f"{stem}.json", gt_file, schema


def cmd_score_dir(args):
    rows, missing = [], 0
    for stem, pred, gt, schema in _pairs(args.gt_dir, args.pred_dir, args.schema_dir):
        r = _score_one(pred, gt, schema)
        if r is None:
            missing += 1
            rows.append((stem, 0.0, None))
            continue
        rows.append((stem, r["leaf_accuracy"], r))
    if not rows:
        print(f"no ground-truth files found in {args.gt_dir}", file=sys.stderr)
        return 1
    for stem, score, r in sorted(rows, key=lambda x: x[1]):
        cov = "" if r else "   (no output -> 0)"
        print(f"  {score:6.2f}  {stem}{cov}")
    scored = [s for _, s, r in rows if r]
    print(f"\ndocuments      {len(rows)}")
    print(f"returned       {len(scored)}  ({100 * len(scored) // max(len(rows), 1)}% coverage)")
    print(f"score          {_mean([s for _, s, _ in rows]):.2f}   (no output counted as 0)")
    print(f"on returned    {_mean(scored):.2f}")
    return 0


def cmd_leaderboard(args):
    providers = sorted(p for p in Path(args.pred_root).iterdir() if p.is_dir())
    if not providers:
        print(f"no provider directories under {args.pred_root}", file=sys.stderr)
        return 1
    docs = [g.stem for g in sorted(Path(args.gt_dir).glob("*.json"))]
    print(f"{'provider':22}{'score':>9}{'coverage':>10}{'on returned':>13}")
    print("-" * 54)
    table = []
    for prov in providers:
        scores, returned = [], 0
        for stem in docs:
            schema = Path(args.schema_dir) / f"{stem}.json" if args.schema_dir else None
            r = _score_one(prov / f"{stem}.json", Path(args.gt_dir) / f"{stem}.json", schema)
            # A document a provider did not return scores 0; it is never dropped from the
            # mean, or a provider that fails on hard documents would look better than one
            # that attempts them.
            scores.append(r["leaf_accuracy"] if r else 0.0)
            returned += 1 if r else 0
        table.append((prov.name, _mean(scores), returned, len(docs),
                      _mean([s for s in scores if s > 0])))
    for name, score, ret, total, on_ret in sorted(table, key=lambda t: -t[1]):
        print(f"{name:22}{score:>9.2f}{f'{ret}/{total}':>10}{on_ret:>13.2f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omni-extract-bench",
                                 description="Score document-extraction predictions.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score one prediction")
    s.add_argument("--pred", required=True)
    s.add_argument("--gt", required=True)
    s.add_argument("--schema")
    s.set_defaults(fn=cmd_score)

    d = sub.add_parser("score-dir", help="score a directory of predictions")
    d.add_argument("--pred-dir", required=True)
    d.add_argument("--gt-dir", required=True)
    d.add_argument("--schema-dir")
    d.set_defaults(fn=cmd_score_dir)

    b = sub.add_parser("leaderboard", help="score every provider under a root directory")
    b.add_argument("--pred-root", required=True)
    b.add_argument("--gt-dir", required=True)
    b.add_argument("--schema-dir")
    b.set_defaults(fn=cmd_leaderboard)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
