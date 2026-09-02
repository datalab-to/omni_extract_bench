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


def _leaderboard_rows(pred_root, gt_dir, schema_dir):
    """(provider, scores-with-zeros, returned, n_docs) for one directory of ground truth."""
    providers = sorted(p for p in Path(pred_root).iterdir() if p.is_dir())
    docs = [g.stem for g in sorted(Path(gt_dir).glob("*.json"))]
    rows = []
    for prov in providers:
        scores, returned = [], 0
        for stem in docs:
            schema = Path(schema_dir) / f"{stem}.json" if schema_dir else None
            r = _score_one(prov / f"{stem}.json", Path(gt_dir) / f"{stem}.json", schema)
            # A document a provider did not return scores 0; it is never dropped from the
            # mean, or a provider that fails on hard documents would look better than one
            # that attempts them.
            scores.append(r["leaf_accuracy"] if r else 0.0)
            returned += 1 if r else 0
        rows.append((prov.name, scores, returned, len(docs)))
    return rows


def cmd_leaderboard_subsets(args):
    """The published headline: METRIC_SPEC section 5, UNIFIED = mean of the subset scores.

    Subsets differ ~10x in size (329 documents vs 35). A document-mean lets the largest subset
    decide the benchmark and silently re-weights it whenever a subset grows; the spec therefore
    declares equal weight per subset. This command computes exactly that, and prints the
    document-mean beside it, labelled, so the two are never confused. The two differ by 2-20
    points per provider on the reference corpus.

    Expects the HF dataset layout: <data-root>/<subset>/<doc>/{ground_truth,schema}.json with
    predictions at <pred-root>/<provider>/<subset>/<doc>.json.
    """
    root = Path(args.data_root)
    subsets = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not subsets:
        print(f"no subset directories under {root}", file=sys.stderr)
        return 1
    per = {}                       # provider -> subset -> (scores, returned, n)
    for sub in subsets:
        gt_dir = root / sub
        docs = sorted(d for d in gt_dir.iterdir() if d.is_dir())
        for prov in sorted(p for p in Path(args.pred_root).iterdir() if p.is_dir()):
            scores, returned = [], 0
            for d in docs:
                r = _score_one(prov / sub / f"{d.name}.json", d / "ground_truth.json",
                               d / "schema.json")
                scores.append(r["leaf_accuracy"] if r else 0.0)
                returned += 1 if r else 0
            per.setdefault(prov.name, {})[sub] = (scores, returned, len(docs))
    head = f"{'provider':22}{'UNIFIED':>9}{'doc-mean':>10}{'coverage':>11}" + "".join(f"{s[:10]:>12}" for s in subsets)
    print(head); print("-" * len(head))
    table = []
    for prov, by in per.items():
        sub_means = [_mean(by[s][0]) for s in subsets if s in by]
        macro = _mean(sub_means)
        allscores = [x for s in subsets if s in by for x in by[s][0]]
        ret = sum(by[s][1] for s in subsets if s in by); n = sum(by[s][2] for s in subsets if s in by)
        table.append((prov, macro, _mean(allscores), ret, n, sub_means))
    for prov, macro, dm, ret, n, subs in sorted(table, key=lambda t: -t[1]):
        print(f"{prov:22}{macro:>8.2f}{dm:>10.2f}{f'{ret}/{n}':>11}" + "".join(f"{v:>12.2f}" for v in subs))
    print(f"\nUNIFIED = mean of per-subset means (METRIC_SPEC section 5) -- the headline. doc-mean is shown for reference only.")
    print("doc-mean = mean over all documents, shown for reference; NOT the headline.")
    return 0


def cmd_leaderboard(args):
    if getattr(args, "data_root", None):
        return cmd_leaderboard_subsets(args)
    if not args.gt_dir:
        print("leaderboard needs --gt-dir (one subset) or --data-root (all subsets, UNIFIED)", file=sys.stderr)
        return 2
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
    b.add_argument("--gt-dir", help="one directory of ground truth (document-mean over it)")
    b.add_argument("--schema-dir")
    b.add_argument("--data-root", help="HF-layout root with <subset>/<doc>/ dirs: reports the "
                                        "declared headline, UNIFIED = mean of per-subset means")
    b.set_defaults(fn=cmd_leaderboard)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
