"""Command-line interface.

    omni-extract-bench score --pred p.json --gt g.json --schema s.json
    omni-extract-bench score-dir --pred-dir preds/ --gt-dir gt/ --schema-dir schemas/
    omni-extract-bench leaderboard --pred-root baselines/ --gt-dir gt/ --schema-dir schemas/

`score-dir` pairs files by basename. `leaderboard` expects one sub-directory per provider
under `--pred-root` and scores them all over the same document list, which is what makes the
comparison fair: every provider is scored over an identical denominator, and a document a
provider failed to return scores 0 rather than being dropped from its mean.

A schema is required, and is passed through `strip_benchmark_keys` and `resolve_refs` first --
the scorer refuses a schema it cannot see through. A document that cannot be scored at all is
recorded and reported at the end rather than aborting the run, and is kept distinct from a
document a provider simply did not return: those are a harness problem and a vendor problem,
and reporting them together would hide the first as the second.

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


def _score_or_note(pred_path, gt_path, schema_path, failures, stem):
    """`_score_one`, with the failure recorded instead of raised."""
    try:
        return _score_one(pred_path, gt_path, schema_path)
    except Failed as exc:
        failures.append((stem, str(exc)))
        return None


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _report_failures(failures):
    """Say which documents could not be scored, and why. Never swallow them.

    A document that failed to score is not the same as a provider that returned nothing, and
    reporting them together would hide a broken harness as a poor vendor.
    """
    if not failures:
        return
    print(f"\n{len(failures)} document(s) could not be scored:", file=sys.stderr)
    for stem, why in failures[:20]:
        print(f"   {stem}: {why}", file=sys.stderr)
    if len(failures) > 20:
        print(f"   ... and {len(failures) - 20} more", file=sys.stderr)


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


def _pairs(gt_dir, pred_dir, schema_dir):
    for gt_file in sorted(Path(gt_dir).glob("*.json")):
        stem = gt_file.stem
        schema = Path(schema_dir) / f"{stem}.json" if schema_dir else None
        yield stem, Path(pred_dir) / f"{stem}.json", gt_file, schema


def cmd_score_dir(args):
    # Three outcomes, kept apart on purpose. A provider that returned nothing scores 0 and
    # stays in the mean, because dropping it would reward failing on hard documents. A
    # document THIS harness could not score is not a measurement of the provider at all, so
    # it is excluded from the mean and reported loudly. Merging the two would let a broken
    # harness read as a poor vendor.
    graded, empty, failures = [], [], []
    for stem, pred, gt, schema in _pairs(args.gt_dir, args.pred_dir, args.schema_dir):
        before = len(failures)
        r = _score_or_note(pred, gt, schema, failures, stem)
        if len(failures) > before:
            continue
        (graded if r else empty).append((stem, r))
    total = len(graded) + len(empty) + len(failures)
    if not total:
        print(f"no ground-truth files found in {args.gt_dir}", file=sys.stderr)
        return 1

    print(f"  {'acc':>6} {'f1':>6} {'prec':>6} {'rec':>6}  {'fab':>4} {'inv':>4}  document")
    for stem, r in sorted(graded, key=lambda x: x[1]["accuracy"]):
        inv = r["invented_item"] + r["invented_field"]
        print(f"  {r['accuracy']:6.2f} {r['f1'] * 100:6.2f} {r['precision'] * 100:6.2f}"
              f" {r['recall'] * 100:6.2f}  {r['fabricated']:4} {inv:4}  {stem}")
    for stem, _ in empty:
        print(f"  {0.0:6.2f} {'':>6} {'':>6} {'':>6}  {'':>4} {'':>4}  {stem}"
              f"   (no output -> 0)")

    scores = [r["accuracy"] for _, r in graded] + [0.0] * len(empty)
    measured = len(graded) + len(empty)
    print(f"\ndocuments      {total}")
    print(f"scored         {measured}   ({len(graded)} returned, {len(empty)} empty -> 0)")
    if failures:
        print(f"NOT SCORED     {len(failures)}   excluded from the mean -- harness, "
              f"not vendor")
    print(f"score          {_mean(scores):.2f}   over the {measured} scored")
    print(f"on returned    {_mean([r['accuracy'] for _, r in graded]):.2f}")
    _report_failures(failures)
    return 0


def cmd_leaderboard(args):
    providers = sorted(p for p in Path(args.pred_root).iterdir() if p.is_dir())
    if not providers:
        print(f"no provider directories under {args.pred_root}", file=sys.stderr)
        return 1
    docs = [g.stem for g in sorted(Path(args.gt_dir).glob("*.json"))]
    print(f"{'provider':22}{'score':>9}{'coverage':>10}{'on returned':>13}")
    print("-" * 54)
    table, failures = [], []
    for prov in providers:
        scores, returned = [], 0
        for stem in docs:
            schema = Path(args.schema_dir) / f"{stem}.json"
            before = len(failures)
            r = _score_or_note(prov / f"{stem}.json", Path(args.gt_dir) / f"{stem}.json",
                               schema, failures, f"{prov.name}/{stem}")
            if len(failures) > before:
                continue                      # harness, not vendor -- see cmd_score_dir
            # A document a provider did not return scores 0; it is never dropped from the
            # mean, or a provider that fails on hard documents would look better than one
            # that attempts them.
            scores.append(r["accuracy"] if r else 0.0)
            returned += 1 if r else 0
        table.append((prov.name, _mean(scores), returned, len(scores),
                      _mean([s for s in scores if s > 0])))
    for name, score, ret, scored, on_ret in sorted(table, key=lambda t: -t[1]):
        print(f"{name:22}{score:>9.2f}{f'{ret}/{scored}':>10}{on_ret:>13.2f}")
    if failures:
        print(f"\n{len(failures)} provider-document pair(s) not scored; those documents are "
              f"excluded from the means above, so a provider's denominator may differ.",
              file=sys.stderr)
    _report_failures(failures)
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

    d = sub.add_parser("score-dir", help="score a directory of predictions")
    d.add_argument("--pred-dir", required=True)
    d.add_argument("--gt-dir", required=True)
    d.add_argument("--schema-dir", required=True)
    d.set_defaults(fn=cmd_score_dir)

    # The corpus-level commands. These read the benchmark layout -- a directory per document
    # holding ground_truth.json and schema.json -- rather than the flat per-type directories
    # the three commands above take.
    from . import run as _run

    n = sub.add_parser("bench", help="score a directory of predictions against a corpus")
    n.add_argument("--predictions", required=True, help="directory of <doc_id>.json")
    n.add_argument("--corpus", help="default: download the published benchmark")
    n.add_argument("--out", help="write summary.parquet and verdicts/ here")
    n.add_argument("--no-verdicts", action="store_true",
                   help="skip the per-address table; roughly halves the time")
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

    b = sub.add_parser("leaderboard", help="score every provider under a root directory")
    b.add_argument("--pred-root", required=True)
    b.add_argument("--gt-dir", required=True)
    b.add_argument("--schema-dir", required=True)
    b.set_defaults(fn=cmd_leaderboard)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
