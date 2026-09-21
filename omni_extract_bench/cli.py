"""Command-line interface.

    oeb score      --pred p.json --gt g.json --schema s.json [--verdicts]
    oeb benchmark  --providers datalab reducto --out runs/
    oeb predict    --provider datalab --doc x.pdf --schema s.json
    oeb providers [datalab]

`score` grades one document: three JSON files in, the metrics dict out as JSON on stdout.
`benchmark` is the whole published benchmark -- fetch the corpus, run it through each
vendor, score, write it down -- and lives in `benchmark.py`, which is a script rather than
library code precisely because it makes the orchestration decisions the library refuses to.

There is no manifest and no runner. Scoring a corpus is a loop over this command, or better
over `score` itself -- written the way your corpus is laid out, rather than the way a table
would have to be. That loop is the caller's, because it is where the decisions live that this
package has no business making: which documents, in what order, how many at once, what to do
with a prediction that came back as a recorded failure, and how to aggregate at the end
(`docs/METRIC_SPEC.md` section 7 on why a flat mean over documents is not the right one).

A subcommand rather than a bare `oeb`, because producing predictions is the other half of this
repository and will want a verb of its own.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys


def read_json(path: str):
    """Parse one local JSON file, naming the file when it will not parse.

    The bare `json` message says line and column and not which of the three documents it came
    from, which is the only thing a reader needs when two of them are vendor output.
    """
    try:
        return json.loads(pathlib.Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not JSON -- {exc}") from None


def cmd_score(args) -> int:
    from .metric import score

    result = score(read_json(args.pred), read_json(args.gt), read_json(args.schema),
                   order_matters=args.order_matters or (), verdicts=args.verdicts)
    if args.verdicts:
        result["verdicts"] = [v._asdict() for v in result["verdicts"]]
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


EXAMPLE_MODELS = ("openai/gpt-5.6-sol", "anthropic/claude-opus-5", "google/gemini-3.7-flash")


MAX_DEFAULT = 48


def show_default(value) -> str:
    """One default, fit to a column: `''` for empty, cut with `...` where it runs long."""
    text = "''" if value == "" else str(value)
    return text if len(text) <= MAX_DEFAULT else text[:MAX_DEFAULT - 3] + "..."


def cmd_providers(args) -> int:
    """The names `--providers` accepts, or one provider's options in detail.

        oeb providers            every name, one per line
        oeb providers datalab    what `--options` takes for it, and what each is by default

    ONE VERB, NOT TWO. The list is the question usually being asked and the options are a
    follow-up about one entry -- printing every provider's defaults in the list drowned the
    names. Bare, it stays one column, so `oeb providers | ...` is still a clean list.

    The options come from `vendor.settings_for`, which reads the adapter's own `Config`, so
    this cannot drift from what the adapter accepts.
    """
    from .harness import PROVIDERS
    from .harness.vendor import settings_for

    if not args.provider:
        for provider in (*PROVIDERS, *EXAMPLE_MODELS):
            print(provider)
        return 0

    options = settings_for(args.provider)
    print(args.provider)
    if not options:
        print("\n  no options: it takes the document and the schema and nothing else")
        return 0

    width = max(len(key) for key in options)
    print()
    for key, value in options.items():
        print(f"  {key:<{width}}  {show_default(value)}")
    print(f"\n  oeb benchmark --providers {args.provider} "
          f"--options '{{\"{args.provider}\": {{\"{next(iter(options))}\": ...}}}}'")
    return 0


def read_options(value: str | None, *, per_provider: bool = True) -> dict:
    """`--options` as JSON, given literally or as a path to a file.

    Keyed by provider for `benchmark`, which runs several; flat for `predict`, which runs one.

    JSON rather than an inline `provider:key=value` syntax, because OpenRouter model ids
    already use the colon -- `mistralai/mistral-medium-3-5:batch`, `:free`, `:nitro` -- so any
    colon-separated form is ambiguous exactly where model ids appear. JSON also keeps types:
    `8000` stays an integer instead of arriving as a string the adapter has to guess about.
    """
    if not value:
        return {}
    text = value
    path = pathlib.Path(value)
    if not value.lstrip().startswith("{"):
        if not path.exists():
            raise ValueError(f"--options: {value!r} is neither JSON nor a file that exists")
        text = path.read_text()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--options: not JSON -- {exc}") from None
    if not isinstance(parsed, dict):
        raise ValueError("--options must be a JSON object")
    def ok(value):
        return (isinstance(value, dict)
                or (isinstance(value, list) and value
                    and all(isinstance(x, dict) for x in value)))

    if per_provider and not all(ok(v) for v in parsed.values()):
        raise ValueError('--options maps a provider to its options, or to a LIST of them to '
                         'run it once for each:\n'
                         '    \'{"datalab": {"mode": "accurate"}}\'\n'
                         '    \'{"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]}\'')
    return parsed


def read_workers(value: str | None) -> dict[str, int]:
    """`--predict-workers` as `{provider: count}`, `"*"` being a bare number meaning all.

        --predict-workers 25                     -> {"*": 25}
        --predict-workers reducto=25,datalab=40  -> {"reducto": 25, "datalab": 40}

    Per provider rather than one global number, because the right value is a fact about the
    vendor: these are documents in flight at it, and a run naming several vendors has no one
    number that fits them all.
    """
    if not value:
        return {}
    if value.strip().isdigit():
        return {"*": _count(value, value)}
    out = {}
    for part in value.split(","):
        name, sep, count = part.partition("=")
        if not sep:
            raise ValueError(f"--predict-workers: {part.strip()!r} is neither a number nor "
                             f"NAME=COUNT, e.g. '25' or 'reducto=25,datalab=40'")
        out[name.strip()] = _count(count, part)
    return out


def _count(text: str, shown: str) -> int:
    """One `--predict-workers` count: a whole number of documents, so at least one."""
    if not text.strip().isdigit() or int(text) < 1:
        raise ValueError(f"--predict-workers: {shown.strip()!r} needs a count of 1 or more")
    return int(text)


def cmd_predict(args) -> int:
    """One document through one vendor. The mirror of `oeb score`: files in, JSON out."""
    from .harness import (AccountFailure, MissingCredential, MissingDependency, VendorError,
                          predict)

    try:
        record = predict(args.provider, args.doc, read_json(args.schema), timeout=args.timeout,
                         **read_options(args.options, per_provider=False))
    except (MissingCredential, MissingDependency, AccountFailure, VendorError) as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1
    json.dump(record, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 1 if record.get("error") else 0


def cmd_benchmark(args) -> int:
    from .benchmark import run

    from .harness import AccountFailure, MissingCredential, MissingDependency

    try:
        summary = run(args.providers, out=args.out, data_root=args.data_root,
                      suites=args.suites,
                      limit=args.limit, timeout=args.timeout,
                      predict_workers=read_workers(args.predict_workers),
                      score_workers=args.score_workers,
                      verdicts=args.verdicts, rescore=args.rescore,
                      score_only=args.score_only,
                      options=read_options(args.options))
    except (MissingCredential, MissingDependency, AccountFailure) as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oeb",
                                 description="Score one document-extraction prediction.")
    ap.add_argument("-q", "--quiet", action="store_true", help="progress off; results only")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score one prediction against one ground truth")
    s.add_argument("--pred", required=True, help="the prediction, JSON")
    s.add_argument("--gt", required=True, help="the ground truth, JSON")
    s.add_argument("--schema", required=True,
                   help="the JSON Schema the prediction was generated against")
    s.add_argument("--verdicts", action="store_true",
                   help="include one verdict per address: what happened there, and what each "
                        "side was compared as")
    s.add_argument("--order-matters", nargs="*", metavar="ARRAY",
                   help="arrays whose order is part of the answer, e.g. steps, "
                        "'books[*].chapters'. Default: none -- the order rows appear in a "
                        "document is usually an accident of layout")
    s.set_defaults(fn=cmd_score)

    b = sub.add_parser("benchmark",
                       help="fetch the corpus, run it through each vendor, score it")
    b.add_argument("--providers", nargs="+", required=True, metavar="NAME",
                   help="vendors and/or model ids, e.g. datalab reducto "
                        "openai/gpt-5.6-sol. `oeb providers` lists them")
    b.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs"),
                   help="where predictions, scores and the summary go. Default: runs/")
    b.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("benchmark"),
                   help="where the corpus is downloaded to. Default: benchmark/")
    b.add_argument("--suites", nargs="+", help="limit to these suites")
    b.add_argument("--limit", type=int, default=0, help="first N documents; for a smoke test")
    b.add_argument("--timeout", type=float, default=1800,
                   help="seconds one document may take, the same for every vendor")
    b.add_argument("--predict-workers", metavar="N|NAME=N,...",
                   help="documents in flight at one vendor -- one number for every provider, "
                        "or per provider: 'reducto=25,datalab=40'. Model ids share one "
                        "budget, being one OpenRouter key. Default: the harness's per-vendor "
                        "limit")
    b.add_argument("--score-workers", type=int, default=0, metavar="N",
                   help="processes used to grade. Scoring is the CPU-bound half and is "
                        "independent per document. Default: one per core, capped at 8; "
                        "1 grades in this process")
    b.add_argument("--verdicts", action="store_true",
                   help="also write one verdict per address, per document")
    b.add_argument("--options", metavar="JSON",
                   help='per-provider options, as JSON or a path to a JSON file: '
                        '\'{"datalab": {"mode": "accurate"}}\'. A LIST runs that provider '
                        'once per entry, so \'{"datalab": [{"mode": "balanced"}, '
                        '{"mode": "accurate"}]}\' compares its tiers in one run. Each is '
                        "recorded in the run's settings.json and on every document, and "
                        "digested into the directory name so two of them cannot mix")
    b.add_argument("--rescore", action="store_true",
                   help="grade every document again, ignoring the scores already on disk. "
                        "Grading otherwise resumes per document, so reach for this after "
                        "changing the metric -- nothing else notices that")
    b.add_argument("--score-only", action="store_true",
                   help="score the predictions already on disk; call no vendor")
    b.set_defaults(fn=cmd_benchmark)

    d = sub.add_parser("predict", help="run one document through one vendor")
    d.add_argument("--provider", required=True,
                   help="a vendor or a model id; `oeb providers` lists them")
    d.add_argument("--doc", required=True, help="the document, PDF")
    d.add_argument("--schema", required=True, help="the JSON Schema to extract against")
    d.add_argument("--timeout", type=float, default=1800,
                   help="seconds this document may take, end to end")
    d.add_argument("--options", metavar="JSON",
                   help='options for this provider, as JSON or a path: \'{"mode": "accurate"}\'')
    d.set_defaults(fn=cmd_predict)

    pl = sub.add_parser("providers",
                        help="list the vendors benchmark can run, or detail one of them")
    pl.add_argument("provider", nargs="?",
                    help="name one to see what --options takes for it, and the defaults")
    pl.set_defaults(fn=cmd_providers)

    args = ap.parse_args(argv)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.WARNING)
    logging.getLogger(__package__).setLevel(logging.WARNING if args.quiet else logging.INFO)

    try:
        return args.fn(args)
    except (ValueError, TypeError, OSError, ImportError) as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
