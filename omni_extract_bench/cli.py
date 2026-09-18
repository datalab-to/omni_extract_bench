"""Command-line interface.

    oeb score      --pred p.json --gt g.json --schema s.json [--verdicts]
    oeb benchmark  --providers datalab reducto --out runs/
    oeb predict    --provider datalab --doc x.pdf --schema s.json
    oeb providers

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
        # A Verdict is a NamedTuple, which json would write as an array; as an object each
        # field is named at the point someone reads it.
        result["verdicts"] = [v._asdict() for v in result["verdicts"]]
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


def cmd_providers(args) -> int:
    """Every name `oeb benchmark --providers` accepts, one per line.

    The named vendors are a closed set; the model ids are examples, since any OpenRouter
    `org/model` works. One column so the output pipes into something else without cutting.

    A verb of its own rather than a flag on `benchmark`, because listing them means importing
    the harness -- more than `--help` should do -- and `--providers` is required on that verb.
    """
    from .harness import PROVIDERS

    for provider in PROVIDERS:
        print(provider)
    for example in ("openai/gpt-5.6-sol", "anthropic/claude-opus-5", "google/gemini-3.7-flash"):
        print(example)
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
    if per_provider and not all(isinstance(v, dict) for v in parsed.values()):
        raise ValueError('--options must map a provider to its options, e.g. '
                         '\'{"datalab": {"mode": "accurate"}}\'')
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
        # Handled here rather than in `main`'s boundary, because these types live behind the
        # harness extra and naming them in an `except` clause up there would import it for
        # every verb -- including `oeb score`, which needs no vendor SDK at all.
        print(f"  {exc}", file=sys.stderr)
        return 1
    # The whole record by default -- what was asked, what came back, what it cost -- because
    # that is what the library produces and what makes an answer checkable. `--result-only`
    # gives the bare extraction, so `oeb predict ... --result-only > p.json` feeds `oeb score`.
    # The whole record, always. It carries the answer AND what it cost, what was sent and what
    # came back -- and a flag to print just the answer would be a way to throw that away by
    # accident, which is the habit this harness exists to prevent. `| jq .result` is one pipe,
    # and explicit where someone reads it.
    json.dump(record, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 1 if record.get("error") else 0




def cmd_benchmark(args) -> int:
    # Imported here rather than at module scope: `oeb score` is the base install, and nothing
    # on that path should need the benchmark extra present until this verb is actually used.
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
    # The provider names are not `choices`: listing them means importing the harness, which
    # installs the transport taps as a side effect, and `oeb --help` should not monkey-patch
    # anyone's HTTP stack. `benchmark.run` validates them and names them all when it can't.
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
                        "or per provider: 'reducto=25,datalab=40'. Default: the harness's "
                        "per-vendor limit")
    b.add_argument("--score-workers", type=int, default=0, metavar="N",
                   help="processes used to grade. Scoring is the CPU-bound half and is "
                        "independent per document. Default: one per core, capped at 8; "
                        "1 grades in this process")
    b.add_argument("--verdicts", action="store_true",
                   help="also write one verdict per address, per document")
    b.add_argument("--options", metavar="JSON",
                   help='per-provider options, as JSON or a path to a JSON file: '
                        '\'{"datalab": {"mode": "accurate"}}\'. Each is recorded in the '
                        "document's run_manifest, because a run that is not stock must say so")
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

    pl = sub.add_parser("providers", help="list the vendors benchmark can run, and their tiers")
    pl.set_defaults(fn=cmd_providers)

    args = ap.parse_args(argv)
    # Progress on stderr, results on stdout, so `oeb score ... | jq` stays a pipe.
    #
    # The ROOT logger stays at WARNING and only OUR namespace is turned up. The obvious thing
    # -- `basicConfig(level=INFO)` -- sets the level for every library in the process, and then
    # each one has to be muted by name: httpx logs a line per request, and `openai` logs its
    # own copy of the same line through `openai._base_client`, so silencing httpx was not
    # enough. That list would need extending for every dependency anyone ever adds. Turning up
    # one namespace instead means a new library is quiet by default and still free to warn.
    handler = logging.StreamHandler()          # stderr
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.WARNING)
    logging.getLogger(__package__).setLevel(logging.WARNING if args.quiet else logging.INFO)

    try:
        return args.fn(args)
    except (ValueError, TypeError, OSError, ImportError) as exc:
        # One boundary for everything the three documents can get wrong -- a file that is not
        # there, one that will not parse, a ground truth that is not an object, a schema that
        # is missing or still holds `$ref`. `score` raises these with the reason and the fix
        # already written out, so print the message and not a traceback. ImportError joins
        # them because a missing extra is the same kind of thing -- something to install, not
        # a defect -- and the messages raised for it name what to install.
        print(f"  {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
