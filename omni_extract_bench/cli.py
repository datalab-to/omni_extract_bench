"""The `oeb` command: a thin wrapper over `benchmark`, `harness` and the scorer.

    oeb run        --providers datalab reducto --out runs/       predict, then grade
    oeb grade      mine/ [--dataset ...] [--manifest ...]          grade a folder of predictions/<doc_id>.json
    oeb score      --pred p.json --gold g.json --schema s.json     score one prediction against one gold
    oeb predict    --provider datalab --doc x.pdf --schema s.json  one document through one system
    oeb check      [--dataset ...] [--manifest ...]                whether a dataset is one oeb can run
    oeb providers  [datalab]                                       the providers, or one provider's options

Results go to stdout as JSON; the plan and progress go to stderr. Commands and flags from before 1.0 still
work, each printing what replaced it (`deprecated`).
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import Any

from omni_bench_core import errors

from . import data

EXAMPLE_MODELS = ("openai/gpt-5.6-sol", "anthropic/claude-opus-5", "google/gemini-3.7-flash")


def deprecated(old: str, new: str) -> None:
    """Says on stderr that `old` is deprecated, and what to use instead.

    The commands and flags from before 1.0 that still work:

        oeb benchmark      is `oeb run`, with the same flags
        --score-only       on `oeb run`: grades the runs already on disk and predicts nothing; use `oeb grade`
        --score-workers    on `oeb run` and `oeb grade`: ignored; the scorer sets how many processes it uses
        --gt               on `oeb score`: is `--gold`
    """
    print(f"oeb: {old} is deprecated and will be removed in a later release; use {new}", file=sys.stderr)


def read_object(path: str, flag: str) -> dict:
    """The JSON object in the file `path` that `flag` names."""
    try:
        got = json.loads(pathlib.Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise errors.UsageError(f"{flag} {path}: no such file") from None
    except IsADirectoryError:
        raise errors.UsageError(f"{flag} {path} is a folder; pass a JSON file") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise errors.UsageError(f"{flag} {path} isn't JSON: {exc}") from None
    if not isinstance(got, dict):
        raise errors.UsageError(f"{flag} {path} holds a JSON "
                                f"{'null' if got is None else type(got).__name__}; it must be a JSON object")
    return got


def read_object_or_json(value: str, flag: str) -> dict:
    """A JSON object given inline, or the path of a file holding one."""
    if not value.lstrip().startswith("{"):
        return read_object(value, flag)
    try:
        got = json.loads(value)
    except json.JSONDecodeError as exc:
        raise errors.UsageError(f"{flag} isn't JSON: {exc}") from None
    if not isinstance(got, dict):
        raise errors.UsageError(f"{flag} must be a JSON object, not {value!r:.80}")
    return got


def read_options(value: str | None) -> dict[str, list[dict[str, Any]]]:
    """`--options` as each provider's list of option sets, given as inline JSON or a path to a file.

    A provider maps to one object of options, or to a list of them to run it once for each:

        '{"datalab": {"mode": "accurate"}}'
        '{"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]}'

    JSON rather than a `provider:key=value` syntax, because OpenRouter model ids already use the colon
    (`:batch`, `:free`), and JSON keeps types: `8000` stays an integer."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        if not pathlib.Path(value).is_file():
            raise errors.UsageError(f"--options {value!r} is neither inline JSON nor a file that exists; pass a "
                                    "JSON object, or the path of a file holding one") from None
        parsed = read_object(value, "--options")
    if not isinstance(parsed, dict):
        raise errors.UsageError(f"--options must be a JSON object, not {value!r:.80}")
    out = {}
    for k, v in parsed.items():
        xs = v if isinstance(v, list) else [v]
        if not xs or not all(isinstance(x, dict) for x in xs):
            raise errors.UsageError(f"--options gives {k!r} {v!r:.80}; give it an object of options, or a non-empty "
                                    "list of them to run it once for each")
        out[k] = xs
    return out


def read_ids(path: pathlib.Path, flag: str) -> frozenset[str]:
    """A file of ids, one per line; blank lines and `#` comments are skipped."""
    if not path.is_file():
        raise errors.UsageError(f"{flag} {path}: no such file; it holds ids, one per line")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return frozenset(x for line in lines if (x := line.split("#", 1)[0].strip()))


def read_workers(value: str | None) -> int | dict[str, int]:
    """`--workers` as one number for every adapter, or a JSON object of a number per adapter."""
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        pass
    try:
        out = json.loads(value)
    except json.JSONDecodeError:
        raise errors.UsageError(f"--workers is a number, or a JSON object like '{{\"datalab\": 20}}', not "
                                f"{value!r}") from None
    if not isinstance(out, dict):
        raise errors.UsageError(f"--workers must be a number or a JSON object, not {value!r}")
    return out


def systems(providers: list[str], options: dict[str, list[dict[str, Any]]]) -> list:
    """An adapter for each provider and each of its option sets (`harness.system`)."""
    from . import harness

    if unknown := sorted(set(options) - set(providers)):
        raise errors.UsageError(f"--options names {unknown[0]!r}, which is not in --providers ({', '.join(providers)}); "
                                "its options would be ignored")
    return [harness.system(p, o) for p in providers for o in options.get(p, [{}])]


def confirm(text: str) -> bool:
    """Shows the plan on stderr and asks. With no terminal to ask (a pipe, CI), it goes ahead."""
    print(text, file=sys.stderr)
    if not sys.stdin.isatty(): return True
    sys.stderr.write("  proceed? [y/N] ")
    sys.stderr.flush()
    try:
        return input().strip().lower() in ("y", "yes")
    except EOFError:
        return False


def cmd_run(args) -> int:
    from . import benchmark

    b = benchmark.default(args.dataset, args.manifest)
    only = read_ids(args.docs, "--docs") if args.docs else None
    p = b.plan(systems(args.providers, read_options(args.options)), args.out, suites=args.suites, limit=args.limit,
               sample=args.sample, only=only, timeout=args.timeout, workers=read_workers(args.workers),
               retry_failed=args.retry_failed)
    if args.score_only:
        deprecated("--score-only", "`oeb grade <run folder>`")
        if never := [str(r.folder) for r in p.runs if not r.predictions.is_dir()]:
            raise errors.UsageError(f"no run at {never[0]}: this provider with these options was never predicted "
                                    "under this --out")
        _print({r.name: b.grade(r, details=args.verdicts) for r in p.runs})
        return 0
    if not (args.yes or confirm(b.describe(p))):
        print("oeb run: not run", file=sys.stderr)
        return 1
    try:
        b.predict(p)
    except Exception:
        # Note: what was predicted is graded even when predicting failed part way, as one JSON object, the
        # same as when it finishes; not after an interrupt, since whoever stopped it wants it stopped.
        _print({r.name: b.grade(r, details=args.verdicts) for r in p.runs if r.predictions.is_dir()})
        raise
    _print({r.name: b.grade(r, details=args.verdicts) for r in p.runs})
    return 0


def cmd_grade(args) -> int:
    from omni_bench_core import run

    from . import benchmark

    folder = run.Run.at(args.folder)
    said = folder.settings()
    dataset = args.dataset or said.get("dataset") or data.DEFAULT_DATASET
    if args.dataset is None and said.get("commit"): dataset = f"{dataset}@{said['commit']}"
    manifest = args.manifest or said.get("manifest") or data.MANIFEST
    _print(benchmark.default(dataset, manifest).grade(folder, details=args.verdicts))
    return 0


def cmd_score(args) -> int:
    from .scorers import field_matching

    if args.gt is not None:
        deprecated("--gt", "--gold")
        args.gold = args.gold or args.gt
    if args.gold is None:
        raise errors.UsageError("--gold is required: the ground truth, JSON")
    pred, gold, schema = (read_object(path, flag) for flag, path in (
        ("--pred", args.pred), ("--gold", args.gold), ("--schema", args.schema)))
    # Note: a bare --order-matters is every array, as the library's `order_matters=True`.
    order = True if args.order_matters == [] else tuple(args.order_matters or ())
    result = field_matching.score(pred, gold, schema, order_matters=order, verdicts=args.verdicts)
    if args.verdicts:
        result["verdicts"] = [v._asdict() for v in result["verdicts"]]
    _print(result)
    return 0


def cmd_predict(args) -> int:
    """One document through one system, by the rules a run uses; the record on stdout, and exit 1 when the
    record is a failure."""
    from omni_bench_core import adapter

    from . import harness, task

    system = harness.system(args.provider, read_object_or_json(args.options, "--options") if args.options else {})
    doc = pathlib.Path(args.doc)
    if not doc.is_file():
        raise errors.UsageError(f"--doc {doc}: no such file")
    # Note: predicting reads no gold, so the row's gold path is never opened.
    row = data.Doc(doc.stem, data.NO_SUITE, doc, doc, read_object(args.schema, "--schema"))
    rec = adapter.predict(system, task.Extract().input(row), timeout=args.timeout)
    _print(rec)
    return 1 if rec["error"] else 0


def cmd_check(args) -> int:
    """Whether a dataset is one oeb can run: its manifest, every input file, and every gold."""
    from . import benchmark

    _print({"dataset": args.dataset, "manifest": args.manifest, **benchmark.default(args.dataset, args.manifest).check()})
    return 0


def cmd_providers(args) -> int:
    """The names `--providers` accepts, or one provider's options and their defaults. The options are the
    adapter class's own fields, so this can't drift from what the adapter accepts."""
    import dataclasses

    from . import harness

    if not args.provider:
        for name in (*harness.PROVIDERS, *EXAMPLE_MODELS): print(name)
        print("\nthe model ids are examples: any OpenRouter org/model id works.", file=sys.stderr)
        return 0
    cls = harness.adapter_class(args.provider)
    fields = [f for f in dataclasses.fields(cls) if not ("/" in args.provider and f.name == "model")]
    if not fields:
        print(f"{args.provider}: no options; it takes the document and the schema and nothing else")
        return 0
    for f in fields:
        default = f.default if f.default is not dataclasses.MISSING else "(required)"
        print(f"{f.name:<20} {default!r:<30} {f.metadata.get('help', '')}")
    return 0


def _print(x: Any) -> None:
    json.dump(x, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="oeb", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", aliases=["benchmark"], help="predict with each provider, then grade")
    r.add_argument("--providers", nargs="+", required=True, metavar="NAME",
                   help="vendors (`oeb providers`) or OpenRouter model ids")
    r.add_argument("--options", metavar="JSON", help="each provider's options, inline JSON or a file")
    r.add_argument("--dataset", default=data.DEFAULT_DATASET,
                   help="a dataset folder, or hf://datasets/<org>/<name>[@<revision>]; default the public one")
    r.add_argument("--manifest", default=data.MANIFEST, help="a manifest inside the dataset")
    r.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs"), help="the parent of the run folders")
    r.add_argument("--suites", nargs="+", help="only these suites")
    r.add_argument("--limit", type=int, default=0, help="the first N documents; for a smoke test")
    r.add_argument("--sample", type=int, default=0, metavar="N", help="N documents spread over every suite")
    r.add_argument("--docs", type=pathlib.Path, metavar="FILE", help="only the documents whose ids this file lists, "
                                                                    "one per line")
    r.add_argument("--timeout", type=float, default=1800, help="seconds each document may take")
    r.add_argument("--workers", metavar="N|JSON", help="documents in flight per adapter: a number, or a JSON object")
    r.add_argument("--retry-failed", action="store_true",
                   help="predict again the documents whose recorded failure may not recur (a timeout, an outage)")
    r.add_argument("--no-verdicts", dest="verdicts", action="store_false", help="don't write per-address verdicts")
    r.add_argument("-y", "--yes", action="store_true", help="don't ask before predicting")
    r.add_argument("--score-only", action="store_true", help=argparse.SUPPRESS)   # Note: deprecated; see the docstring.
    r.add_argument("--score-workers", type=int, help=argparse.SUPPRESS)
    r.set_defaults(fn=cmd_run)

    g = sub.add_parser("grade", help="grade a folder of predictions/<doc_id>.json: yours, or a run's")
    g.add_argument("folder", help="the folder holding predictions/")
    g.add_argument("--dataset", default=None, help="default: the folder's own, else the public dataset")
    g.add_argument("--manifest", default=None, help="default: the folder's own, else manifest.parquet")
    g.add_argument("--no-verdicts", dest="verdicts", action="store_false", help="don't write per-address verdicts")
    g.add_argument("--score-workers", type=int, help=argparse.SUPPRESS)
    g.set_defaults(fn=cmd_grade)

    s = sub.add_parser("score", help="score one prediction against one ground truth")
    s.add_argument("--pred", required=True, help="the prediction, JSON")
    s.add_argument("--gold", help="the ground truth, JSON (required)")
    s.add_argument("--gt", help=argparse.SUPPRESS)
    s.add_argument("--schema", required=True, help="the JSON Schema the prediction was generated against")
    s.add_argument("--verdicts", action="store_true", help="include one verdict per address")
    s.add_argument("--order-matters", nargs="*", metavar="ARRAY",
                   help="arrays whose order is part of the answer; given bare, every array")
    s.set_defaults(fn=cmd_score)

    d = sub.add_parser("predict", help="run one document through one system")
    d.add_argument("--provider", required=True, help="a vendor (`oeb providers`) or an OpenRouter model id")
    d.add_argument("--doc", required=True, help="the document, PDF")
    d.add_argument("--schema", required=True, help="the JSON Schema to extract against")
    d.add_argument("--options", metavar="JSON", help="the provider's options, a JSON object, inline or a file")
    d.add_argument("--timeout", type=float, default=1800, help="seconds the document may take")
    d.set_defaults(fn=cmd_predict)

    c = sub.add_parser("check", help="whether a dataset is one oeb can run: its manifest, files and gold")
    c.add_argument("--dataset", default=data.DEFAULT_DATASET, help="a dataset folder, or hf://datasets/<org>/<name>")
    c.add_argument("--manifest", default=data.MANIFEST, help="a manifest inside the dataset")
    c.set_defaults(fn=cmd_check)

    pl = sub.add_parser("providers", help="the providers, or one provider's options")
    pl.add_argument("provider", nargs="?", help="a provider, for its options")
    pl.set_defaults(fn=cmd_providers)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Runs one `oeb` command and returns its exit code.

    `run` shows its plan and waits for a yes, unless given `-y` or run with no terminal. A `UsageError` (what
    was asked is wrong) or a `Stop` (an account, a key or a dependency) is printed as
    `oeb <command>: <what was wrong>` and exits 1; an interrupt exits 130. Anything else is a bug, and is
    raised as it is."""
    args = parser().parse_args(argv)
    if args.cmd == "benchmark":
        deprecated("`oeb benchmark`", "`oeb run`, which takes the same flags")
        args.cmd = "run"
    if getattr(args, "score_workers", None) is not None:
        deprecated("--score-workers", "nothing: the scorer sets how many processes it uses")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().handlers[:] = [handler]
    logging.getLogger().setLevel(logging.WARNING)
    for name in (__package__, "omni_bench_core"): logging.getLogger(name).setLevel(logging.INFO)
    try:
        return args.fn(args)
    except errors.UsageError as exc:
        print(f"oeb {args.cmd}: {exc}", file=sys.stderr)
        return 1
    except errors.Stop as exc:
        print(f"oeb {args.cmd}: {type(exc).__name__}: {exc}" + ("; fix it and rerun to resume" if args.cmd == "run" else ""),
              file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"oeb {args.cmd}: interrupted" + ("; rerun to resume" if args.cmd == "run" else ""), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
