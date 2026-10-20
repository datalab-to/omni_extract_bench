"""The `oeb` command: a thin wrapper over `benchmark`, `harness` and the scorer.

    oeb run        --system datalab --out runs/                    predict, then grade
    oeb grade      mine/ [--partial] [--dataset ...]               grade a folder of predictions/<doc_id>.json
    oeb score      --pred p.json --gold g.json --schema s.json     score one prediction against one gold
    oeb predict    --system datalab --doc x.pdf --schema s.json    one document through one system
    oeb check      [--dataset ...] [--manifest ...]                whether a dataset is one oeb can run
    oeb systems    [datalab]                                       the systems, or one system's options

A run is one system, so `oeb run` refuses several. To compare systems, run one command per system, with the
same choice of documents:

    oeb run --system datalab --options '{"mode": "accurate"}' --sample 20 -y
    oeb run --system reducto --sample 20 -y

Results go to stdout as JSON; the plan and progress go to stderr. Commands and flags from before 1.0 still
work, each printing what replaced it (`deprecated`). `__version__` is the installed package's version.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import pathlib
import shlex
import sys
from typing import Any

from omni_bench_core import errors

from . import data

EXAMPLE_MODELS = ("openai/gpt-5.6-sol", "anthropic/claude-opus-5", "google/gemini-3.7-flash")
__version__ = importlib.metadata.version("omni-extract-bench")


def deprecated(old: str, new: str) -> None:
    """Says on stderr that `old` is deprecated, and what to use instead.

    The commands and flags from before 1.0 that still work:

        oeb benchmark      is `oeb run`, with the same flags
        oeb providers      is `oeb systems`
        --systems          on `oeb run`, with one name: is `--system`. Several names are refused (`one_per_system`)
        --providers        on `oeb run`: is `--systems`, so the same
        --provider         on `oeb predict`: is `--system`
        --docs             on `oeb run`: is `--only`
        --no-verdicts      on `oeb run` and `oeb grade`: is `--no-details`
        --verdicts         on `oeb score`: is `--details`
        --score-only       on `oeb run`: grades the runs already on disk and predicts nothing; use `oeb grade`
        --score-workers    on `oeb run` and `oeb grade`: is ignored; the scorer sets how many processes it uses
        --gt               on `oeb score`: is `--gold`
    """
    print(f"oeb: {old} is deprecated and will be removed in a later release; use {new}", file=sys.stderr)


# Note: each pre-1.0 flag that was renamed, as (its dest, the new flag's dest, the old flag, the new flag).
RENAMED = (
    ("provider", "system", "--provider", "--system"),
    ("docs", "only", "--docs", "--only"),
    ("no_verdicts", "no_details", "--no-verdicts", "--no-details"),
    ("verdicts", "details", "--verdicts", "--details"),
    ("gt", "gold", "--gt", "--gold"),
)


def renamed(args: argparse.Namespace) -> None:
    """Moves each renamed pre-1.0 flag given onto its new name, saying so (`deprecated`)."""
    for old, new, old_flag, new_flag in RENAMED:
        if getattr(args, old, None) in (None, False): continue
        deprecated(old_flag, new_flag)
        if getattr(args, new, None) not in (None, False):
            raise errors.UsageError(f"{old_flag} is the old name of {new_flag}; give {new_flag} only")
        setattr(args, new, getattr(args, old))


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


def system_options(value: str | None, system: str) -> dict[str, Any]:
    """`--options` of `oeb run` and `oeb predict`: the system's options as a bare object (`{"mode": "accurate"}`),
    or keyed by the system (`{"datalab": {"mode": "accurate"}}`). An object keyed by other systems is refused."""
    from . import harness

    if not value: return {}
    got = read_object_or_json(value, "--options")
    if list(got) != [system]:
        if others := [k for k, v in got.items() if isinstance(v, dict) and (k in harness.PROVIDERS or "/" in k)]:
            raise errors.UsageError(f"--options gives options for {', '.join(others)}, but this command runs {system}; "
                                    f"give {system}'s options bare, e.g. '{{\"mode\": \"accurate\"}}'")
        return got
    if not isinstance(got[system], dict):
        raise errors.UsageError(f"--options gives {system!r} {got[system]!r:.80}; one command runs one set of "
                                f"options, so give one object, e.g. '{{\"{system}\": {{...}}}}'")
    return got[system]


def read_ids(path: pathlib.Path, flag: str) -> frozenset[str]:
    """A file of ids, one per line; blank lines and `#` comments are skipped."""
    if not path.is_file():
        raise errors.UsageError(f"{flag} {path}: no such file; it holds ids, one per line")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return frozenset(x for line in lines if (x := line.split("#", 1)[0].strip()))


def confirm(text: str) -> bool:
    """Shows the plan on stderr and asks. With no terminal to ask (a pipe, CI), it goes ahead. Ctrl-D is a no."""
    print(text, file=sys.stderr)
    if not sys.stdin.isatty(): return True
    sys.stderr.write("  proceed? [y/N] ")
    sys.stderr.flush()
    try:
        return input().strip().lower() in ("y", "yes")
    except EOFError:
        # Note: Ctrl-D ends no line, so the next message would follow the prompt on the same one.
        sys.stderr.write("\n")
        return False


def said(command: str, exc: errors.UsageError | errors.Stop) -> str:
    """The line `oeb <command>` prints on stderr when `exc` stops it."""
    if not isinstance(exc, errors.Stop): return f"oeb {command}: {exc}"
    # Note: the core's messages say what to do; an account failure carries only the system's own words.
    resume = isinstance(exc, errors.AccountFailure) and command == "run" and "again" not in str(exc)
    return f"oeb {command}: {type(exc).__name__}: {exc}" + ("; fix it, then run the same command again to resume"
                                                            if resume else "")


def predicted(p: Any) -> None:
    """For the deprecated `--score-only`: refuses a run that was never predicted."""
    if not p.run.predictions.is_dir():
        raise errors.UsageError(f"no run at {p.run.folder}: this system with these options and documents was never "
                                "predicted under this --out")


SYSTEM_FLAGS = ("--system", "--systems", "--providers")


def one_system(args: argparse.Namespace) -> str:
    """The one system `oeb run` was given. Several are refused before anything else, with one command per
    system (`one_per_system`); one name under the deprecated `--systems` or `--providers` is `--system`."""
    old = "--providers" if args.providers else "--systems" if args.systems else None
    names = args.system or args.providers or args.systems
    if len(names) > 1:
        raise errors.UsageError(one_per_system(args.argv, names))
    if old: deprecated(old, "--system")
    return names[0]


def one_per_system(argv: list[str], names: list[str]) -> str:
    """The refusal of a run given several systems: one command per system, each the user's own `argv` with
    `--system` naming one system. A keyed `--options` becomes each system's own bare object."""
    rest, skip = [], False
    for tok in argv[1:]:   # Note: argv[0] is the command, `run` or its old name
        if skip and not tok.startswith("-"): continue
        skip = tok in SYSTEM_FLAGS
        if skip or tok.split("=", 1)[0] in SYSTEM_FLAGS: continue
        rest.append(tok)

    def options(value: str) -> dict[str, str | None]:
        try:
            got = read_object_or_json(value, "--options")
        except errors.UsageError:
            return {n: value for n in names}
        if got and all(k in names and isinstance(v, dict) for k, v in got.items()):
            return {n: json.dumps(got[n], ensure_ascii=False) if n in got else None for n in names}
        return {n: value for n in names}

    def command(name: str) -> str:
        out, xs = ["oeb", "run", "--system", name], iter(rest)
        for tok in xs:
            if tok == "--options" or tok.startswith("--options="):
                value = tok.split("=", 1)[1] if "=" in tok else next(xs, "")
                if (o := options(value)[name]) is not None: out += ["--options", o]
            else:
                out.append(tok)
        return " ".join(shlex.quote(x) for x in out)

    return (f"a run is one system, but {len(names)} were given ({', '.join(names)}). Run one command per system:\n"
            + "".join(f"    {command(n)}\n" for n in names)
            + "Use the same --sample (or --limit, --only, --suites) in each, so they answer the same rows.")


def cmd_run(args) -> int:
    """Plans, asks, predicts, then grades one system's run, and prints the grading. A run that stops part way
    is left for the same command, run again."""
    from . import benchmark, harness

    system = one_system(args)
    b = benchmark.default(args.dataset, args.manifest)
    only = read_ids(args.only, "--only") if args.only else None
    kw = dict(suites=args.suites, limit=args.limit, sample=args.sample, only=only, timeout=args.timeout,
              workers=args.workers, retry_failed=args.retry_failed)
    p = b.plan(harness.system(system, system_options(args.options, system)), args.out, **kw)
    if args.score_only:
        deprecated("--score-only", "`oeb grade <run folder>`")
        predicted(p)
    elif not (args.yes or confirm(b.describe(p))):
        print("oeb run: not run", file=sys.stderr)
        return 1
    else:
        b.predict(p)
    _print(b.grade(p.run, details=not args.no_details))
    return 0


def cmd_grade(args) -> int:
    from omni_bench_core import run

    from . import benchmark

    folder = run.Run.at(args.folder)
    said = folder.settings()
    dataset = args.dataset or said.get("dataset") or data.DEFAULT_DATASET
    if args.dataset is None and said.get("commit"): dataset = f"{dataset}@{said['commit']}"
    manifest = args.manifest or said.get("manifest") or data.MANIFEST
    overlay = (said.get("task") or {}).get("options", {}).get("overlay", True)
    _print(benchmark.default(dataset, manifest, overlay=overlay).grade(folder, partial=args.partial,
                                                                       details=not args.no_details))
    return 0


def cmd_score(args) -> int:
    from .scorers import field_matching

    pred, gold, schema = (read_object(path, flag) for flag, path in (
        ("--pred", args.pred), ("--gold", args.gold), ("--schema", args.schema)))
    # Note: a bare --order-matters is every array, as the library's `order_matters=True`.
    order = True if args.order_matters == [] else tuple(args.order_matters or ())
    result = field_matching.score(pred, gold, schema, order_matters=order, verdicts=args.details)
    if args.details:
        result["verdicts"] = [v._asdict() for v in result["verdicts"]]
    _print(result)
    return 0


def cmd_predict(args) -> int:
    """One document through one system, by the rules a run uses; the record on stdout, and exit 1 when the
    record is a failure. A refused connection stops it, and any other exception from the adapter is a bug in
    the adapter (`errors.bug`), as in a run."""
    from omni_bench_core import adapter

    from . import harness, task
    from .harness import contract

    system = harness.system(args.system, system_options(args.options, args.system))
    doc = pathlib.Path(args.doc)
    if not doc.is_file():
        raise errors.UsageError(f"--doc {doc}: no such file")
    # Note: predicting reads no gold, so the row's gold path is never opened.
    row = data.Doc(doc.stem, data.NO_SUITE, doc, doc, read_object(args.schema, "--schema"))
    try:
        rec = adapter.predict(system, task.Extract().input(row), timeout=args.timeout)
    except errors.Stop:
        raise
    except Exception as exc:  # noqa: BLE001 -- the adapter's code raised; say whose code it is, as a run does
        if errors.refused(exc): raise contract.refused(system, exc) from None
        raise errors.bug(f"predicting {doc}", exc, type(system)) from None
    _print(rec)
    return 1 if rec["error"] else 0


def cmd_check(args) -> int:
    """Whether a dataset is one oeb can run: its manifest, every input file, and every gold."""
    from . import benchmark

    _print({"dataset": args.dataset, "manifest": args.manifest, **benchmark.default(args.dataset, args.manifest).check()})
    return 0


def cmd_systems(args) -> int:
    """The names `--system` accepts, or one system's options, their defaults, and the environment variables
    it needs (`Adapter.keys`). Both are read off the adapter class, so this can't drift from the adapter."""
    import dataclasses

    from . import harness

    if not args.system:
        for name in (*harness.PROVIDERS, *EXAMPLE_MODELS): print(name)
        print("\nthe model ids are examples: any OpenRouter org/model id works.", file=sys.stderr)
        return 0
    cls = harness.adapter_class(args.system)
    fields = [f for f in dataclasses.fields(cls) if not ("/" in args.system and f.name == "model")]
    if not fields:
        print(f"{args.system}: no options; it takes the document and the schema and nothing else")
    for f in fields:
        default = f.default if f.default is not dataclasses.MISSING else "(required)"
        print(f"{f.name:<20} {default!r:<30} {f.metadata.get('help', '')}")
    print(f"\nneeds {', '.join(cls.keys)} in the environment: {' and '.join(f'export {k}=...' for k in cls.keys)}"
          if cls.keys else "\nneeds no key")
    return 0


def _print(x: Any) -> None:
    json.dump(x, sys.stdout, indent=2, default=str, ensure_ascii=False)
    sys.stdout.write("\n")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="oeb", description=__doc__.split("\n\n")[0], allow_abbrev=False)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", aliases=["benchmark"], help="predict with one system, then grade", allow_abbrev=False)
    # Note: a required flag and its deprecated forms are a required group, so usage shows the flag as required.
    rs = r.add_mutually_exclusive_group(required=True)
    # Note: `nargs="+"` only so several systems are refused with one command per system (`one_system`), not
    # with argparse's "unrecognized arguments".
    rs.add_argument("--system", nargs="+", metavar="NAME", help="a vendor (`oeb systems`) or an OpenRouter model id")
    rs.add_argument("--systems", nargs="+", help=argparse.SUPPRESS)
    rs.add_argument("--providers", nargs="+", help=argparse.SUPPRESS)
    r.add_argument("--options", metavar="JSON", help="the system's options, inline JSON or a file: an object, e.g. "
                                                      "'{\"mode\": \"accurate\"}'")
    r.add_argument("--dataset", default=data.DEFAULT_DATASET,
                   help="a dataset folder, or hf://datasets/<org>/<name>[@<revision>] (default: %(default)s, the "
                        "public one)")
    r.add_argument("--manifest", default=data.MANIFEST, help="a manifest inside the dataset (default: %(default)s)")
    r.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs"),
                   help="the parent of the run folders (default: runs/)")
    r.add_argument("--suites", nargs="+", help="only these suites")
    r.add_argument("--limit", type=int, default=0, help="the first N documents; for a smoke test")
    r.add_argument("--sample", type=int, default=0, metavar="N", help="N documents spread over every suite")
    r.add_argument("--only", type=pathlib.Path, metavar="FILE", help="only the documents whose ids this file lists, "
                                                                    "one per line")
    r.add_argument("--docs", type=pathlib.Path, help=argparse.SUPPRESS)
    r.add_argument("--timeout", type=float, default=1800, help="seconds each document may take (default: %(default)g)")
    r.add_argument("--workers", type=int, default=0, metavar="N",
                   help="documents in flight at once (default: the system's own)")
    r.add_argument("--retry-failed", action="store_true",
                   help="predict again the documents whose recorded failure may not recur (a timeout, an outage)")
    r.add_argument("--no-details", action="store_true", help="don't write each document's per-address breakdown")
    r.add_argument("--no-verdicts", action="store_true", help=argparse.SUPPRESS)
    r.add_argument("-y", "--yes", action="store_true", help="don't ask before predicting")
    r.add_argument("--score-only", action="store_true", help=argparse.SUPPRESS)   # Note: deprecated; see the docstring.
    r.add_argument("--score-workers", type=int, help=argparse.SUPPRESS)
    r.set_defaults(fn=cmd_run)

    g = sub.add_parser("grade", help="grade a folder of predictions/<doc_id>.json: yours, or a run's",
                       allow_abbrev=False)
    g.add_argument("folder", help="the folder holding predictions/")
    g.add_argument("--partial", action="store_true",
                   help="grade a run that isn't finished: its unfinished documents count as 0, and the summary "
                        "counts them under \"partial\"; scores.jsonl names each")
    g.add_argument("--dataset", default=None, help="a dataset folder, or hf://datasets/<org>/<name>[@<revision>] "
                                                    "(default: the run's own, else the public dataset, "
                                                    f"{data.DEFAULT_DATASET})")
    g.add_argument("--manifest", default=None, help="a manifest inside the dataset (default: the run's own, else "
                                                     "manifest.parquet)")
    g.add_argument("--no-details", action="store_true", help="don't write each document's per-address breakdown")
    g.add_argument("--no-verdicts", action="store_true", help=argparse.SUPPRESS)
    g.add_argument("--score-workers", type=int, help=argparse.SUPPRESS)
    g.set_defaults(fn=cmd_grade)

    s = sub.add_parser("score", help="score one prediction against one ground truth", allow_abbrev=False)
    s.add_argument("--pred", required=True, help="the prediction, JSON")
    sg = s.add_mutually_exclusive_group(required=True)
    sg.add_argument("--gold", help="the ground truth, JSON")
    sg.add_argument("--gt", help=argparse.SUPPRESS)
    s.add_argument("--schema", required=True, help="the JSON Schema the prediction was generated against")
    s.add_argument("--details", action="store_true", help="include the breakdown: one verdict per address")
    s.add_argument("--verdicts", action="store_true", help=argparse.SUPPRESS)
    s.add_argument("--order-matters", nargs="*", metavar="ARRAY",
                   help="arrays whose order is part of the answer; given bare, every array")
    s.set_defaults(fn=cmd_score)

    d = sub.add_parser("predict", help="run one document through one system", allow_abbrev=False)
    ds = d.add_mutually_exclusive_group(required=True)
    ds.add_argument("--system", help="a vendor (`oeb systems`) or an OpenRouter model id")
    ds.add_argument("--provider", help=argparse.SUPPRESS)
    d.add_argument("--doc", required=True, help="the document, PDF")
    d.add_argument("--schema", required=True, help="the JSON Schema to extract against")
    d.add_argument("--options", metavar="JSON", help="the system's options, inline JSON or a file: an object, "
                                                      "e.g. '{\"mode\": \"accurate\"}'")
    d.add_argument("--timeout", type=float, default=1800, help="seconds the document may take (default: %(default)g)")
    d.set_defaults(fn=cmd_predict)

    c = sub.add_parser("check", help="whether a dataset is one oeb can run: its manifest, files and gold",
                       allow_abbrev=False)
    c.add_argument("--dataset", default=data.DEFAULT_DATASET,
                   help="a dataset folder, or hf://datasets/<org>/<name>[@<revision>] (default: %(default)s)")
    c.add_argument("--manifest", default=data.MANIFEST, help="a manifest inside the dataset (default: %(default)s)")
    c.set_defaults(fn=cmd_check)

    pl = sub.add_parser("systems", aliases=["providers"], help="the systems, or one system's options and keys",
                        allow_abbrev=False)
    pl.add_argument("system", nargs="?", help="a system, for its options and the environment variables it needs")
    pl.set_defaults(fn=cmd_systems)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Runs one `oeb` command and returns its exit code.

    `run` shows its plan and waits for a yes, unless given `-y` or run with no terminal. A `UsageError` (what
    was asked is wrong, or a run graded before it is finished) or a `Stop` (an account, a key, a dependency,
    or a bug in an adapter) is printed as `oeb <command>: <what was wrong>` and exits 1; an interrupt exits
    130. Anything else is a bug in oeb, and is raised as it is."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    args.argv = argv
    if args.command == "benchmark":
        deprecated("`oeb benchmark`", "`oeb run`, which takes the same flags")
        args.command = "run"
    if args.command == "providers":
        deprecated("`oeb providers`", "`oeb systems`")
        args.command = "systems"
    if getattr(args, "score_workers", None) is not None:
        print("oeb: --score-workers is ignored and will be removed; the scorer sets how many processes it uses",
              file=sys.stderr)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().handlers[:] = [handler]
    logging.getLogger().setLevel(logging.WARNING)
    for name in (__package__, "omni_bench_core"): logging.getLogger(name).setLevel(logging.INFO)
    try:
        renamed(args)
        return args.fn(args)
    except (errors.UsageError, errors.Stop) as exc:
        print(said(args.command, exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"oeb {args.command}: interrupted" + ("; run the same command again to resume" if args.command == "run"
                                                    else ""), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
