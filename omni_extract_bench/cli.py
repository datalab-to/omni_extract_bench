"""The `oeb` command: a thin wrapper over `benchmark`, `harness` and the scorer.

    oeb init       datalab > datalab.yaml                          a config: one run, every key written out
    oeb run        --config datalab.yaml [--set rows.sample=20]    predict, then grade
    oeb grade      mine/ [--partial] [--dataset ...]               grade a folder of predictions/<doc_id>.json
    oeb score      --pred p.json --gold g.json --schema s.json     score one prediction against one gold
    oeb predict    --system datalab --doc x.pdf --schema s.json    one document through one system
    oeb check      [--dataset ...] [--manifest ...]                whether a dataset is one oeb can run
    oeb systems    [datalab]                                       the systems, or one system's options

A run is one config: the dataset, its manifest, the scorers, which rows, and one system with every option
(omni-bench-core's `config`). `--set path=value` changes one value the file has. To compare systems, write one
config per system, with the same `rows`:

    oeb init datalab > datalab.yaml && oeb init reducto > reducto.yaml
    oeb run --config datalab.yaml --set system.datalab.mode=accurate --set rows.sample=20 -y
    oeb run --config reducto.yaml --set rows.sample=20 -y

Each run folder keeps its config, resolved, as `config.yaml`, so `oeb run --config <run>/config.yaml` asks it
again. Results go to stdout as JSON; the plan and progress go to stderr. Commands and flags from before 1.0 still
work, each printing what replaced it (`deprecated`). `__version__` is the installed package's version.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import pathlib
import sys
from typing import Any

from omni_bench_core import config, errors

from . import data

EXAMPLE_MODELS = ("openai/gpt-5.6-sol", "anthropic/claude-opus-5", "google/gemini-3.7-flash")
__version__ = importlib.metadata.version("omni-extract-bench")


def deprecated(old: str, new: str) -> None:
    """Says on stderr that `old` is deprecated, and what to use instead.

    The commands and flags from before 1.0 that still work:

        oeb providers      is `oeb systems`
        --provider         on `oeb predict`: is `--system`
        --no-verdicts      on `oeb grade`: is `--no-details`
        --verdicts         on `oeb score`: is `--details`
        --score-workers    on `oeb grade`: is ignored; the scorer sets how many processes it uses
        --gt               on `oeb score`: is `--gold`
    """
    print(f"oeb: {old} is deprecated and will be removed in a later release; use {new}", file=sys.stderr)


# Note: each pre-1.0 flag that was renamed, as (its dest, the new flag's dest, the old flag, the new flag).
RENAMED = (
    ("provider", "system", "--provider", "--system"),
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
            raise ValueError(f"{old_flag} is the old name of {new_flag}; give {new_flag} only")
        setattr(args, new, getattr(args, old))


def read_object(path: str, flag: str) -> dict:
    """The JSON object in the file `path` that `flag` names."""
    try:
        got = json.loads(pathlib.Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ValueError(f"{flag} {path}: no such file") from None
    except IsADirectoryError:
        raise ValueError(f"{flag} {path} is a folder; pass a JSON file") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{flag} {path} isn't JSON: {exc}") from None
    if not isinstance(got, dict):
        raise ValueError(f"{flag} {path} holds a JSON "
                         f"{'null' if got is None else type(got).__name__}; it must be a JSON object")
    return got


def read_object_or_json(value: str, flag: str) -> dict:
    """A JSON object given inline, or the path of a file holding one."""
    if not value.lstrip().startswith("{"):
        return read_object(value, flag)
    try:
        got = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag} isn't JSON: {exc}") from None
    if not isinstance(got, dict):
        raise ValueError(f"{flag} must be a JSON object, not {value!r:.80}")
    return got


def system_options(value: str | None, system: str) -> dict[str, Any]:
    """`--options` of `oeb predict`: the system's options as a bare object (`{"mode": "accurate"}`),
    or keyed by the system (`{"datalab": {"mode": "accurate"}}`). An object keyed by other systems is refused."""
    from . import harness

    if not value: return {}
    got = read_object_or_json(value, "--options")
    if list(got) != [system]:
        if others := [k for k, v in got.items() if isinstance(v, dict) and (k in harness.PROVIDERS or "/" in k)]:
            raise ValueError(f"--options gives options for {', '.join(others)}, but this command runs {system}; "
                             f"give {system}'s options bare, e.g. '{{\"mode\": \"accurate\"}}'")
        return got
    if not isinstance(got[system], dict):
        raise ValueError(f"--options gives {system!r} {got[system]!r:.80}; one command runs one set of "
                         f"options, so give one object, e.g. '{{\"{system}\": {{...}}}}'")
    return got[system]


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


def scorers() -> dict[str, type]:
    """The scorers a config may name, by name."""
    from .scorers import field_matching

    return {field_matching.FieldMatching.name: field_matching.FieldMatching}


def cmd_init(args) -> int:
    """A config for one system on the public dataset, graded by field matching, as YAML on stdout: every row, and
    every option of the system at its default, so the file says the whole run."""
    import yaml

    from . import harness

    c = config.template(harness.system, args.system, dataset=args.dataset, manifest=args.manifest,
                        scorers={"field_matching": {}}, timeout=1800)
    sys.stdout.write(yaml.safe_dump(c, sort_keys=False, width=1000, allow_unicode=True))
    return 0


def cmd_run(args) -> int:
    """Plans, asks, predicts, then grades the run a config asks, and prints the grading: a new run, or the one
    `--resume` names. A run that stops part way is continued with `--resume <its folder name>`."""
    from . import harness, task

    c = config.load(args.config, args.set)
    b = config.benchmark(c, task.Extract(), scorers())
    s = config.system(c, harness.system)
    p = config.plan(c, b, s, args.out, resume=args.resume, retry_failed=args.retry_failed)
    if not (args.yes or confirm(b.describe(p))):
        print("oeb run: not run", file=sys.stderr)
        return 1
    config.write(c, s, p.run.folder)
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
    _print(benchmark.default(dataset, manifest).grade(folder, partial=args.partial, details=not args.no_details))
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
    """One document through one system, by a run's rules: the record on stdout, and exit 1 when it is a
    failure. Anything the adapter raises is raised as it is, with a note naming the adapter."""
    from omni_bench_core import adapter

    from . import harness, task

    system = harness.system(args.system, system_options(args.options, args.system))
    doc = pathlib.Path(args.doc)
    if not doc.is_file():
        raise ValueError(f"--doc {doc}: no such file")
    # Note: predicting reads no gold, so the row's gold path is never opened.
    row = data.Doc(doc.stem, data.NO_SUITE, doc, doc, read_object(args.schema, "--schema"))
    try:
        rec = adapter.predict(system, task.Extract().input(row), timeout=args.timeout)
    except Exception as exc:
        # Note: the core's `predict` already noted a refused connection; anything else is the adapter's bug.
        if not errors.refused(exc): errors.noted(exc, type(system), f"predicting {doc}")
        raise
    _print(rec)
    return 1 if rec["error"] else 0


def cmd_check(args) -> int:
    """Whether a dataset is one oeb can run: its manifest, every input file, and every gold."""
    from . import benchmark

    _print({"dataset": args.dataset, "manifest": args.manifest, **benchmark.default(args.dataset, args.manifest).check()})
    return 0


def cmd_systems(args) -> int:
    """The names `--system` accepts, or one system's options, their defaults and the environment variables
    it needs, all read off the adapter class."""
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

    n = sub.add_parser("init", help="a config for one system, as YAML on stdout", allow_abbrev=False)
    n.add_argument("system", help="a vendor (`oeb systems`) or an OpenRouter model id")
    n.add_argument("--dataset", default=data.DEFAULT_DATASET,
                   help="a dataset folder, or hf://datasets/<org>/<name>[@<revision>] (default: %(default)s, the "
                        "public one)")
    n.add_argument("--manifest", default=data.MANIFEST, help="a manifest inside the dataset (default: %(default)s)")
    n.set_defaults(fn=cmd_init)

    r = sub.add_parser("run", help="predict with one config's system, then grade", allow_abbrev=False)
    r.add_argument("--config", required=True, metavar="YAML", help="the run: a config, as `oeb init` writes one")
    r.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                   help="change one value the config has, read as YAML, e.g. rows.sample=20; repeatable")
    r.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs"),
                   help="the parent of the run folders (default: runs/)")
    r.add_argument("--resume", metavar="RUN",
                   help="continue the run of this folder name under --out, asking the same question; it pays only "
                        "for documents with no answer yet (default: a new run)")
    r.add_argument("--retry-failed", action="store_true",
                   help="with --resume, predict again the documents whose recorded failure may not recur (a "
                        "timeout, an outage)")
    r.add_argument("--no-details", action="store_true", help="don't write each document's per-address breakdown")
    r.add_argument("-y", "--yes", action="store_true", help="don't ask before predicting")
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
    """Runs one `oeb` command and returns its exit code. `run` shows its plan and waits for a yes, unless
    given `-y` or run with no terminal. An interrupt exits 130; anything else is raised as it is, with
    its traceback."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
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
    except KeyboardInterrupt:
        print(f"oeb {args.command}: interrupted" + ("; continue it with --resume <its folder name>" if args.command == "run"
                                                    else ""), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
