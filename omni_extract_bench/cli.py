"""Command-line interface.

    oeb score      --pred p.json --gt g.json --schema s.json [--verdicts]
    oeb benchmark  --providers datalab reducto --out runs/
    oeb grade      mine/ [--dataset ...] [--manifest ...] [--suites ...]
    oeb check      [--dataset ...] [--manifest ...]       # is the dataset one oeb can run?
    oeb predict    --provider datalab --doc x.pdf --schema s.json
    oeb providers [datalab]

`score` grades one document: three JSON files in, the metrics dict out as JSON on stdout.
A mistake in what was asked (a missing file, an unknown provider, a malformed option) is one line
on stderr, `oeb <command>: <what was wrong>`, and exit 1; a traceback is a bug in oeb.
`benchmark` is the whole published benchmark -- fetch the corpus, run it through each
vendor, score, write it down -- and lives in `benchmark.py`, which is a script rather than
library code precisely because it makes the orchestration decisions the library refuses to.
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
    except FileNotFoundError:
        raise ValueError(f"{path}: no such file") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not JSON -- {exc}") from None


def cmd_score(args) -> int:
    from .scorers.field_matching import score

    pred, gt, schema = (read_json(f) for f in (args.pred, args.gt, args.schema))
    files = (("--pred", args.pred, pred), ("--gt", args.gt, gt), ("--schema", args.schema, schema))
    if bad := next(((flag, path, x) for flag, path, x in files if not isinstance(x, dict)), None):
        raise ValueError(f"{bad[0]} {bad[1]} is a JSON {type(bad[2]).__name__}; it must be a JSON object")
    result = score(pred, gt, schema, order_matters=args.order_matters or (), verdicts=args.verdicts)
    if args.verdicts:
        result["verdicts"] = [v._asdict() for v in result["verdicts"]]
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


EXAMPLE_MODELS = ("openai/gpt-5.6-sol", "anthropic/claude-opus-5", "google/gemini-3.7-flash")


def cmd_providers(args) -> int:
    """The names `--providers` accepts, or one provider's options in detail.

        oeb providers            every name
        oeb providers datalab    what `--options` takes for it, and what each is by default

    ONE VERB, NOT TWO. The list is the question usually being asked and the options are a
    follow-up about one entry -- printing every provider's defaults in the list drowned the
    names. PIPED, IT IS STILL ONE COLUMN OF NAMES: the table is for a reader, and
    `oeb providers | ...` gets the list it has always got.

    The options come from the adapter's own `Config` -- its fields are the options and their
    defaults -- so this cannot drift from what the adapter accepts.
    """
    import dataclasses

    from rich import box
    from rich.console import Console
    from rich.table import Table

    from .harness import PROVIDERS
    from .harness.registry import config_for

    console = Console()

    if not args.provider:
        if not console.is_terminal:
            for provider in (*PROVIDERS, *EXAMPLE_MODELS):
                print(provider)
            return 0
        table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
        table.add_column("provider")
        for provider in PROVIDERS:
            table.add_row(provider)
        table.add_section()
        for model in EXAMPLE_MODELS:
            table.add_row(model)
        console.print(table)
        print("\nthe three model ids are examples: any OpenRouter org/model id works.")
        return 0

    config = config_for(args.provider)
    fields = dataclasses.fields(config)
    if not fields:
        console.print(f"{args.provider}: no options -- it takes the document and the schema "
                      f"and nothing else")
        return 0

    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False,
                  title=args.provider, title_justify="left")
    table.add_column("option")
    table.add_column("default", overflow="fold")
    for field in fields:
        value = getattr(config, field.name)
        table.add_row(field.name, "''" if value == "" else str(value))
    console.print(table)
    # Note: a model id's `model` is its name, which --options refuses, so the example is another field.
    if example := next((f.name for f in fields if f.name != "model"), None):
        print(f"\noeb benchmark --providers {args.provider} "
              f"--options '{{\"{args.provider}\": {{\"{example}\": ...}}}}'")
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
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        if not (path := pathlib.Path(value)).is_file():
            raise FileNotFoundError(f"--options: {value!r} is not JSON, and there is no file of that "
                                    "name; pass a JSON object, or the path of a file holding one") from None
        try:
            parsed = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"--options: {path} is not JSON -- {exc}") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"--options must be a JSON object, not {value!r:.80}")
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


def read_workers(value: str | None) -> int | dict[str, int]:
    """`--workers` as one number for every adapter, or a JSON object of a number per adapter.

        --workers 25                              -> 25
        --workers '{"reducto": 25, "datalab": 40}' -> {"reducto": 25, "datalab": 40}

    Per adapter rather than one global number, because the right value is a fact about the vendor
    account: these are documents in flight at it, and a run naming several vendors has no one number
    that fits them all."""
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        pass
    try:
        out = json.loads(value)
    except json.JSONDecodeError:
        raise ValueError(f"--workers is a number, or a JSON object like '{{\"datalab\": 20}}', "
                         f"not {value!r}") from None
    if isinstance(out, dict):
        return out
    raise ValueError(f"--workers must be a number or a JSON object, not {value!r}")


def cmd_predict(args) -> int:
    """One document through one vendor. The mirror of `oeb score`: files in, JSON out."""
    from .harness import predict

    record = predict(args.provider, args.doc, read_json(args.schema), timeout=args.timeout,
                     **read_options(args.options, per_provider=False))
    json.dump(record, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 1 if record.get("error") else 0


def confirm_plan(plan) -> bool:
    """Show what the benchmark is about to do, and ask. `-y` / `--yes` skips this entirely.

    A benchmark run costs real money and takes hours, so the default is to say what it will
    cost before it starts. The plan goes to stderr with the rest of the progress, leaving
    stdout for the summary.

    NOBODY TO ASK IS NOT THE SAME AS NO. A pipe, a cron job or a CI step has no terminal, so
    it proceeds -- it was scripted, which is consent. Only an interactive session is asked.
    """
    from rich.console import Console

    Console(stderr=True, highlight=False).print(plan)
    if not sys.stdin.isatty():
        return True
    try:
        return input("  proceed? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _ids(path: pathlib.Path) -> frozenset[str]:
    """The doc ids a `--docs` file names, one per line."""
    if not path.is_file():
        raise FileNotFoundError(f"--docs {path}: no such file; it holds doc ids, one per line")
    return frozenset(path.read_text().split())


def cmd_benchmark(args) -> int:
    from . import benchmark, data

    p = benchmark.plan(args.providers, out=args.out,
                       dataset=args.dataset or data.DEFAULT_DATASET,
                       manifest=args.manifest, suites=args.suites,
                       limit=args.limit, sample=args.sample,
                       only=_ids(args.docs) if args.docs else None,
                       timeout=args.timeout,
                       workers=read_workers(args.workers),
                       score_workers=args.score_workers,
                       verdicts=args.verdicts,
                       score_only=args.score_only, retry_failed=args.retry_failed,
                       options=read_options(args.options))
    if args.yes:
        for line in benchmark.describe(p).splitlines():
            logging.getLogger(__package__).info("%s", line)
    elif not confirm_plan(benchmark._plan_view(p)):
        logging.getLogger(__package__).warning("nothing run")
        return 1
    json.dump(benchmark.execute(p), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_check(args) -> int:
    """`oeb check`: whether a dataset is one oeb can run. It checks, stopping at the first problem:

        1. the manifest, as `data.read_manifest` does: the columns doc_id, suite, doc_path, gt_path
           and schema are there; each doc_id is a str and a filename; each suite is a string, or
           null for "other"; each doc_path is a relative path inside the dataset to a file (one with
           an extension); each gt_path is a relative path inside the dataset named `<doc_id>.json`;
           each schema is a non-empty JSON object; doc ids are unique, ignoring case; and there is at
           least one row;
        2. every doc_path and gt_path exists as a file; and
        3. every gold reads as a JSON object (`field_matching.check`).

    It does not open an input (a PDF isn't checked to be a PDF), check a gold against its schema, or
    run the scorer. The problem goes to stderr, naming its row; a dataset that passes is described on
    stdout, as JSON, and exits 0."""
    from . import data
    from .scorers import field_matching

    _, docs = data.read_manifest(args.dataset or data.DEFAULT_DATASET, args.manifest)
    if not docs:
        raise ValueError(f"the manifest {args.manifest} names no documents")
    if missing := [str(f) for d in docs for f in (d.doc_path, d.gt_path) if not f.is_file()]:
        raise ValueError(f"{len(missing)} files the manifest names don't exist, e.g. {missing[:3]}")
    field_matching.check(docs)
    print(json.dumps({"ok": True, "manifest": args.manifest, "documents": len(docs),
                      "suites": len({d.suite for d in docs}), "inputs": len({d.doc_path for d in docs})}))
    return 0


def cmd_grade(args) -> int:
    """Grade a folder of predictions/<doc_id>.json, whatever made them, and print its summary. A
    document with no prediction fails: the folder is your submission, so a document left out is one
    your system didn't answer. `--suites` grades part of the dataset on purpose."""
    from . import benchmark, data

    out = pathlib.Path(args.run)
    if not (out / "predictions").is_dir():
        raise FileNotFoundError(f"{out} has no predictions/ folder; it holds predictions/<doc_id>.json, "
                                "one per document you predicted")
    said = json.loads((out / "settings.json").read_text()) if (out / "settings.json").is_file() else {}
    provider = said.get("provider") or out.name
    # Note: a run folder says what it was predicted on; graded against another dataset, its scores
    # would be overwritten with a grading against gold it never saw.
    dataset = args.dataset or said.get("dataset") or data.DEFAULT_DATASET
    if (was := said.get("dataset")) and data.dataset_id(dataset) != was:
        raise ValueError(f"{out} is a run on dataset {was}, not {data.dataset_id(dataset)}, and a run never "
                         f"mixes datasets; pass --dataset {was}, or grade a copy")
    manifest = args.manifest or said.get("manifest") or data.MANIFEST
    if args.suites and (out / "records").is_dir():
        logging.getLogger(__package__).warning(
            "%s: grading only %s replaces the run's summary with these suites'; grade without --suites "
            "for the whole run", out, ", ".join(args.suites))
    _, docs = data.read_manifest(dataset, manifest)
    summary = benchmark.grade(benchmark.Run(provider, {}, out), list(data.select(docs, args.suites)),
                              verdicts=args.verdicts, score_workers=args.score_workers)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oeb", description="Benchmark document extraction: score one "
                                 "prediction, grade a folder of them, check a dataset, or predict and "
                                 "grade with each vendor.")
    ap.add_argument("-q", "--quiet", action="store_true", help="progress off; results only")
    # Note: also after the command (`oeb grade x -q`); SUPPRESS so a command's default can't undo it.
    quiet = argparse.ArgumentParser(add_help=False)
    quiet.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                       help="progress off; results only")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", parents=[quiet], help="score one prediction against one ground truth")
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

    b = sub.add_parser("benchmark", parents=[quiet],
                       help="fetch the corpus, run it through each vendor, score it")
    b.add_argument("--providers", nargs="+", required=True, metavar="NAME",
                   help="vendors and/or model ids, e.g. datalab reducto "
                        "openai/gpt-5.6-sol. `oeb providers` lists them")
    b.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs"),
                   help="where predictions, scores and the summary go. Default: runs/")
    # Note: a str, not a Path, because a Path folds the '//' out of an hf:// address.
    b.add_argument("--dataset", default=None,
                   help="the dataset: a folder on disk, or hf://datasets/<org>/<name>[@<revision>], "
                        "which is fetched into the HuggingFace cache. Default: ours, "
                        "hf://datasets/datalab-to/omni_extract_bench")
    b.add_argument("--manifest", default="manifest.parquet",
                   help="which documents of the dataset to run: a parquet inside it, e.g. "
                        "manifests/sample40-predictive-across-vendors-2026-09-22.parquet. Columns: "
                        "doc_id, suite, doc_path, gt_path, and a schema column holding JSON; "
                        "paths resolve against the dataset's folder. Default: manifest.parquet")
    b.add_argument("--suites", nargs="+", help="limit to these suites")
    b.add_argument("--limit", type=int, default=0, help="first N documents; for a smoke test")
    b.add_argument("--sample", type=int, default=0, metavar="N",
                   help="N documents spread over every suite, the same ones each time; a smoke test "
                        "that sees every kind of document")
    b.add_argument("--docs", type=pathlib.Path, metavar="FILE",
                   help="a file of doc ids, one per line: run only these documents")
    b.add_argument("--timeout", type=float, default=1800,
                   help="seconds one document may take, the same for every vendor")
    b.add_argument("--workers", metavar="N|JSON",
                   help="documents in flight per adapter, shared by its runs: N for every adapter, or "
                        'a number per adapter, e.g. \'{"datalab": 40, "llm_single_shot": 3}\' '
                        "(llm_single_shot: every OpenRouter model id; default: a table per vendor)")
    b.add_argument("--score-workers", type=int, default=0, metavar="N",
                   help="processes used to grade. Scoring is the CPU-bound half and is "
                        "independent per document. Default: one per core, capped at 8; "
                        "1 grades in this process")
    b.add_argument("--no-verdicts", dest="verdicts", action="store_false",
                   help="skip the per-address verdicts. They are why a number can be argued "
                        "with rather than only quoted, so they are written by default -- but "
                        "they are the bulk of what a run writes, around five times the gold "
                        "they grade")
    b.add_argument("--options", metavar="JSON",
                   help='per-provider options, as JSON or a path to a JSON file: '
                        '\'{"datalab": {"mode": "accurate"}}\'. A LIST runs that provider '
                        'once per entry, so \'{"datalab": [{"mode": "balanced"}, '
                        '{"mode": "accurate"}]}\' compares its tiers in one run. Each is '
                        "recorded in the run's settings.json and on every document, and "
                        "digested into the directory name so two of them cannot mix")
    b.add_argument("--retry-failed", action="store_true",
                   help="predict again documents whose recorded failure may not recur: a timeout, an "
                        "outage, a 429 or 5xx. A 400 or a refusal is an answer, and is not retried")
    b.add_argument("--score-only", action="store_true",
                   help="score the predictions already on disk; call no vendor")
    b.add_argument("-y", "--yes", action="store_true",
                   help="do not show the plan and wait for approval before spending. A run "
                        "with no terminal to ask -- a pipe, a cron job -- never waits anyway")
    b.set_defaults(fn=cmd_benchmark)

    g = sub.add_parser("grade", parents=[quiet], help="grade a folder of predictions/<doc_id>.json: yours, or a run's")
    g.add_argument("run", help="the folder holding predictions/")
    g.add_argument("--dataset", default=None,
                   help="the dataset: a folder on disk, or hf://datasets/<org>/<name>[@<revision>]. "
                        "Default: the one the folder's settings.json names, else ours")
    g.add_argument("--manifest", default=None,
                   help="which documents of the dataset: a parquet inside it. Default: the one the "
                        "folder's settings.json names, else manifest.parquet")
    g.add_argument("--suites", nargs="+",
                   help="grade only these suites; otherwise every document, and one with no prediction fails")
    g.add_argument("--score-workers", type=int, default=0, metavar="N",
                   help="processes used to grade. Default: one per core, capped at 8; 1 grades in this process")
    g.add_argument("--no-verdicts", dest="verdicts", action="store_false",
                   help="skip the per-address verdicts")
    g.set_defaults(fn=cmd_grade)

    c = sub.add_parser("check", parents=[quiet], help="whether a dataset is one oeb can run: its manifest, files and gold")
    c.add_argument("--dataset", default=None,
                   help="the dataset: a folder on disk, or hf://datasets/<org>/<name>[@<revision>]. "
                        "Default: ours")
    c.add_argument("--manifest", default="manifest.parquet",
                   help="which documents of the dataset: a parquet inside it. Default: manifest.parquet")
    c.set_defaults(fn=cmd_check)

    d = sub.add_parser("predict", parents=[quiet], help="run one document through one vendor")
    d.add_argument("--provider", required=True,
                   help="a vendor or a model id; `oeb providers` lists them")
    d.add_argument("--doc", required=True, help="the document, PDF")
    d.add_argument("--schema", required=True, help="the JSON Schema to extract against")
    d.add_argument("--timeout", type=float, default=1800,
                   help="seconds this document may take, end to end")
    d.add_argument("--options", metavar="JSON",
                   help='options for this provider, as JSON or a path: \'{"mode": "accurate"}\'')
    d.set_defaults(fn=cmd_predict)

    pl = sub.add_parser("providers", parents=[quiet],
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
    except (ValueError, FileNotFoundError) as exc:   # Note: a ManifestError is a ValueError.
        print(f"oeb {args.cmd}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
