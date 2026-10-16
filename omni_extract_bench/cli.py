"""The `oeb` command: a thin wrapper over `score`, `predict` and `benchmark`.

    oeb score      --pred p.json --gold g.json --schema s.json [--verdicts]
    oeb benchmark  --providers datalab reducto --out runs/
    oeb grade      mine/ [--dataset ...] [--manifest ...]
    oeb check      [--dataset ...] [--manifest ...]       # is the dataset one oeb can run?
    oeb predict    --provider datalab --doc x.pdf --schema s.json
    oeb providers [datalab]

Results go to stdout as JSON; progress and messages go to stderr. `benchmark` shows its plan and waits
for a yes unless given `-y`. A `UsageError` (what was asked is wrong) or a `Stop` (an account, a key or a
dependency) is printed as `oeb <command>: <what was wrong>` and exits 1, after a `benchmark` stopped by a
`Stop` prints the summaries it wrote; an interrupt exits 130; anything else, a scorer that raises among
them, is a bug of ours, and is raised as it is, with no scores or summary written.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

from . import errors

EXAMPLE_MODELS = ("openai/gpt-5.6-sol", "anthropic/claude-opus-5", "google/gemini-3.7-flash")


def read_object(path: str, flag: str) -> dict:
    """The JSON object in the file `path` that `flag` names."""
    try:
        got = json.loads(pathlib.Path(path).read_text())
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
            raise errors.UsageError(f"--options {value!r} is neither inline JSON nor a file that exists; "
                                    "pass a JSON object, or the path of a file holding one") from None
        try:
            parsed = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise errors.UsageError(f"--options: {path} isn't JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise errors.UsageError(f"--options must be a JSON object, not {value!r:.80}")
    def ok(value):
        return (isinstance(value, dict)
                or (isinstance(value, list) and value
                    and all(isinstance(x, dict) for x in value)))

    if per_provider and not all(ok(v) for v in parsed.values()):
        raise errors.UsageError('--options maps a provider to its options, or to a LIST of them to '
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
        raise errors.UsageError(f"--workers is a number, or a JSON object like '{{\"datalab\": 20}}', "
                                f"not {value!r}") from None
    if isinstance(out, dict):
        return out
    raise errors.UsageError(f"--workers must be a number or a JSON object, not {value!r}")


def read_ids(path: pathlib.Path, flag: str) -> frozenset[str]:
    """A file of ids, one per line; blank lines and `#` comments are skipped."""
    if not path.is_file():
        raise errors.UsageError(f"{flag} {path}: no such file; it holds ids, one per line")
    return frozenset(x for line in path.read_text().splitlines() if (x := line.split("#", 1)[0].strip()))


def confirm_plan(plan) -> bool:
    """Show what the benchmark is about to do, and ask. `-y` / `--yes` skips this entirely.

    A benchmark run costs real money and takes hours, so the default is to say what it will
    cost before it starts. The plan goes to stderr with the rest of the progress, leaving
    stdout for the summary.

    NOBODY TO ASK IS NOT THE SAME AS NO. A pipe, a cron job or a CI step has no terminal, so
    it proceeds -- it was scripted, which is consent. Only an interactive session is asked.
    """
    from rich import console

    console.Console(stderr=True, highlight=False).print(plan)
    if not sys.stdin.isatty():
        return True
    try:
        return input("  proceed? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _say(m: str) -> None:
    """A grading's closing line, or advice: to stderr, beside the progress."""
    print(m, file=sys.stderr)


def _written(p) -> dict[str, dict]:
    """Each run's summary.json, by run name, as grading wrote it."""
    return {r.out.name: json.loads((r.out / "summary.json").read_text())
            for r in p.runs if (r.out / "summary.json").is_file()}


def _retryable(p) -> int:
    """The documents whose failure may not recur, which `--retry-failed` would predict again."""
    from . import benchmark

    return sum(benchmark.needs(r, d, True) and not benchmark.needs(r, d) for r in p.runs for d in p.docs)


def cmd_score(args) -> int:
    from .scorers import field_matching

    pred, gold, schema = (read_object(path, flag) for flag, path in (
        ("--pred", args.pred), ("--gold", args.gold), ("--schema", args.schema)))
    # Note: a bare --order-matters is every array, as the library's `order_matters=True`.
    order = True if args.order_matters == [] else tuple(args.order_matters or ())
    result = field_matching.score(pred, gold, schema, order_matters=order, verdicts=args.verdicts)
    if args.verdicts:
        result["verdicts"] = [v._asdict() for v in result["verdicts"]]
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


def cmd_predict(args) -> int:
    """One document through one vendor. The mirror of `oeb score`: files in, JSON out."""
    from . import harness

    record = harness.predict(args.provider, args.doc, read_object(args.schema, "--schema"),
                             timeout=args.timeout, **read_options(args.options, per_provider=False))
    json.dump(record, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 1 if record.get("error") else 0


def cmd_benchmark(args) -> int:
    from . import benchmark, data, progress

    p = benchmark.plan(args.providers, out=args.out,
                       dataset=args.dataset or data.DEFAULT_DATASET,
                       manifest=args.manifest, suites=args.suites,
                       limit=args.limit, sample=args.sample,
                       only=read_ids(args.docs, "--docs") if args.docs else None,
                       timeout=args.timeout,
                       workers=read_workers(args.workers),
                       score_workers=args.score_workers,
                       verdicts=args.verdicts,
                       score_only=args.score_only, retry_failed=args.retry_failed,
                       options=read_options(args.options))
    # Note: nothing to ask before grading alone, which spends nothing.
    if args.yes or args.score_only:
        print(benchmark.describe(p), file=sys.stderr)
    elif not confirm_plan(progress.plan_view(p)):
        print("oeb benchmark: nothing run", file=sys.stderr)
        return 1
    try:
        summaries = benchmark.execute(p, _say)
    except errors.Stop:
        # Note: `execute` grades what was predicted before it raises, so every run has its summary.
        json.dump(_written(p), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        raise
    json.dump(summaries, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    if n := _retryable(p):
        _say(f"{errors.counted(n, 'document')} failed in a way that may not recur: --retry-failed predicts "
             f"{'it' if n == 1 else 'them'} again")
    return 0


def cmd_grade(args) -> int:
    """Grade a folder of predictions/<doc_id>.json, whatever made them, against its whole manifest, and
    print its summary. The folder's settings.json says its dataset and manifest; a folder of your own
    gets them there once its first grading succeeds."""
    from . import benchmark, data

    out = pathlib.Path(args.run)
    benchmark.gradable(out)   # Note: first, so a mistyped folder is named before a dataset is read.
    said = benchmark.settled(out, args.dataset, args.manifest)
    dataset = args.dataset or said.get("dataset") or data.DEFAULT_DATASET
    manifest = args.manifest or said.get("manifest") or data.MANIFEST
    _, docs = data.read_manifest(dataset, manifest)
    run = benchmark.Run(said.get("provider") or out.name, {}, out)
    summary = benchmark.grade(run, docs, _say, settings=said, verdicts=args.verdicts,
                              score_workers=args.score_workers)
    if not benchmark.is_run(said):
        # Note: only once grading succeeded, so a failed one leaves the folder as it was.
        benchmark.declare(out, dataset, manifest, said)
    json.dump(summary, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


def cmd_check(args) -> int:
    """`oeb check`: whether a dataset is one oeb can run, stopping at the first problem: its manifest
    (`data.read_manifest`), every file the manifest names on disk, and every gold (`field_matching.check`).
    It opens no input and runs no scorer; docs/API.md lists every rule."""
    from . import data
    from .scorers import field_matching

    _, docs = data.read_manifest(args.dataset or data.DEFAULT_DATASET, args.manifest)
    missing = [f"{d.doc_id} {col} {f}" for d in docs for col, f in (("doc_path", d.doc_path), ("gt_path", d.gt_path))
               if not f.is_file()]
    if missing:
        raise errors.UsageError(f"files the manifest names that don't exist: "
                                f"{errors.counted(len(missing), 'file')}, e.g. "
                                + "; ".join(missing[:3]))
    field_matching.check(docs)
    print(json.dumps({"ok": True, "manifest": args.manifest, "documents": len(docs),
                      "suites": len({d.suite for d in docs}), "inputs": len({d.doc_path for d in docs})}))
    return 0


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

    from rich import box, console, table

    from . import harness

    out = console.Console()

    if not args.provider:
        if not out.is_terminal:
            for provider in (*harness.PROVIDERS, *EXAMPLE_MODELS):
                print(provider)
            return 0
        sheet = table.Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
        sheet.add_column("provider")
        for provider in harness.PROVIDERS:
            sheet.add_row(provider)
        sheet.add_section()
        for model in EXAMPLE_MODELS:
            sheet.add_row(model)
        out.print(sheet)
        print("\nthe three model ids are examples: any OpenRouter org/model id works.")
        return 0

    config = harness.registry.config_for(args.provider)
    fields = dataclasses.fields(config)
    if not fields:
        out.print(f"{args.provider}: no options -- it takes the document and the schema "
                  f"and nothing else")
        return 0

    sheet = table.Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False,
                        title=args.provider, title_justify="left")
    sheet.add_column("option")
    sheet.add_column("default", overflow="fold")
    for field in fields:
        value = getattr(config, field.name)
        sheet.add_row(field.name, "''" if value == "" else str(value))
    out.print(sheet)
    # Note: a model id's `model` is its name, which --options refuses, so the example is another field.
    if example := next((f.name for f in fields if f.name != "model"), None):
        print(f"\noeb benchmark --providers {args.provider} "
              f"--options '{{\"{args.provider}\": {{\"{example}\": ...}}}}'")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oeb", description="Benchmark document extraction: score one "
                                 "prediction, grade a folder of them, check a dataset, or predict and "
                                 "grade with each vendor.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score one prediction against one ground truth")
    s.add_argument("--pred", required=True, help="the prediction, JSON")
    s.add_argument("--gold", required=True, help="the ground truth, JSON")
    s.add_argument("--schema", required=True,
                   help="the JSON Schema the prediction was generated against")
    s.add_argument("--verdicts", action="store_true",
                   help="include one verdict per address: what happened there, and what each "
                        "side was compared as")
    s.add_argument("--order-matters", nargs="*", metavar="ARRAY",
                   help="arrays whose order is part of the answer, e.g. steps, "
                        "'books[*].chapters'; given bare, every array. Default: none -- the order "
                        "rows appear in a document is usually an accident of layout")
    s.set_defaults(fn=cmd_score)

    b = sub.add_parser("benchmark",
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
                   help="a file of doc ids, one per line (blank lines and # comments skipped): run "
                        "only these documents")
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

    g = sub.add_parser("grade", help="grade a folder of predictions/<doc_id>.json: yours, or a run's")
    g.add_argument("run", help="the folder holding predictions/")
    g.add_argument("--dataset", default=None,
                   help="the dataset: a folder on disk, or hf://datasets/<org>/<name>[@<revision>]. "
                        "Default: the one the folder's settings.json names, else ours")
    g.add_argument("--manifest", default=None,
                   help="which documents of the dataset: a parquet inside it. Default: the one the "
                        "folder's settings.json names, else manifest.parquet")
    g.add_argument("--score-workers", type=int, default=0, metavar="N",
                   help="processes used to grade. Default: one per core, capped at 8; 1 grades in this "
                        "process")
    g.add_argument("--no-verdicts", dest="verdicts", action="store_false",
                   help="skip the per-address verdicts")
    g.set_defaults(fn=cmd_grade)

    c = sub.add_parser("check", help="whether a dataset is one oeb can run: its manifest, files and gold")
    c.add_argument("--dataset", default=None,
                   help="the dataset: a folder on disk, or hf://datasets/<org>/<name>[@<revision>]. "
                        "Default: ours")
    c.add_argument("--manifest", default="manifest.parquet",
                   help="which documents of the dataset: a parquet inside it. Default: manifest.parquet")
    c.set_defaults(fn=cmd_check)

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
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().handlers[:] = [handler]
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger(__package__).setLevel(logging.INFO)

    resume = args.cmd == "benchmark"
    try:
        return args.fn(args)
    except errors.UsageError as exc:
        print(f"oeb {args.cmd}: {exc}", file=sys.stderr)
        return 1
    except errors.Stop as exc:
        print(f"oeb {args.cmd}: {type(exc).__name__}: {exc}"
              + ("; fix it and rerun to resume" if resume else ""), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"oeb {args.cmd}: interrupted" + ("; rerun to resume" if resume else ""), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
