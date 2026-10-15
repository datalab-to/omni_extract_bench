#!/usr/bin/env python3
"""The whole benchmark: fetch the corpus, predict with each vendor, score, write it down.

    oeb benchmark --providers datalab reducto --out runs/
    oeb benchmark --providers datalab --limit 5              # smoke test

Two levels of data, each a NamedTuple, and functions over them:

    Plan   the invocation      every Run, and the documents they cover; `plan(...)` builds it
    Run    one configuration   a provider plus its options, and the directory it writes to

`execute(plan)` is `predict_all`, then `grade_all`. `describe(plan)` says what it would do, with
no vendor call. Concurrency is per adapter, not per run: every run of one adapter (two datalab
modes, every OpenRouter model) shares one pool of `workers_for` its adapter.

Resumable and idempotent: a document is predicted again only when it has no record, graded
again only when its line is missing or stale: each line says which prediction, which gold and which
scoring code it graded, and a line whose prediction, gold or scorer has changed is graded again
(`current`), which calls no vendor. So a finished grading's lines are one scorer's against one gold.
`--rescore` grades every document again anyway.

Credentials come from the environment and everything else from `--options`, so
`run_manifest.settings` is a complete account of what each vendor was asked.

    pip install 'omni-extract-bench[benchmark,harness]'
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import threading
import warnings
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, NamedTuple

from . import data, score, scorer
from rich import box
from rich.console import Console, Group
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .progress import NULL, Progress
from .harness import (AccountFailure, MissingCredential, MissingDependency,
                      predict)


log = logging.getLogger(__name__)

PLAN_WIDTH = 120


def plural(n: int, word: str) -> str:
    """`1 document`, `2 documents`. A plan that says "1 documents" reads as a bug."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# The code a verdict depends on: a change to any of it is a new scorer.
SCORING = ("scorer.py", "matching.py", "values.py", "recognise.py")


def scorer_version() -> str:
    """The package's version and a digest of its scoring code, stamped on each score row so a run
    graded by another scorer is known, and graded again (`current`).

    Note: the version alone is the same across every commit between releases, so it can't say the
    scoring changed; the digest of the files that decide a verdict can. The version is kept for the
    reader: which release graded a row.
    """
    try:
        version = importlib.metadata.version("omni-extract-bench")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    h = hashlib.sha256()
    for name in SCORING:
        h.update((Path(__file__).parent / name).read_bytes())
    return f"{version}+{h.hexdigest()[:12]}"


SCORER = scorer_version()


def scoring(stamp: str | None) -> str | None:
    """The scoring code a scorer stamp names: its digest, without the version, so a release that
    changes no scoring file grades as the one before it did."""
    return None if stamp is None else stamp.rpartition("+")[2]




def write_json_atomic(path: Path, obj, *, indent: int | None = None) -> None:
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(obj, default=str, indent=indent))
    os.replace(tmp, path)


class Run(NamedTuple):
    """One configuration and its folder: an address, as a `Path` is. What a run holds is whatever
    its folder holds when it is read, so it is resumed, graded or shared between processes as it
    stands. The functions below are the bookkeeping of any loop over its documents:

        prepare(run, corpus, timeout)          # once: the folder, and settings.json
        if needs(run, doc):                    # no readable record yet
            record(run, doc.doc_id, harness.predict(...))
        grade(run, docs)                       # scores.jsonl and summary.json; the summary
    """

    provider: str
    options: dict
    out: Path

    @property
    def predictions(self) -> Path:
        """The bare extractions. What scoring reads, and all it reads."""
        return self.out / "predictions"

    @property
    def records(self) -> Path:
        """The schema sent, the cost, the manifest. What an audit reads."""
        return self.out / "records"


def settings(run: Run) -> dict:
    """The whole resolved configuration, not just what the caller passed."""
    from .harness.registry import settings_for

    return settings_for(run.provider, run.options)


def head(run: Run) -> dict:
    """What every file a run writes says about itself, before any numbers."""
    return {"run": run.out.name, "provider": run.provider, "settings": settings(run)}


def prepare(run: Run, dataset: str, timeout: float, commit: str | None = None,
            manifest: str = data.MANIFEST) -> None:
    """The run's folder made, and what it asks written to its settings.json before it asks:
    provider, settings, dataset (and the commit of a HuggingFace one), manifest and the timeout
    per document. `summary.json` comes after grading, which an interrupted run never reaches."""
    run.predictions.mkdir(parents=True, exist_ok=True)
    run.records.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run.out / "settings.json", {
        **head(run), "dataset": data.dataset_id(dataset), **({"commit": commit} if commit else {}),
        "manifest": manifest, "timeout_s": timeout}, indent=2)


def recorded(run: Run, doc_id: str) -> dict | None:
    """What the run has recorded for the document, or None when nothing (or nothing readable)."""
    path = run.records / f"{doc_id}.json"
    try:
        got = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("%s isn't readable (%s); its document counts as not predicted", path, exc)
        return None
    return got if isinstance(got, dict) else None


def needs(run: Run, doc: data.Doc) -> bool:
    """True when the run has no readable record of the document: it was never attempted."""
    return recorded(run, doc.doc_id) is None


def record(run: Run, doc_id: str, result: dict) -> None:
    """One answer written down: its prediction, if it has one, then the record. predictions/ holds
    only answers; a failure is described in the record alone. The record is the marker `needs`
    reads, so a kill between the two predicts the document again rather than skipping it."""
    if result["error"] is None:
        write_json_atomic(run.predictions / f"{doc_id}.json", result["result"])
    write_json_atomic(run.records / f"{doc_id}.json", result)


def owed(run: Run, docs: list[data.Doc], verdicts: bool = True, rescore: bool = False) -> tuple[int, int]:
    """How many of `docs` the run would predict, and then grade. The two are independent: every
    prediction can be on disk with every grade still owed."""
    _, wanted = grading_split(run.out, docs, verdicts=verdicts, rescore=rescore)
    return sum(needs(run, doc) for doc in docs), len(wanted)


def grade(run: Run, docs: list[data.Doc], verdicts: bool = True, workers: int = 0,
          rescore: bool = False, progress=NULL) -> dict:
    """Grade the run's predictions, resumable per document: one line each in scores.jsonl, and
    summary.json over every line the run has, written once grading finishes (an interrupted
    grading writes none). Returns the summary of `docs`' lines, as omni-parse-bench's `grade`
    returns a summary; the lines themselves are `read_scores(run.out)`."""
    known, wanted = grading_split(run.out, docs, verdicts=verdicts, rescore=rescore)
    if len(wanted) < len(docs):
        log.info("%s: %d of %d documents already scored, %d to grade",
                 run.out.name, len(docs) - len(wanted), len(docs), len(wanted))
    progress.start(len(wanted))

    one = functools.partial(score_one, run=run, verdicts=verdicts)
    graded: dict[str, dict] = {}

    def so_far() -> list[dict]:
        """Every row known, old and new, in corpus order. Merged over what is on disk, so a
        `--limit 2` run cannot rewrite the file with two lines."""
        merged = {**known, **graded}
        return sorted(merged.values(), key=lambda r: (r.get("suite", ""), r["doc_id"]))

    remaining = wanted
    try:
        if remaining and (workers or SCORE_WORKERS) > 1:
            try:
                with cf.ProcessPoolExecutor(max_workers=workers or SCORE_WORKERS) as pool:
                    futures = {pool.submit(one, doc): doc.doc_id for doc in remaining}
                    try:
                        for future in cf.as_completed(futures):
                            graded[futures[future]] = row = future.result()
                            progress.record(error=row["status"] != "scored")
                    except BaseException:
                        pool.shutdown(wait=True, cancel_futures=True)
                        raise
                remaining = []
            except BrokenProcessPool:
                if graded:
                    raise
                log.warning("scoring pool stopped before any document was graded, so "
                            "grading continues in this process. If that was a Ctrl-C, "
                            "press it again; if it was not, put the call under "
                            "`if __name__ == \"__main__\":` or pass score_workers=1.")
        for doc in remaining:
            graded[doc.doc_id] = row = one(doc)
            progress.record(error=row["status"] != "scored")
    finally:
        progress.finish()
        write_scores(run.out, so_far())
        if len(graded) < len(wanted):
            log.warning("scoring stopped after %d of %d documents; the rest are graded "
                        "on the next run", len(graded), len(wanted))
    write_json_atomic(run.out / "summary.json", {**head(run), "scorer": SCORER,
                                                 **scorer.summarize(so_far())}, indent=2)
    # Note: the return answers what this invocation asked for, `docs`; summary.json, every line.
    answered = {**known, **graded}
    return {**head(run), "scorer": SCORER,
            **scorer.summarize([answered[d.doc_id] for d in docs if d.doc_id in answered])}


def asked(run: Run) -> list[str]:
    """Every resolved setting the run sends the vendor, one string each."""
    return [f"{k}={v if v != '' else chr(39) * 2}" for k, v in sorted(settings(run).items())]


SCORE_WORKERS = min(8, os.cpu_count() or 1)


def score_one(doc: data.Doc, *, run: Run, verdicts: bool = False) -> dict:
    """Grade one prediction on disk and return its row. Module-level and picklable for the
    process pool, and it never raises: one bad document is an error row.

    A document fails when its record says the call failed, or when it has no prediction: a
    folder of predictions you bring is your submission, so a missing file counts against
    coverage rather than being left out of it."""
    row = {"doc_id": doc.doc_id, "suite": doc.suite, "provider": run.provider,
           "scorer": SCORER, "gold": gold_digest(doc), "prediction": prediction_digest(run.out, doc.doc_id)}
    if failed := (recorded(run, doc.doc_id) or {}).get("error"):
        return {**row, "status": "error", "error": f"{failed['type']}: {failed['message']}"}
    try:
        pred = json.loads((run.predictions / f"{doc.doc_id}.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {**row, "status": "error", "error": "no prediction on disk"}
    if not (isinstance(pred, dict) and pred):
        return {**row, "status": "error",
                "error": "empty prediction" if pred == {} else "not an object"}
    try:
        result = score(pred, scorer.gold(doc), doc.schema, verdicts=verdicts)
        if verdicts:
            (run.out / "verdicts").mkdir(parents=True, exist_ok=True)
            (run.out / "verdicts" / f"{doc.doc_id}.jsonl").write_text(
                "\n".join(json.dumps(v._asdict(), default=str)
                          for v in result.pop("verdicts")))
    except Exception as exc:                            # noqa: BLE001 -- the message is the row
        return {**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {**row, "status": "scored", **result}


def read_scores(out: Path, verdicts: bool = False) -> dict[str, dict]:
    """Scores already on disk for this provider, keyed by document."""
    rows: dict[str, dict] = {}
    path = out / "scores.jsonl"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            doc_id = row["doc_id"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if verdicts and not (out / "verdicts" / f"{doc_id}.jsonl").exists():
            continue
        rows[doc_id] = row
    return rows


def grading_split(out: Path, docs: list[data.Doc], verdicts: bool = False,
                  rescore: bool = False) -> tuple[dict[str, dict], list[data.Doc]]:
    """The lines this grading may keep, and the documents still to grade. One definition, so
    what `describe` promises and what `grade` then does cannot drift. A selected document's line is
    kept while it is `current`; a line of a document not selected is kept while its scorer is this
    one, so a run's lines are never two scorers'. `rescore` keeps none."""
    if rescore:
        return {}, list(docs)
    now, by = scoring(SCORER), {d.doc_id: d for d in docs}
    known = {i: r for i, r in read_scores(out, verdicts=verdicts).items()
             if (current(out, r, by[i]) if i in by else scoring(r.get("scorer")) == now)}
    return known, [d for d in docs if d.doc_id not in known]


def current(out: Path, line: dict, doc: data.Doc) -> bool:
    """Whether a document's line still stands: it graded the prediction on disk now
    (`prediction_digest`), against the document's gold as it is (`gold_digest`), with this scoring
    code (`scoring`). Otherwise the document is graded again, which calls no vendor."""
    return (scoring(line.get("scorer")) == scoring(SCORER) and line.get("gold") == gold_digest(doc)
            and line.get("prediction") == prediction_digest(out, doc.doc_id))


def prediction_digest(out: Path, doc_id: str) -> str | None:
    """Which prediction a line grades: a digest of its file in `out/predictions`, None when it has none."""
    try:
        return hashlib.sha256((out / "predictions" / f"{doc_id}.json").read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def gold_digest(doc: data.Doc) -> str | None:
    """Which gold a document is graded against: a digest of what grading reads, its id, suite and
    schema and its gold file; None when the gold can't be read, which its row then says."""
    try:
        gold = doc.gt_path.read_bytes()
    except OSError:
        return None
    h = hashlib.sha256(json.dumps([doc.doc_id, doc.suite, doc.schema], sort_keys=True).encode())
    h.update(gold)
    return h.hexdigest()[:16]


def write_scores(out: Path, rows: list[dict]) -> None:
    """One line per document, written wherever scoring stops. Atomic because it is read back: a
    truncated file would silently lose everything past the cut."""
    path = out / "scores.jsonl"
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))
    os.replace(tmp, path)


#: Documents in flight per adapter, keyed by adapter module so every OpenRouter model shares
#: one budget. What each vendor tolerated in practice, not a published limit.
WORKERS = {
    "datalab": 10,
    "reducto": 3,
    "llamaextract": 3,
    "azure_cu": 3,
    "llm_single_shot": 5,
    "mistral": 5,
    "extend": 5,
}

DEFAULT_WORKERS = 5


def workers_for(providers, requested: dict[str, int] | int | None) -> int:
    """How many documents to have in flight at one vendor."""
    from .harness.registry import resolve

    names = [providers] if isinstance(providers, str) else list(providers)
    if isinstance(requested, int):
        requested = {"*": requested} if requested else {}
    requested = requested or {}
    asked = [requested[n] for n in names if requested.get(n)]
    return (min(asked) if asked else
            requested.get("*") or WORKERS.get(resolve(names[0]), DEFAULT_WORKERS))


def configurations(providers: list[str], options: dict | None = None) -> list[tuple[str, dict]]:
    """Each (provider, options) pair to run. `--options` may give one provider a LIST, and each
    entry is a run of its own."""
    unknown = sorted(set(options or {}) - set(providers))
    if unknown:
        raise ValueError(
            f"--options names {unknown[0]!r}, which is not in --providers "
            f"({', '.join(providers)}). Its options would be silently ignored.")
    out = []
    for provider in providers:
        asked = (options or {}).get(provider)
        if isinstance(asked, list) and not asked:
            raise ValueError(f"--options gives {provider!r} an empty list, so it would not "
                             f"run at all. Give it options, or leave it out.")
        out += [(provider, one or {}) for one in (asked if isinstance(asked, list) else [asked])]
    return out


class Plan(NamedTuple):
    """One invocation: every run, the documents they cover, and how. `plan(...)` builds it, and
    `describe(p)` says what `execute(p)` would do, with no vendor call."""

    runs: tuple[Run, ...]
    docs: tuple[data.Doc, ...]
    dataset: str
    manifest: str
    root: Path                             # the dataset's folder on disk, `fetch`
    timeout: float
    workers: dict[str, int] | int | None   # documents in flight per adapter, `workers_for`
    score_workers: int
    verdicts: bool
    rescore: bool
    score_only: bool


def plan(providers: list[str], *, out: Path = Path("runs"), dataset: str | Path = data.DEFAULT_DATASET,
         manifest: str = data.MANIFEST, suites: list[str] | None = None, limit: int = 0,
         timeout: float = 1800.0, predict_workers: dict[str, int] | int | None = None,
         score_workers: int = 0, verdicts: bool = True, rescore: bool = False,
         score_only: bool = False, options: dict | None = None) -> Plan:
    """Every run this invocation makes and the documents it covers, checked before any call: a
    bad provider, option or `--out` fails here. Runs are deduplicated by folder."""
    from .harness.registry import out_name, resolve

    for provider in providers:
        resolve(provider)
    runs = {}
    for provider, options_ in configurations(providers, options):
        name = out_name(provider, options_)
        runs.setdefault(name, Run(provider, options_, Path(out) / name))
    ident = data.dataset_id(dataset)
    for run in runs.values():
        # Note: a folder is named for provider and options, not dataset, so two datasets would
        # share one. Two manifests of one dataset may: they are the same documents and gold.
        path = run.out / "settings.json"
        if path.exists() and (was := json.loads(path.read_text()).get("dataset")) not in (None, ident):
            raise ValueError(f"{run.out} is a run on dataset {was}, not {ident}, and a run never "
                             f"mixes datasets; pass another --out, or --dataset {was}")
    root, every = data.read_manifest(dataset, manifest)
    if not (docs := data.select(every, suites, limit)):
        raise ValueError("no documents selected: check --suites and --limit")
    return Plan(tuple(runs.values()), tuple(docs), str(dataset), manifest, root, timeout,
                predict_workers, score_workers, verdicts, rescore, score_only)


def predict_all(p: Plan) -> None:
    """Predict every document each run has no record of. Every run of one adapter drains through
    that adapter's one pool (`workers_for`), the adapters go at once, and a live table shows each
    run. Predicting is network wait, so the adapters do not slow each other."""
    from .harness.registry import resolve

    # Note: set on an interrupt, so no run starts another document; a call already made is let
    # finish, and lands.
    stop = threading.Event()
    by: dict[str, list[Run]] = {}
    for run in p.runs:
        prepare(run, p.dataset, p.timeout, None if data.hf_dataset(p.dataset) is None else p.root.name,
                p.manifest)
        by.setdefault(resolve(run.provider), []).append(run)

    def drain(adapter: str, runs: list[Run], bars: Progress) -> None:
        """Every document of every run of one adapter, through its one pool."""
        workers = workers_for([r.provider for r in runs], p.workers)
        bar = {r.out.name: bars.reporter(r.out.name) for r in runs}
        left, work = {}, []
        for r in runs:
            todo = [d for d in p.docs if needs(r, d)]
            log.info("%s: %d documents, %d to run", r.out.name, len(p.docs), len(todo))
            bar[r.out.name].start(len(todo), workers)
            if not todo:
                bar[r.out.name].finish()
            left[r.out.name] = len(todo)
            work += [(r, d) for d in todo]
        halt = threading.Event()

        def one(r: Run, d: data.Doc) -> None:
            if stop.is_set() or halt.is_set():
                return
            if not (file := Path(d.doc_path)).exists():
                # Note: a document whose file is missing isn't the vendor's failure: nothing is
                # recorded, so it is neither graded as the vendor's nor kept from a resume.
                log.warning("%s: no file for %s at %s", r.out.name, d.doc_id, file)
                bar[r.out.name].record(error=True)
                return
            try:
                with bar[r.out.name].calling():
                    got = predict(r.provider, file, d.schema, timeout=p.timeout, **r.options)
                record(r, d.doc_id, got)
            except BaseException:
                # Note: set here, by the worker that failed, not when the main thread gets to
                # it: the other workers would otherwise send documents queued behind it.
                halt.set()
                raise
            cost = got.get("cost") or {}
            bar[r.out.name].record(error=bool(got.get("error")), usd=cost.get("usd"),
                                   credits=cost.get("credits"), wall_s=cost.get("wall_s"))

        ex = cf.ThreadPoolExecutor(workers)
        try:
            fs = {ex.submit(one, r, d): r.out.name for r, d in work}
            for f in cf.as_completed(fs):
                f.result()
                # Note: a run is finished when its own last document lands, not when the pool
                # it shares drains, so each run is timed for itself.
                left[name := fs[f]] -= 1
                if not left[name]:
                    bar[name].finish()
        except BaseException as exc:
            # Note: an account or credential failure is the same for every document and every
            # run of the adapter (one key), so all of them stop now and nothing queued is sent.
            ex.shutdown(wait=False, cancel_futures=True)
            if isinstance(exc, Exception):
                log.error("%s: stopped its runs (%s): %s: %s", adapter,
                          ", ".join(r.out.name for r in runs), type(exc).__name__,
                          str(exc)[:300])
            raise
        finally:
            ex.shutdown(wait=True)

    failures: list[Exception] = []
    with Progress([r.out.name for r in p.runs]) as bars:
        with cf.ThreadPoolExecutor(max(1, len(by))) as ex:
            fs = [ex.submit(drain, adapter, runs, bars) for adapter, runs in by.items()]
            try:
                for f in cf.as_completed(fs):
                    # Note: one adapter's failure (its key, its account) is its own, so the
                    # others go on; an interrupt stops them all.
                    try:
                        f.result()
                    except Exception as exc:  # noqa: BLE001
                        failures.append(exc)
            except BaseException:
                stop.set()
                log.warning("interrupted -- finishing the calls already in flight; "
                            "documents not yet written come back on the next run")
                raise
    if failures:
        raise failures[0]


def grade_all(p: Plan) -> dict[str, dict]:
    """Grade every run, serially, and publish each as soon as it is graded. Serial because
    grading saturates every core. Returns each run's summary over the selected documents
    (`grade`'s), by run name; its summary.json covers every document it has a line for.

    A SECOND PHASE, NOT A SECOND DISPLAY. Grading starts once every vendor has answered, so
    it gets a table of its own in the same shape -- with no cost and nothing in flight,
    because there is no vendor. Streaming it instead, so a run that finished predicting
    began grading, would put a process pool beside the live thread pools to save minutes at
    the end of a run that spends hours at the start of it.
    """
    summary = {}
    with Progress([r.out.name for r in p.runs]) as bars:
        for run in p.runs:
            summary[run.out.name] = s = grade(run, list(p.docs), verdicts=p.verdicts,
                                              workers=p.score_workers, rescore=p.rescore,
                                              progress=bars.reporter(run.out.name))
            log.info("%s: accuracy %.4f over %d suites, coverage %d/%d", run.out.name,
                     s["accuracy"], len(s["per_suite"]), s["scored"], s["documents"])
    return summary


def execute(p: Plan) -> dict[str, dict]:
    """Predict, then grade. Returns each run's summary, by run name."""
    if not p.score_only:
        predict_all(p)
    return grade_all(p)


def settings_cell(asked: list[str]) -> Group:
    """One setting per line, ruled off from the next. A `Rule` sizes itself to the cell, whose
    width nothing here knows in advance."""
    if not asked:
        return Group(Text("-", style="dim"))       # mistral: nothing to ask at all
    rows: list = []
    for n, setting in enumerate(asked):
        if n:
            rows.append(Rule(style="dim"))
        rows.append(Text(setting, style="cyan"))
    return Group(*rows)


def facts(p: Plan) -> Table:
    """What this invocation is, one labelled row each. A row rather than another clause in a
    sentence, so a new flag cannot lengthen a line that already ran to three of them."""
    from .harness.registry import resolve

    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column("", style="dim", no_wrap=True)
    table.add_column("", overflow="fold")
    table.add_row("out", str(p.runs[0].out.parent) if p.runs else "")
    table.add_row("runs", f"{len(p.runs)} over "
                          f"{plural(len({resolve(r.provider) for r in p.runs}), 'adapter')}")
    table.add_row("dataset", p.dataset)
    table.add_row("manifest", p.manifest)
    table.add_row("documents", f"{plural(len(p.docs), 'document')} selected")
    table.add_row("timeout", f"{p.timeout:.0f}s per document")
    # Shown even when false: `score only  false` is what says this will spend money.
    for name, on in (("score only", p.score_only), ("rescoring", p.rescore)):
        table.add_row(name, Text(str(on).lower(), style="bold" if on else "dim"))
    return table


def plan_view(p: Plan) -> Group:
    """The whole plan as one renderable: each run, its settings, and what it still owes.
    `describe` renders it as text for a log, the command line prints it in colour."""
    from .harness.registry import resolve

    by: dict[str, list[Run]] = {}
    for run in p.runs:
        by.setdefault(resolve(run.provider), []).append(run)
    # A table, so a cell too wide for the terminal wraps rather than being cut.
    table = Table(box=box.ROUNDED, header_style="dim", expand=False)
    table.add_column("adapter", overflow="fold")
    table.add_column("at once", justify="right", no_wrap=True)
    table.add_column("run", overflow="fold")
    table.add_column("settings", overflow="fold")
    table.add_column("predict", justify="right", no_wrap=True)
    table.add_column("grade", justify="right", no_wrap=True)
    for adapter, runs in by.items():
        workers = workers_for([r.provider for r in runs], p.workers)
        for n, run in enumerate(runs):
            to_predict, to_grade = owed(run, list(p.docs), verdicts=p.verdicts,
                                        rescore=p.rescore)
            to_predict = 0 if p.score_only else to_predict
            table.add_row(Text(adapter if n == 0 else "", style="bold"),
                          Text(str(workers) if n == 0 else ""),
                          Text(run.out.name), settings_cell(asked(run)),
                          Text(str(to_predict), style="bold" if to_predict else ""),
                          Text(str(to_grade), style="bold" if to_grade else ""),
                          end_section=n == len(runs) - 1)
    return Group(Text("benchmark", style="bold"), facts(p), Text(""), table)


def describe(p: Plan) -> str:
    """`plan_view` as plain text, for a log that has no colour and no width to ask about."""
    console = Console(file=io.StringIO(), width=PLAN_WIDTH, no_color=True, highlight=False)
    console.print(plan_view(p))
    return "\n".join(line.rstrip() for line in console.file.getvalue().splitlines())


def run(providers: list[str], **kwargs) -> dict:
    """Plan, log the plan, predict, grade. Keyword arguments are `plan`'s, and no argparse, so it
    stays callable from a notebook, and it never reads stdin."""
    p = plan(providers, **kwargs)
    for line in describe(p).splitlines():
        log.info("%s", line)
    return execute(p)


# Note: the names that moved to `data`, kept for one release. Each warns, naming its replacement.
MOVED = {name: f"data.{name}" for name in
         ("Doc", "COLUMNS", "REPO", "MANIFEST", "HF_PREFIX", "DEFAULT_DATASET", "hf_dataset", "dataset_id")}
MOVED |= {"summarise": "scorer.summarize", "over": "scorer._over", "ERRORS": "scorer.ERRORS"}


def __getattr__(name: str) -> Any:
    if name not in MOVED: raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _deprecated(f"benchmark.{name}", MOVED[name])
    module, _, attr = MOVED[name].partition(".")
    return getattr(data if module == "data" else scorer, attr)


def fetch(dataset: str | Path, manifest: str = data.MANIFEST) -> Path:
    """Deprecated: `data.download_dataset(dataset)`."""
    _deprecated("benchmark.fetch(dataset, manifest)", "data.download_dataset(dataset)")
    data._on_hub(dataset, manifest)
    return data.download_dataset(dataset)


def read_manifest(path: Path, root: Path, suites=None, limit: int = 0) -> list[data.Doc]:
    """Deprecated: `data.select(data.read_manifest(dataset, manifest)[1], suites, limit)`."""
    _deprecated("benchmark.read_manifest(path, root, ...)",
                "data.select(data.read_manifest(dataset, manifest)[1], suites, limit)")
    rel = Path(path).relative_to(root) if Path(path).is_absolute() == Path(root).is_absolute() else Path(path)
    return list(data.select(data.read_manifest(root, str(rel))[1], suites, limit))


def corpus(dataset: str | Path, manifest: str = data.MANIFEST, suites: list[str] | None = None,
           limit: int = 0) -> tuple[list[data.Doc], Path]:
    """Deprecated: `data.read_manifest(dataset, manifest)`, then `data.select`."""
    _deprecated("benchmark.corpus(...)", "data.read_manifest(dataset, manifest), then data.select")
    root, every = data.read_manifest(dataset, manifest)
    if not (docs := data.select(every, suites, limit)):
        raise ValueError("no documents selected: check --suites and --limit")
    return list(docs), root


def document(p: Plan, doc: data.Doc) -> Path:
    """Deprecated: `doc.doc_path`, a full local path once `data.read_manifest` has read it."""
    _deprecated("benchmark.document(plan, doc)", "doc.doc_path")
    return Path(doc.doc_path)


def _deprecated(old: str, new: str) -> None:
    warnings.warn(f"{old} is deprecated: use {new}. The old name will be removed in a later release.",
                  DeprecationWarning, stacklevel=3)
