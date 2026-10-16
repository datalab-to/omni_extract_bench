#!/usr/bin/env python3
"""The whole benchmark: fetch the corpus, predict with each vendor, score, write it down.

    oeb benchmark --providers datalab reducto --out runs/
    oeb benchmark --providers datalab --limit 5              # smoke test

Two levels of data, each a NamedTuple, and functions over them:

    Plan   the invocation      every Run, and the documents they cover; `plan(...)` builds it
    Run    one configuration   a provider plus its options, and the directory it writes to

`execute(plan)` is `predict_all`, then `grade_all`. `describe(plan)` says what it would do, with
no vendor call. Concurrency is per adapter, not per run: every run of one adapter (two datalab
modes, every OpenRouter model) shares one pool of `_workers_for` its adapter.

Resumable: a document is predicted again only when it has no record. Grading is fresh every time:
every document is graded again, which calls no vendor, and scores.jsonl and summary.json are written
whole when it finishes, so a run's scores are always one grading's. Reusing lines a grading already
wrote is an optimization for later.

Credentials come from the environment and everything else from `--options`, so
`run_manifest.settings` is a complete account of what each vendor was asked.

    pip install 'omni-extract-bench[benchmark,harness]'
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import io
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import threading
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import NamedTuple

from . import data
from .scorers import field_matching as scorer
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


def _plural(n: int, word: str) -> str:
    """`1 document`, `2 documents`. A plan that says "1 documents" reads as a bug."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# The code a verdict depends on: a change to any of it is a new scorer.
SCORING = ("scorers/field_matching/__init__.py", "scorers/field_matching/matching.py",
           "scorers/field_matching/values.py", "scorers/field_matching/recognise.py")


def scorer_version() -> str:
    """The package's version and a digest of its scoring code, stamped on each score line: which scorer
    graded it.

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


def _write_json_atomic(path: Path, obj) -> None:
    """Written whole, then moved into place, so a crash never leaves half a file."""
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(obj, default=str, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


class Run(NamedTuple):
    """One configuration and its folder: an address, as a `Path` is. What a run holds is whatever
    its folder holds when it is read, so it is resumed, graded or shared between processes as it
    stands. The functions below are the bookkeeping of any loop over its documents:

        prepare(run, dataset, timeout)         # once: the folder, and settings.json
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


def _settings(run: Run) -> dict:
    """The whole resolved configuration, not just what the caller passed."""
    from .harness.registry import settings_for

    return settings_for(run.provider, run.options)


def _head(run: Run) -> dict:
    """What every file a run writes says about itself, before any numbers. A folder of your own
    predictions may name a provider omni-extract-bench has no adapter for: its settings are then {}."""
    try:
        settings = _settings(run)
    except ValueError:
        settings = {}
    return {"run": run.out.name, "provider": run.provider, "settings": settings}


def prepare(run: Run, dataset: str, timeout: float, manifest: str = data.MANIFEST) -> None:
    """The run's folder made, and what it asks written to its settings.json before it asks:
    provider, settings, dataset (and the commit of a HuggingFace one, the downloaded folder's name),
    manifest and the timeout per document, whole. `summary.json` comes after grading, which an interrupted
    run never reaches."""
    root, _ = data.read_manifest(dataset, manifest)
    commit = root.name if data._hf_dataset(dataset) is not None else None
    run.predictions.mkdir(parents=True, exist_ok=True)
    run.records.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(run.out / "settings.json", {
        **_head(run), "dataset": data.dataset_id(dataset), **({"commit": commit} if commit else {}),
        "manifest": manifest, "timeout_s": timeout})


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


def needs(run: Run, doc: data.Doc, retry_failed: bool = False) -> bool:
    """True when the run has no readable record of the document (it was never attempted), or, with
    `retry_failed`, when its record is a failure that may not recur (`_retryable`: a timeout, an
    outage, a 429 or 5xx). A failure that is an answer (a 400, a refusal) is never retried."""
    if (got := recorded(run, doc.doc_id)) is None:
        return True
    return retry_failed and _retryable(got.get("error"))


def _retryable(error: dict | None) -> bool:
    """Whether a recorded failure may go the other way on another try, by omni-parse-bench's rule: a
    timeout, an outage (`VendorUnreachable`), or a transient HTTP status.

    Note: read off the type and status, not the record's `transient`, which says whether `predict`
    retried within the call: it never does for a timeout or a failed poll, since the first would spend a
    second budget and the second may buy the extraction twice."""
    from .harness.errors import TRANSIENT_STATUSES

    if not error: return False
    return (error.get("type") in ("VendorTimeout", "VendorUnreachable")
            or error.get("status") in TRANSIENT_STATUSES)


def record(run: Run, doc_id: str, result: dict) -> dict:
    """What `harness.predict` returned for the document, written down and returned, as the record it now
    has: its prediction, if it has one, then the record. predictions/ holds only answers; a failure is
    described in the record alone. The record is the marker `needs` reads, so a kill between the two
    predicts the document again rather than skipping it."""
    if result["error"] is None:
        _write_json_atomic(run.predictions / f"{doc_id}.json", result["result"])
    _write_json_atomic(run.records / f"{doc_id}.json", result)
    return result


def owed(p: Plan, run: Run) -> tuple[int, int]:
    """How many documents the run would predict, and then grade, by the rules `predict_all` and
    `grade_all` follow: it predicts the plan's documents it has no record of, and grades every
    document of the manifest that will have been attempted (the rest are pending)."""
    coming = set() if p.score_only else {d.doc_id for d in p.docs if needs(run, d, p.retry_failed)}
    _, every = data.read_manifest(p.dataset, p.manifest)
    keeps = not p.score_only or run.records.is_dir()
    to_grade = sum(not keeps or d.doc_id in coming or recorded(run, d.doc_id) is not None
                   or predicted(run, d.doc_id) is not None for d in every)
    return len(coming), to_grade


def grade(run: Run, docs: list[data.Doc], verdicts: bool = True, score_workers: int = 0,
          progress=NULL, *, among: Sequence[data.Doc] | None = None) -> dict:
    """Grade every document of `docs` (usually the whole manifest, as `grade_all` passes it), fresh, and
    write scores.jsonl (one line each, in their order), summary.json and, with `verdicts`, verdicts/
    (one file per scored document) whole once grading finishes: they are always one grading's, of
    `docs`. An interrupted grading writes nothing, so the last finished one's files stand; without
    `verdicts`, an old verdicts/ is removed. Returns the summary, as omni-parse-bench's `grade` does;
    the lines themselves are `read_scores(run)`. Files in predictions/ that match no document of
    the manifest (`among`, when `docs` is a part of it), and predictions that can't be read (graded
    as failed), are counted in a warning. A prediction is read only from a file named exactly
    `<doc_id>.json`, so a name that differs in case is a stray file on every system, macOS included.

    While any document is pending, the summary's `headline` (and that of each suite with pending
    documents) is null: a headline is a finished run's, and every other number is there as it stands.

    A document is scored when it has a prediction, and failed (an error line, counted against
    coverage) when its record says the call failed or it has no prediction. It is pending (no line,
    counted in the summary's `pending`) when it was never attempted: the run keeps records, as a run
    our loop predicts does, and has neither a record nor a prediction for it. A folder of
    predictions you bring keeps no records, so a document missing from it is failed: it is your
    submission. A folder with no predictions/ is refused: it is most likely another one (another
    option, another --out)."""
    if not run.predictions.is_dir():
        raise ValueError(f"no run at {run.out}: it has no predictions/ folder, which holds "
                         "predictions/<doc_id>.json, one per document predicted. Check the folder's path")
    if not docs:
        raise ValueError(f"no documents to grade {run.out} against: the manifest, or the suites chosen, "
                         "hold none, and grading would write an empty summary over its scores")
    partial = run.out / ".verdicts.partial"
    shutil.rmtree(partial, ignore_errors=True)   # Note: an interrupted grading's.
    keeps = run.records.is_dir()
    listed = frozenset(f.name for f in run.predictions.iterdir() if f.is_file())
    attempted = [d for d in docs if not keeps or recorded(run, d.doc_id) is not None
                 or f"{d.doc_id}.json" in listed]
    progress.start(len(attempted))
    one = functools.partial(_score_one, run=run, verdicts=partial if verdicts else None, listed=listed)
    graded: dict[str, dict] = {}
    remaining = list(attempted)
    try:
        if remaining and (score_workers or SCORE_WORKERS) > 1:
            try:
                with cf.ProcessPoolExecutor(max_workers=score_workers or SCORE_WORKERS) as pool:
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
    lines = [graded[d.doc_id] for d in attempted]
    shutil.rmtree(run.out / "verdicts", ignore_errors=True)
    if partial.is_dir(): partial.rename(run.out / "verdicts")
    _write_scores(run.out, lines)
    summary = _with_pending({**_head(run), "scorer": SCORER, **scorer.summarize(lines)},
                            [d for d in docs if d.doc_id not in graded])
    _write_json_atomic(run.out / "summary.json", summary)
    _warn_unusable(run, docs, among or docs, listed)
    log.info("%s", _closing(run.out.name, summary))
    return summary


def _with_pending(summary: dict, pending: list[data.Doc]) -> dict:
    """`summary` with the count of documents never attempted, overall and per suite, and a null
    headline wherever there are any."""
    by: dict[str, int] = collections.Counter(d.suite for d in pending)
    empty = {k: v for k, v in scorer.summarize([]).items() if k != "per_suite"}
    per = {s: summary["per_suite"].get(s, empty) | {"pending": by.get(s, 0)}
           for s in sorted({*summary["per_suite"], *by})}
    for s in by: per[s]["headline"] = None
    return summary | {"headline": None if pending else summary["headline"], "pending": len(pending),
                      "per_suite": per}


def _closing(name: str, s: dict) -> str:
    """A grading's one line: the headline, or why there is none yet, and coverage over every document."""
    total = s["documents"] + s["pending"]
    if s["headline"] is None:
        return (f"{name}: headline - ({s['pending']} pending; a headline needs every document), "
                f"coverage {s['scored']}/{total}")
    return f"{name}: headline {s['headline']:.3f}, coverage {s['scored']}/{total}"


def _warn_unusable(run: Run, docs: list[data.Doc], among: Sequence[data.Doc],
                   listed: frozenset[str]) -> None:
    """A warning for each kind of file in predictions/ that grading couldn't use: one named for no
    document of the manifest `among` (a misnamed file is otherwise only a lower coverage), a folder,
    and a prediction of `docs` that can't be read, which was graded as failed."""
    names = {f"{d.doc_id}.json" for d in among}
    stray = sorted(listed - names)
    stray += sorted(f"{f.name}/" for f in run.predictions.iterdir() if f.is_dir())
    if stray:
        log.warning("%s: %d files in predictions/ are named for no document of the manifest, e.g. %s; a "
                    "prediction is predictions/<doc_id>.json, named exactly (case included)",
                    run.out, len(stray), stray[:3])
    unread = [d.doc_id for d in docs if f"{d.doc_id}.json" in listed
              and recorded(run, d.doc_id) is None and predicted(run, d.doc_id) is None]
    if unread:
        log.warning("%s: %d predictions can't be read as a JSON object, and were graded as failed, e.g. %s",
                    run.out, len(unread), unread[:3])


def _asked(run: Run) -> list[str]:
    """Every resolved setting the run sends the vendor, one string each."""
    return [f"{k}={v if v != '' else chr(39) * 2}" for k, v in sorted(_settings(run).items())]


SCORE_WORKERS = min(8, os.cpu_count() or 1)


def _score_one(doc: data.Doc, *, run: Run, verdicts: Path | None = None,
               listed: frozenset[str] | None = None) -> dict:
    """Grade one prediction on disk and return its row. Module-level and picklable for the
    process pool, and it never raises: one bad document is an error row. Its verdicts are written
    into the folder `verdicts`, when given. `listed`, predictions/'s file names, when given, is what
    says whether the document has a prediction, by exact name.

    A document fails when its record says the call failed, or when it has no prediction: a
    folder of predictions you bring is your submission, so a missing file counts against
    coverage rather than being left out of it."""
    row = {"doc_id": doc.doc_id, "suite": doc.suite, "provider": run.provider,
           "scorer": SCORER}
    if failed := (recorded(run, doc.doc_id) or {}).get("error"):
        return {**row, "status": "error", "error": f"{failed['type']}: {failed['message']}"}
    if listed is not None and f"{doc.doc_id}.json" not in listed:
        return {**row, "status": "error", "error": f"no prediction: predictions/{doc.doc_id}.json"}
    if (pred := predicted(run, doc.doc_id)) is None:
        return {**row, "status": "error", "error": "the prediction isn't readable as a JSON object"}
    if not pred:
        return {**row, "status": "error", "error": "empty prediction"}
    try:
        result = scorer.score(pred, scorer.gold(doc), doc.schema, verdicts=verdicts is not None)
        if verdicts is not None:
            verdicts.mkdir(parents=True, exist_ok=True)
            (verdicts / f"{doc.doc_id}.jsonl").write_text(
                "\n".join(json.dumps(v._asdict(), default=str)
                          for v in result.pop("verdicts")))
    except Exception as exc:                            # noqa: BLE001 -- the message is the row
        return {**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {**row, "status": "scored", **result}


def predicted(run: Run, doc_id: str) -> dict | None:
    """The document's prediction (`predictions/`), the extracted JSON object; None when it has none,
    or none readable as an object. Quiet: `grade` says once how many it couldn't read."""
    try:
        got = json.loads((run.predictions / f"{doc_id}.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return got if isinstance(got, dict) else None


def read_scores(run: Run) -> list[dict]:
    """The run's score lines (scores.jsonl), in file order: one per graded document; [] for a run never
    graded, and FileNotFoundError for a folder that doesn't exist. A line that doesn't parse is left
    out, and logged."""
    if not run.out.is_dir():
        raise FileNotFoundError(f"no run at {run.out}")
    path = run.out / "scores.jsonl"
    try:
        text = path.read_text()
    except OSError:
        return []
    out = []
    for n, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("%s: line %d isn't readable JSON; left out", path, n + 1)
    return out


def _write_scores(out: Path, rows: list[dict]) -> None:
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


def _workers_for(adapter: str, requested: int | Mapping[str, int] = 0) -> int:
    """Documents in flight at once for every run of this adapter (`registry.resolve`) together: what
    was requested (one number for every adapter, or a number per adapter), else `WORKERS`."""
    if isinstance(requested, Mapping): requested = requested.get(adapter, 0)
    return requested or WORKERS.get(adapter, DEFAULT_WORKERS)


def _per_adapter(workers: Mapping[str, int]) -> dict[str, int]:
    """`--workers` as a number per adapter, checked. The limit is a vendor account's, shared by every
    run of its adapter (every OpenRouter model id shares llm_single_shot's), so it is set per adapter,
    never per provider or model."""
    from .harness import registry

    # Note: the adapters filed so far, which include every plugin a provider of this plan loaded;
    # resolving every provider would load every plugin, and a broken one would stop any run.
    adapters = {registry.resolve(n) for n in registry.ADAPTERS} | {"llm_single_shot"}
    for key, n in workers.items():
        if key not in adapters:
            try:
                a = registry.resolve(key)
                hint = f"{key} uses the {a} adapter, whose pool all its runs share; write {{\"{a}\": {n}}}"
            except ValueError:
                hint = f"the adapters are {sorted(adapters)}"
            raise ValueError(f"--workers is per adapter, and {key!r} isn't one: {hint}")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"--workers gives {key!r} {n!r}; documents at once is a whole number, 1 or more")
    return dict(workers)


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
    timeout: float
    workers: int | dict[str, int]          # documents in flight per adapter, `_workers_for`
    score_workers: int
    verdicts: bool
    score_only: bool
    retry_failed: bool = False             # predict again documents whose recorded failure was transient


def plan(providers: list[str], *, out: Path = Path("runs"), dataset: str | Path = data.DEFAULT_DATASET,
         manifest: str = data.MANIFEST, suites: list[str] | None = None, limit: int = 0,
         sample: int = 0, only: Iterable[str] | None = None,
         timeout: float = 1800.0, workers: int | Mapping[str, int] = 0,
         score_workers: int = 0, verdicts: bool = True,
         score_only: bool = False, retry_failed: bool = False, options: dict | None = None) -> Plan:
    """Every run this invocation makes and the documents it covers, checked before any call: a
    bad provider, option or `--out` fails here.

    `workers` is the documents in flight per adapter, shared by its runs: one number for every
    adapter, or a number per adapter (`_per_adapter`); 0, or an adapter left out, takes `WORKERS`."""
    from .harness.registry import out_name, resolve

    data._names(providers, "providers")
    data._names(suites, "suites")
    if isinstance(workers, bool) or not isinstance(workers, int | Mapping):
        raise TypeError(f"workers is {workers!r}; it is a number for every adapter, or a mapping of "
                        "adapter to number")
    counts = [("--limit", limit, "no limit"), ("--sample", sample, "no sample"),
              ("--score-workers", score_workers, "the default")]
    if isinstance(workers, int): counts.append(("--workers", workers, "the default per vendor"))
    for flag, n, zero in counts:
        if n < 0: raise ValueError(f"{flag} is {n}, but it counts, so it can't be negative; 0 means {zero}")
    if not timeout > 0:
        raise ValueError(f"--timeout is {timeout}, but it is the seconds a document may take; give a "
                         "positive number")
    for provider in providers:
        resolve(provider)
    if isinstance(workers, Mapping): workers = _per_adapter(workers)
    runs = {}
    for provider, options_ in configurations(providers, options):
        name = out_name(provider, options_)
        if name in runs:
            raise ValueError(f"run {name} is asked for more than once: the same provider with the same "
                             "settings (an option set to its default is the same setting); drop the "
                             "repeat from --providers or --options")
        runs[name] = Run(provider, options_, Path(out) / name)
    ident = data.dataset_id(dataset)
    for run in runs.values():
        # Note: a folder is named for provider and options, not dataset or manifest, so two of either
        # would share one, and a grading of one would replace the other's summary.
        said = json.loads(f.read_text()) if (f := run.out / "settings.json").exists() else {}
        if (was := said.get("dataset")) not in (None, ident):
            raise ValueError(f"{run.out} is a run on dataset {was}, not {ident}, and a run never "
                             f"mixes datasets; pass another --out, or --dataset {was}")
        if (was := said.get("manifest")) not in (None, manifest):
            raise ValueError(f"{run.out} is a run on manifest {was}, not {manifest}, and a run never "
                             f"mixes manifests; pass --manifest {was}, or another --out")
    root, every = data.read_manifest(dataset, manifest)
    commit = root.name if data._hf_dataset(dataset) is not None else None
    for run in runs.values():
        if msg := _commit_moved(run, commit):
            log.warning(msg)
    if not every:
        raise ValueError(f"the manifest {manifest} of {dataset} names no documents")
    if not (docs := data.select(every, suites, limit, sample, only)):
        given = [f for f, v in (("--suites", suites), ("--limit", limit), ("--sample", sample),
                                ("--docs", only is not None)) if v]
        raise ValueError(f"no documents selected by {', '.join(given) or 'the manifest'}; check them")
    return Plan(tuple(runs.values()), tuple(docs), str(dataset), manifest, timeout,
                workers, score_workers, verdicts, score_only, retry_failed)


def _commit_moved(run: Run, commit: str | None) -> str | None:
    """The warning for a run made at another commit of its dataset than `commit`, or `None`."""
    path = run.out / "settings.json"
    if not commit or not path.exists() or (was := json.loads(path.read_text()).get("commit")) in (None, commit):
        return None
    return (f"{run.out} was made at commit {was} of its dataset, and this plan uses {commit}: its "
            f"predictions made at {was} will be graded against the gold at {commit}. Pass another "
            f"--out to keep them apart.")


def predict_all(p: Plan) -> None:
    """Predict every document each run has no record of. Every run of one adapter drains through
    that adapter's one pool (`_workers_for`), the adapters go at once, and a live table shows each
    run. Predicting is network wait, so the adapters do not slow each other."""
    from .harness.registry import resolve

    # Note: set on an interrupt, so no run starts another document; a call already made is let
    # finish, and lands.
    stop = threading.Event()
    by: dict[str, list[Run]] = {}
    for run in p.runs:
        prepare(run, p.dataset, p.timeout, p.manifest)
        by.setdefault(resolve(run.provider), []).append(run)

    def drain(adapter: str, runs: list[Run], bars: Progress) -> None:
        """Every document of every run of one adapter, through its one pool."""
        workers = _workers_for(adapter, p.workers)
        bar = {r.out.name: bars.reporter(r.out.name) for r in runs}
        left, work = {}, []
        for r in runs:
            todo = [d for d in p.docs if needs(r, d, p.retry_failed)]
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
            except BaseException as exc:
                # Note: set here, by the worker that failed, not when the main thread gets to
                # it: the other workers would otherwise send documents queued behind it.
                halt.set()
                if isinstance(exc, Exception) and not isinstance(
                        exc, AccountFailure | MissingCredential | MissingDependency):
                    # Note: `predict` records every failure that is a fact about the document, so
                    # this is a bug in the adapter or in ours; recorded, it would pass as the vendor's.
                    exc.add_note(f"{r.out.name}: raised while predicting {d.doc_id} with the {adapter} "
                                 "adapter, a bug rather than the vendor's answer, so its runs stop; "
                                 "what was recorded is kept, and a run after the fix resumes")
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
                log.error("%s: stopped its runs (%s): %s: %s%s", adapter,
                          ", ".join(r.out.name for r in runs), type(exc).__name__,
                          str(exc)[:300], "".join(f"\n  {n}" for n in getattr(exc, "__notes__", ())))
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
    grading saturates every core. Each run is graded over the whole manifest, whatever `--suites`
    and `--limit` chose to predict, so a smaller invocation never shrinks a run's scores: a document
    not predicted yet is pending. Returns each run's summary, by run name.

    A SECOND PHASE, NOT A SECOND DISPLAY. Grading starts once every vendor has answered, so
    it gets a table of its own in the same shape -- with no cost and nothing in flight,
    because there is no vendor. Streaming it instead, so a run that finished predicting
    began grading, would put a process pool beside the live thread pools to save minutes at
    the end of a run that spends hours at the start of it.
    """
    summary = {}
    _, every = data.read_manifest(p.dataset, p.manifest)
    with Progress([r.out.name for r in p.runs]) as bars:
        for run in p.runs:
            summary[run.out.name] = grade(run, list(every), verdicts=p.verdicts,
                                          score_workers=p.score_workers,
                                          progress=bars.reporter(run.out.name))
    return summary


def execute(p: Plan) -> dict[str, dict]:
    """Predict, then grade. Returns each run's summary, by run name. What was predicted is graded
    even when predicting failed part way.

    Note: not when it was interrupted (Ctrl-C): whoever stopped it wants it stopped, grading takes
    minutes, and `--score-only` grades what is there."""
    try:
        if not p.score_only:
            predict_all(p)
    except Exception:
        grade_all(p)
        raise
    return grade_all(p)


def _settings_cell(asked: list[str]) -> Group:
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


def _facts(p: Plan) -> Table:
    """What this invocation is, one labelled row each. A row rather than another clause in a
    sentence, so a new flag cannot lengthen a line that already ran to three of them."""
    from .harness.registry import resolve

    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column("", style="dim", no_wrap=True)
    table.add_column("", overflow="fold")
    table.add_row("out", str(p.runs[0].out.parent) if p.runs else "")
    table.add_row("runs", f"{len(p.runs)} over "
                          f"{_plural(len({resolve(r.provider) for r in p.runs}), 'adapter')}")
    table.add_row("dataset", p.dataset)
    table.add_row("manifest", p.manifest)
    table.add_row("documents", f"{_plural(len(p.docs), 'document')} selected")
    table.add_row("timeout", f"{p.timeout:.0f}s per document")
    # Shown even when false: `score only  false` is what says this will spend money.
    for name, on in (("score only", p.score_only),):
        table.add_row(name, Text(str(on).lower(), style="bold" if on else "dim"))
    return table


def _plan_view(p: Plan) -> Group:
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
        workers = _workers_for(adapter, p.workers)
        for n, run in enumerate(runs):
            to_predict, to_grade = owed(p, run)
            to_predict = 0 if p.score_only else to_predict
            table.add_row(Text(adapter if n == 0 else "", style="bold"),
                          Text(str(workers) if n == 0 else ""),
                          Text(run.out.name), _settings_cell(_asked(run)),
                          Text(str(to_predict), style="bold" if to_predict else ""),
                          Text(str(to_grade), style="bold" if to_grade else ""),
                          end_section=n == len(runs) - 1)
    return Group(Text("benchmark", style="bold"), _facts(p), Text(""), table)


def describe(p: Plan) -> str:
    """`_plan_view` as plain text, for a log that has no colour and no width to ask about."""
    console = Console(file=io.StringIO(), width=PLAN_WIDTH, no_color=True, highlight=False)
    console.print(_plan_view(p))
    return "\n".join(line.rstrip() for line in console.file.getvalue().splitlines())
