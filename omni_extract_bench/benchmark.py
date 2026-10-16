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
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, NamedTuple

from . import data, harness
from .errors import Stop, UsageError
from .harness.errors import TRANSIENT_STATUSES
from .progress import NULL, Progress
from .scorers import field_matching as scorer

log = logging.getLogger(__name__)

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


def write_json(path: Path, obj: Any) -> None:
    """Written whole, then moved into place, so a crash never leaves half a file."""
    _write_text(path, json.dumps(obj, default=str, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: Path, text: str) -> None:
    """`write_json` for any text."""
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(text)
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


def prepare(run: Run, dataset: str, timeout: float, manifest: str = data.MANIFEST) -> None:
    """The run's folder made, and what it asks written to its settings.json (`declare`) before it asks:
    provider, settings and the timeout per document."""
    run.predictions.mkdir(parents=True, exist_ok=True)
    run.records.mkdir(parents=True, exist_ok=True)
    declare(run.out, dataset, manifest, {"run": run.out.name, "provider": run.provider,
                                         "settings": _settings(run), "timeout_s": timeout})


def settled(out: Path, dataset: str | None = None, manifest: str | None = None) -> dict:
    """The folder's settings.json ({} when it has none), checked against the dataset and manifest asked for
    (None: whatever it says). A folder holds one dataset and one manifest, so another is refused: graded
    against other gold, its scores would be replaced under the same name."""
    said = _said(out)
    for key, now in (("dataset", dataset and data.dataset_id(dataset)), ("manifest", manifest)):
        if now and (was := said.get(key)) and was != now:
            raise UsageError(f"{out} holds a run on {key} {was}, not {now}: a folder holds one dataset and "
                             f"one manifest; pass --{key} {was}, or another folder")
    return said


def declare(out: Path, dataset: str, manifest: str, fields: Mapping[str, Any] | None = None) -> None:
    """Writes the folder's settings.json, whole: `fields`, then the dataset (and the commit it was read at)
    and the manifest it is graded on. `prepare` writes a run's; `oeb grade`, a folder of your own's once its
    first grading succeeds."""
    root, _ = data.read_manifest(dataset, manifest)
    commit = data.commit(dataset, root)
    write_json(out / "settings.json", {**(fields or {}), "dataset": data.dataset_id(dataset),
                                       **({"commit": commit} if commit else {}), "manifest": manifest})


def _said(out: Path) -> dict:
    """The folder's settings.json as it stands; {} when it has none."""
    path = out / "settings.json"
    try:
        said = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UsageError(f"{path} isn't readable JSON ({exc}); fix it, or remove it to start the folder's "
                         "settings over") from None
    if not isinstance(said, dict):
        raise UsageError(f"{path} holds a JSON {type(said).__name__}; settings.json is an object")
    return said


def recorded(run: Run, doc_id: str) -> dict | None:
    """What the run has recorded for the document, or None when nothing (or nothing readable, which is
    logged)."""
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
    """Whether a recorded failure may go the other way on another try: a timeout, an outage
    (`VendorUnreachable`), or a transient HTTP status.

    Note: read off the type and status, not the record's `transient`, which says whether `predict`
    retried within the call: it never does for a timeout or a failed poll, since the first would spend a
    second budget and the second may buy the extraction twice."""
    if not error: return False
    return error.get("type") in ("VendorTimeout", "VendorUnreachable") or error.get("status") in TRANSIENT_STATUSES


def record(run: Run, doc_id: str, result: dict) -> dict:
    """What `harness.predict` returned for the document, written down and returned, as the record it now
    has: its prediction, if it has one, then the record. predictions/ holds only answers; a failure is
    described in the record alone. The record is the marker `needs` reads, so a kill between the two
    predicts the document again rather than skipping it."""
    if result["error"] is None:
        write_json(run.predictions / f"{doc_id}.json", result["result"])
    write_json(run.records / f"{doc_id}.json", result)
    return result


def _listed(run: Run) -> frozenset[str]:
    """The file names in predictions/; a prediction is read only under its exact name, so the score and the
    stray-file warning agree on every disk, case-insensitive ones included."""
    return frozenset(f.name for f in run.predictions.iterdir()) if run.predictions.is_dir() else frozenset()


def _tried(run: Run, doc_id: str, listed: frozenset[str]) -> bool:
    """Whether the run attempted the document: it has a record, or a prediction (`listed`, by exact name). In
    a folder that keeps records, a document never tried is pending; `grade` and `owed` both follow this."""
    return f"{doc_id}.json" in listed or recorded(run, doc_id) is not None


def owed(p: Plan, run: Run) -> tuple[int, int]:
    """How many documents the run would predict, and then grade, by the rules `predict_all` and
    `grade_all` follow: it predicts the plan's documents it has no record of, and grades every document of
    the manifest it will have attempted (`_tried`; the rest are pending), or every document in a folder
    without records/."""
    coming = set() if p.score_only else {d.doc_id for d in p.docs if needs(run, d, p.retry_failed)}
    _, every = data.read_manifest(p.dataset, p.manifest)
    if p.score_only and not run.records.is_dir(): return 0, len(every)
    listed = _listed(run)
    return len(coming), sum(d.doc_id in coming or _tried(run, d.doc_id, listed) for d in every)


def gradable(out: Path) -> None:
    """Refuses a folder grading can't read: one that doesn't exist, or that holds no predictions/."""
    if not out.is_dir():
        raise UsageError(f"no folder at {out}; pass the folder holding predictions/<doc_id>.json")
    if not (out / "predictions").is_dir():
        hint = (f"; {out} looks like the predictions/ folder itself: pass its parent, {out.parent}"
                if out.name == "predictions" else "")
        raise UsageError(f"{out} has no predictions/ folder, so nothing was predicted there; a folder to "
                         f"grade holds predictions/<doc_id>.json{hint}")


def scorer_errors(summary: dict) -> int:
    """Documents whose grading raised in our code (status `error`): a scorer bug, left out of the scores;
    `oeb benchmark` and `oeb grade` exit 2 when there are any."""
    return summary.get("errors", 0)


def _closing(s: dict) -> str:
    """A grading's one line: the headline, or why there is none, and how many documents were graded and
    failed."""
    why = (f" ({s['pending']} pending: a headline needs every document)" if s["pending"]
           else " (no documents)")
    head = f"headline {s['headline']:.3f}" if s["headline"] is not None else "headline —" + why
    return (f"{s['run']}: {head}; {s['documents']} of {s['documents'] + s['pending']} documents graded, "
            f"{s['failed']} failed"
            + (f"; {e} errored in our code" if (e := scorer_errors(s)) else ""))


def grade(run: Run, docs: Sequence[data.Doc], say: Callable[[str], None] = print, *,
          settings: Mapping[str, Any] | None = None, verdicts: bool = True, score_workers: int = 0,
          progress=NULL) -> dict:
    """Grade the run's folder against `docs`, the manifest's (`grade_all` passes them), write its
    verdicts/ (with `verdicts`), scores.jsonl and summary.json, and return the summary. `settings` is the
    folder's settings.json as `oeb grade` will write it once grading succeeds; otherwise it is read from the
    folder.

    A document's prediction is `predictions/<doc_id>.json`, named exactly. A document never attempted
    (`_tried`) in a folder that keeps records is pending; otherwise one with no usable prediction fails.
    docs/API.md has the summary's fields."""
    gradable(run.out)
    said = _said(run.out) if settings is None else settings
    partial = run.out / ".verdicts.partial"
    lines, pending = _graded(run, docs, partial if verdicts else None, score_workers, progress)
    summary = _summary(run, said, lines, pending)
    shutil.rmtree(run.out / "verdicts", ignore_errors=True)
    if partial.is_dir(): partial.rename(run.out / "verdicts")
    _write_text(run.out / "scores.jsonl", "".join(json.dumps(x, default=str) + "\n" for x in lines))
    # Note: last, so summary.json always stands beside the scores it summarizes.
    write_json(run.out / "summary.json", summary)
    _warn_unusable(run, docs, lines)
    say(_closing(summary))
    return summary


def _graded(run: Run, docs: Sequence[data.Doc], verdicts: Path | None, score_workers: int,
            progress) -> tuple[list[dict], list[data.Doc]]:
    """The score lines of the documents the folder attempted (`_tried`), in their order, each verdict file
    written into `verdicts` when given; and the documents pending."""
    if verdicts is not None: shutil.rmtree(verdicts, ignore_errors=True)   # Note: an interrupted grading's.
    keeps, listed = run.records.is_dir(), _listed(run)
    attempted = [d for d in docs if not keeps or _tried(run, d.doc_id, listed)]
    progress.start(len(attempted))
    one = functools.partial(_score_one, run=run, verdicts=verdicts, listed=listed)
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
    return [graded[d.doc_id] for d in attempted], [d for d in docs if d.doc_id not in graded]


def _summary(run: Run, said: Mapping[str, Any], lines: list[dict], pending: list[data.Doc]) -> dict:
    """`scorer.summarize` of the lines, beside what graded them, the pending count, and the cost; the
    headline, overall and per suite, is None while any document there is pending: a run has one only once
    it is finished."""
    s = scorer.summarize(lines)
    by = collections.Counter(d.suite for d in pending)
    none = {k: v for k, v in scorer.summarize([]).items() if k != "per_suite"}   # a suite all pending
    per = {k: s["per_suite"].get(k, none) | {"pending": by.get(k, 0)} for k in sorted({*s["per_suite"], *by})}
    for k in by: per[k]["headline"] = None
    costs = [c for x in lines if (c := (recorded(run, x["doc_id"]) or {}).get("cost"))]
    credits = [c["credits"] for c in costs if c.get("credits") is not None]
    return {"run": run.out.name, "provider": run.provider, "settings": said.get("settings", {}),
            "scorer": SCORER, **s, "headline": None if pending else s["headline"], "per_suite": per,
            "pending": len(pending), "usd": round(sum(c.get("usd") or 0 for c in costs), 4),
            "credits": round(sum(credits), 4) if credits else None}


#: The error of a line whose prediction is on disk but isn't a JSON object.
UNREADABLE = "the prediction isn't readable as a JSON object"


def _warn_unusable(run: Run, docs: Sequence[data.Doc], lines: list[dict]) -> None:
    """Logs the predictions grading couldn't use: files named for no document of the manifest `docs` (left
    out), and those that couldn't be read as a JSON object (graded as failed). A naming slip
    (`<doc_id>.pdf.json`, another case) otherwise shows only as a lower coverage."""
    names = {f"{d.doc_id}.json" for d in docs}
    stray = sorted(n + ("/" if (run.predictions / n).is_dir() else "") for n in _listed(run) - names)
    if stray:
        log.warning("%s: %d files in predictions/ are named for no document of the manifest, so they were "
                    "left out, e.g. %s; a prediction is predictions/<doc_id>.json, named exactly (case too)",
                    run.out, len(stray), stray[:3])
    if unread := [x["doc_id"] for x in lines if x.get("error") == UNREADABLE]:
        log.warning("%s: %d predictions couldn't be read as a JSON object, so their documents were graded "
                    "as failed, e.g. %s", run.out, len(unread), unread[:3])


SCORE_WORKERS = min(8, os.cpu_count() or 1)


def _score_one(doc: data.Doc, *, run: Run, verdicts: Path | None, listed: frozenset[str]) -> dict:
    """Grade one prediction on disk and return its line. Module-level and picklable for the process pool,
    and it never raises: one bad document is one line. Its verdicts are written into the folder `verdicts`,
    when given; `listed`, predictions/'s file names, says by exact name whether it has a prediction.

    A document fails (status `failed`) when its record says the call failed, or when it has no usable
    prediction: a folder of predictions you bring is your submission, so a missing file counts against
    coverage rather than being left out of it. A grading that raises is ours (status `error`), left out of
    the scores."""
    row = {"run": run.out.name, "doc_id": doc.doc_id, "suite": doc.suite, "provider": run.provider,
           "scorer": SCORER}
    if failed := (recorded(run, doc.doc_id) or {}).get("error"):
        return {**row, "status": "failed", "error": f"{failed['type']}: {failed['message']}"}
    if f"{doc.doc_id}.json" not in listed:
        return {**row, "status": "failed", "error": f"no prediction: predictions/{doc.doc_id}.json"}
    if (pred := predicted(run, doc.doc_id)) is None:
        return {**row, "status": "failed", "error": UNREADABLE}
    if not pred:
        return {**row, "status": "failed", "error": "empty prediction"}
    try:
        result = scorer.score(pred, scorer.gold(doc), doc.schema, verdicts=verdicts is not None)
        if verdicts is not None:
            verdicts.mkdir(parents=True, exist_ok=True)
            (verdicts / f"{doc.doc_id}.jsonl").write_text(
                "\n".join(json.dumps(v._asdict(), default=str) for v in result.pop("verdicts")))
    except Exception as exc:                            # noqa: BLE001 -- the message is the line
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
    graded. A line that doesn't parse is left out, and logged. A folder that doesn't exist is refused: a
    mistyped path isn't a run with no scores."""
    if not run.out.is_dir():
        raise UsageError(f"no folder at {run.out}; pass a run's folder, which holds scores.jsonl")
    path = run.out / "scores.jsonl"
    if not path.exists(): return []
    out = []
    for n, line in enumerate(path.read_text().splitlines()):
        if not line.strip(): continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("%s: line %d isn't readable JSON; left out", path, n + 1)
    return out


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
    adapters = registry.adapter_names()
    for key, n in workers.items():
        if key not in adapters:
            try:
                a = registry.resolve(key)
                hint = f"{key} uses the {a} adapter, whose pool all its runs share; write {{\"{a}\": {n}}}"
            except UsageError:
                hint = f"the adapters are {sorted(adapters)}"
            raise UsageError(f"--workers is per adapter, and {key!r} isn't one: {hint}")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise UsageError(f"--workers gives {key!r} {n!r}; documents at once is a whole number, 1 or more")
    return dict(workers)


def pools(p: Plan) -> dict[str, int]:
    """Each adapter of the plan's runs (`registry.resolve`) and its documents in flight, shared by its runs."""
    from .harness.registry import resolve

    return {a: _workers_for(a, p.workers) for a in sorted({resolve(r.provider) for r in p.runs})}


def configurations(providers: list[str], options: dict | None = None) -> list[tuple[str, dict]]:
    """Each (provider, options) pair to run. `--options` may give one provider a LIST, and each
    entry is a run of its own."""
    unknown = sorted(set(options or {}) - set(providers))
    if unknown:
        raise UsageError(
            f"--options names {unknown[0]!r}, which is not in --providers "
            f"({', '.join(providers)}). Its options would be silently ignored.")
    out = []
    for provider in providers:
        asked = (options or {}).get(provider)
        if isinstance(asked, list) and not asked:
            raise UsageError(f"--options gives {provider!r} an empty list, so it would not "
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


def plan(providers: Sequence[str], *, out: str | Path = Path("runs"), dataset: str | Path = data.DEFAULT_DATASET,
         manifest: str = data.MANIFEST, suites: Iterable[str] | None = None, limit: int = 0,
         sample: int = 0, only: Iterable[str] | None = None,
         timeout: float = 1800.0, workers: int | Mapping[str, int] = 0,
         score_workers: int = 0, verdicts: bool = True,
         score_only: bool = False, retry_failed: bool = False, options: dict | None = None) -> Plan:
    """Every run this invocation makes and the documents it covers, checked before any call: a bad
    provider, option, count or folder fails here.

    `workers` is the documents in flight per adapter, shared by its runs: one number for every
    adapter, or a number per adapter (`_per_adapter`); 0, or an adapter left out, takes `WORKERS`."""
    from .harness.registry import out_name, resolve

    if isinstance(providers, str):
        raise TypeError(f"providers is a list of names, not the string {providers!r}; pass [{providers!r}]")
    if isinstance(workers, bool) or not isinstance(workers, int | Mapping):
        raise TypeError(f"workers is a number for every adapter, or a mapping of adapter to number, not "
                        f"{workers!r}")
    counts = [("--score-workers", score_workers, "the default")]
    if isinstance(workers, int): counts.append(("--workers", workers, "each adapter's default"))
    for flag, n, zero in counts:
        if n < 0: raise UsageError(f"{flag} is {n}, but it counts, so it can't be negative; 0 means {zero}")
    if not timeout > 0:
        raise UsageError(f"--timeout is {timeout}, but it is the seconds a document may take; give a "
                         "positive number")
    for provider in providers:
        resolve(provider)
    if isinstance(workers, Mapping): workers = _per_adapter(workers)
    runs = {}
    for provider, options_ in configurations(providers, options):
        name = out_name(provider, options_)
        if name in runs:
            raise UsageError(f"run {name} is asked for more than once: the same provider with the same "
                             "settings (an option set to its default is the same setting); drop the "
                             "repeat from --providers or --options")
        runs[name] = Run(provider, options_, Path(out) / name)
    root, every = data.read_manifest(dataset, manifest)
    if not (docs := data.select(every, suites, limit, sample, only)):
        given = [f for f, v in (("--suites", suites), ("--limit", limit), ("--sample", sample),
                                ("--docs", only is not None)) if v]
        raise UsageError(f"no documents selected: {' and '.join(given)} chose none of the manifest's "
                         f"{len(every)} documents")
    commit = data.commit(dataset, root)
    for run in runs.values():
        if score_only and not run.predictions.is_dir():
            raise UsageError(f"no run at {run.out}: {run.provider} with these settings was never predicted "
                             "under this --out; check --out and --options, or drop --score-only")
        said = settled(run.out, str(dataset), manifest)
        if msg := _commit_moved(run, said.get("commit"), commit):
            log.warning(msg)
    return Plan(tuple(runs.values()), tuple(docs), str(dataset), manifest, timeout,
                workers, score_workers, verdicts, score_only, retry_failed)


def _commit_moved(run: Run, was: str | None, commit: str | None) -> str | None:
    """The warning for a run made at commit `was` of its dataset, planned at `commit`, or `None`.

    Note: warned, not refused: a new commit is usually a gold correction, which grading takes in on its
    own; but a document that changed would be graded on an old prediction."""
    if not commit or was in (None, commit):
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
                    got = harness.predict(r.provider, file, d.schema, timeout=p.timeout, **r.options)
                record(r, d.doc_id, got)
            except BaseException as exc:
                # Note: set here, by the worker that failed, not when the main thread gets to
                # it: the other workers would otherwise send documents queued behind it.
                halt.set()
                if isinstance(exc, Exception) and not isinstance(exc, Stop):
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


def grade_all(p: Plan, say: Callable[[str], None] = print) -> dict[str, dict]:
    """`grade` each run of the plan over the whole manifest, serially, since grading saturates every core;
    their summaries by run name. A smaller invocation (`--suites`, `--limit`) never shrinks a run's scores:
    a document not predicted yet is pending.

    Note: a second phase with a table of its own, not streamed beside predicting: that would put a process
    pool beside the live thread pools to save minutes at the end of a run that spends hours at its start."""
    summary = {}
    _, every = data.read_manifest(p.dataset, p.manifest)
    with Progress([r.out.name for r in p.runs]) as bars:
        for run in p.runs:
            summary[run.out.name] = grade(run, every, say, verdicts=p.verdicts, score_workers=p.score_workers,
                                          progress=bars.reporter(run.out.name))
    return summary


def execute(p: Plan, say: Callable[[str], None] = print) -> dict[str, dict]:
    """Predict, then grade. Returns each run's summary, by run name. What was predicted is graded
    even when predicting failed part way.

    Note: not when it was interrupted (Ctrl-C): whoever stopped it wants it stopped, grading takes
    minutes, and `--score-only` grades what is there."""
    try:
        if not p.score_only:
            predict_all(p)
    except Exception:
        grade_all(p, say)
        raise
    return grade_all(p, say)


def describe(p: Plan) -> str:
    """What `execute(p)` would do, as plain text (`progress.plan_view`), with no vendor call."""
    from . import progress

    return progress.describe(p)
