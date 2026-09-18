#!/usr/bin/env python3
"""The whole benchmark: fetch the corpus, predict with each vendor, score, write it down.

    oeb benchmark --providers datalab reducto --out runs/
    oeb benchmark --providers datalab --limit 5              # smoke test

    fetch          the corpus from HuggingFace: manifest, PDFs and gold
    read_manifest  620 rows -> Doc(doc_id, suite, pdf, gt, schema)
    predict_all    each document through one vendor, under the harness's parity rules
    score_all      score() each prediction against its gold
    summarise      per suite, then UNIFIED

RESUMABLE, AND SAFE TO RUN TWICE. A document is predicted again only when it has no record,
and graded again only when it has no row in `scores.jsonl` -- so an interrupted run carries on
instead of paying twice. Nothing checks the metric has not changed under a resume; `--rescore`
is how to say it has.

THE ORCHESTRATION, DELIBERATELY NOT THE LIBRARY. `score` grades one document and
`harness.predict` produces one prediction; which documents, in what order and how many at once
are decisions about a corpus. What this file does, you can do differently.

    pip install 'omni-extract-bench[benchmark,harness]'

Credentials come from the environment. Everything else a vendor is told comes from `--options`,
so `run_manifest.settings` is a complete account of what each vendor was asked.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import functools
import json
import logging
import os
import threading
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import NamedTuple

from . import score
from .progress import NULL, Progress
from .harness import (WORKERS, AccountFailure, MissingCredential, MissingDependency,
                      predict)


log = logging.getLogger(__name__)

REPO = "datalab-to/omni_extract_bench"
MANIFEST = "manifest.parquet"


class Doc(NamedTuple):
    """One benchmark document, resolved once from the manifest.

    The schema is parsed here and passed as a dict, because the vendor and the scorer must
    get the same object.
    """

    doc_id: str
    suite: str
    pdf: Path
    gt: Path
    schema: dict


# ── 1. the corpus ────────────────────────────────────────────────────────────────────────
def fetch(root: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is needed to fetch the benchmark corpus, and is not installed:\n"
            "    pip install 'omni-extract-bench[benchmark]'\n"
            "The scorer itself needs none of it -- `score` and `oeb score` work without."
        ) from None

    log.info("fetching the corpus into %s", root)
    return Path(snapshot_download(REPO, repo_type="dataset", local_dir=str(root)))


def read_manifest(path: Path, root: Path, suites=None, limit: int = 0) -> list[Doc]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is needed to read the benchmark manifest, and is not installed:\n"
            "    pip install 'omni-extract-bench[benchmark]'"
        ) from None

    docs = []
    for row in pq.read_table(path).to_pylist():
        if suites and row["suite"] not in suites:
            continue
        docs.append(Doc(doc_id=row["doc_id"],
                        suite=row["suite"],
                        pdf=root / row["doc_path"],
                        gt=root / row["gt_path"],
                        schema=json.loads(row["schema"])))
    docs.sort(key=lambda d: (d.suite, d.doc_id))       # a stable order, so runs are comparable
    return docs[:limit] if limit else docs


# ── 2. predictions ───────────────────────────────────────────────────────────────────────
def write_json(path: Path, obj, *, indent: int | None = None) -> None:
    """Write a file that is either wholly there or not there at all.

    `write_text` killed part way through leaves a truncated file that still EXISTS, and a
    resume asking "was this attempted?" would skip it forever. The temp file shares a
    directory with the target, because rename is only atomic within a filesystem.

    `indent` for the few files a person opens; the per-document ones are read by the machine
    and there are 620 of them per run.
    """
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(obj, default=str, indent=indent))
    os.replace(tmp, path)


def needs_run(record_path: Path) -> bool:
    """True when this document has not been attempted yet.

    A record means the vendor was called and answered, well or badly; whether a failure
    deserved another try was `predict`'s decision and the attempts are spent. The record is
    written LAST, so its presence means the prediction beside it is complete. Parsed rather
    than stat-ed: a file that will not parse is from a run that died.
    """
    if not record_path.exists():
        return True
    try:
        json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    return False


def predict_all(docs: list[Doc], provider: str, out: Path, timeout: float, workers: int,
                options: dict | None = None, progress=NULL, label: str | None = None,
                stop: threading.Event | None = None) -> None:
    """Every document through one vendor, concurrently, leaving the answer and the evidence.

        <out>/predictions/<doc_id>.json    the bare extraction -- what scoring reads
        <out>/records/<doc_id>.json        the schema sent, every HTTP call, the cost

    Two files because they are read at different times and are different sizes: scoring wants
    the answer, an audit wants everything, and only one of them is worth loading 620 of.
    """
    # What this run is CALLED, which is the provider plus its options; `provider` alone would
    # log two configurations of one vendor under the same name.
    label = label or provider
    preds, records = out / "predictions", out / "records"
    preds.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)

    todo = [d for d in docs if needs_run(records / f"{d.doc_id}.json")]
    log.info("%s: %d documents, %d to run", label, len(docs), len(todo))
    progress.start(len(todo), workers)
    if not todo:
        progress.finish()
        return

    # TWO SCOPES: the caller's `stop` is a Ctrl-C and halts every vendor, `mine` is an unset
    # key or a credit ceiling and halts this one. Sharing a single event made a healthy vendor
    # publish 17% coverage because another ran out of credits.
    mine = threading.Event()
    stopping = lambda: mine.is_set() or (stop is not None and stop.is_set())   # noqa: E731
    fatal: list[Exception] = []
    tally: collections.Counter[str] = collections.Counter()   # main thread only: no lock
    spend: list[float] = []
    billed: list[float] = []          # vendors that price in credits, not dollars
    took: list[float] = []            # wall-clock per document

    # One place a provider gives up, so the three ways cannot drift. Locked because
    # check-then-set is not atomic: two workers failing together both passed the test.
    halt = threading.Lock()

    def give_up(exc: Exception, why: str = "") -> tuple:
        """Stop this vendor, remember why, and report this document as never attempted."""
        with halt:
            if not mine.is_set():
                if why:
                    log.error("%s: %s -- %s", label, why, exc)
                fatal.append(exc)
                mine.set()
        return "skipped", None, None, None

    def run_one(doc: Doc) -> tuple:
        if stopping():
            return "skipped", None, None, None
        try:
            # The block is exactly the call, so the in-flight count is what the vendor holds,
            # not what our pool has queued.
            with progress.calling():
                record = predict(provider, doc.pdf, doc.schema, timeout=timeout,
                                 **(options or {}))
            # ORDER IS THE COMMIT: the prediction lands first, the record is the marker
            # `needs_run` reads. Killed between them the document looks unattempted and is run
            # again -- the cheap mistake. INSIDE the guard, because a full disk fails here,
            # and these writes once sat past the last `except`: the vendor was paid for every
            # remaining document while not one answer could be stored.
            write_json(preds / f"{doc.doc_id}.json", record["result"])
            write_json(records / f"{doc.doc_id}.json", record)
        except (MissingDependency, MissingCredential) as exc:
            # Ours, identical for every document, not transient.
            return give_up(exc)
        except AccountFailure as exc:
            # RAISED, not merely logged: its skipped documents have no prediction, so the
            # summary would report a vendor that was never asked as one that could not answer.
            return give_up(exc, "STOPPED, account-level failure")
        except Exception as exc:          # noqa: BLE001 -- unknown, so assume it is ours
            # Anything `predict` did not turn into a record, or anything the writes raise. It
            # will repeat, so stop rather than buy the same failure 600 more times.
            return give_up(exc)
        cost = record.get("cost") or {}
        return ("error" if record.get("error") else "ok",
                cost.get("usd"), cost.get("credits"), cost.get("wall_s"))

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, doc) for doc in todo]
        try:
            for i, future in enumerate(cf.as_completed(futures), 1):
                status, usd, credits, wall_s = future.result()
                tally[status] += 1
                if usd is not None:
                    spend.append(usd)
                if credits is not None:
                    billed.append(credits)
                if wall_s is not None:
                    took.append(wall_s)
                # A skipped document was never reached, so it is not progress: it would fill
                # the bar on the way out of a run that failed on its first document.
                if status != "skipped":
                    progress.record(error=status == "error", usd=usd, credits=credits,
                                    wall_s=wall_s)
        except KeyboardInterrupt:
            # Before the pool's `__exit__`, which waits for every queued future -- without
            # this the interrupt works through the whole corpus first. Only reachable when
            # `predict_all` is called directly; under `run` the handler is there.
            mine.set()
            for future in futures:
                future.cancel()
            log.warning("%s: interrupted -- %s; documents not yet written come back on the "
                        "next run", provider, dict(tally))
            raise

    # `len(spend)` is reported because several vendors price only some documents: a bare
    # total over a corpus where half reported nothing reads as the bill and is a fraction.
    if took:
        # The mean AND the range: 4s-and-1800s is a different proposition from a steady 900s.
        log.info("%s: %.1fs per document on average (min %.1f, max %.1f over %d)",
                 provider, sum(took) / len(took), min(took), max(took), len(took))
    if spend:
        log.info("%s: $%.2f reported over %d of %d documents", provider, sum(spend),
                 len(spend), len(todo))
    if billed:
        # Credits stay in the vendor's own unit; the rate is contract-specific.
        log.info("%s: %g credits reported over %d of %d documents", provider, sum(billed),
                 len(billed), len(todo))
    if todo and not spend and not billed:
        log.info("%s: no per-document cost reported (billed out of band)", provider)

    progress.finish()
    # After the pool drains, so no worker is still writing. The documents it stopped keep no
    # file, so they stay resumable.
    if fatal:
        raise fatal[0]


# ── 3. scores ────────────────────────────────────────────────────────────────────────────

#: Capped rather than taken from the core count: the heaviest arrays here need ~1.8 GB each
#: (`matching.MAX_CELLS`), so a machine should not be run out of memory by its own core count.
SCORE_WORKERS = min(8, os.cpu_count() or 1)


def score_one(doc: Doc, *, provider: str, out: Path, verdicts: bool = False) -> dict:
    """Grade one prediction on disk and return its row.

    Module-level and picklable, so `score_all` can hand it to a process pool. It never
    raises: one bad document is an error row, not the end of the grading pass.
    """
    row = {"doc_id": doc.doc_id, "suite": doc.suite, "provider": provider}
    path = out / "predictions" / f"{doc.doc_id}.json"
    try:
        pred = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {**row, "status": "error", "error": "no prediction on disk"}
    # ONE definition of "is this something to score". It was two once and they disagreed, so
    # a vendor reported 100% coverage while 37 of its 45 outputs were empty.
    if not (isinstance(pred, dict) and pred and "__error__" not in pred):
        reason = (pred or {}).get("__error__") if isinstance(pred, dict) else "not an object"
        return {**row, "status": "error", "error": str(reason or "empty prediction")}
    try:
        result = score(pred, json.loads(doc.gt.read_text()), doc.schema, verdicts=verdicts)
        if verdicts:
            # INSIDE the guard: it writes to disk, and an escape from here is not one bad
            # row but the whole grading pass. `exist_ok` -- workers arrive together.
            (out / "verdicts").mkdir(parents=True, exist_ok=True)
            (out / "verdicts" / f"{doc.doc_id}.jsonl").write_text(
                "\n".join(json.dumps(v._asdict(), default=str)
                          for v in result.pop("verdicts")))
    except Exception as exc:                            # noqa: BLE001 -- the message is the row
        return {**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {**row, "status": "scored", **result}


def read_scores(out: Path, verdicts: bool = False) -> dict[str, dict]:
    """Scores already on disk for this provider, keyed by document.

    What makes grading resumable per document. It does NOT check the metric is the one that
    produced them -- editing the scorer and resuming would mix two definitions, and that is
    the caller's to avoid with `rescore`. One unreadable line costs one document, not the file.
    """
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
        # A row scored without verdicts cannot answer a run that wants them: the per-address
        # file was never written, and reusing the row leaves a hole no later run fills.
        if verdicts and not (out / "verdicts" / f"{doc_id}.jsonl").exists():
            continue
        rows[doc_id] = row
    return rows


def score_all(docs: list[Doc], provider: str, out: Path, verdicts: bool = False,
              workers: int = 0, rescore: bool = False) -> list[dict]:
    """Grade the predictions on disk, and write one line per document.

    EVERY DOCUMENT COMES BACK, including ones with no usable prediction: coverage is only
    visible if a failure occupies a row. Such a row carries NO metrics -- just the status and
    what happened -- rather than zeros, which would claim the model tried and missed every
    field. `summarise` is where they count as zero, when it averages.

    PROCESSES, not threads: this is the one CPU-bound half of a run, and
    `_worth_if_paired`'s recursion is interpreted Python that would serialise on the GIL.

    RESUMABLE PER DOCUMENT -- a document already in `scores.jsonl` is not graded again;
    `rescore=True` grades everything regardless. The file is in document order whatever order
    the workers finish in, so a run stays comparable with the one before it.
    """
    known = read_scores(out, verdicts=verdicts)
    if rescore:
        # This selection only: rows the run did not select are still theirs to keep.
        selected = {d.doc_id for d in docs}
        known = {k: v for k, v in known.items() if k not in selected}
    wanted = [d for d in docs if d.doc_id not in known]
    if len(wanted) < len(docs):
        log.info("%s: %d of %d documents already scored, %d to grade",
                 provider, len(docs) - len(wanted), len(docs), len(wanted))

    grade = functools.partial(score_one, provider=provider, out=out, verdicts=verdicts)
    graded: dict[str, dict] = {}

    def so_far() -> list[dict]:
        """Every row known, old and new, in corpus order.

        Merged over WHAT IS ON DISK, not over `docs`: a `--limit 2` run would otherwise
        rewrite the file with two lines and throw away a full run's grading.
        """
        merged = {**known, **graded}
        return sorted(merged.values(), key=lambda r: (r.get("suite", ""), r["doc_id"]))

    remaining = wanted
    try:
        if remaining and (workers or SCORE_WORKERS) > 1:
            try:
                with cf.ProcessPoolExecutor(max_workers=workers or SCORE_WORKERS) as pool:
                    # `as_completed`, not `map`: `map` yields in input order, so one heavy
                    # document at the front holds back everything finished behind it. A Ctrl-C
                    # kept nothing while nine documents sat completed behind document 0.
                    futures = {pool.submit(grade, doc): doc.doc_id for doc in remaining}
                    for future in cf.as_completed(futures):
                        graded[futures[future]] = future.result()
                remaining = []
            except BrokenProcessPool:
                # WHERE IT BROKE IS THE DIAGNOSIS. Part way through is a worker out of
                # memory, or children cut down by a Ctrl-C -- starting the corpus over is the
                # wrong answer to either. Before anything was graded it is usually the `spawn`
                # guard (a process re-imports __main__, so a bare script re-runs itself in
                # every worker) -- but a Ctrl-C lands here too, so the message says both.
                if graded:
                    raise
                log.warning("scoring pool stopped before any document was graded, so grading "
                            "continues in this process. If that was a Ctrl-C, press it again; "
                            "if it was not, put the call under `if __name__ == \"__main__\":` "
                            "or pass score_workers=1.")
        for doc in remaining:
            graded[doc.doc_id] = grade(doc)
    finally:
        # ONE WRITE, ON EVERY PATH OUT: finished, interrupted, broken pool, disk error. It
        # replaced a save-and-re-raise in each handler, which is how the path that needed it
        # most -- an exception from a worker -- came to be the one without it.
        write_scores(out, so_far())
        if len(graded) < len(wanted):
            log.warning("scoring stopped after %d of %d documents; the rest are graded on "
                        "the next run", len(graded), len(wanted))

    # THE FILE AND THE ANSWER ARE DIFFERENT LISTS. The file is every document ever graded
    # here; the return is this run's selection, because `summarise` counts what it is given --
    # handed the file, a `--limit 2` run reported five documents.
    answered = {**known, **graded}
    return [answered[d.doc_id] for d in docs if d.doc_id in answered]


def write_scores(out: Path, rows: list[dict]) -> None:
    """One line per document, written wherever scoring stops rather than only where it ends.

    ATOMIC because it is READ BACK: an interrupt landing mid-write would truncate the very
    file it was saving, and `read_scores` would silently drop everything past the cut.
    """
    path = out / "scores.jsonl"
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))
    os.replace(tmp, path)


# ── 4. the number ────────────────────────────────────────────────────────────────────────
def summarise(rows: list[dict]) -> dict:
    """Per suite, then UNIFIED: the mean of the suite means, not the mean over documents.

    Equal weight per suite so a large one cannot dominate (`docs/METRIC_SPEC.md` section 7).
    The flat mean is reported too, because seeing the two differ is the point.

    A document with no usable prediction scores ZERO here while its row keeps null metrics:
    `coverage` says how often the vendor answered, the score says what it is worth to someone
    who has to run every document. Averaging only successes would pay for failing on hard ones.
    """
    by_suite: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        by_suite[row["suite"]].append(row["accuracy"] if row["status"] == "scored" else 0.0)

    per_suite = {suite: {"documents": len(v), "accuracy": sum(v) / len(v)}
                 for suite, v in sorted(by_suite.items())}
    scored = [r for r in rows if r["status"] == "scored"]
    return {
        "documents": len(rows),
        "scored": len(scored),
        "coverage": len(scored) / len(rows) if rows else 0.0,
        "unified": (sum(s["accuracy"] for s in per_suite.values()) / len(per_suite)
                    if per_suite else 0.0),
        "flat_mean": sum(sum(v) for v in by_suite.values()) / len(rows) if rows else 0.0,
        "mean_over_scored": (sum(r["accuracy"] for r in scored) / len(scored)
                             if scored else 0.0),
        "per_suite": per_suite,
    }


# ── the run ──────────────────────────────────────────────────────────────────────────────
#: For a vendor `harness.WORKERS` says nothing about.
DEFAULT_WORKERS = 5


def workers_for(provider: str, requested: dict[str, int] | int | None) -> int:
    """How many documents to have in flight at one vendor.

    A worker holds its document through the whole poll loop, so this IS the number of jobs in
    flight server-side, not a request rate -- at five-second polls, ten in flight is two
    requests a second. Named providers beat a bare number, which beats the per-vendor default.
    """
    if isinstance(requested, int):
        requested = {"*": requested} if requested else {}
    requested = requested or {}
    return (requested.get(provider) or requested.get("*")
            or WORKERS.get(provider, DEFAULT_WORKERS))


def plan(providers: list[str], options: dict | None = None) -> list[tuple[str, str, dict]]:
    """(label, provider, options) for every configuration this invocation measures.

    A RUN IS A PROVIDER PLUS ITS OPTIONS, not a provider. `--options` may give one provider a
    LIST of option sets, and each is its own run -- which is how one invocation compares a
    vendor's tiers:

        --providers datalab --options '{"datalab": [{"mode": "balanced"},
                                                    {"mode": "accurate"}]}'

    The label is `out_name`, so the same string is the directory, the summary key and the
    progress line. Deduplicating on it means two spellings of one configuration cannot be
    bought twice -- `--providers datalab datalab` is one run, and so is naming the stock
    settings explicitly.
    """
    from .harness.vendor import out_name

    # A KEY THAT MATCHES NO PROVIDER IS A TYPO, and a silent one: the options are dropped and
    # the run reports as stock, which is a steered measurement wearing a stock label.
    unknown = sorted(set(options or {}) - set(providers))
    if unknown:
        raise ValueError(
            f"--options names {unknown[0]!r}, which is not in --providers "
            f"({', '.join(providers)}). Its options would be silently ignored.")

    runs: dict[str, tuple[str, str, dict]] = {}
    for provider in providers:
        asked = (options or {}).get(provider)
        if isinstance(asked, list) and not asked:
            # Otherwise the provider silently contributes no runs, and an empty `runs` reaches
            # the pool as `max_workers=0`.
            raise ValueError(f"--options gives {provider!r} an empty list, so it would not "
                             f"run at all. Give it options, or leave it out.")
        for one in (asked if isinstance(asked, list) else [asked]):
            label = out_name(provider, one)
            runs.setdefault(label, (label, provider, one or {}))
    return list(runs.values())


def run(providers: list[str], *, out: Path = Path("runs"),
        data_root: Path = Path("benchmark"), suites: list[str] | None = None, limit: int = 0,
        timeout: float = 1800.0, predict_workers: dict[str, int] | int | None = None,
        score_workers: int = 0, verdicts: bool = False, rescore: bool = False,
        score_only: bool = False, options: dict | None = None) -> dict:
    """Fetch, predict, score, write it down. Returns the summary it also writes to `out`.

    Keyword arguments and no argparse, so this stays callable from a notebook; it raises
    rather than exits, for the same reason.

        <out>/<provider>-<digest>/
            settings.json                      what this run asked, before it asked it
            summary.json                       what it came to, as soon as it is graded
            predictions/<doc_id>.json          the bare extraction -- what scoring reads
            records/<doc_id>.json              the schema sent, the cost, the manifest
            scores.jsonl                       one graded row per document

    A run directory is self-contained, and there is no table across them: `runs/*/summary.json`
    is one, aggregated however its reader likes, and a file here would only be one opinion
    about that written down -- stale the moment another run lands beside it.

    `options` is `{provider: {option: value}}` and lands in `run_manifest.settings`, because
    a run that turned a vendor down must not be able to look stock afterwards.

    `predict_workers` is `{provider: count}` (see `workers_for`). It is separate from
    `score_workers` because they buy different resources: documents a vendor holds at once,
    against local cores that grade.
    """
    from .harness.vendor import resolve, settings_for

    for provider in providers:
        resolve(provider)          # raises ValueError naming the vendors, before any download
    runs = plan(providers, options)

    root = fetch(data_root)
    docs = read_manifest(root / MANIFEST, root, suites=suites, limit=limit)
    if not docs:
        raise ValueError("no documents selected: check --suites and --limit")

    # WHAT THIS RUN IS, WRITTEN BEFORE IT STARTS. The directory is `<provider>-<digest of the
    # settings>`, which keeps two configurations apart and says nothing a person can read.
    # This is where they read it -- before the first document, because `summary.json` is
    # written after grading and a run interrupted, or stopped by a credit ceiling, never
    # reaches one. The alternative was a record: they carry `run_manifest.settings` too, but
    # only once a document has landed, and a directory should not need one to say what it is.
    dirs = {label: out / label for label, _, _ in runs}
    for label, provider, opts in runs:
        dirs[label].mkdir(parents=True, exist_ok=True)
        write_json(dirs[label] / "settings.json",
                   {"provider": provider, "settings": settings_for(provider, opts),
                    "timeout_s": timeout}, indent=2)

    # PREDICT EVERY VENDOR AT ONCE, THEN GRADE. Predicting is network wait -- a document's
    # `wall_s` is the vendor's own server-side time plus a couple of seconds -- so vendors do
    # not slow each other down. Grading is the opposite and saturates every core, and letting
    # it overlap a vendor call would put local CPU load inside a published latency figure.
    stop = threading.Event()           # one Ctrl-C stops every provider, not just one
    if not score_only:
        # THE CAP IS THE VENDOR'S, AND IT IS SHARED. Two runs of one provider go at once, so
        # a cap applied to each would put twice as many documents in flight as the vendor
        # tolerates -- measured, `WORKERS["datalab"] = 10` reached 20. Split it between them.
        per_vendor = collections.Counter(provider for _, provider, _ in runs)
        with Progress([label for label, _, _ in runs]) as bars, \
                cf.ThreadPoolExecutor(max_workers=len(runs)) as pool:
            futures = {pool.submit(predict_all, docs, provider, dirs[label], timeout=timeout,
                                   workers=max(1, workers_for(provider, predict_workers)
                                               // per_vendor[provider]),
                                   options=opts, label=label,
                                   progress=bars.reporter(label),
                                   stop=stop): label
                       for label, provider, opts in runs}
            # `result()` re-raises what a provider raised. Leaving the loop still drains the
            # pool, so the other vendors finish and their predictions are on disk.
            try:
                for future in cf.as_completed(futures):
                    future.result()
            except KeyboardInterrupt:
                # THE INTERRUPT LANDS HERE, not in `predict_all`: Python delivers it to the
                # main thread, and the providers are in workers where a handler can never see
                # it. Without this the pool's `__exit__` waits for every provider to work
                # through the whole corpus -- a Ctrl-C that does nothing.
                #
                # Both halves are needed. `cancel_futures` drops providers that have not
                # started; only the flag reaches the ones that have, because a running thread
                # cannot be cancelled in Python -- stopping it is always cooperative.
                stop.set()
                pool.shutdown(wait=False, cancel_futures=True)
                log.warning("interrupted -- finishing the calls already in flight; "
                            "documents not yet written come back on the next run")
                raise

    # KEYED BY THE RUN, NOT BY THE VENDOR, and it is the same string that named the directory.
    # A measured configuration is (provider, what it was steered with); keyed by the vendor
    # alone, a steered run and a stock one are one row, and the second silently replaces the
    # first. `provider` and `settings` ride along in the row, so a published table can be
    # labelled however its author likes without parsing the key back apart.
    summary = {}
    for label, provider, opts in runs:
        head = {"run": label, "provider": provider, "settings": settings_for(provider, opts)}
        mine = score_all(docs, provider, dirs[label], verdicts=verdicts,
                         workers=score_workers, rescore=rescore)
        summary[label] = {**head, **summarise(mine)}
        # WRITTEN AS SOON AS THIS RUN IS GRADED, so a run that finished is readable whether or
        # not the ones after it do. Grading is serial, and holding every summary until the last
        # vendor is how an invocation that stops half way leaves directories of answers and
        # nothing that reads them.
        #
        # AND IT DESCRIBES THE DIRECTORY, NOT THIS INVOCATION -- `summarise` over every row in
        # `scores.jsonl`, not over the selection. `runs/*/summary.json` is what a reader
        # aggregates, and what it says must not depend on how the last run happened to be
        # limited: `--limit 2` against a graded 620 would leave a two-document summary sitting
        # on a full run, and a table built from those is wrong by a factor of 300. The RETURN
        # value answers the other question -- what did I just run -- so it stays the selection.
        write_json(dirs[label] / "summary.json",
                   {**head, **summarise(list(read_scores(dirs[label]).values()))}, indent=2)
        s = summary[label]
        log.info("%s: unified %.4f over %d suites, coverage %d/%d",
                 label, s["unified"], len(s["per_suite"]), s["scored"], s["documents"])

    return summary
