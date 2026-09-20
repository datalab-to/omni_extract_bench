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


class Leg(NamedTuple):
    """One run: what it asks, of whom, where it writes, which bar it advances.

    `label` is `out_name(provider, options)` -- the string that already names the directory,
    the summary row and the progress line. It is a FIELD rather than implied by the call
    because one pool now drains several runs, so every task has to say which one it came from
    on the way back.

    `provider` is here for the same reason and is NOT the group: legs are grouped by adapter,
    and `openai/gpt-5.6-sol` and `anthropic/claude-opus-5` share `llm_single_shot` while being
    different models. The group decides the pool; the leg decides what gets asked.
    """

    label: str
    provider: str
    options: dict
    out: Path
    progress: object = NULL


def predict_provider(legs: list[Leg], docs: list[Doc], adapter: str, timeout: float,
                     workers: int, stop: threading.Event | None = None) -> None:
    """Every document of every run on ONE adapter, through one pool sized to that service.

        <leg.out>/predictions/<doc_id>.json    the bare extraction -- what scoring reads
        <leg.out>/records/<doc_id>.json        the schema sent, the cost, the manifest

    Two files because they are read at different times and are different sizes: scoring wants
    the answer, an audit wants everything, and only one of them is worth loading 620 of.

    ONE POOL PER VENDOR, NOT PER RUN. The cap is a fact about the vendor -- how many documents
    it will hold at once -- so the pool that enforces it has to be the vendor's too. A pool per
    run put `WORKERS["datalab"] = 10` in flight twice over when one invocation measured two
    datalab tiers, and dividing the cap between the runs instead fixes the count while leaving
    it static: the tier that finishes first hands its half back to nobody. One queue drains
    across every run of the vendor, so the whole cap is always in use and no run can exceed it.

    It also makes `mine` a fact about the right thing. A credit ceiling or an unset key is true
    of the ACCOUNT, so it stops every run of that vendor rather than each discovering it in
    turn, a wave of documents apart.
    """
    by_label = {leg.label: leg for leg in legs}
    work: list[tuple[Leg, Doc]] = []
    todo_of: dict[str, int] = {}
    seen = {leg.label: {"tally": collections.Counter(), "spend": [], "billed": [], "took": []}
            for leg in legs}
    for leg in legs:
        (leg.out / "predictions").mkdir(parents=True, exist_ok=True)
        (leg.out / "records").mkdir(parents=True, exist_ok=True)
        todo = [d for d in docs if needs_run(leg.out / "records" / f"{d.doc_id}.json")]
        log.info("%s: %d documents, %d to run", leg.label, len(docs), len(todo))
        # The denominator is the VENDOR's cap, which every run of it shares. Two legs both
        # reading `/10` are reading the same ten.
        leg.progress.start(len(todo), workers)
        todo_of[leg.label] = len(todo)
        work += [(leg, doc) for doc in todo]
        if not todo:
            leg.progress.finish()

    left = dict(todo_of)               # documents still outstanding, per run
    mine = threading.Event()
    stopping = lambda: mine.is_set() or (stop is not None and stop.is_set())   # noqa: E731
    fatal: list[Exception] = []

    # One place a vendor gives up, so the three ways cannot drift. Locked because
    # check-then-set is not atomic: two workers failing together both passed the test.
    halt = threading.Lock()

    def give_up(leg: Leg, exc: Exception, why: str = "") -> tuple:
        """Stop this vendor, remember why, and report this document as never attempted."""
        with halt:
            if not mine.is_set():
                if why:
                    log.error("%s: %s -- %s", leg.label, why, exc)
                fatal.append(exc)
                mine.set()
        return leg.label, "skipped", None, None, None

    def run_one(leg: Leg, doc: Doc) -> tuple:
        if stopping():
            return leg.label, "skipped", None, None, None
        try:
            # The block is exactly the call, so the in-flight count is what the vendor holds,
            # not what our pool has queued.
            with leg.progress.calling():
                record = predict(leg.provider, doc.pdf, doc.schema, timeout=timeout,
                                 **leg.options)
            # ORDER IS THE COMMIT: the prediction lands first, the record is the marker
            # `needs_run` reads. Killed between them the document looks unattempted and is run
            # again -- the cheap mistake.
            write_json(leg.out / "predictions" / f"{doc.doc_id}.json", record["result"])
            write_json(leg.out / "records" / f"{doc.doc_id}.json", record)
        except (MissingDependency, MissingCredential) as exc:
            # Ours, identical for every document, not transient.
            return give_up(leg, exc)
        except AccountFailure as exc:
            # RAISED, not merely logged: its skipped documents have no prediction, so the
            # summary would report a vendor that was never asked as one that could not answer.
            return give_up(leg, exc, "STOPPED, account-level failure")
        except Exception as exc:          # noqa: BLE001 -- unknown, so assume it is ours
            # Anything `predict` did not turn into a record, or anything the writes raise. It
            # will repeat, so stop rather than buy the same failure 600 more times.
            return give_up(leg, exc)
        cost = record.get("cost") or {}
        return (leg.label, "error" if record.get("error") else "ok",
                cost.get("usd"), cost.get("credits"), cost.get("wall_s"))

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, leg, doc) for leg, doc in work]
        try:
            for future in cf.as_completed(futures):
                label, status, usd, credits, wall_s = future.result()
                got = seen[label]
                got["tally"][status] += 1
                for key, value in (("spend", usd), ("billed", credits), ("took", wall_s)):
                    if value is not None:
                        got[key].append(value)
                # A skipped document was never reached, so it is not progress: it would fill
                # the bar on the way out of a run that failed on its first document.
                if status != "skipped":
                    by_label[label].progress.record(error=status == "error", usd=usd,
                                                    credits=credits, wall_s=wall_s)
                # WHEN THIS RUN'S LAST DOCUMENT LANDS, not when the vendor's queue empties.
                # One queue serves every run of a vendor, so finishing them together reported
                # the whole vendor's wall time on each line -- two reducto tiers both "done in
                # 4m44s" while one of them had been finished for minutes.
                left[label] -= 1
                if not left[label]:
                    by_label[label].progress.finish()
        except KeyboardInterrupt:
            mine.set()
            for future in futures:
                future.cancel()
            log.warning("%s: interrupted -- %s; documents not yet written come back on the "
                        "next run", adapter,
                        {k: dict(v["tally"]) for k, v in seen.items()})
            raise

    for leg in legs:
        got, n = seen[leg.label], todo_of[leg.label]
        # `len(spend)` is reported because several vendors price only some documents: a bare
        # total over a corpus where half reported nothing reads as the bill and is a fraction.
        if got["took"]:
            # The mean AND the range: 4s-and-1800s is a different proposition from a steady
            # 900s.
            log.info("%s: %.1fs per document on average (min %.1f, max %.1f over %d)",
                     leg.label, sum(got["took"]) / len(got["took"]), min(got["took"]),
                     max(got["took"]), len(got["took"]))
        if got["spend"]:
            log.info("%s: $%.2f reported over %d of %d documents", leg.label,
                     sum(got["spend"]), len(got["spend"]), n)
        if got["billed"]:
            # Credits stay in the vendor's own unit; the rate is contract-specific.
            log.info("%s: %g credits reported over %d of %d documents", leg.label,
                     sum(got["billed"]), len(got["billed"]), n)
        if n and not got["spend"] and not got["billed"]:
            log.info("%s: no per-document cost reported (billed out of band)", leg.label)

    # After the pool drains, so no worker is still writing. The documents it stopped keep no
    # file, so they stay resumable.
    if fatal:
        raise fatal[0]


def predict_all(docs: list[Doc], provider: str, out: Path, timeout: float, workers: int,
                options: dict | None = None, progress=NULL, label: str | None = None,
                stop: threading.Event | None = None) -> None:
    """One run of one vendor: `predict_provider` with a single leg."""
    predict_provider([Leg(label or provider, provider, options or {}, out, progress)],
                     docs, provider, timeout, workers, stop)


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
    if not (isinstance(pred, dict) and pred and "__error__" not in pred):
        reason = (pred or {}).get("__error__") if isinstance(pred, dict) else "not an object"
        return {**row, "status": "error", "error": str(reason or "empty prediction")}
    try:
        result = score(pred, json.loads(doc.gt.read_text()), doc.schema, verdicts=verdicts)
        if verdicts:
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
                    futures = {pool.submit(grade, doc): doc.doc_id for doc in remaining}
                    for future in cf.as_completed(futures):
                        graded[futures[future]] = future.result()
                remaining = []
            except BrokenProcessPool:
                if graded:
                    raise
                log.warning("scoring pool stopped before any document was graded, so grading "
                            "continues in this process. If that was a Ctrl-C, press it again; "
                            "if it was not, put the call under `if __name__ == \"__main__\":` "
                            "or pass score_workers=1.")
        for doc in remaining:
            graded[doc.doc_id] = grade(doc)
    finally:
        write_scores(out, so_far())
        if len(graded) < len(wanted):
            log.warning("scoring stopped after %d of %d documents; the rest are graded on "
                        "the next run", len(graded), len(wanted))
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
#: The five ways an address can go wrong. `matched` is the sixth bucket and the only good one;
#: together they partition every address (`METRIC_SPEC.md` section 4). Reported as RATES here,
#: and named `_rate` for it: `scores.jsonl` spells these same five words as counts, and a
#: reader moving between the two files should not have to work out which one they are holding.
ERRORS = ("misread", "unfound", "fabricated", "invented_item", "invented_field")


def over(rows: list[dict]) -> dict:
    """How a set of documents went: coverage, the three metrics, and where the errors were.

    A MEAN OVER DOCUMENTS. Every document counts once, however many fields it holds -- the
    corpus runs from 26 addresses to 35,239, and a ratio of sums would let that one document
    decide the number for all of them (`METRIC_SPEC.md` section 7 -- a value's weight is
    inversely proportional to the size of the document holding it).

    OVER THE DOCUMENTS THAT SCORED, all of it, so one denominator holds for the whole block.
    `precision` is why: a document with no usable prediction asserted nothing, and scoring its
    `matched / asserted` zero would say everything it claimed was wrong. What the failures cost
    is `coverage`, right beside these -- and `accuracy * coverage` is the mean over every
    document with the failures counted as zero, if that is the number you want.
    """
    scored = [r for r in rows if r["status"] == "scored"]

    def mean(of) -> float:
        return sum(of(r) for r in scored) / len(scored) if scored else 0.0

    return {
        "documents": len(rows),
        "scored": len(scored),
        "coverage": len(scored) / len(rows) if rows else 0.0,
        "accuracy": mean(lambda r: r["accuracy"]),
        "precision": mean(lambda r: r["precision"]),
        "recall": mean(lambda r: r["recall"]),
        **{f"{e}_rate": mean(lambda r, e=e: r[e] / r["total"] if r["total"] else 0.0)
           for e in ERRORS},
    }


def summarise(rows: list[dict]) -> dict:
    """The run, and then each suite, in the same shape.

    NO CORPUS NUMBER IS PICKED FOR YOU. Weighting the suites equally rather than by size is a
    real choice (`METRIC_SPEC.md` section 7) and this does not make it: `per_suite` carries the
    same block per suite, so the suite-weighted figure is a mean of four numbers you can take
    yourself, for any of these and not just accuracy.

        unified = mean(s["accuracy"] for s in summary["per_suite"].values())

    The suites are not close to the same size -- 329, 202, 47, 42 -- so the top-level figures
    and that mean are different numbers, and which one belongs in a table is the table author's
    to say rather than this function's.
    """
    by_suite: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_suite[row["suite"]].append(row)
    return {**over(rows), "per_suite": {s: over(v) for s, v in sorted(by_suite.items())}}


# ── the run ──────────────────────────────────────────────────────────────────────────────
#: For an adapter `harness.WORKERS` says nothing about -- a new one, before anyone has looked.
DEFAULT_WORKERS = 5


def workers_for(providers, requested: dict[str, int] | int | None) -> int:
    """How many documents to have in flight at one vendor.

    A worker holds its document through the whole poll loop, so this IS the number of jobs in
    flight server-side, not a request rate -- at five-second polls, ten in flight is two
    requests a second.

    `providers` is every provider NAME sharing one adapter, because the pool is the adapter's:
    `openai/gpt-5.6-sol` and `anthropic/claude-opus-5` are two names for one OpenRouter key.
    `--predict-workers` still takes the names people type, so where several of them name
    different numbers the SMALLEST wins -- a cap is a ceiling somebody asked for, and honouring
    the lowest cannot exceed any of them.

    A named provider beats a bare number, which beats the adapter's own limit.
    """
    from .harness.vendor import resolve

    names = [providers] if isinstance(providers, str) else list(providers)
    if isinstance(requested, int):
        requested = {"*": requested} if requested else {}
    requested = requested or {}
    asked = [requested[n] for n in names if requested.get(n)]
    return (min(asked) if asked else
            requested.get("*") or WORKERS.get(resolve(names[0]), DEFAULT_WORKERS))


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

    `predict_workers` is `{provider: count}` (see `workers_for`) -- the names people type, of
    which several can share one budget: every `org/model` id is one OpenRouter key. It is
    separate from `score_workers` because they buy different resources: documents a vendor
    holds at once, against local cores that grade.
    """
    from .harness.vendor import resolve, settings_for

    for provider in providers:
        resolve(provider)          # raises ValueError naming the vendors, before any download
    runs = plan(providers, options)

    root = fetch(data_root)
    docs = read_manifest(root / MANIFEST, root, suites=suites, limit=limit)
    if not docs:
        raise ValueError("no documents selected: check --suites and --limit")

    # The directory is `<provider>-<digest of the settings>`
    dirs = {label: out / label for label, _, _ in runs}
    for label, provider, opts in runs:
        dirs[label].mkdir(parents=True, exist_ok=True)
        write_json(dirs[label] / "settings.json",
                   {"provider": provider, "settings": settings_for(provider, opts),
                    "timeout_s": timeout}, indent=2)


    stop = threading.Event()           # one Ctrl-C stops every provider, not just one
    if not score_only:
        with Progress([label for label, _, _ in runs]) as bars:
            # GROUPED BY VENDOR, because that is what the concurrency cap is about. Two
            # datalab tiers are two runs and one vendor: they get a bar each, their own
            # directories and their own rows, and share the ten documents datalab will hold.
            # KEYED BY ADAPTER, not by the name typed: every `org/model` id is one
            # `llm_single_shot` against one OpenRouter key, so three of them named separately
            # are still one budget.
            legs: dict[str, list[Leg]] = collections.defaultdict(list)
            for label, provider, opts in runs:
                legs[resolve(provider)].append(
                    Leg(label, provider, opts, dirs[label], bars.reporter(label)))
            with cf.ThreadPoolExecutor(max_workers=len(legs)) as pool:
                futures = {pool.submit(predict_provider, group, docs, adapter, timeout,
                                       workers_for([leg.provider for leg in group],
                                                   predict_workers),
                                       stop): adapter
                           for adapter, group in legs.items()}
                # `result()` re-raises what a provider raised. Leaving the loop still drains
                # the pool, so the other vendors finish and their predictions are on disk.
                try:
                    for future in cf.as_completed(futures):
                        future.result()
                except KeyboardInterrupt:
                    # THE INTERRUPT LANDS HERE, not in `predict_provider`: Python delivers it
                    # to the main thread, and the vendors are in workers where a handler can
                    # never see it. Both halves are needed -- `cancel_futures` drops vendors
                    # that have not started, and only the flag reaches the ones that have,
                    # because a running thread cannot be cancelled in Python.
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
        log.info("%s: accuracy %.4f over %d suites, coverage %d/%d",
                 label, s["accuracy"], len(s["per_suite"]), s["scored"], s["documents"])

    return summary
