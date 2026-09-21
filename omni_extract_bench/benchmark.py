#!/usr/bin/env python3
"""The whole benchmark: fetch the corpus, predict with each vendor, score, write it down.

    oeb benchmark --providers datalab reducto --out runs/
    oeb benchmark --providers datalab --limit 5              # smoke test

Three levels of abstraction == three classes.

    BenchmarkRun   the invocation      every Run, grouped by adapter, predicted then graded
    ProviderRun    one adapter         every Run on it, through one pool sized to that service
    Run            one configuration   a provider plus its options 

    >>> print(BenchmarkRun(["datalab"],
    ...                    options={"datalab": [{"mode": "balanced"},
    ...                                         {"mode": "accurate"}]}).describe())
    benchmark: 2 run(s) over 1 adapter(s), 1800s per document -> runs
      datalab            2 run(s), 10 document(s) at a time
          datalab-f46415c9                     mode=balanced
          datalab-01a72762                     mode=accurate

That costs nothing -- no download, no vendor call -- so the plan can be read before the money
is spent, and `go()` logs the same lines on its way in so a run that did spend says what it
bought.

Resumable and idempotent. A document is predicted again only when it has no record,
and graded again only when it has no row in `scores.jsonl` -- so an interrupted run carries on
instead of paying twice. Nothing checks the metric has not changed under a resume; `--rescore`
is how to say it has.

This is the orchestration around primitives `score` and `grade`. `score` grades one document and
`harness.predict` produces one prediction.

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
    docs.sort(key=lambda d: (d.suite, d.doc_id))
    return docs[:limit] if limit else docs


def write_json_atomic(path: Path, obj, *, indent: int | None = None) -> None:
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(obj, default=str, indent=indent))
    os.replace(tmp, path)


def needs_run(record_path: Path) -> bool:
    """True when this document has not been attempted yet."""
    if not record_path.exists():
        return True
    try:
        json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    return False


class Run:
    """One configuration this benchmark measures: a provider plus its options.

    THE UNIT EVERYTHING IS KEYED BY. `label` is `out_name(provider, options)`, and that one
    string names the directory, the summary row, the progress line and the log lines -- so two
    tiers of one vendor cannot be mistaken for each other anywhere.

    `provider` is separate from the grouping: Runs are grouped by ADAPTER, and
    `openai/gpt-5.6-sol` and `anthropic/claude-opus-5` share `llm_single_shot` while being
    different models. The group decides the pool; the Run decides what gets asked.

    MUTABLE, BUT ONLY ON THE MAIN THREAD. Workers READ a Run -- its provider, its options, its
    paths -- and never touch the counters, which `ProviderRun` folds in from its own
    `as_completed` loop. So none of this needs a lock, and nothing here may start being written
    from a worker without one.
    """

    def __init__(self, label: str, provider: str, options: dict, out: Path, progress=NULL):
        self.label, self.provider, self.options = label, provider, options
        self.out, self.progress = out, progress
        self.todo = self.left = 0
        self.counts: collections.Counter[str] = collections.Counter()
        self.spend: list[float] = []
        self.billed: list[float] = []
        self.took: list[float] = []

    def __repr__(self) -> str:
        return f"Run({self.label})"

    def settings(self) -> dict:
        """The whole resolved configuration, not just what the caller passed."""
        from .harness.vendor import settings_for

        return settings_for(self.provider, self.options)

    def head(self) -> dict:
        """What every file this Run writes says about itself, before any numbers."""
        return {"run": self.label, "provider": self.provider, "settings": self.settings()}

    def outstanding(self, docs: list[Doc], verdicts: bool = False,
                    rescore: bool = False) -> tuple[int, int]:
        """How many of `docs` this Run would predict, and how many it would then grade.

        What a resume COSTS, which is the question worth answering before one starts: the two
        halves are independent, and a Run with every prediction on disk can still owe 620
        grades after a `--rescore`.
        """
        to_predict = sum(1 for doc in docs if self.needs(doc))
        _, wanted = grading_split(self.out, docs, verdicts=verdicts, rescore=rescore)
        return to_predict, len(wanted)

    def score(self, docs: list[Doc], verdicts: bool = False, workers: int = 0,
              rescore: bool = False) -> list[dict]:
        """Grade this Run's predictions, and write one line per document.

        The other half of `outstanding`, on the same flags: that one says how many this would
        grade, this one grades them.

        `self` NEVER REACHES THE PROCESS POOL. `score_one` is a module function handed the
        provider and the path as plain values, because a Run carrying a live progress reporter
        cannot be pickled and the pool would die on the first submit.

        PROCESSES, not threads: this is the one CPU-bound half of a run, and the matcher's
        recursion is interpreted Python that would serialise on the GIL.

        RESUMABLE PER DOCUMENT -- a document already in `scores.jsonl` is not graded again;
        `rescore=True` grades everything regardless. The file is in document order whatever
        order the workers finish in, so a run stays comparable with the one before it.
        """
        known, wanted = grading_split(self.out, docs, verdicts=verdicts, rescore=rescore)
        if len(wanted) < len(docs):
            log.info("%s: %d of %d documents already scored, %d to grade",
                     self.label, len(docs) - len(wanted), len(docs), len(wanted))

        grade = functools.partial(score_one, provider=self.provider, out=self.out,
                                  verdicts=verdicts)
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
                    log.warning("scoring pool stopped before any document was graded, so "
                                "grading continues in this process. If that was a Ctrl-C, "
                                "press it again; if it was not, put the call under "
                                "`if __name__ == \"__main__\":` or pass score_workers=1.")
            for doc in remaining:
                graded[doc.doc_id] = grade(doc)
        finally:
            write_scores(self.out, so_far())
            if len(graded) < len(wanted):
                log.warning("scoring stopped after %d of %d documents; the rest are graded "
                            "on the next run", len(graded), len(wanted))

        answered = {**known, **graded}
        return [answered[d.doc_id] for d in docs if d.doc_id in answered]

    def describe(self, work: tuple[int, int] | None = None) -> str:
        """One line: what this Run asks, and what it still owes when `work` is known.

        The provider is not a column of its own because `label` already carries it -- it is
        `out_name(provider, options)`, so `openai__gpt-5.6-sol-28890c89` names its model.
        """
        asked = ", ".join(f"{k}={v}" for k, v in sorted(self.options.items())) or "stock"
        line = f"{self.label:<36} {asked:<22}"
        if work is None:
            return line.rstrip()
        return f"{line} {work[0]:>5} to predict, {work[1]:>5} to grade"

    @property
    def predictions(self) -> Path:
        """The bare extractions. What scoring reads, and all it reads."""
        return self.out / "predictions"

    @property
    def records(self) -> Path:
        """The schema sent, the cost, the manifest. What an audit reads."""
        return self.out / "records"

    def needs(self, doc: Doc) -> bool:
        """True when this Run has not attempted this document yet."""
        return needs_run(self.records / f"{doc.doc_id}.json")

    def store(self, doc: Doc, record: dict) -> None:
        """Write one answer down. THE ORDER IS THE COMMIT.

        The prediction lands first and the record is the marker `needs` reads, so a process
        killed between them leaves a document that looks unattempted and is run again -- the
        cheap mistake. The other order buys the expensive one: a record with no answer beside
        it, which no resume ever revisits.
        """
        write_json_atomic(self.predictions / f"{doc.doc_id}.json", record["result"])
        write_json_atomic(self.records / f"{doc.doc_id}.json", record)

    def begin(self, docs: list[Doc], workers: int) -> list[Doc]:
        """Make room, work out what is left to do, and open the progress line for it."""
        self.predictions.mkdir(parents=True, exist_ok=True)
        self.records.mkdir(parents=True, exist_ok=True)
        todo = [doc for doc in docs if self.needs(doc)]
        self.todo = self.left = len(todo)
        log.info("%s: %d documents, %d to run", self.label, len(docs), len(todo))
        self.progress.start(len(todo), workers)
        if not todo:
            self.progress.finish()
        return todo

    def landed(self, status: str, usd, credits, wall_s) -> None:
        """One document came back, well or badly."""
        self.counts[status] += 1
        for bucket, value in ((self.spend, usd), (self.billed, credits), (self.took, wall_s)):
            if value is not None:
                bucket.append(value)
        if status != "skipped":
            self.progress.record(error=status == "error", usd=usd, credits=credits,
                                 wall_s=wall_s)
        self.left -= 1
        if not self.left:
            self.progress.finish()

    def report(self) -> None:
        """This Run's last word, for a log with no bar to look at."""
        if self.took:
            log.info("%s: %.1fs per document on average (min %.1f, max %.1f over %d)",
                     self.label, sum(self.took) / len(self.took), min(self.took),
                     max(self.took), len(self.took))
        if self.spend:
            log.info("%s: $%.2f reported over %d of %d documents", self.label,
                     sum(self.spend), len(self.spend), self.todo)
        if self.billed:
            log.info("%s: %g credits reported over %d of %d documents", self.label,
                     sum(self.billed), len(self.billed), self.todo)
        if self.todo and not self.spend and not self.billed:
            log.info("%s: no per-document cost reported (billed out of band)", self.label)


class ProviderRun:
    """Every Run on one adapter, through one pool sized to that service.

    ONE POOL PER ADAPTER, NOT PER RUN. The cap is a fact about the vendor -- how many documents
    it will hold at once -- so the pool that enforces it has to be the vendor's too. A pool per
    Run put `WORKERS["datalab"] = 10` in flight twice over when one invocation measured two
    datalab tiers, and dividing the cap between them fixes the count while leaving it static:
    the tier that finishes first hands its half back to nobody. One queue drains across every
    Run of the adapter, so the whole cap is always in use and no Run can exceed it.

    THE ADAPTER, NOT THE PROVIDER NAME, and for six of seven vendors those are the same thing.
    They part company for model ids: `openai/gpt-5.6-sol` and `anthropic/claude-opus-5` are two
    names for one `llm_single_shot`, one OpenRouter endpoint and one key.
    """

    def __init__(self, adapter: str, runs: list[Run], workers: int):
        self.adapter, self.runs, self.workers = adapter, runs, workers
        self.mine = threading.Event()
        self.fatal: list[Exception] = []
        self.halt = threading.Lock()

    def __repr__(self) -> str:
        return f"ProviderRun({self.adapter}, {len(self.runs)} runs, {self.workers} at a time)"

    def describe(self, work: dict[str, tuple[int, int]] | None = None) -> str:
        """This adapter and the Runs under it, indented beneath a line naming the cap."""
        head = (f"  {self.adapter:<18} {len(self.runs)} run(s), "
                f"{self.workers} document(s) at a time")
        return "\n".join([head] + [f"      {r.describe((work or {}).get(r.label))}"
                                   for r in self.runs])

    def _give_up(self, run: Run, exc: Exception, why: str = "") -> tuple:
        """Stop this vendor, remember why, and report this document as never attempted."""
        with self.halt:
            if not self.mine.is_set():
                if why:
                    log.error("%s: %s -- %s", run.label, why, exc)
                self.fatal.append(exc)
                self.mine.set()
        return run.label, "skipped", None, None, None

    def _predict_one(self, run: Run, doc: Doc, timeout: float, stopping) -> tuple:
        if stopping():
            return run.label, "skipped", None, None, None
        try:
            with run.progress.calling():
                record = predict(run.provider, doc.pdf, doc.schema, timeout=timeout,
                                 **run.options)
            run.store(doc, record)
        except (MissingDependency, MissingCredential) as exc:
            return self._give_up(run, exc)
        except AccountFailure as exc:
            return self._give_up(run, exc, "STOPPED, account-level failure")
        except Exception as exc:          # noqa: BLE001 -- unknown, so assume it is ours
            return self._give_up(run, exc)
        cost = record.get("cost") or {}
        return (run.label, "error" if record.get("error") else "ok",
                cost.get("usd"), cost.get("credits"), cost.get("wall_s"))

    def predict(self, docs: list[Doc], timeout: float,
                stop: threading.Event | None = None) -> None:
        """Every document of every Run here, and raise whatever stopped the vendor."""
        by_label = {run.label: run for run in self.runs}
        work = [(run, doc) for run in self.runs for doc in run.begin(docs, self.workers)]
        stopping = lambda: self.mine.is_set() or (stop is not None and stop.is_set())  # noqa: E731

        with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._predict_one, run, doc, timeout, stopping)
                       for run, doc in work]
            try:
                for future in cf.as_completed(futures):
                    label, *outcome = future.result()
                    by_label[label].landed(*outcome)
            except KeyboardInterrupt:
                self.mine.set()
                for future in futures:
                    future.cancel()
                log.warning("%s: interrupted -- %s; documents not yet written come back on "
                            "the next run", self.adapter,
                            {r.label: dict(r.counts) for r in self.runs})
                raise

        for run in self.runs:
            run.report()

        if self.fatal:
            raise self.fatal[0]


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


def grading_split(out: Path, docs: list[Doc], verdicts: bool = False,
                  rescore: bool = False) -> tuple[dict[str, dict], list[Doc]]:
    """The grades this selection may reuse, and the documents still to grade.

    ONE DEFINITION, because `describe` promises what a resume will cost and `score_all` is
    what it then costs. Two copies of this rule would drift, and the promise is the half that
    would be wrong -- a plan that says 8 documents and then grades 620.
    """
    known = read_scores(out, verdicts=verdicts)
    if rescore:
        selected = {d.doc_id for d in docs}
        known = {k: v for k, v in known.items() if k not in selected}
    return known, [d for d in docs if d.doc_id not in known]


def write_scores(out: Path, rows: list[dict]) -> None:
    """One line per document, written wherever scoring stops rather than only where it ends.

    ATOMIC because it is READ BACK: an interrupt landing mid-write would truncate the very
    file it was saving, and `read_scores` would silently drop everything past the cut.
    """
    path = out / "scores.jsonl"
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))
    os.replace(tmp, path)


ERRORS = ("misread", "unfound", "fabricated", "invented_item", "invented_field")


def over(rows: list[dict]) -> dict:
    """How a set of documents went: coverage, the three metrics, and where the errors were."""
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
    """The run, and then each suite, in the same shape."""
    by_suite: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_suite[row["suite"]].append(row)
    return {**over(rows), "per_suite": {s: over(v) for s, v in sorted(by_suite.items())}}


DEFAULT_WORKERS = 5


def workers_for(providers, requested: dict[str, int] | int | None) -> int:
    """How many documents to have in flight at one vendor."""
    from .harness.vendor import resolve

    names = [providers] if isinstance(providers, str) else list(providers)
    if isinstance(requested, int):
        requested = {"*": requested} if requested else {}
    requested = requested or {}
    asked = [requested[n] for n in names if requested.get(n)]
    return (min(asked) if asked else
            requested.get("*") or WORKERS.get(resolve(names[0]), DEFAULT_WORKERS))


def plan(providers: list[str], options: dict | None = None,
         out: Path = Path("runs")) -> list[Run]:
    """A `Run` for every configuration this invocation measures.

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

    unknown = sorted(set(options or {}) - set(providers))
    if unknown:
        raise ValueError(
            f"--options names {unknown[0]!r}, which is not in --providers "
            f"({', '.join(providers)}). Its options would be silently ignored.")

    runs: dict[str, Run] = {}
    for provider in providers:
        asked = (options or {}).get(provider)
        if isinstance(asked, list) and not asked:
            raise ValueError(f"--options gives {provider!r} an empty list, so it would not "
                             f"run at all. Give it options, or leave it out.")
        for one in (asked if isinstance(asked, list) else [asked]):
            label = out_name(provider, one)
            runs.setdefault(label, Run(label, provider, one or {}, out / label))
    return list(runs.values())


class BenchmarkRun:
    """One invocation: every Run, grouped by adapter, predicted then graded then written down.

        <out>/<provider>-<digest>/
            settings.json     what this Run asked, before it asked it
            summary.json      what it came to, as soon as it is graded
            predictions/<doc_id>.json    the bare extraction -- what scoring reads
            records/<doc_id>.json        the schema sent, the cost, the manifest
            scores.jsonl                 one graded row per document

    A Run directory is self-contained, and there is no table across them:
    `runs/*/summary.json` is one, aggregated however its reader likes, and a file here would
    only be one opinion about that written down -- stale the moment another Run lands beside it.

    THE PLAN IS AVAILABLE BEFORE ANYTHING IS SPENT. Construct one and call `describe()`: it
    validates the providers and the options, works out the Runs and their caps, and says so --
    all without a download or a vendor call. `go()` logs the same thing on its way in, so a run
    that did cost money says what it bought.
    """

    def __init__(self, providers: list[str], *, out: Path = Path("runs"),
                 data_root: Path = Path("benchmark"), suites: list[str] | None = None,
                 limit: int = 0, timeout: float = 1800.0,
                 predict_workers: dict[str, int] | int | None = None, score_workers: int = 0,
                 verdicts: bool = False, rescore: bool = False, score_only: bool = False,
                 options: dict | None = None):
        from .harness.vendor import resolve

        for provider in providers:
            resolve(provider)
        self.out, self.data_root, self.suites, self.limit = out, data_root, suites, limit
        self.timeout, self.score_workers = timeout, score_workers
        self.verdicts, self.rescore, self.score_only = verdicts, rescore, score_only
        self.runs = plan(providers, options, out)
        grouped: dict[str, list[Run]] = collections.defaultdict(list)
        for run in self.runs:
            grouped[resolve(run.provider)].append(run)
        self.providers = [ProviderRun(adapter, group,
                                      workers_for([r.provider for r in group], predict_workers))
                          for adapter, group in grouped.items()]

    def __repr__(self) -> str:
        return (f"BenchmarkRun({len(self.runs)} runs over "
                f"{len(self.providers)} adapters -> {self.out})")

    def describe(self, docs: list[Doc] | None = None) -> str:
        """The whole plan. With `docs`, what each Run still owes; without, what it would ask.

        Both are useful and at different moments: before the corpus is on disk there is no
        such thing as "612 to predict", and once it is, that is the only number worth reading.
        """
        work = None if docs is None else {
            r.label: r.outstanding(docs, verdicts=self.verdicts, rescore=self.rescore)
            for r in self.runs}
        scope = [f"{len(self.runs)} run(s) over {len(self.providers)} adapter(s)",
                 f"{self.timeout:.0f}s per document"]
        if self.suites:
            scope.append(f"suites {', '.join(self.suites)}")
        if self.limit:
            scope.append(f"first {self.limit} documents")
        if self.score_only:
            scope.append("SCORE ONLY, no vendor is called")
        if self.rescore:
            scope.append("RESCORING, grades on disk ignored")
        if docs is not None:
            scope.append(f"{len(docs)} documents selected")
        return "\n".join([f"benchmark: {', '.join(scope)} -> {self.out}"]
                          + [p.describe(work) for p in self.providers])

    def prepare(self) -> None:
        """Each Run's directory, and what it asked, written BEFORE it asks.

        `summary.json` is written after grading, so a Run interrupted or stopped by a credit
        ceiling never reaches one -- and its directory is a digest, which says nothing a person
        can read. This is where they read it.
        """
        for run in self.runs:
            run.out.mkdir(parents=True, exist_ok=True)
            write_json_atomic(run.out / "settings.json",
                              {**run.head(), "timeout_s": self.timeout}, indent=2)

    def predict(self, docs: list[Doc]) -> None:
        """Every adapter at once. Predicting is network wait, so they do not slow each other."""
        stop = threading.Event()
        with Progress([r.label for r in self.runs]) as bars:
            for run in self.runs:
                run.progress = bars.reporter(run.label)
            with cf.ThreadPoolExecutor(max_workers=len(self.providers)) as pool:
                futures = {pool.submit(p.predict, docs, self.timeout, stop): p.adapter
                           for p in self.providers}
                try:
                    for future in cf.as_completed(futures):
                        future.result()
                except KeyboardInterrupt:
                    stop.set()
                    pool.shutdown(wait=False, cancel_futures=True)
                    log.warning("interrupted -- finishing the calls already in flight; "
                                "documents not yet written come back on the next run")
                    raise

    def score(self, docs: list[Doc]) -> dict:
        """Grade every Run, serially, and publish each as soon as it is graded.

        SERIAL because grading saturates every core: two Runs at once would only contend, and
        letting it overlap a vendor call would put local CPU load inside a published latency.
        """
        summary = {}
        for run in self.runs:
            mine = run.score(docs, verdicts=self.verdicts, workers=self.score_workers,
                             rescore=self.rescore)
            summary[run.label] = {**run.head(), **summarise(mine)}
            write_json_atomic(run.out / "summary.json",
                              {**run.head(),
                               **summarise(list(read_scores(run.out).values()))}, indent=2)
            s = summary[run.label]
            log.info("%s: accuracy %.4f over %d suites, coverage %d/%d",
                     run.label, s["accuracy"], len(s["per_suite"]), s["scored"],
                     s["documents"])
        return summary

    def corpus(self) -> list[Doc]:
        """The documents this invocation selects, fetching them if they are not here yet."""
        root = fetch(self.data_root)
        docs = read_manifest(root / MANIFEST, root, suites=self.suites, limit=self.limit)
        if not docs:
            raise ValueError("no documents selected: check --suites and --limit")
        return docs

    def go(self, confirm=None) -> dict:
        """Fetch, predict, score, write it down. Returns the summary it also writes.

        `confirm` is handed the plan and decides whether to go on, which is how a command line
        asks before spending money. Returning False runs nothing and gives back `{}` -- and
        that is the only way `{}` comes back, since `plan` refuses to produce no runs at all.

        THE CALLBACK OWNS SHOWING IT. Given one, this does not log the plan: the caller is
        about to put it in front of somebody and two copies help nobody. Without one, it logs,
        because a run that spends money should say what it bought.
        """
        docs = self.corpus()
        plan = self.describe(docs)
        if confirm is None:
            for line in plan.splitlines():
                log.info("%s", line)
        elif not confirm(plan):
            log.warning("nothing run")
            return {}
        self.prepare()
        if not self.score_only:
            self.predict(docs)
        return self.score(docs)


def run(providers: list[str], *, confirm=None, **kwargs) -> dict:
    """Fetch, predict, score, write it down. Returns the summary it also writes to `out`.

    Keyword arguments and no argparse, so this stays callable from a notebook; it raises rather
    than exits, for the same reason. `BenchmarkRun(providers, **kwargs)` is the same thing with
    the plan available first -- `describe()` says what it would do, and costs nothing.

    Nothing here reads stdin. `confirm` is a callable the CLI supplies, so a library call is
    never the thing that blocks waiting for somebody to type y.
    """
    return BenchmarkRun(providers, **kwargs).go(confirm=confirm)
