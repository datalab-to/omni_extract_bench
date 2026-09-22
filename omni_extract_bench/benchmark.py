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
    benchmark
    out         runs
    runs        2 over 1 adapter
    corpus      huggingface datalab-to/omni_extract_bench
    timeout     1800s per document
    score only  false
    rescoring   false
    <BLANKLINE>
    ╭─────────┬─────────┬──────────────────┬─────────────────────────────────┬─────────┬───────╮
    │ adapter │ at once │ run              │ settings                        │ predict │ grade │
    ├─────────┼─────────┼──────────────────┼─────────────────────────────────┼─────────┼───────┤
    │ datalab │      10 │ datalab-f46415c9 │ base_url=https://www.datalab.to │         │       │
    │         │         │                  │ ─────────────────────────────── │         │       │
    │         │         │                  │ mode=balanced                   │         │       │
    │         │         │                  │ ─────────────────────────────── │         │       │
    │         │         │                  │ poll_interval=5.0               │         │       │
    │         │         │ datalab-01a72762 │ base_url=https://www.datalab.to │         │       │
    │         │         │                  │ ─────────────────────────────── │         │       │
    │         │         │                  │ mode=accurate                   │         │       │
    │         │         │                  │ ─────────────────────────────── │         │       │
    │         │         │                  │ poll_interval=5.0               │         │       │
    ╰─────────┴─────────┴──────────────────┴─────────────────────────────────┴─────────┴───────╯

That costs nothing -- no download, no vendor call -- and `execute()` logs it on the way in, so
that did spend says what it bought.

Resumable and idempotent: a document is predicted again only when it has no record, graded
again only when it has no row. Nothing notices a changed metric under a resume, which is what
`--rescore` is for.

Credentials come from the environment and everything else from `--options`, so
`run_manifest.settings` is a complete account of what each vendor was asked.

    pip install 'omni-extract-bench[benchmark,harness]'
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import io
import functools
import json
import logging
import os
import threading
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import NamedTuple

from . import score
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


REPO = "datalab-to/omni_extract_bench"
MANIFEST = "manifest.parquet"
#: What a manifest row says, and all it has to say. Named here because the error that reports
#: a missing one quotes them, and so does `--manifest`'s help. It matches `Doc`'s fields today
#: and is still written out rather than read off them: this is the shape of files already on
#: disk, so a field `Doc` grows later must not silently invalidate every manifest there is.
COLUMNS = ("doc_id", "suite", "doc_path", "gt_path", "schema")


class Doc(NamedTuple):
    """One benchmark element."""

    doc_id: str
    suite: str
    doc_path: Path
    gt_path: Path
    schema: dict


def fetch(root: Path | None = None, repo: str = REPO) -> Path:
    """The corpus on disk. With no `root` it lands in the HuggingFace cache, which is where it
    belongs: shared between every checkout and every working directory, so the second folder
    you run from resolves it for free.
    """
    try:
        from huggingface_hub import file_exists, snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is needed to fetch the benchmark corpus, and is not installed:\n"
            "    pip install 'omni-extract-bench[benchmark]'\n"
            "The scorer itself needs none of it -- `score` and `oeb score` work without."
        ) from None

    # Asked BEFORE the download, because a corpus is gigabytes and a dataset without a
    # manifest is not a corpus at all -- waiting for all of it to be told so is the wrong
    # order. A hub that will not answer (offline, a cached corpus) is not an answer of no:
    # the download runs, and `read_manifest` says it if the manifest really is missing.
    try:
        missing = not file_exists(repo, MANIFEST, repo_type="dataset")
    except OSError:
        missing = False
    if missing:
        raise ValueError(
            f"the HuggingFace dataset {repo} has no {MANIFEST} at its root, so there is "
            f"nothing to benchmark. It must follows the same structure as ours at "
            f"https://huggingface.co/datasets/{REPO}. Point --manifest at a parquet of your "
            f"own to benchmark documents that are already on disk."
        )

    log.info("fetching the corpus into %s", root or "the huggingface cache")
    return Path(snapshot_download(repo, repo_type="dataset",
                                  **({"local_dir": str(root)} if root else {})))


def read_manifest(path: Path, root: Path, suites=None, limit: int = 0) -> list[Doc]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is needed to read the benchmark manifest, and is not installed:\n"
            "    pip install 'omni-extract-bench[benchmark]'"
        ) from None
    if not path.exists():
        raise ValueError(
            f"there is no manifest at {path}. It is a parquet file with one row per "
            f"document, as {', '.join(COLUMNS)}; ours is at "
            f"https://huggingface.co/datasets/{REPO}, and a corpus of your own follows the "
            f"same convention."
        )
    table = pq.read_table(path)
    # Every column, once, before any row: a manifest short of one is short of it everywhere,
    # and read off a row it is a `KeyError` with a column name and no file in it.
    absent = [c for c in COLUMNS if c not in table.column_names]
    if absent:
        has = ", ".join(table.column_names) or "no columns"
        raise ValueError(f"{path} has no {', '.join(absent)} column"
                         f"{'s' if len(absent) > 1 else ''}. A manifest names one document "
                         f"per row, as {', '.join(COLUMNS)}; this one has {has}.")
    docs = []
    for n, row in enumerate(table.to_pylist()):
        if suites and row["suite"] not in suites:
            continue
        # A doc_id IS a filename -- `predictions/<doc_id>.json` -- and the key a resume reads.
        # Caught here, where the row can be named; caught downstream it is a FileNotFoundError
        # inside a worker thread, hours in.
        doc_id = row["doc_id"]
        if not doc_id or doc_id in (".", "..") or Path(doc_id).name != doc_id:
            raise ValueError(f"{path}: row {n} has doc_id {doc_id!r}, which is not a filename. "
                             f"It names this document's prediction, its record and its row in "
                             f"scores.jsonl, so it has to be one path component.")
        # Same reason: an unreadable schema is this row's, and says so here rather than as a
        # bare `Expecting value: line 1 column 1` over a corpus of thousands.
        try:
            schema = json.loads(row["schema"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: row {n} ({doc_id}) has a schema column that is not "
                             f"JSON: {exc}") from None
        docs.append(Doc(doc_id=doc_id,
                        suite=row["suite"],
                        doc_path=root / row["doc_path"],
                        gt_path=root / row["gt_path"],
                        schema=schema))
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
    """One configuration measured: a provider plus its options, and the counters it fills as it
    goes. Workers only ever read one, so none of it needs a lock."""

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
        from .harness.registry import settings_for

        return settings_for(self.provider, self.options)

    def head(self) -> dict:
        """What every file this Run writes says about itself, before any numbers."""
        return {"run": self.label, "provider": self.provider, "settings": self.settings()}

    def outstanding(self, docs: list[Doc], verdicts: bool = True,
                    rescore: bool = False) -> tuple[int, int]:
        """How many of `docs` this Run would predict, and then grade. The two are independent:
        every prediction can be on disk with every grade still owed."""
        to_predict = sum(1 for doc in docs if self.needs(doc))
        _, wanted = grading_split(self.out, docs, verdicts=verdicts, rescore=rescore)
        return to_predict, len(wanted)

    def score(self, docs: list[Doc], verdicts: bool = True, workers: int = 0,
              rescore: bool = False, progress=NULL) -> list[dict]:
        """Grade this Run's predictions, resumable per document, and write one line each. `self`
        never reaches the process pool: a Run holding a live reporter cannot be pickled."""
        known, wanted = grading_split(self.out, docs, verdicts=verdicts, rescore=rescore)
        if len(wanted) < len(docs):
            log.info("%s: %d of %d documents already scored, %d to grade",
                     self.label, len(docs) - len(wanted), len(docs), len(wanted))
        progress.start(len(wanted))

        grade = functools.partial(score_one, provider=self.provider, out=self.out,
                                  verdicts=verdicts)
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
                        futures = {pool.submit(grade, doc): doc.doc_id for doc in remaining}
                        for future in cf.as_completed(futures):
                            graded[futures[future]] = row = future.result()
                            progress.record(error=row["status"] != "scored")
                    remaining = []
                except BrokenProcessPool:
                    if graded:
                        raise
                    log.warning("scoring pool stopped before any document was graded, so "
                                "grading continues in this process. If that was a Ctrl-C, "
                                "press it again; if it was not, put the call under "
                                "`if __name__ == \"__main__\":` or pass score_workers=1.")
            for doc in remaining:
                graded[doc.doc_id] = row = grade(doc)
                progress.record(error=row["status"] != "scored")
        finally:
            progress.finish()
            write_scores(self.out, so_far())
            if len(graded) < len(wanted):
                log.warning("scoring stopped after %d of %d documents; the rest are graded "
                            "on the next run", len(graded), len(wanted))

        answered = {**known, **graded}
        return [answered[d.doc_id] for d in docs if d.doc_id in answered]

    def asked(self) -> list[str]:
        """Every resolved setting this Run sends the vendor, one string each."""
        return [f"{k}={v if v != '' else chr(39) * 2}"
                for k, v in sorted(self.settings().items())]

    def cells(self, work: tuple[int, int] | None = None) -> list[str]:
        """This Run as table cells: its name, and what it still owes when `work` is known. Cells
        rather than a padded line, so the column is sized by the data."""
        return [self.label, *(("", "") if work is None else (str(work[0]), str(work[1])))]

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
        """Write one answer down. The prediction lands first and the record is the marker
        `needs` reads, so a kill between them re-runs the document rather than skipping it."""
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
    """Every Run on one adapter, through one pool sized to that service. The cap belongs to the
    vendor rather than to a Run, and every model id shares one adapter."""

    def __init__(self, adapter: str, runs: list[Run], workers: int):
        self.adapter, self.runs, self.workers = adapter, runs, workers
        self.mine = threading.Event()
        self.fatal: list[Exception] = []
        self.halt = threading.Lock()

    def __repr__(self) -> str:
        return f"ProviderRun({self.adapter}, {len(self.runs)} runs, {self.workers} at a time)"

    def heading(self) -> str:
        """The adapter and the cap every Run under it shares."""
        return self.adapter

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
                record = predict(run.provider, doc.doc_path, doc.schema, timeout=timeout,
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

        def stopping() -> bool:
            return self.mine.is_set() or (stop is not None and stop.is_set())


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
    """Grade one prediction on disk and return its row. Module-level and picklable for the
    process pool, and it never raises: one bad document is an error row."""
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
        result = score(pred, json.loads(doc.gt_path.read_text()), doc.schema, verdicts=verdicts)
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
    """The grades this selection may reuse, and the documents still to grade. One definition, so
    what `describe` promises and what `Run.score` then does cannot drift."""
    known = read_scores(out, verdicts=verdicts)
    if rescore:
        selected = {d.doc_id for d in docs}
        known = {k: v for k, v in known.items() if k not in selected}
    return known, [d for d in docs if d.doc_id not in known]


def write_scores(out: Path, rows: list[dict]) -> None:
    """One line per document, written wherever scoring stops. Atomic because it is read back: a
    truncated file would silently lose everything past the cut."""
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


#: Documents in flight per adapter. Keyed by ADAPTER MODULE name (`harness.resolve`), not by
#: provider name, so every OpenRouter model id shares one budget instead of taking one each.
#:
#: It lives here, not in the harness: how many documents to have in flight at once is a
#: decision about a corpus, and `harness` declares orchestration absent on purpose. The numbers
#: are what each vendor tolerated in practice, not a published limit.
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


def plan(providers: list[str], options: dict | None = None,
         out: Path = Path("runs")) -> list[Run]:
    """A `Run` per configuration measured, deduplicated by label. `--options` may give one
    provider a LIST, and each entry is a Run of its own."""
    from .harness.registry import out_name

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


class BenchmarkRun:
    """One invocation: every Run, grouped by adapter, predicted then graded then written down.
    `describe()` says what it would do, with no download and no vendor call."""

    def __init__(self, providers: list[str], *, out: Path = Path("runs"),
                 data_root: Path | None = None, manifest: Path | None = None,
                 repo: str | None = None,
                 suites: list[str] | None = None,
                 limit: int = 0, timeout: float = 1800.0,
                 predict_workers: dict[str, int] | int | None = None, score_workers: int = 0,
                 verdicts: bool = True, rescore: bool = False, score_only: bool = False,
                 options: dict | None = None):
        from .harness.registry import resolve

        for provider in providers:
            resolve(provider)
        self.out, self.data_root, self.manifest = out, data_root, manifest
        self.repo = repo or REPO           # `None` rather than REPO, so the cli has no copy
        #: Names the corpus, for `settings.json` to record and `prepare` to check. Not
        #: `corpus`, which is the method that fetches it.
        self.corpus_id = (f"manifest {manifest}" if manifest
                          else f"huggingface {self.repo}")
        self.suites, self.limit = suites, limit
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

    def facts(self, docs: list[Doc] | None = None) -> Table:
        """What this invocation is, one labelled row each. A row rather than another clause in
        a sentence, so a new flag cannot lengthen a line that already ran to three of them."""
        table = Table(box=None, show_header=False, pad_edge=False)
        table.add_column("", style="dim", no_wrap=True)
        table.add_column("", overflow="fold")
        table.add_row("out", str(self.out))
        table.add_row("runs", f"{len(self.runs)} over "
                              f"{plural(len(self.providers), 'adapter')}")
        table.add_row("corpus", self.corpus_id)
        if docs is not None:
            table.add_row("documents", f"{plural(len(docs), 'document')} selected")
        elif self.limit:
            table.add_row("documents", f"first {self.limit}")
        table.add_row("timeout", f"{self.timeout:.0f}s per document")
        if self.suites:
            # Its own row, so the commas in it cannot read as more facts -- joined into one
            # sentence, `suites a, b, c` made `b` look like a clause of its own.
            table.add_row("suites", ", ".join(self.suites))
        # Both, always, even when false: `score only  false` is what says this will call a
        # vendor and spend money, and saying it only by absence is not saying it.
        for name, on in (("score only", self.score_only), ("rescoring", self.rescore)):
            table.add_row(name, Text(str(on).lower(), style="bold" if on else "dim"))
        return table

    def plan_view(self, docs: list[Doc] | None = None) -> Group:
        """The whole plan as one renderable; with `docs`, what each Run still owes. `describe`
        renders it as text for a log, the command line prints it in colour."""
        work = None if docs is None else {
            r.label: r.outstanding(docs, verdicts=self.verdicts, rescore=self.rescore)
            for r in self.runs}

        # A REAL TABLE, so a cell too wide for the terminal WRAPS rather than being cut. The
        # adapter is its own column instead of a heading row: in column one a heading is the
        # widest cell there is, and it was padding every label out to its width.
        table = Table(box=box.ROUNDED, header_style="dim", expand=False)
        table.add_column("adapter", overflow="fold")
        table.add_column("at once", justify="right", no_wrap=True)
        table.add_column("run", overflow="fold")
        table.add_column("settings", overflow="fold")
        table.add_column("predict", justify="right", no_wrap=True)
        table.add_column("grade", justify="right", no_wrap=True)
        for provider in self.providers:
            for n, run in enumerate(provider.runs):
                label, to_predict, to_grade = run.cells((work or {}).get(run.label))
                table.add_row(Text(provider.heading() if n == 0 else "", style="bold"),
                              Text(str(provider.workers) if n == 0 else ""),
                              Text(label), settings_cell(run.asked()),
                              Text(to_predict, style="" if to_predict in ("", "0") else "bold"),
                              Text(to_grade, style="" if to_grade in ("", "0") else "bold"),
                              end_section=n == len(provider.runs) - 1)
        return Group(Text("benchmark", style="bold"), self.facts(docs), Text(""), table)

    def describe(self, docs: list[Doc] | None = None) -> str:
        """`plan_view` as plain text, for a log that has no colour and no width to ask about."""
        console = Console(file=io.StringIO(), width=PLAN_WIDTH, no_color=True, highlight=False)
        console.print(self.plan_view(docs))
        return "\n".join(line.rstrip() for line in console.file.getvalue().splitlines())

    def prepare(self) -> None:
        """Each Run's directory, and what it asked, written before it asks -- `summary.json` comes
        after grading, which an interrupted run never reaches."""
        for run in self.runs:
            run.out.mkdir(parents=True, exist_ok=True)
            settings = run.out / "settings.json"
            if settings.exists():
                # A directory is named for the provider and its options, never for the corpus,
                # so two corpora would land in one and `summary.json` would average both.
                was = json.loads(settings.read_text()).get("corpus")
                if was and was != self.corpus_id:
                    raise ValueError(f"{run.out} holds {run.provider} against {was}, and this "
                                     f"run is against {self.corpus_id}. Use a different --out.")
            write_json_atomic(settings, {**run.head(), "timeout_s": self.timeout,
                                         "corpus": self.corpus_id}, indent=2)

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
        """Grade every Run, serially, and publish each as soon as it is graded. Serial because
        grading saturates every core.

        A SECOND PHASE, NOT A SECOND DISPLAY. Grading starts once every vendor has answered, so
        it gets a table of its own in the same shape -- with no cost and nothing in flight,
        because there is no vendor. Streaming it instead, so a run that finished predicting
        began grading, would put a process pool beside the live thread pools to save minutes at
        the end of a run that spends hours at the start of it.
        """
        summary = {}
        with Progress([r.label for r in self.runs]) as bars:
            for run in self.runs:
                mine = run.score(docs, verdicts=self.verdicts, workers=self.score_workers,
                                 rescore=self.rescore, progress=bars.reporter(run.label))
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
        """The documents this invocation selects, fetching ours when no manifest was brought.
        A relative path resolves against `--data-root`, else the manifest's own directory; ours
        goes to the HuggingFace cache unless `--data-root` names somewhere else."""
        if self.manifest is not None:
            root = self.data_root if self.data_root is not None else self.manifest.parent
            path = self.manifest
        else:
            root = fetch(self.data_root, self.repo)
            path = root / MANIFEST
        docs = read_manifest(path, root, suites=self.suites, limit=self.limit)
        if not docs:
            raise ValueError("no documents selected: check --suites and --limit")
        return docs

    def execute(self, confirm=None) -> dict:
        """Fetch, predict, score, write it down. `confirm` is handed the plan; returning
        False runs nothing and gives back `{}`."""
        docs = self.corpus()
        if confirm is None:
            for line in self.describe(docs).splitlines():
                log.info("%s", line)
        elif not confirm(self.plan_view(docs)):
            log.warning("nothing run")
            return {}
        self.prepare()
        if not self.score_only:
            self.predict(docs)
        return self.score(docs)


def run(providers: list[str], *, confirm=None, **kwargs) -> dict:
    """Fetch, predict, score, write it down. Keyword arguments and no argparse so it stays
    callable from a notebook, and it never reads stdin -- `confirm` is the CLI's."""
    return BenchmarkRun(providers, **kwargs).execute(confirm=confirm)
