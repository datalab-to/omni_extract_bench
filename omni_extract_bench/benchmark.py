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
    corpus      hf://datasets/datalab-to/omni_extract_bench/manifest.parquet
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
HF_DATASETS = "hf://datasets/"
DEFAULT_MANIFEST = f"{HF_DATASETS}{REPO}/{MANIFEST}"
#: Written out rather than read off `Doc`: manifests on disk have this shape, and a field `Doc`
#: grows later must not invalidate them.
COLUMNS = ("doc_id", "suite", "doc_path", "gt_path", "schema")


class Doc(NamedTuple):
    """One benchmark element."""

    doc_id: str
    suite: str
    doc_path: Path
    gt_path: Path
    schema: dict


class HfPath(NamedTuple):
    """A file in a HuggingFace dataset, as `hf://datasets/<repo>[@<revision>]/<path>`."""

    repo: str
    revision: str | None
    path: str

    def __str__(self) -> str:
        at = f"@{self.revision}" if self.revision else ""
        return f"{HF_DATASETS}{self.repo}{at}/{self.path}"


def hf_path(manifest: str | Path) -> HfPath | None:
    """The HuggingFace file `manifest` addresses, or `None` when it is a path on disk."""
    form = f"hf://datasets/<org>/<name>[@<revision>]/<path>, e.g. {DEFAULT_MANIFEST}"
    s = str(manifest)
    if isinstance(manifest, Path) and s.startswith("hf:"):
        raise ValueError(f"the manifest {s!r} was made a Path, which folds the '//' out of an "
                         f"address. Pass it as a str: {form}")
    if not s.startswith("hf://"):
        return None
    if not s.startswith(HF_DATASETS):
        raise ValueError(f"{s!r} is not a HuggingFace dataset; a manifest is read from one "
                         f"as {form}")
    org, _, rest = s.removeprefix(HF_DATASETS).partition("/")
    name, _, path = rest.partition("/")
    name, at, revision = name.partition("@")
    if not org or not name or not path or (at and not revision):
        raise ValueError(f"{s!r} does not name a file in a HuggingFace dataset. Write it as "
                         f"{form}")
    return HfPath(f"{org}/{name}", revision or None, path)


def fetch(at: HfPath) -> Path:
    """The dataset `at` is in, on disk: the snapshot folder, named for its commit. It lands in
    the HuggingFace cache, which is where it belongs: shared between every checkout and every
    working directory, so the second folder you run from resolves it for free.
    """
    try:
        from huggingface_hub import file_exists, snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is needed to fetch the benchmark corpus, and is not installed:\n"
            "    pip install 'omni-extract-bench[benchmark]'\n"
            "The scorer itself needs none of it -- `score` and `oeb score` work without."
        ) from None

    # Asked before a download of gigabytes. A hub that will not answer (offline, cached) is not
    # a no: the download runs, and `read_manifest` reports a manifest that really is missing.
    try:
        missing = not file_exists(at.repo, at.path, repo_type="dataset", revision=at.revision)
    except OSError:
        missing = False
    if missing:
        raise ValueError(
            f"there is no manifest at {at}, so there is nothing to benchmark. A dataset holds "
            f"one or more, each following the same structure as ours at "
            f"https://huggingface.co/datasets/{REPO}. Point --manifest at a parquet of your "
            f"own to benchmark documents that are already on disk."
        )

    log.info("fetching %s into the huggingface cache", at)
    return Path(snapshot_download(at.repo, repo_type="dataset", revision=at.revision))


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
        # A doc_id is a filename (`predictions/<doc_id>.json`); caught here, the row can be named.
        doc_id = row["doc_id"]
        if not doc_id or doc_id in (".", "..") or Path(doc_id).name != doc_id:
            raise ValueError(f"{path}: row {n} has doc_id {doc_id!r}, which is not a filename. "
                             f"It names this document's prediction, its record and its row in "
                             f"scores.jsonl, so it has to be one path component.")
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


class Run(NamedTuple):
    """One configuration and its folder: an address, as a `Path` is. What a run holds is whatever
    its folder holds when it is read, so it is resumed, graded or shared between processes as it
    stands. The functions below are the bookkeeping of any loop over its documents:

        prepare(run, corpus, timeout)          # once: the folder, and settings.json
        if needs(run, doc):                    # no readable record yet
            record(run, doc.doc_id, harness.predict(...))
        grade(run, docs)                       # scores.jsonl, one row per document
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


def prepare(run: Run, corpus: str, timeout: float, commit: str | None = None) -> None:
    """The run's folder made, and what it asks written to its settings.json before it asks:
    `summary.json` comes after grading, which an interrupted run never reaches. A folder that
    holds another corpus is refused: it is named for provider and options, not corpus."""
    run.predictions.mkdir(parents=True, exist_ok=True)
    run.records.mkdir(parents=True, exist_ok=True)
    path = run.out / "settings.json"
    if path.exists() and (was := json.loads(path.read_text()).get("corpus")) not in (None, corpus):
        raise ValueError(f"{run.out} holds {run.provider} against {was}, and this run is against "
                         f"{corpus}. Use a different --out.")
    write_json_atomic(path, {**head(run), "timeout_s": timeout, "corpus": corpus,
                             **({"corpus_commit": commit} if commit else {})}, indent=2)


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


def needs(run: Run, doc: Doc) -> bool:
    """True when the run has no readable record of the document: it was never attempted."""
    return recorded(run, doc.doc_id) is None


def record(run: Run, doc_id: str, result: dict) -> None:
    """One answer written down: its prediction, if it has one, then the record. predictions/ holds
    only answers; a failure is described in the record alone. The record is the marker `needs`
    reads, so a kill between the two predicts the document again rather than skipping it."""
    if result["error"] is None:
        write_json_atomic(run.predictions / f"{doc_id}.json", result["result"])
    write_json_atomic(run.records / f"{doc_id}.json", result)


def owed(run: Run, docs: list[Doc], verdicts: bool = True, rescore: bool = False) -> tuple[int, int]:
    """How many of `docs` the run would predict, and then grade. The two are independent: every
    prediction can be on disk with every grade still owed."""
    _, wanted = grading_split(run.out, docs, verdicts=verdicts, rescore=rescore)
    return sum(needs(run, doc) for doc in docs), len(wanted)


def grade(run: Run, docs: list[Doc], verdicts: bool = True, workers: int = 0,
          rescore: bool = False, progress=NULL) -> list[dict]:
    """Grade the run's predictions, resumable per document, and write one line each."""
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

    answered = {**known, **graded}
    return [answered[d.doc_id] for d in docs if d.doc_id in answered]


def asked(run: Run) -> list[str]:
    """Every resolved setting the run sends the vendor, one string each."""
    return [f"{k}={v if v != '' else chr(39) * 2}" for k, v in sorted(settings(run).items())]


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

    def _give_up(self, run: Run, exc: Exception, why: str = "") -> None:
        """Stop this vendor and remember why. The document counts as never attempted."""
        with self.halt:
            if not self.mine.is_set():
                if why:
                    log.error("%s: %s -- %s", run.out.name, why, exc)
                self.fatal.append(exc)
                self.mine.set()

    def _predict_one(self, run: Run, doc: Doc, timeout: float, stopping, bar) -> None:
        if stopping():
            return
        try:
            with bar.calling():
                got = predict(run.provider, doc.doc_path, doc.schema, timeout=timeout,
                              **run.options)
            record(run, doc.doc_id, got)
        except (MissingDependency, MissingCredential) as exc:
            return self._give_up(run, exc)
        except AccountFailure as exc:
            return self._give_up(run, exc, "STOPPED, account-level failure")
        except Exception as exc:          # noqa: BLE001 -- unknown, so assume it is ours
            return self._give_up(run, exc)
        cost = got.get("cost") or {}
        bar.record(error=bool(got.get("error")), usd=cost.get("usd"),
                   credits=cost.get("credits"), wall_s=cost.get("wall_s"))

    def predict(self, docs: list[Doc], timeout: float, stop: threading.Event | None = None,
                bars: Progress | None = None) -> None:
        """Every document of every Run here, and raise whatever stopped the vendor."""
        reporters = {run.out.name: bars.reporter(run.out.name) if bars else NULL for run in self.runs}
        left, work = {}, []
        for run in self.runs:
            run.predictions.mkdir(parents=True, exist_ok=True)
            run.records.mkdir(parents=True, exist_ok=True)
            todo = [doc for doc in docs if needs(run, doc)]
            log.info("%s: %d documents, %d to run", run.out.name, len(docs), len(todo))
            reporters[run.out.name].start(len(todo), self.workers)
            if not todo:
                reporters[run.out.name].finish()
            left[run.out.name] = len(todo)
            work += [(run, doc) for doc in todo]

        def stopping() -> bool:
            return self.mine.is_set() or (stop is not None and stop.is_set())

        with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._predict_one, run, doc, timeout, stopping,
                                   reporters[run.out.name]): run.out.name for run, doc in work}
            try:
                for future in cf.as_completed(futures):
                    future.result()
                    # Note: finished when its own last document lands, not when the pool it
                    # shares drains, so each run is timed for itself.
                    name = futures[future]
                    left[name] -= 1
                    if not left[name]:
                        reporters[name].finish()
            except KeyboardInterrupt:
                self.mine.set()
                for future in futures:
                    future.cancel()
                log.warning("%s: interrupted; documents not yet written come back on the next "
                            "run", self.adapter)
                raise

        if self.fatal:
            raise self.fatal[0]


SCORE_WORKERS = min(8, os.cpu_count() or 1)


def score_one(doc: Doc, *, run: Run, verdicts: bool = False) -> dict:
    """Grade one prediction on disk and return its row. Module-level and picklable for the
    process pool, and it never raises: one bad document is an error row.

    A document fails when its record says the call failed, or when it has no prediction: a
    folder of predictions you bring is your submission, so a missing file counts against
    coverage rather than being left out of it."""
    row = {"doc_id": doc.doc_id, "suite": doc.suite, "provider": run.provider}
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
        result = score(pred, json.loads(doc.gt_path.read_text()), doc.schema, verdicts=verdicts)
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


def plan(providers: list[str], options: dict | None = None,
         out: Path = Path("runs")) -> list[Run]:
    """A `Run` per configuration measured, deduplicated by folder. `--options` may give one
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
            name = out_name(provider, one)
            runs.setdefault(name, Run(provider, one or {}, out / name))
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
                 data_root: Path | None = None, manifest: str | Path | None = None,
                 suites: list[str] | None = None,
                 limit: int = 0, timeout: float = 1800.0,
                 predict_workers: dict[str, int] | int | None = None, score_workers: int = 0,
                 verdicts: bool = True, rescore: bool = False, score_only: bool = False,
                 options: dict | None = None):
        from .harness.registry import resolve

        for provider in providers:
            resolve(provider)
        self.out, self.data_root = out, data_root
        self.manifest = manifest or DEFAULT_MANIFEST
        if hf_path(self.manifest) is not None and data_root is not None:
            raise ValueError(f"--data-root {data_root} cannot move {self.manifest}: a manifest "
                             f"in a HuggingFace dataset resolves its paths against that "
                             f"dataset. Drop --data-root, or point --manifest at a parquet on "
                             f"disk.")
        self.corpus_id = str(self.manifest)
        self.corpus_commit: str | None = None
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
            table.add_row("suites", ", ".join(self.suites))
        # Shown even when false: `score only  false` is what says this will spend money.
        for name, on in (("score only", self.score_only), ("rescoring", self.rescore)):
            table.add_row(name, Text(str(on).lower(), style="bold" if on else "dim"))
        return table

    def plan_view(self, docs: list[Doc] | None = None) -> Group:
        """The whole plan as one renderable; with `docs`, what each Run still owes. `describe`
        renders it as text for a log, the command line prints it in colour."""
        work = None if docs is None else {
            r.out.name: owed(r, docs, verdicts=self.verdicts, rescore=self.rescore)
            for r in self.runs}

        # A table, so a cell too wide for the terminal wraps rather than being cut.
        table = Table(box=box.ROUNDED, header_style="dim", expand=False)
        table.add_column("adapter", overflow="fold")
        table.add_column("at once", justify="right", no_wrap=True)
        table.add_column("run", overflow="fold")
        table.add_column("settings", overflow="fold")
        table.add_column("predict", justify="right", no_wrap=True)
        table.add_column("grade", justify="right", no_wrap=True)
        for provider in self.providers:
            for n, run in enumerate(provider.runs):
                got = (work or {}).get(run.out.name)
                to_predict, to_grade = ("", "") if got is None else map(str, got)
                table.add_row(Text(provider.heading() if n == 0 else "", style="bold"),
                              Text(str(provider.workers) if n == 0 else ""),
                              Text(run.out.name), settings_cell(asked(run)),
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
        for run in self.runs:
            prepare(run, self.corpus_id, self.timeout, self.corpus_commit)

    def predict(self, docs: list[Doc]) -> None:
        """Every adapter at once. Predicting is network wait, so they do not slow each other."""
        stop = threading.Event()
        with Progress([r.out.name for r in self.runs]) as bars:
            with cf.ThreadPoolExecutor(max_workers=len(self.providers)) as pool:
                futures = {pool.submit(p.predict, docs, self.timeout, stop, bars): p.adapter
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
        with Progress([r.out.name for r in self.runs]) as bars:
            for run in self.runs:
                mine = grade(run, docs, verdicts=self.verdicts, workers=self.score_workers,
                             rescore=self.rescore, progress=bars.reporter(run.out.name))
                summary[run.out.name] = {**head(run), **summarise(mine)}
                write_json_atomic(run.out / "summary.json",
                                  {**head(run),
                                   **summarise(list(read_scores(run.out).values()))}, indent=2)
                s = summary[run.out.name]
                log.info("%s: accuracy %.4f over %d suites, coverage %d/%d",
                         run.out.name, s["accuracy"], len(s["per_suite"]), s["scored"],
                         s["documents"])
        return summary

    def corpus(self) -> list[Doc]:
        """The documents this invocation selects, fetching the dataset when the manifest is in
        one. A relative path in a manifest on disk resolves against `--data-root`, else the
        manifest's own directory; in a dataset, against the dataset's root."""
        if (at := hf_path(self.manifest)) is None:
            path = Path(self.manifest)
            root = self.data_root if self.data_root is not None else path.parent
        else:
            root = fetch(at)
            path = root / at.path
            self.corpus_commit = root.name
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
