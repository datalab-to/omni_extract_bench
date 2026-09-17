#!/usr/bin/env python3
"""The whole benchmark, end to end: fetch the corpus, predict with each vendor, score, write it down.

    oeb benchmark --providers datalab reducto --out runs/
    oeb benchmark --providers datalab --limit 5              # smoke test

`run()` is the whole thing; the command line for it is in `cli.py`, so this file holds no
argparse and can be called directly from a script or a notebook.

Four steps, in the order they run, each one function below:

    fetch          the corpus from HuggingFace: manifest, PDFs and gold
    read_manifest  620 rows -> Doc(doc_id, suite, pdf, gt, schema)
    predict_all    each document through one vendor, under the harness's parity rules
    score_all      score() each prediction against its gold
    summarise      per suite, then UNIFIED

IT IS RESUMABLE, AND RUNNING IT TWICE IS SAFE. A document is re-attempted only when it has no
prediction, or the prediction on disk is a TRANSIENT failure -- so an interrupted run continues
where it stopped, and a run that finished costs nothing to score again. Scoring is pure and
always redone: it is free, and it is the half most likely to change under you.

THIS SCRIPT IS THE ORCHESTRATION, AND IT IS DELIBERATELY NOT IN THE LIBRARY. `score` scores one
document and `harness.predict` produces one prediction; which documents, in what order, how many
at once, and what to retry are decisions about a corpus, and a library that made them would be
choosing the shape of every benchmark built on it. What this file does, you can do differently.

    pip install 'omni-extract-bench[benchmark,harness]'

Credentials for each vendor come from the environment; see `harness/README.md`.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import json
import logging
import os
import threading
from pathlib import Path
from typing import NamedTuple

from . import score
from .harness import AccountFailure, MissingCredential, MissingDependency, predict


log = logging.getLogger(__name__)

REPO = "datalab-to/omni_extract_bench"
MANIFEST = "manifest.parquet"


class Doc(NamedTuple):
    """One benchmark document, with everything needed to run and score it already resolved.

    Built once from the manifest so that nothing downstream re-reads parquet, re-resolves a
    path, or re-parses a schema -- the schema in particular is parsed here and passed around
    as a dict, because it is handed to both the vendor and the scorer and they must get the
    same object.
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
def write_json(path: Path, obj) -> None:
    """Write a file that is either wholly there or not there at all.

    `write_text` is not atomic: killed part way through, it leaves a truncated file that still
    EXISTS -- and a resume that asks "was this attempted?" would skip it forever. Writing to a
    neighbour and renaming is atomic on POSIX, so an interrupted write leaves the old file (or
    no file), never half of one. The temp file must share a directory with the target, because
    rename is only atomic within a filesystem.
    """
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(obj, default=str))
    os.replace(tmp, path)


def needs_run(record_path: Path) -> bool:
    """True when this document has not been attempted yet.

    A record on disk means the vendor was called and answered -- well or badly. Whether a
    failure was worth another try is `predict`'s business, decided inside the call with the
    HTTP status in hand; by the time a record exists those attempts are spent. A resume that
    second-guessed it would pay again for answers already bought.

    The record is written LAST and atomically, so its presence means the prediction beside it
    is complete too. It is parsed rather than merely stat-ed: a file that will not parse is a
    file from a run that died, and skipping it would lose the document for good.
    """
    if not record_path.exists():
        return True
    try:
        json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    return False


def predict_all(docs: list[Doc], provider: str, out: Path, timeout: float, workers: int,
                options: dict | None = None) -> None:
    """Every document through one vendor, concurrently, leaving the answer and the evidence.

        <out>/predictions/<doc_id>.json    the bare extraction -- what scoring reads
        <out>/records/<doc_id>.json        the schema sent, every HTTP call, the cost

    Two files because they are read at different times and are different sizes: scoring wants
    the answer, an audit wants everything, and only one of them is worth loading 620 of.
    """
    preds, records = out / "predictions", out / "records"
    preds.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)

    todo = [d for d in docs if needs_run(records / f"{d.doc_id}.json")]
    log.info("%s: %d documents, %d to run", provider, len(docs), len(todo))
    if not todo:
        return

    # An account failure is not about the document, so it stops this provider rather than being
    # written down 600 times: one vendor once marked 174 documents failed in 60 seconds after
    # hitting a credit ceiling. A raise in one pool worker does not halt the others, hence the
    # flag. Documents left untouched keep no file, so the next run picks them up.
    stop = threading.Event()
    fatal: list[Exception] = []
    tally: collections.Counter[str] = collections.Counter()   # main thread only: no lock
    spend: list[float] = []
    billed: list[float] = []          # vendors that price in credits, not dollars
    took: list[float] = []            # wall-clock per document

    def run_one(doc: Doc) -> str:
        if stop.is_set():
            return "skipped", None, None, None
        try:
            record = predict(provider, doc.pdf, doc.schema, timeout=timeout,
                             **(options or {}))
        except (MissingDependency, MissingCredential) as exc:
            # A missing SDK or an unset API key: ours, and identical for every document. It
            # stops the provider and is re-raised below rather than written down 620 times.
            if not stop.is_set():
                fatal.append(exc)
                stop.set()
            return "skipped", None, None, None
        except AccountFailure as exc:
            if not stop.is_set():
                log.error("%s: STOPPED, account-level failure -- %s", provider, exc)
                stop.set()
            return "skipped", None, None, None
        # ORDER IS THE COMMIT. The prediction lands first; the record is the marker `needs_run`
        # reads, so it is written only once the prediction it describes is safely on disk.
        # Killed in between, the document simply looks unattempted and is run again -- which is
        # the cheap mistake. The expensive one is looking done while the prediction is missing.
        write_json(preds / f"{doc.doc_id}.json", record["result"])
        write_json(records / f"{doc.doc_id}.json", record)
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
                if i % 10 == 0 or i == len(todo):
                    log.info("  %s [%d/%d] %s", provider, i, len(todo), dict(tally))
        except KeyboardInterrupt:
            # Ctrl-C. This MUST happen before the pool's `__exit__`, which waits for every
            # future already submitted -- and `submit` queued all of them up front, so without
            # this the interrupt would quietly work through the whole corpus before returning.
            # Setting the flag first makes the queued ones return immediately; cancelling stops
            # the ones that have not started at all. Calls already in flight finish their write,
            # so the run stops on a document boundary.
            stop.set()
            for future in futures:
                future.cancel()
            log.warning("%s: interrupted -- %s; documents not yet written come back on the "
                        "next run", provider, dict(tally))
            raise

    # What it cost, said where it was spent. `len(spend)` is there because several vendors
    # report no per-call cost at all: a bare total over a corpus where half the documents
    # reported nothing would read as the bill and be a fraction of it.
    if took:
        # The MEAN and the range, because a vendor whose documents take 4s and 1800s is a
        # different proposition from one that reliably takes 900s, and an average alone
        # cannot tell them apart. Wall-clock as the harness saw it: queueing and retries
        # included, because that is what running the benchmark actually costs in time.
        log.info("%s: %.1fs per document on average (min %.1f, max %.1f over %d)",
                 provider, sum(took) / len(took), min(took), max(took), len(took))
    if spend:
        log.info("%s: $%.2f reported over %d of %d documents", provider, sum(spend),
                 len(spend), len(todo))
    if billed:
        # Credits, in the vendor's own unit. Never converted: the rate is contract-specific,
        # so a dollar figure derived from it would be invented rather than measured.
        log.info("%s: %g credits reported over %d of %d documents", provider, sum(billed),
                 len(billed), len(todo))
    if todo and not spend and not billed:
        log.info("%s: no per-document cost reported (billed out of band)", provider)

    # Raised after the pool drains, so no in-flight worker is still writing when it surfaces.
    # The documents it stopped keep no file at all, and so stay resumable once it is fixed.
    if fatal:
        raise fatal[0]


# ── 3. scores ────────────────────────────────────────────────────────────────────────────

def score_all(docs: list[Doc], provider: str, out: Path, verdicts: bool = False) -> list[dict]:
    """Grade every prediction on disk, and write one line per document.

    EVERY DOCUMENT COMES BACK, including the ones with no usable prediction. Coverage is only
    visible if a failure occupies a row. Such a row carries `status="error"` and NULL metrics
    rather than zero, because a zero claims the model tried and missed every field -- the
    summary then reports coverage beside the score instead of hiding the difference in it.
    """
    rows = []
    for doc in docs:
        row = {"doc_id": doc.doc_id, "suite": doc.suite, "provider": provider}
        path = out / "predictions" / f"{doc.doc_id}.json"
        try:
            pred = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            rows.append({**row, "status": "error", "error": "no prediction on disk"})
            continue
        # ONE definition of "is this something to score", and it lives here because here is
        # the only place that asks. It was two once, and they disagreed: one path treated an
        # empty `{}` as "never attempted" while the other graded it as 0.0 and counted it as
        # attempted -- so a vendor reported 100% coverage while 37 of its 45 outputs were empty.
        if not (isinstance(pred, dict) and pred and "__error__" not in pred):
            reason = (pred or {}).get("__error__") if isinstance(pred, dict) else "not an object"
            rows.append({**row, "status": "error", "error": str(reason or "empty prediction")})
            continue
        try:
            result = score(pred, json.loads(doc.gt.read_text()), doc.schema,
                           verdicts=verdicts)
        except Exception as exc:                        # noqa: BLE001 -- the message is the row
            rows.append({**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            continue
        if verdicts:
            (out / "verdicts").mkdir(parents=True, exist_ok=True)
            (out / "verdicts" / f"{doc.doc_id}.jsonl").write_text(
                "\n".join(json.dumps(v._asdict(), default=str) for v in result.pop("verdicts")))
        rows.append({**row, "status": "scored", **result})

    with (out / "scores.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    return rows


# ── 4. the number ────────────────────────────────────────────────────────────────────────
def summarise(rows: list[dict]) -> dict:
    """Per suite, then UNIFIED: the mean of the suite means, not the mean over documents.

    Equal weight per suite, so a large suite cannot dominate -- `docs/METRIC_SPEC.md` section 7.
    A flat mean over documents is reported too, because it is what people compute by hand and
    seeing the two differ is the point.

    A document with no usable prediction scores ZERO here, while its row keeps null metrics.
    Both are honest and they answer different questions: `coverage` says how often the vendor
    answered at all, and the score says what the benchmark is worth to someone who has to run
    every document. Averaging only what succeeded would pay a vendor for failing on hard ones.
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
def run(providers: list[str], *, out: Path = Path("runs"),
        data_root: Path = Path("benchmark"), suites: list[str] | None = None, limit: int = 0,
        timeout: float = 1800.0, workers: int = 0, verdicts: bool = False,
        score_only: bool = False, options: dict | None = None) -> dict:
    """Fetch, predict, score, write it down. Returns the summary it also writes to `out`.

    Plain keyword arguments rather than a parsed namespace, and no argparse in this file: the
    command line lives in `cli.py`, so this stays callable from a notebook or another script
    without one. It raises rather than exiting, for the same reason.

    `options` is `{provider: {option: value}}`, passed through to that provider's adapter. Each
    one is recorded in the document's `run_manifest.overrides`, because the benchmark's claim
    is that every vendor ran at its maximum and a run that turned something down is a different
    measurement -- one that should not be able to look stock afterwards.
    """
    from .harness import WORKERS
    from .harness.vendor import out_name, resolve

    for provider in providers:
        resolve(provider)          # raises ValueError naming the vendors, before any download

    root = fetch(data_root)
    docs = read_manifest(root / MANIFEST, root, suites=suites, limit=limit)
    if not docs:
        raise ValueError("no documents selected: check --suites and --limit")

    # One provider at a time, on purpose: concurrency limits are per vendor, and interleaving
    # them would make a slow vendor's latency a function of who else was running.
    summary = {}
    for provider in providers:
        provider_out = out / out_name(provider)      # `openai/gpt-5.6-sol` would nest
        provider_out.mkdir(parents=True, exist_ok=True)
        if not score_only:
            predict_all(docs, provider, provider_out, timeout=timeout,
                        workers=workers or WORKERS.get(provider, 5),
                        options=(options or {}).get(provider))
        summary[provider] = summarise(
            score_all(docs, provider, provider_out, verdicts=verdicts))
        s = summary[provider]
        log.info("%s: unified %.4f over %d suites, coverage %d/%d",
                 provider, s["unified"], len(s["per_suite"]), s["scored"], s["documents"])

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
