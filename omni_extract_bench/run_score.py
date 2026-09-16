"""Scoring a table of documents: one manifest in, two tables out.

    oeb score --manifest jobs.parquet --out run/

    run/scores.parquet/part-00000.parquet      one row per manifest row
    run/verdicts.parquet/part-00000-0.parquet  one row per address

**The manifest is the benchmark.** It says which documents to score and where their bytes
are, and nothing else does -- no corpus directory, no atlas, no `<doc_id>.json` naming rule,
no discovery. A row naming three blobs is a unit of work and a table of them is a run.
Whatever produced the table owns dataset management; this module never learns what a subset,
a vendor or a version is.

Required columns::

    doc_id
    gt_path      the ground truth, as JSON
    pred_path    the prediction, as JSON
    schema       the schema itself, inline

Paths are read through fsspec, so `s3://`, `gs://` and a local path are one code path.

Each output is a DIRECTORY of parts carrying the `.parquet` name, because every reader opens
one exactly as it opens a single file -- so a manifest that is one file and a manifest that is
a hundred parts look alike wherever one is named. One part per batch, written under a name
nothing else writes, so a fan-out needs no concatenate step: a part is complete or absent.
A directory that already holds a run is refused rather than written into, since parts are
named by batch and a second, differently sized run would replace some and leave the rest.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import itertools
import json
import logging
import pathlib
import time
import traceback
from typing import Any

from .harness.dialects import resolve_refs, strip_benchmark_keys
from .harness.prediction_io import usable
from .score import grade, show

#: Progress goes to a logger, results are returned. A library that prints has decided where
#: its output belongs, and that is the caller's decision -- `cli` is what turns this on.
log = logging.getLogger(__name__)

REQUIRED = ("doc_id", "gt_path", "pred_path", "schema")

#: What a scored row carries, and the arrow type it carries it as. Declared rather than
#: inferred so every part of a fan-out agrees: a batch where every prediction failed has
#: nothing to infer `accuracy` from, and a null column will not concatenate with a double one.
SCORE_FIELDS: list[tuple[str, str]] = [
    ("status", "string"), ("error", "string"),
    ("accuracy", "float64"), ("precision", "float64"), ("recall", "float64"),
    ("matched", "int64"), ("misread", "int64"), ("unfound", "int64"),
    ("fabricated", "int64"), ("invented_item", "int64"), ("invented_field", "int64"),
    ("total", "int64"), ("asserted", "int64"),
    ("gt_rows", "int64"), ("pred_rows", "int64"), ("matched_rows", "int64"),
    ("matching_exact", "bool"),
    # What this row cost to grade. Recorded because scheduling a fan-out wants to know, and
    # measured time is the only estimate that is not a guess: the spread across documents is
    # four orders of magnitude, and file size predicts it at 0.87 where a schema predicts it
    # at 0.34. A run's own numbers make the next run's packing exact.
    ("seconds", "float64"),
]

VERDICT_FIELDS: list[tuple[str, str]] = [
    ("doc_id", "string"), ("address", "string"),
    ("gold_raw", "string"), ("gold_canon", "string"),
    ("pred_raw", "string"), ("pred_canon", "string"), ("verdict", "string"),
]

#: Names this module owns. A manifest may not reuse one: it would then mean two different
#: things depending on which row you read.
RESERVED = frozenset(name for name, _ in SCORE_FIELDS)


# ── the table layer, shared with run_predict ─────────────────────────────────────
def open_uri(uri: str, mode: str = "rb"):
    """Open one path, wherever it lives: `s3://...`, `gs://...` or local.

    fsspec rather than a client of our own, opened inside whichever process does the work: a
    filesystem object does not survive being pickled to a worker, while fsspec's instance
    cache gives each process its own without being asked.

    A local path falls back to the builtin when fsspec is not installed, so that the base
    install -- the scorer and scipy, which is what the README calls it -- can grade a pair of
    files on disk. fsspec arrives with the `benchmark` extra alongside pyarrow, because a
    manifest needs both; one local file needs neither.
    """
    try:
        import fsspec
    except ImportError:
        if "://" in uri:
            raise ImportError(
                f"reading {uri!r} needs fsspec: pip install 'omni-extract-bench[benchmark]'"
                " (add [s3] for a bucket)") from None
        return open(uri, mode)

    return fsspec.open(uri, mode)


def open_manifest(manifest: str):
    """The manifest as a pyarrow dataset, read through fsspec.

    A dataset rather than a file reader, for three things that come free with it: the schema
    is known before a single row is read, so a manifest that cannot mean what it says is
    refused without doing any work; batches stream, so a manifest of a hundred thousand rows
    is never in memory whole; and a DIRECTORY of parts is a dataset too -- which is what a
    predict run writes, so its output is a manifest without being assembled first.

    Parquet or CSV, by extension. In a CSV the `schema` cell is JSON text, which `json.loads`
    reads exactly as it reads the bytes a parquet column holds.
    """
    # A manifest is the one thing the base install genuinely cannot read: a parquet dataset
    # is pyarrow by definition. Named here rather than left as a bare ImportError, because
    # the extra that supplies it is not guessable from "No module named 'pyarrow'".
    try:
        import fsspec
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError(f"reading a manifest needs {exc.name}: "
                          "pip install 'omni-extract-bench[benchmark]'"
                          " (add [s3] for a manifest or paths in a bucket)") from None

    fs, path = fsspec.core.url_to_fs(manifest)
    return ds.dataset(path, filesystem=fs,
                      format="csv" if manifest.endswith(".csv") else "parquet")


def check_manifest(data, required, reserved) -> None:
    """Refuse a manifest that cannot mean what it says, before any work is done.

    Including the repeats: doc_id is the only thing identifying a row, so without unique ids
    the verdicts table cannot be joined back to the scores table. Checked on the dataset's
    doc_id column rather than while batches go past, so a repeat in the fifth batch is caught
    before the first four parts are written -- and one columnar pass names every offender
    instead of stopping at the first.
    """
    import pyarrow.compute as pc

    names = data.schema.names
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"manifest has no {', '.join(missing)} column"
                         f"{'s' if len(missing) > 1 else ''}. It has: {', '.join(names)}")
    clash = sorted(set(names) & reserved)
    if clash:
        raise ValueError(
            f"manifest column(s) {', '.join(clash)} collide with this run's own output "
            f"columns; rename them. Reserved: {', '.join(sorted(reserved))}")

    counts = pc.value_counts(data.to_table(columns=["doc_id"]).column("doc_id")).to_pylist()
    repeated = sorted(str(c["values"]) for c in counts if c["counts"] > 1)
    if repeated:
        raise ValueError(f"duplicate doc_id in the manifest: {', '.join(repeated[:5])}"
                         + (f" and {len(repeated) - 5} more" if len(repeated) > 5 else "")
                         + ". doc_id is the only thing identifying a row, so it has to be "
                           "unique -- the verdicts table joins back on it.")


def resolve(path: str, root: str) -> str:
    """Where a path in a manifest actually points.

    **Relative means owned by the benchmark** and is taken from `root`; **absolute, or a URI,
    means external** and is used as written. Delta Lake draws the same line -- a relative path
    in its log is a file the table owns, an absolute one is a file somewhere else -- and COCO
    and HuggingFace both hold relative names and take the root from the caller.

    The root is named by the caller rather than inferred from where the manifest sits, which is
    what keeps a chain working: `predict`'s output becomes `score`'s input, and a base guessed
    from the second manifest's own location would resolve the first manifest's paths against
    the wrong directory. `.` by default, so a manifest of absolute paths needs no root at all.
    """
    if not path or "://" in path or path.startswith("/") or root in ("", "."):
        return path
    return f"{root.rstrip('/')}/{path}"


def read_json(uri: str) -> Any:
    with open_uri(uri) as fh:
        return json.loads(fh.read())


def write_part(rows: list[dict], fields, path: str, carry=None) -> None:
    """One part: the declared fields, plus `carry`'s columns exactly as they arrived."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(
        rows, schema=pa.schema([(n, pa.type_for_alias(t)) for n, t in fields]))
    if carry is not None:
        for name in reversed(carry.schema.names):
            table = table.add_column(0, carry.schema.field(name), carry.column(name))
    with open_uri(path, "wb") as fh:
        pq.write_table(table, fh)


def prepare_out(out: str, overwrite: bool = False,
                names: tuple[str, ...] = ("scores.parquet", "verdicts.parquet")) -> None:
    """Refuse to lay a run on top of another one, or clear it out if that is what was meant.

    A run is a directory of parts named by batch, so a second run into the same `--out`
    overwrites the parts it reaches and leaves the rest: score forty documents, then one, and
    the directory reads back as thirty-one rows from two different runs. Nothing about the
    table says so, which makes it the kind of number someone puts in a post.

    Refusing rather than clearing, because the common way to arrive here is a typo in `--out`,
    and deleting the run someone spent an afternoon on is worse than stopping.
    """
    import fsspec

    fs, path = fsspec.core.url_to_fs(out.rstrip("/"))
    existing = [f"{path}/{n}" for n in names if fs.exists(f"{path}/{n}")]
    if not existing:
        return
    if not overwrite:
        raise ValueError(
            f"{out} already holds a run ({', '.join(pathlib.Path(d).name for d in existing)}). "
            f"Writing into it again would mix the two, because parts are named by batch and "
            f"only the ones this run reaches get overwritten. Point --out somewhere new, or "
            f"pass --overwrite to replace what is there.")
    for d in existing:
        fs.rm(d, recursive=True)


def in_order(submit, items, window: int):
    """Results in input order, with at most `window` units of work in flight.

    `Executor.map` submits the whole batch at once and holds every finished result until the
    slowest earlier one catches up, which puts the memory ceiling back where the point was to
    remove it. Scoring cost is wildly uneven -- one document can take minutes while its
    neighbours take milliseconds -- so the window is what keeps the ceiling flat.
    """
    items = iter(items)
    futures = collections.deque(submit(x) for x in itertools.islice(items, window))
    for item in items:
        yield futures.popleft().result()
        futures.append(submit(item))
    while futures:
        yield futures.popleft().result()


def arrow_schema(fields):
    import pyarrow as pa

    return pa.schema([(n, pa.type_for_alias(t)) for n, t in fields])


def write_streaming(batches, fields, base_dir: str, prefix: str) -> None:
    """Write an ITERATOR of row lists as a parquet dataset, pulling as it writes.

    `ds.write_dataset` is pyarrow's own streaming writer: it consumes the iterator lazily and
    decides row groups and file rotation itself, so nothing here buffers and no row group is
    sized by accident. Used for verdicts, which arrive one document at a time and are two
    orders of magnitude larger than scores.
    """
    import fsspec
    import pyarrow as pa
    import pyarrow.dataset as ds

    schema = arrow_schema(fields)
    fs, path = fsspec.core.url_to_fs(base_dir)
    # A RecordBatchReader is the library's container for a lazy, typed stream: it carries the
    # schema, so the writer never has to guess it from a sample -- which would mean grading the
    # first batch twice, on documents that can take minutes.
    reader = pa.RecordBatchReader.from_batches(
        schema, (pa.RecordBatch.from_pylist(rows, schema=schema) for rows in batches if rows))
    ds.write_dataset(reader, base_dir=path, filesystem=fs, format="parquet",
                     basename_template=prefix + "-{i}.parquet",
                     # Parts from this run's other batches already sit here, under their own
                     # prefix, so an existing directory is expected rather than an error.
                     existing_data_behavior="overwrite_or_ignore")


def carried(batch, consumed):
    """The batch's own columns, less the ones this run consumed."""
    import pyarrow as pa

    return pa.Table.from_batches([batch]).drop_columns(
        [c for c in batch.schema.names if c in consumed])


# ── scoring one row ──────────────────────────────────────────────────────────────
def failed(why: str) -> dict:
    """A row that was not graded: every metric null, so nothing averages a failure as zero."""
    return {**{name: None for name, _ in SCORE_FIELDS}, "status": "error", "error": why}


def score_row(row: dict, root: str = ".") -> tuple[dict, list[dict]]:
    """Grade one manifest row, timed. Never raises -- a run that stops on its first bad row is
    a run you cannot finish, and what went wrong belongs on the row, not in a traceback."""
    started = time.perf_counter()
    scored, verdicts = _grade_row(row, root)
    scored["seconds"] = round(time.perf_counter() - started, 4)
    return scored, verdicts


def _grade_row(row: dict, root: str) -> tuple[dict, list[dict]]:
    try:
        gt = read_json(resolve(row["gt_path"], root))
        # Both transforms are required, not tidy: the scorer refuses a schema it cannot see
        # through, since an additionalProperties object behind a $ref would be graded while
        # the same object written inline is skipped.
        schema = resolve_refs(strip_benchmark_keys(json.loads(row["schema"])))
    except Exception:                                                   # noqa: BLE001
        return failed(traceback.format_exc()), []
    try:
        pred = read_json(resolve(row["pred_path"], root))
    except Exception:                                                   # noqa: BLE001
        return failed(traceback.format_exc()), []
    if not usable(pred):
        # No exception happened here, so there is no traceback to give: a prediction that
        # records its own failure is reported in the words it recorded.
        reason = pred.get("__error__") if isinstance(pred, dict) else None
        return failed(str(reason) if reason else "prediction has no fields to score"), []
    try:
        result = grade(pred, gt, schema, verdicts=True)
    except Exception:                                                   # noqa: BLE001
        # Ours, not theirs: a schema this scorer cannot see through, or a bug here.
        return failed(traceback.format_exc()), []

    scored = {**failed(None), "status": "scored",
              **{k: v for k, v in result.items() if k in RESERVED}}
    return scored, [{"doc_id": row["doc_id"],
                     "address": show(v.address),
                     # JSON-encoded, not str(): `None` and the string "None" must not collapse
                     # into one cell, and a float must come back as the number it was.
                     "gold_raw": None if v.gold_raw is None else json.dumps(v.gold_raw),
                     "gold_canon": v.gold_canon,
                     "pred_raw": None if v.pred_raw is None else json.dumps(v.pred_raw),
                     "pred_canon": v.pred_canon,
                     "verdict": v.verdict}
                    for v in result.get("verdicts", [])]


def score_batch(batch, out: str, name: str, root: str = ".") -> dict:
    """Score one batch, write its two parts, and return a status tally.

    **This is the unit of work, everywhere.** One process grades the rows in order; whoever
    wants parallelism runs more of these. Locally that is `run`'s pool; on a fan-out it is one
    container per call. Neither needs a scoring path of its own, and nothing here has an
    opinion about how many of it are running.

    `name` is what the parts are called and the caller owns it: two callers passing the same
    name overwrite each other's work.

    Memory is one document at a time. Verdicts go to the dataset writer as each document
    finishes rather than being gathered, which matters because addresses per document span
    four orders of magnitude.
    """
    out = out.rstrip("/")
    tally = {"scored": 0, "error": 0}
    scores: list[dict] = []

    def verdicts_of(rows):
        """Yield each document's verdicts as it is graded, keeping the score row."""
        for row in rows:
            scored, verdicts = score_row(row, root)
            tally[scored["status"]] += 1
            scores.append(scored)
            yield verdicts

    write_streaming(verdicts_of(batch.to_pylist()), VERDICT_FIELDS,
                    f"{out}/verdicts.parquet", name)
    # The score rows are sixteen small fields each and the batch they attach to is already in
    # memory, so they are written once the batch is done rather than streamed.
    write_part(scores, SCORE_FIELDS, f"{out}/scores.parquet/{name}.parquet", carried(batch, ()))
    return tally


def run(manifest: str, out: str, jobs: int = 1, batch_size: int | None = None,
        root: str = ".", overwrite: bool = False) -> dict:
    """Score a manifest into `<out>/scores.parquet` and `<out>/verdicts.parquet`.

    Returns a status tally.

    Parallelism is here and only here: `jobs` workers, each handed whole batches, each writing
    its own parts. Nothing is pickled back but a tally -- a worker's verdicts go straight from
    that worker to storage, and a batch that draws a slow document delays only itself.

    `batch_size` defaults to about four batches per worker, capped at 256: enough batches for
    the pool to pull around a slow one, few enough that the output is not thousands of tiny
    files. On a corpus with a heavy tail it is worth lowering -- see docs/FOLLOW_UPS.md, which
    records what a measured run says about it.
    """
    if jobs < 1 or (batch_size is not None and batch_size < 1):
        raise ValueError(f"jobs and batch_size are counts, so both are at least 1; "
                         f"got jobs={jobs}, batch_size={batch_size}")
    data = open_manifest(manifest)
    check_manifest(data, REQUIRED, RESERVED)
    prepare_out(out, overwrite)
    total = data.count_rows()
    batch_size = batch_size or max(1, min(256, total // (jobs * 4) or 1))
    log.info("scoring %d documents into %s (%d per batch, %d job%s)",
             total, out, batch_size, jobs, "" if jobs == 1 else "s")

    tally = {"scored": 0, "error": 0}
    work = ((batch, f"part-{n:05d}")
            for n, batch in enumerate(data.to_batches(batch_size=batch_size)))

    def done(counts, n):
        for status, count in counts.items():
            tally[status] += count
        log.info("part-%05d: %d/%d documents  %s", n, sum(tally.values()), total,
                 "  ".join(f"{k}={v}" for k, v in tally.items()))

    if jobs == 1:
        for n, (batch, name) in enumerate(work):
            done(score_batch(batch, out, name, root), n)
    else:
        with cf.ProcessPoolExecutor(max_workers=jobs) as pool:
            # A bounded window rather than submitting every batch at once: a queued batch holds
            # its rows, and a manifest is meant to stream rather than land in the pipe whole.
            counts = in_order(lambda item: pool.submit(score_batch, item[0], out, item[1], root),
                              work, jobs * 2)
            for n, c in enumerate(counts):
                done(c, n)
    return tally
