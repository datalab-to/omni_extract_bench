"""Scoring a whole benchmark: documents in, rows out.

`score.py` grades one prediction against one ground truth. This is the layer above it -- the
one that knows a benchmark is a set of documents, that a prediction may be missing or may be
an error blob, and that the answer is a table. It knows nothing about parquet, HuggingFace or
R2, and nothing about where any of its inputs came from.

**A corpus is its atlas.**

    <corpus>/corpus.parquet          which documents are in the benchmark
    <corpus>/<doc_id>/ground_truth.json
    <corpus>/<doc_id>/schema.json

Scoring runs over the rows in `corpus.parquet`, and each row's files are checked against the
hashes it records. See `corpus.py` for why. Anything else in the directory is ignored --
`document.pdf`, `source.json`, page images -- and a document the atlas does not list is not in
the benchmark, which is how curation works.

The published corpus is one instance of this, not a special case: the same `documents()` reads
both, and the extra columns ours carries are additive.

Predictions are `<doc_id>.json`, holding the bare extraction. No envelope, no atlas, nothing
to unwrap.

Every stage is an iterable of plain data, so any of them can be replaced without the others
noticing -- a different corpus loader, predictions pulled from a bucket, rows written as
JSONL instead of parquet:

    cases(documents(corpus), predictions(preds))  ->  score(case)  ->  rows

The one thing that is not swappable is the metric.
"""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple

from . import corpus as corpus_atlas
from .corpus import GROUND_TRUTH, SCHEMA, Stale
from .layout import prediction_id
from .dialects import resolve_refs, strip_benchmark_keys
from .prediction_io import usable
from .score import grade, show

class Document(NamedTuple):
    """One document of a benchmark, before any prediction.

    The hashes are what let a score record *what it scored*. A corrected ground truth is a
    different `gt_sha256`, so the corrected score is a different row rather than the same row
    disagreeing with itself.
    """

    doc_id: str
    gt: object
    schema: object
    gt_sha256: str
    schema_sha256: str


class Case(NamedTuple):
    """One scoreable unit: a document and a prediction for it.

    The prediction is carried as bytes, not as parsed JSON, for two reasons. `prediction_id`
    hashes what was stored, so the bytes are the identity. And a file that will not parse has
    to become an outcome rather than an exception -- otherwise one malformed prediction takes
    the whole run with it, which is the opposite of what a benchmark should do.
    """

    doc: Document
    prediction_id: str
    raw: bytes


class Outcome(NamedTuple):
    """What came of scoring one case.

    Three kinds, and the distinctions are the point:

        graded      scored on its merits
        unusable    the provider returned nothing scoreable -- an error payload, an empty
                    object, bytes that are not JSON. Its fault, and it counts against it.
        failed      THIS harness could not score it. Our fault, and it must not be read as
                    the provider doing badly.

    `prediction_io.usable` exists because the first two were once told apart two different
    ways, and a provider read as 100% coverage while 37 of its 45 outputs were empty. Merging
    the third in would let a broken schema in our corpus read as a poor vendor.

    A caller that averages `accuracy` without looking at `kind` reproduces both mistakes.
    """

    kind: str                 # graded | unusable | failed
    error: str | None
    summary: dict | None
    verdicts: list | None


def document(root: Path, entry) -> Document:
    """Load one document named by an atlas row, checking it is what the atlas says.

    Takes an entry rather than a bare id so a worker can load exactly what it was handed
    without re-reading the atlas. A parsed ground truth can be tens of megabytes, so it is
    read here rather than pickled across.

    Raises:
        Stale: if a file's contents no longer match the atlas. Scoring against data that has
            moved underneath the benchmark is worse than stopping: the numbers would look
            ordinary and mean something else.
    """
    root = Path(root)
    gt_file, schema_file = root / entry.ground_truth_path, root / entry.schema_path
    for f in (gt_file, schema_file):
        if not f.exists():
            raise FileNotFoundError(
                f"{entry.doc_id}: {f.relative_to(root)} is listed in the atlas but missing")
    gt_bytes, schema_bytes = gt_file.read_bytes(), schema_file.read_bytes()
    gt_sha = hashlib.sha256(gt_bytes).hexdigest()
    schema_sha = hashlib.sha256(schema_bytes).hexdigest()
    for got, want, what in ((gt_sha, entry.gt_sha256, GROUND_TRUTH),
                            (schema_sha, entry.schema_sha256, SCHEMA)):
        if got != want:
            raise Stale(
                f"{entry.doc_id}: {what} has changed since the atlas was written.\n"
                f"  If that was intended, record it:  oeb build-corpus --corpus {root} "
                f"--refresh\n"
                f"  If it was not, the benchmark's data has drifted underneath it.")
    return Document(
        doc_id=entry.doc_id,
        gt=json.loads(gt_bytes),
        schema=resolve_refs(strip_benchmark_keys(json.loads(schema_bytes))),
        gt_sha256=gt_sha,
        schema_sha256=schema_sha,
    )


def documents(root: Path) -> Iterator[Document]:
    """Every document the atlas lists, in its order.

    Raises:
        FileNotFoundError: if there is no atlas, or a listed file is missing.
        Stale: if a file no longer matches its recorded hash.
    """
    root = Path(root)
    for entry in corpus_atlas.read(root):
        yield document(root, entry)


def predictions(root: Path) -> Iterator[tuple[str, bytes]]:
    """`<doc_id>.json` for each prediction, as bytes.

    Bytes rather than parsed JSON because `prediction_id` hashes what was stored. Parsing and
    re-dumping would hash our formatting instead of the vendor's.

    Predictions are NOT listed in an atlas. The corpus is a fixed thing that must not drift,
    which is what the atlas is for; a prediction set is whatever you just produced, and
    requiring a manifest to score your own output would be ceremony with nothing to protect.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"predictions {root} is not a directory")
    for f in sorted(root.glob("*.json")):
        yield f.stem, f.read_bytes()


def cases(docs: Iterable[Document], preds: Iterable[tuple[str, bytes]]) -> Iterator[Case]:
    """Join documents to predictions on `doc_id`.

    A document with no prediction yields no case. Absence is a fact about coverage and belongs
    in the caller's report, not in a fabricated zero -- see `Outcome.kind`.

    Raises:
        ValueError: for a prediction naming no document, suggesting near matches. The usual
            cause is a filename convention (`invoice-003.pdf.json`), so the fix is almost
            always visible once you are shown what was close.
    """
    by_id = {d.doc_id: d for d in docs}
    for doc_id, raw in preds:
        doc = by_id.get(doc_id)
        if doc is None:
            close = difflib.get_close_matches(doc_id, by_id, n=3, cutoff=0.6)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ValueError(
                f"prediction {doc_id!r} matches no document in the corpus.{hint} "
                f"A prediction file must be named <doc_id>.json.")
        yield Case(doc=doc, prediction_id=prediction_id(raw), raw=raw)


def score(case: Case, verdicts: bool = True) -> Outcome:
    """Grade one case. Never raises.

    An unusable prediction is not graded. An error blob scores as though the model tried and
    missed every field, which reads identically to a genuine total failure and is not the same
    thing at all.

    Nothing here raises, because a benchmark that stops on its first bad document is a
    benchmark you cannot run. What went wrong is recorded on the row and reported at the end.

    Verdicts come back from the same pass, so asking for them costs the memory of one
    `Verdict` per address and no extra matching.
    """
    try:
        pred = json.loads(case.raw)
    except ValueError as exc:
        return Outcome("unusable", f"not JSON: {exc}", None, None)
    if not usable(pred):
        error = pred.get("__error__") if isinstance(pred, dict) else None
        return Outcome("unusable", str(error) if error is not None else None, None, None)
    try:
        result = grade(pred, case.doc.gt, case.doc.schema, verdicts=verdicts)
    except Exception as exc:                                            # noqa: BLE001
        # Ours, not theirs: a schema this scorer cannot see through, or a bug here.
        return Outcome("failed", f"{type(exc).__name__}: {exc}", None, None)
    return Outcome("graded", None, result, result.pop("verdicts", None))


def key_of(case: Case) -> dict:
    """The columns that identify a scored row.

    A score is a function of the prediction bytes, the ground truth, the schema and the
    scorer's code. The first three are here; the fourth is the run's scorer version. Recording
    fewer would make two legitimately different numbers look like the same number disagreeing
    with itself.
    """
    return {"doc_id": case.doc.doc_id, "prediction_id": case.prediction_id,
            "gt_sha256": case.doc.gt_sha256, "schema_sha256": case.doc.schema_sha256}


def summary_row(case: Case, outcome: Outcome) -> dict:
    """One row for the summary table. Metrics are absent when nothing was graded."""
    row = {**key_of(case), "kind": outcome.kind, "error": outcome.error}
    if outcome.summary is not None:
        row.update({k: v for k, v in outcome.summary.items() if not isinstance(v, list)})
    return row


def verdict_rows(case: Case, outcome: Outcome) -> list[dict]:
    """One row per address: what the gold had, what the prediction had, and the verdict.

    Values are JSON-encoded, not `str()`. `None` and the string `"None"` must not collapse
    into the same cell, and a float must come back as the number it was -- an audit trail that
    cannot be compared byte-for-byte is decoration.
    """
    if not outcome.verdicts:
        return []
    key = key_of(case)
    return [{**key,
             "address": show(v.address),
             "gold": None if v.gold is None else json.dumps(v.gold),
             "pred": None if v.pred is None else json.dumps(v.pred),
             "verdict": v.verdict}
            for v in outcome.verdicts]
