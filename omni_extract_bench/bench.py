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
from typing import Any, Iterable, Iterator, NamedTuple

from . import corpus as corpus_atlas
from .corpus import Entry
from .harness.dialects import resolve_refs, strip_benchmark_keys
from .harness.prediction_io import usable
from .score import grade, show

class Document(NamedTuple):
    """One document of a benchmark, before any prediction."""

    doc_id: str
    gt: object
    schema: object


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


#: Bumping this orphans every score that joined on the old ids. `scores` keys on
#: `prediction_id`, so a change here is a migration, not a refactor. The version travels with
#: the data rather than living only in this constant.
PREDICTION_ID_VERSION = "v1"


def prediction_id(result_bytes: bytes) -> str:
    """Identify an extraction by its stored bytes.

    There is no envelope to see through: a stored prediction is the bare extraction, so this
    is a plain hash of the file. That is what keeps it reproducible in any language, with no
    unwrap rule, no schema dependency and no canonicalisation spec to agree on.
    """
    return hashlib.sha256(result_bytes).hexdigest()


def document(root: Path, entry: Entry) -> Document:
    """Load the document an atlas row names.

    Takes an entry rather than a bare id so a worker can load exactly what it was handed
    without re-reading the atlas. A parsed ground truth can be tens of megabytes, so it is read
    here rather than pickled across.
    """
    root = Path(root)
    gt_file, schema_file = root / entry.ground_truth_path, root / entry.schema_path
    for f in (gt_file, schema_file):
        if not f.exists():
            raise FileNotFoundError(
                f"{entry.doc_id}: {f.relative_to(root)} is listed in the atlas but missing")
    return Document(
        doc_id=entry.doc_id,
        gt=json.loads(gt_file.read_bytes()),
        schema=resolve_refs(strip_benchmark_keys(json.loads(schema_file.read_bytes()))),
    )


def documents(target: Path) -> Iterator[Document]:
    """Every document the atlas lists, in its order. Takes a corpus directory or an atlas file.

    Raises:
        FileNotFoundError: if there is no atlas, or a row names a file that is not there. The
            atlas is what the run follows, so a row pointing at nothing is a stop, not a skip.
    """
    root, _atlas = corpus_atlas.locate(Path(target))
    for entry in corpus_atlas.read(target):
        yield document(root, entry)


#: A prediction set may carry one of these beside its payloads. Optional: its columns are
#: added to the rows for that source, and nothing needs it to be there.
PREDICTION_META = "predictions.parquet"


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


def prediction_meta(root: Path) -> tuple[dict, list[str]]:
    """Whatever a prediction set records about its own predictions, by doc_id.

    How long a vendor took, whether it was recovered after a timeout, what it cost: facts about
    producing the prediction rather than about scoring it. They belong on the row, and nothing
    here can compute them.

    Optional by design. Our runs carry it because `build_vendors.py` writes one; someone
    scoring a directory of JSON they just produced has nothing to carry and should not have to
    invent a manifest to be scored.

    Returns the metadata and the column names dropped for colliding with ours -- `error` means
    the vendor's failure on one side and a scoring failure on the other, and silently letting
    either win would make a column mean two things.
    """
    path = Path(root) / PREDICTION_META
    if not path.exists():
        return {}, []
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    ours = set(SUMMARY_COLUMNS)
    collided = sorted({k for r in rows for k in r} & ours - {"doc_id"})
    return ({r["doc_id"]: {k: v for k, v in r.items() if k not in ours} for r in rows},
            collided)


def cost(node: Any) -> int:
    """Roughly what a ground truth will cost to score: its longest array.

    The scorer solves an assignment problem per array, so the biggest one dominates. Used to
    schedule longest-first, because the spread is extreme -- a median document in the published
    corpus has 6 array rows and the largest has 26,725, so any other order finishes the cheap
    work and then waits on one straggler.

    Computed from the JSON rather than read from an atlas column: it takes half a second across
    660 documents, and an atlas that must carry it is an atlas anyone else has to produce.
    """
    if isinstance(node, dict):
        return max((cost(v) for v in node.values()), default=0)
    if isinstance(node, list):
        return max(len(node), max((cost(v) for v in node), default=0))
    return 0


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


#: Column names this module owns on a summary row. A prediction set's own metadata may not
#: use them, because each would then mean two different things depending on the row.
#: What the verdicts used to be called, before they were renamed to the words the summary
#: already counted them under. A run written by an older scorer still reads, and there is ONE
#: table saying so -- a second reader inventing its own would be the same divergence that
#: `canon_key` exists to prevent, in a smaller place.
#:
#: Delete this once no run worth reading predates the rename.
RENAMED = {"match": "matched", "wrong value": "misread", "missing": "unfound",
           "invented item": "invented_item", "invented field": "invented_field",
           "skipped (open map)": "skipped_open_map"}


def verdict(name: str) -> str:
    """A verdict under the name the scorer uses now, whenever it was written."""
    return RENAMED.get(name, name)


SUMMARY_COLUMNS = frozenset({
    "doc_id", "prediction_id", "kind", "error",
    "accuracy", "f1", "precision", "recall", "found", "read_right", "matched", "total",
    "asserted", "misread", "unfound", "fabricated", "invented_item", "invented_field",
    "gt_rows", "pred_rows", "matched_rows", "matching_exact",
})


def key_of(case: Case) -> dict:
    """The columns that identify a scored row.

    `prediction_id` is the prediction's own bytes hashed, so it still tells a re-run apart from
    an unchanged one -- which is the question the summary is asked most often after "what did it
    score".
    """
    return {"doc_id": case.doc.doc_id, "prediction_id": case.prediction_id}


def summary_row(case: Case, outcome: Outcome) -> dict:
    """One row for the summary table. Metrics are absent when nothing was graded."""
    row = {**key_of(case), "kind": outcome.kind, "error": outcome.error}
    if outcome.summary is not None:
        row.update({k: v for k, v in outcome.summary.items() if not isinstance(v, list)})
    return row


def verdict_rows(case: Case, outcome: Outcome) -> list[dict]:
    """One row per address: what the gold had, what the prediction had, and the verdict.

    Each side twice, raw and canonical. The raw values are JSON-encoded, not `str()`: `None`
    and the string `"None"` must not collapse into the same cell, and a float must come back as
    the number it was -- an audit trail that cannot be compared byte-for-byte is decoration.
    The canonical values are `canon_key`'s own output, already strings, and are what says WHY
    two values counted as equal. Storing them is what stops a reader having to recompute them
    and risk a second opinion about what equal means.
    """
    if not outcome.verdicts:
        return []
    key = key_of(case)
    return [{**key,
             "address": show(v.address),
             "gold_raw": None if v.gold_raw is None else json.dumps(v.gold_raw),
             "gold_canon": v.gold_canon,
             "pred_raw": None if v.pred_raw is None else json.dumps(v.pred_raw),
             "pred_canon": v.pred_canon,
             "verdict": v.verdict}
            for v in outcome.verdicts]
