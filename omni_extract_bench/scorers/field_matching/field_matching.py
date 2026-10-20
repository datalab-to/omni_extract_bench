"""Field matching over a dataset: each document's gold read and checked (`gold`, `check`), every document's
line summarized (`summarize`), and `FieldMatching`, the scorer as omni-bench-core grades a run, which scores each
document with `compare.score`.

It is the one file of the package that imports outside it (omni-bench-core's `Scorer`, and oeb's row type). Every
other file is the comparison itself, which works on its own.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

from omni_bench_core import errors, scorer

from ... import data
from . import compare


class Gold(NamedTuple):
    """What field matching compares a prediction against: the gold extraction, and the schema both were
    asked against."""

    extraction: dict
    schema: dict


@dataclasses.dataclass(frozen=True)
class FieldMatching(scorer.Scorer[data.Doc, Gold, dict[str, Any]]):
    """Field matching as an omni-bench-core scorer.

    Each document's score is `compare.score`'s numbers, and its details are one verdict per address,
    written to `details/<doc_id>.jsonl`. The summary is `summarize`'s; its headline is the
    accuracy over the scored documents, which omni-bench-core multiplies by coverage, so a failed document counts
    as 0.

    Scoring pairs rows with an assignment solver, which is slow on large arrays: one document of the public
    dataset, a list of 1238 entities with arrays inside each, takes about two minutes on its own. So documents
    are scored in a pool of processes, and each score is kept (`cache`), so grading again scores only the
    documents whose prediction or gold changed. Keeping scores is safe because scoring is deterministic."""

    name: ClassVar[str] = "field_matching"
    headline_field: ClassVar[str] = "accuracy"
    workers: ClassVar[int] = min(8, os.cpu_count() or 1)
    cache: ClassVar[bool] = True

    def gold(self, row: data.Doc) -> Gold:
        return Gold(gold(row), row.schema)

    def prediction(self, output: Any) -> dict[str, Any]:
        """The extraction; `errors.SystemFailed` when it isn't a JSON object, so the document fails."""
        if not isinstance(output, dict):
            raise errors.SystemFailed(f"the prediction isn't a JSON object; it holds a JSON "
                                      f"{'null' if output is None else type(output).__name__}")
        return output

    def score(self, pred: dict[str, Any], gold: Gold) -> scorer.Score:
        got = compare.score(pred, gold.extraction, gold.schema, verdicts=True)
        return scorer.Score(got, [v._asdict() for v in got.pop("verdicts")])

    def summarize(self, lines: Sequence[scorer.Line]) -> scorer.Summary:
        """`summarize`, without the counts omni-bench-core adds itself (`rows`, `failed`, `coverage`). Its
        headline is the accuracy over the scored documents, 0 when none was, which omni-bench-core multiplies by
        coverage."""
        s = summarize(lines)
        return {**{k: v for k, v in s.items() if k not in ("rows", "failed", "coverage")},
                "headline": s["accuracy"] or 0.0}


def gold(doc: Any) -> dict:
    """A document's ground truth, the extraction, from its gold file (`doc.gt_path`, a `data.Doc`'s); a
    `GoldError` naming the document when the file can't be read or isn't a JSON object."""
    try:
        g = json.loads(Path(doc.gt_path).read_text())
    except (OSError, ValueError) as exc:
        raise GoldError(f"{doc.doc_id}: its gold {doc.gt_path} can't be read: {exc}") from None
    if isinstance(g, dict): return g
    kind = "null" if g is None else type(g).__name__
    raise GoldError(f"{doc.doc_id}: its gold {doc.gt_path} is a JSON {kind}; it must be a JSON object, the "
                    "extraction")


class GoldError(errors.UsageError):
    """A gold file that can't be read, or isn't the extraction, a JSON object; the message names its document."""


def check(docs: Iterable[Any]) -> None:
    """Raises a `GoldError` naming the first document of `docs` (a manifest's, `data.read_manifest`)
    whose gold file can't be read, isn't JSON, or isn't a JSON object, the one shape `score` takes.
    What `oeb check` runs.

    It does not check a gold against its schema, nor that the scorer can read a schema: finding out
    means scoring, which costs as much as grading, and grading names a document that fails."""
    if hasattr(docs, "gt_path"):
        raise TypeError("check takes the documents, an iterable of them; for one, pass [doc]")
    for d in docs: gold(d)


ERRORS = ("misread", "unfound", "fabricated", "invented_item", "invented_field")


def summarize(lines: Iterable[dict]) -> dict:
    """A run's score lines (scores.jsonl's, one per document) as one summary (`_over`), over every line and
    then per suite, each suite in the same shape. A line carries its document's suite, so the lines are all
    it needs. A line's status is `scored` or `failed` (no usable prediction)."""
    lines = list(lines)
    if not lines:
        raise ValueError("summarize takes at least one score line; there is nothing to summarize")
    if bad := next((x for x in lines if not isinstance(x, dict) or "suite" not in x or "status" not in x), None):
        raise ValueError(f"summarize takes score lines, each with its document's suite and status, as "
                         f"scores.jsonl holds; got {bad!r:.120}. score()'s result "
                         "is one line's numbers, without them")
    by_suite: dict[str, list[dict]] = collections.defaultdict(list)
    for x in lines:
        by_suite[x["suite"]].append(x)
    return {**_over(lines), "per_suite": {s: _over(v) for s, v in sorted(by_suite.items())}}


def _over(lines: list[dict]) -> dict:
    """How a non-empty set of documents went: the headline, the counts, coverage, the three metrics over the
    scored documents, and where the errors were.

    Note: `headline` counts a failed document as 0, so a system can't rank higher by failing on its
    hard documents; `accuracy` is over the scored ones, and `headline` is `accuracy * coverage`. The
    metrics over the scored are None when none was scored: 0.0 would read as everything wrong (and a 0.0
    rate as nothing wrong)."""
    scored = [x for x in lines if x["status"] == "scored"]

    def mean(of) -> float | None:
        return sum(of(x) for x in scored) / len(scored) if scored else None

    return {
        "headline": sum(x["accuracy"] for x in scored) / len(lines),
        "rows": len(lines),
        "scored": len(scored),
        "failed": len(lines) - len(scored),
        "coverage": len(scored) / len(lines),
        "accuracy": mean(lambda x: x["accuracy"]),
        "precision": mean(lambda x: x["precision"]),
        "recall": mean(lambda x: x["recall"]),
        **{f"{e}_rate": mean(lambda x, e=e: x[e] / x["total"] if x["total"] else 0.0)
           for e in ERRORS},
    }
