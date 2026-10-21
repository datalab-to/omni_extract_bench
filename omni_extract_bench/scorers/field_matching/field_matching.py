"""Field matching over a dataset:

- `FieldMatching`: the scorer a run is graded with, scoring each document with `compare.score`;
- `gold` and `check`: each document's gold, read and checked; and
- `summarize` and `metrics`: a whole scores.jsonl, or a set of scored lines, as one summary.

It is the one module of the package that imports omni-bench-core; the rest is the comparison, on its
own."""
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
    """What a prediction is compared against: the gold extraction, and the schema both were asked against."""

    extraction: dict
    schema: dict


@dataclasses.dataclass(frozen=True)
class FieldMatching(scorer.Scorer[data.Doc, Gold, dict[str, Any]]):
    """Field matching as an omni-bench-core scorer.

    A document's result is `compare.score`'s numbers, and its details one verdict per address. The
    summary is `metrics` over the scored documents, overall and per suite, with accuracy as the
    headline. Scoring large arrays is slow (one public document takes about two minutes), so documents
    are scored in a pool of processes and each score is cached."""

    name: ClassVar[str] = "field_matching"
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

    def row_fields(self, row: data.Doc) -> dict[str, Any]:
        return {"suite": row.suite}

    def summarize(self, lines: Sequence[scorer.Line]) -> scorer.Summary:
        """`metrics` of the scored documents, overall and per suite, with `rows`, how many were scored; the
        headline is their accuracy."""
        by_suite: dict[str, list[dict]] = collections.defaultdict(list)
        for x in lines:
            by_suite[x["row"]["suite"]].append(x)
        m = metrics(lines)
        return {"headline": m["accuracy"], "headline_is": "accuracy", "rows": len(lines), **m,
                "per_suite": {s: {"rows": len(v), **metrics(v)} for s, v in sorted(by_suite.items())}}


def gold(doc: Any) -> dict:
    """A document's gold extraction, from its gold file (`doc.gt_path`); a `ValueError` naming the document
    when it can't be read or isn't a JSON object."""
    try:
        g = json.loads(Path(doc.gt_path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{doc.doc_id}: its gold {doc.gt_path} can't be read: {exc}") from None
    if isinstance(g, dict): return g
    kind = "null" if g is None else type(g).__name__
    raise ValueError(f"{doc.doc_id}: its gold {doc.gt_path} is a JSON {kind}; it must be a JSON object, the "
                     "extraction")


def check(docs: Iterable[Any]) -> None:
    """Raises a `ValueError` naming the first document of `docs` whose gold can't be read or isn't a JSON
    object; what `oeb check` runs. Checking a gold against its schema would mean scoring it, so that is
    left to grading."""
    if hasattr(docs, "gt_path"):
        raise TypeError("check takes the documents, an iterable of them; for one, pass [doc]")
    for d in docs: gold(d)


ERRORS = ("misread", "unfound", "fabricated", "invented_item", "invented_field")


def summarize(lines: Iterable[dict]) -> dict:
    """A whole scores.jsonl (one line per document, failed ones included, each with its suite under `row`)
    as one summary: `_over` of every line, and of each suite's lines."""
    lines = list(lines)
    if not lines:
        raise ValueError("summarize takes at least one score line; there is nothing to summarize")
    if bad := next((x for x in lines if not isinstance(x, dict) or "suite" not in x.get("row", {})
                    or "status" not in x), None):
        raise ValueError(f"summarize takes score lines, each with its status and its document's suite under row, "
                         f"as scores.jsonl holds; got {bad!r:.120}. score()'s result "
                         "is one line's numbers, without them")
    by_suite: dict[str, list[dict]] = collections.defaultdict(list)
    for x in lines:
        by_suite[x["row"]["suite"]].append(x)
    return {**_over(lines), "per_suite": {s: _over(v) for s, v in sorted(by_suite.items())}}


def _over(lines: list[dict]) -> dict:
    """How a non-empty set of documents went: the headline, the counts, coverage, the metrics over the
    scored documents, and where the errors were. `headline` is `accuracy * coverage`, so a failed
    document counts as 0. The metrics are None when no document was scored, since 0.0 would read as
    everything wrong."""
    scored = [x for x in lines if x["status"] == "scored"]
    m = metrics(scored) if scored else {k: None for k in ("accuracy", "precision", "recall",
                                                           *(f"{e}_rate" for e in ERRORS))}
    return {
        "headline": sum(x["accuracy"] for x in scored) / len(lines),
        "rows": len(lines),
        "scored": len(scored),
        "failed": len(lines) - len(scored),
        "coverage": len(scored) / len(lines),
        **m,
    }


def metrics(scored: Sequence[dict]) -> dict:
    """The mean accuracy, precision and recall of a non-empty set of scored lines, and the mean rate of
    each kind of error (`ERRORS`)."""
    def mean(of) -> float:
        return sum(of(x) for x in scored) / len(scored)

    return {
        "accuracy": mean(lambda x: x["accuracy"]),
        "precision": mean(lambda x: x["precision"]),
        "recall": mean(lambda x: x["recall"]),
        **{f"{e}_rate": mean(lambda x, e=e: x[e] / x["total"] if x["total"] else 0.0) for e in ERRORS},
    }
