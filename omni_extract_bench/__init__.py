"""Omni Extract Bench: one scorer for document-extraction benchmarks, and the benchmark built on it.

    from omni_extract_bench import score
    result = score(prediction, ground_truth, schema)     # accuracy, f1, precision, recall, ...

Scoring takes three parsed documents and needs nothing else. Running the benchmark (predicting with vendors,
resuming, grading whole runs) is `benchmark.default()`, built on omni-bench-core; the vendor adapters are in
`harness` (`pip install 'omni-extract-bench[harness]'`). The scorer never imports the harness.
"""

from . import data, scorers
from .scorers.field_matching import Verdict, check, gold, score, summarize

__all__ = ["check", "data", "gold", "score", "scorers", "summarize", "Verdict"]
