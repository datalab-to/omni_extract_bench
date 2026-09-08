"""Omni Extract Bench -- one scorer for document-extraction benchmarks.

The public surface is deliberately small:

    from omni_extract_bench import grade, canon_key

    result = grade(prediction, ground_truth, schema)
    result["leaf_accuracy"]   # 0-100, the headline number
    result["recall"]          # gold rows matched / gold rows
    result["precision"]       # gold rows matched / predicted rows
    result["matching_exact"]  # False if an array was too large to solve exactly

`canon_key(value)` is the single definition of "are these two values equal?" used by leaf
scoring, row-pair weighting, and blocking alike.
"""

from .grading import fair_grade as grade
from .grading import fair_grade_value as grade_value
from .grading import canon_key, cmp_leaf
from .matching import match_rows
from .prediction_io import usable

__all__ = [
    "grade",
    "grade_value",
    "canon_key",
    "cmp_leaf",
    "match_rows",
    "usable",
]
__version__ = "0.1.0"
