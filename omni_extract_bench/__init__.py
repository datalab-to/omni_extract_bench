"""Omni Extract Bench -- one scorer for document-extraction benchmarks.

The public surface is deliberately small:

    from omni_extract_bench import grade, canon_key

    result = grade(prediction, ground_truth, schema)
    result["accuracy"]        # 0-100, the headline number
    result["f1"]              # report it next to accuracy: it charges a hopeless guess
    result["recall"]          # matched addresses / gold addresses
    result["precision"]       # matched addresses / asserted addresses
    result["matching_exact"]  # False if an array was too large to solve exactly

An address is a keypath, so `precision` and `recall` treat a keypath-and-value as one
detection -- a value read wrongly is charged on both sides. Row counts are reported
separately as `gt_rows`, `pred_rows` and `matched_rows`.

`canon_key(value)` is the single definition of "are these two values equal?" used by leaf
scoring, row-pair weighting, and blocking alike.
"""

from .matching import match_rows
from .score import explain, grade
from .values import canon_key, cmp_leaf
from .harness.prediction_io import usable

__all__ = [
    "grade",
    "explain",
    "canon_key",
    "cmp_leaf",
    "match_rows",
    "usable",
]
__version__ = "0.1.0"
