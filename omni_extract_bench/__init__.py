"""Omni Extract Bench -- one scorer for document-extraction benchmarks.

The public surface is one function:

    from omni_extract_bench import score

    result = score(prediction, ground_truth, schema)
    result["accuracy"]        # 0 to 1, the headline number
    result["f1"]              # report it next to accuracy: it charges a hopeless guess
    result["recall"]          # matched addresses / gold addresses
    result["precision"]       # matched addresses / asserted addresses
    result["matching_exact"]  # False if an array was too large to solve exactly

An address is a keypath, so `precision` and `recall` treat a keypath-and-value as one
detection -- a value read wrongly is charged on both sides. Row counts are reported
separately as `gt_rows`, `pred_rows` and `matched_rows`.

`score(..., verdicts=True)` adds `result["verdicts"]`: one `Verdict` per address, saying what
happened there and what each side was compared as. It costs almost nothing on top of the
score, because aligning the two documents is nearly all of the work and it is already done.

`canon_key(value)` is the single definition of "are these two values equal?" used by leaf
scoring, row-pair weighting, and blocking alike.

THREE PARSED DOCUMENTS IN, ONE DICT OUT. There is no runner here and no table format: scoring
a corpus is a loop over `score`, written the way your corpus is laid out. To produce the
predictions with the vendor adapters in this repository, see `omni_extract_bench.harness`
(`pip install 'omni-extract-bench[harness]'`); nothing in the scorer imports it, so a machine
that only scores never installs a vendor SDK.
"""

from .matching import match_rows
from .metric import Verdict, score
from .values import canon_key, cmp_leaf

__all__ = [
    "score",
    "Verdict",
    "canon_key",
    "cmp_leaf",
    "match_rows",
]
__version__ = "0.1.2"
