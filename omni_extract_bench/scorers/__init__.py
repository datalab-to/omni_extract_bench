"""The scorers. There is one, field matching: a prediction and its gold extraction compared value by value,
each value addressed by its key path, with rows paired by content.

`field_matching` is the scorer's package. Its functions are the math, usable on their own
(`field_matching.score(pred, gold, schema)`). `field_matching.FieldMatching` is the same scorer as an
omni-bench-core `Scorer`, which grades a whole run.
"""

from . import field_matching  # noqa: F401
