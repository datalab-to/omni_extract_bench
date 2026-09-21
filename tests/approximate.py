#!/usr/bin/env python3
"""Shrink the exactness budget so the greedy fallback engages.

    from tests.approximate import approximate

    with approximate():
        result = score(pred, gt, schema)     # took the greedy path
    assert result["matching_exact"] is False

The greedy path is reachable for real -- three documents in the corpus take it -- but the
ceiling is 20,000 rows, and building an array that size in a test costs minutes for no extra
coverage. This exercises the property that an approximate score ANNOUNCES ITSELF, without
paying for the array.

HERE RATHER THAN IN `matching`, because it is a test's business and not the grader's. It lived
there as `force_approximate()` and was called by nothing the package ships.

All THREE budgets have to come down together. `MAX_CELLS_DENSE` is the ceiling above which
exact is refused however sparse the alternative looks, so leaving it alone lets the density
comparison route the block back to exact and this stops forcing anything.
"""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omni_extract_bench import matching as _matching                         # noqa: E402

TINY = (8, 64, 64)


@contextlib.contextmanager
def approximate():
    """Every `match_rows` inside this block takes the greedy path. Restores on the way out."""
    saved = (_matching.MAX_EXACT, _matching.MAX_CELLS, _matching.MAX_CELLS_DENSE)
    (_matching.MAX_EXACT, _matching.MAX_CELLS, _matching.MAX_CELLS_DENSE) = TINY
    try:
        yield
    finally:
        (_matching.MAX_EXACT, _matching.MAX_CELLS, _matching.MAX_CELLS_DENSE) = saved
