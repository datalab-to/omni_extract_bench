#!/usr/bin/env python3
"""Shrinks the exactness budget, so the greedy pairing fallback is used on small arrays.

    with approximate():
        result = score(pred, gt, schema)     # took the greedy path
    assert result["matching_exact"] is False

The greedy path is real (three documents in the corpus take it), but only above 20,000 rows. All three
budgets come down together: leaving `MAX_CELLS_DENSE` alone would send the block back to the exact path.
"""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omni_extract_bench.scorers.field_matching import matching as _matching                         # noqa: E402

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
