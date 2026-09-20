#!/usr/bin/env python3
"""Provably-optimal array row matching for the benchmark grader."""

import numpy as _np
import scipy.sparse as _sp
from scipy.optimize import linear_sum_assignment as _lsa
from scipy.sparse.csgraph import min_weight_full_bipartite_matching as _sparse_lsa

MAX_EXACT = 20000
MAX_CELLS = 250 * 10**6
MAX_CELLS_DENSE = 500 * 10**6
EXACT_BYTES_PER_CELL = 8
GREEDY_BYTES_PER_PAIR = 72


def _exact_ok(n, m, positives=None):
    """True when one block should be solved exactly rather than greedily.

    Args:
        n, m: the two sides' element counts, in either order.
        positives: an UPPER bound on the pairs that can carry positive weight, or None if the
            caller did not compute one. Only consulted between `MAX_CELLS` and
            `MAX_CELLS_DENSE`, where the two paths have to be compared rather than assumed.

    Returns:
        Whether `optimal_pairs` may be called directly. `match_rows` falls back to greedy when
        this is False, and records that the score is approximate.

    Three bands: below `MAX_CELLS` exact outright, above `MAX_CELLS_DENSE` refused outright
    so its memory is bounded whatever the data does, and between them whichever is cheaper --
    which depends on density. Without `positives` the answer is the same as before it existed;
    an over-estimate biases towards exact, whose cost is known before it runs.
    """
    lo, hi = (n, m) if n <= m else (m, n)
    if lo > MAX_EXACT:
        return False
    cells = lo * hi
    if cells <= MAX_CELLS:
        return True
    if positives is None or cells > MAX_CELLS_DENSE:
        return False
    return cells * EXACT_BYTES_PER_CELL <= positives * GREEDY_BYTES_PER_PAIR


def sparse_pairs(pred_rows, gt_rows, weights):
    """Maximum-weight pairing from a SPARSE weight matrix, exactly, without a dense one.

    The dense solver needs n*m float64 -- 5.7 GB on the largest array here -- which is the
    whole reason `_greedy` exists. When the weights are sparse they are 50 MB instead, and
    scipy's sparse assignment solves them exactly.

    **The escape columns are what make it the same problem.** `_sparse_lsa` finds a FULL
    matching: every row on the smaller side must be assigned. This metric must be free to
    leave a row unpaired -- pairing two rows that share nothing is not a match -- and forcing
    a full matching answers a different, worse question. On a 2x2 where one row can only reach
    the column another row is worth 10 on, the full matching scores 2 against the correct 10.

    So every row is given its own private column, worth less than any real edge can be: real
    weights are scaled by `n + m + 1`, escapes are worth 1, and a matching's escapes can never
    outweigh a single real unit. Taking an escape IS being unmatched. It costs `n` further
    entries, so the sparsity that made this possible survives.
    """
    n, m = weights.shape
    W = weights.tocsr()
    W.eliminate_zeros()
    if W.nnz == 0:
        return [], list(pred_rows), list(gt_rows)
    escape = _sp.csr_matrix((_np.ones(n), (_np.arange(n), _np.arange(n))), shape=(n, n))
    aug = _sp.hstack([W.astype(_np.float64) * (n + m + 1), escape], format="csr")
    rows, cols = _sparse_lsa(aug, maximize=True)

    rows, cols = rows[cols < m], cols[cols < m]
    if len(rows):
        kept = _np.asarray(W[rows, cols]).ravel() > 0
        rows, cols = rows[kept], cols[kept]
    taken_p, taken_g = set(rows.tolist()), set(cols.tolist())
    pairs = [(pred_rows[int(i)], gt_rows[int(j)]) for i, j in zip(rows, cols)]
    up = [r for i, r in enumerate(pred_rows) if i not in taken_p]
    ug = [r for j, r in enumerate(gt_rows) if j not in taken_g]
    return pairs, up, ug


def optimal_pairs(pred_rows, gt_rows, weight, weights=None):
    """Maximum-weight pairing of pred_rows to gt_rows.

    weight(p, g) -> number of matching leaves (higher is better).
    Returns (pairs, unmatched_pred, unmatched_gt) with pairs = [(p_row, g_row), ...].
    Only positive-weight pairs are kept: pairing two rows that share nothing is not a
    "match", it would just relabel two misses as one bad pair.

    `weights` is an optional precomputed array of the same numbers `weight` returns, shaped
    (len(pred_rows), len(gt_rows)). The caller may know how to produce the whole matrix at
    once -- `score._pair_weights` does it with two sparse products -- and filling it here one
    interpreted call at a time is what a heavy document spends most of its time on. It is only
    ever the same numbers: `weight` is still what decides whether a solved pair is kept, so a
    matrix that disagreed with it would be caught rather than believed.
    """
    if not pred_rows or not gt_rows:
        return [], list(pred_rows), list(gt_rows)
    if weights is not None and _sp.issparse(weights):
        return sparse_pairs(pred_rows, gt_rows, weights)
    transposed = len(pred_rows) > len(gt_rows)
    A, B = (gt_rows, pred_rows) if transposed else (pred_rows, gt_rows)
    # cost = -weight, because both solvers minimise
    # The matrix is filled IN a numpy array rather than built as a Python list and converted:
    # a list of lists costs ~4x the array in float objects and list slots before scipy sees any
    # of it. The O(n*m) matrix is what bounds exactness once the solve is compiled, so it is
    # worth not doubling it.
    if weights is not None:
        cost = _np.negative(weights.T if transposed else weights)
    else:
        cost = _np.empty((len(A), len(B)), dtype=float)
        for ia, a in enumerate(A):
            row = cost[ia]
            for jb, b in enumerate(B):
                row[jb] = -(weight(a, b) if not transposed else weight(b, a))
    # scipy solves the rectangular problem directly, returning row/column index arrays rather
    # than a per-row assignment vector.
    rows, cols = _lsa(cost)
    assign = [-1] * len(A)
    for ia, jb in zip(rows.tolist(), cols.tolist()):
        assign[ia] = jb
    pairs, used_b = [], set()
    for ia, jb in enumerate(assign):
        if jb < 0:
            continue
        a, b = A[ia], B[jb]
        p_row, g_row = (b, a) if transposed else (a, b)
        if weight(p_row, g_row) <= 0:
            continue                      # no shared content -> not a pair
        pairs.append((p_row, g_row))
        used_b.add(jb)
    matched_p = {id(p) for p, _ in pairs}
    matched_g = {id(g) for _, g in pairs}
    up = [r for r in pred_rows if id(r) not in matched_p]
    ug = [r for r in gt_rows if id(r) not in matched_g]
    return pairs, up, ug


def match_rows(pred_rows, gt_rows, weight, positives=None, weights=None):
    """Optimal matching over every candidate pair. Returns (pairs, up, ug, exact_flag).

    Every predicted row is priced against every gold row: no partitioning, no key, no
    pre-filter. See the module docstring for why -- in short, a partition can only remove
    pairings the scorer would have accepted, and unlike the greedy fallback it cannot report
    that it did.

    positives: upper bound on pairs that can carry positive weight, for the exact-vs-greedy
    memory comparison in `_exact_ok`.
    weights: an optional precomputed weight matrix; see `optimal_pairs`. Ignored on the greedy
    path, which does not build a matrix at all.
    """
    if not pred_rows or not gt_rows:
        return [], list(pred_rows), list(gt_rows), True
    # A sparse matrix is already small enough to hold, so the ceilings that exist to bound a
    # DENSE one have nothing to say about it. This is the path on which greedy stops being
    # needed: exact, and 50 MB where the dense form would be 5.7 GB.
    if weights is not None and _sp.issparse(weights):
        p2, u2, g2 = sparse_pairs(pred_rows, gt_rows, weights)
        return p2, u2, g2, True
    if not _exact_ok(len(pred_rows), len(gt_rows), positives):
        p2, u2, g2 = _greedy(pred_rows, gt_rows, weight)
        return p2, u2, g2, False
    p2, u2, g2 = optimal_pairs(pred_rows, gt_rows, weight, weights)
    return p2, u2, g2, True


def _greedy(pred_rows, gt_rows, weight):
    """Documented fallback for oversized blocks: best-first greedy."""
    scored = []
    for i, p in enumerate(pred_rows):
        for j, g in enumerate(gt_rows):
            w = weight(p, g)
            if w > 0:
                scored.append((-w, i, j))
    scored.sort()
    up_used, ug_used, pairs = set(), set(), []
    for _, i, j in scored:
        if i in up_used or j in ug_used:
            continue
        up_used.add(i); ug_used.add(j)
        pairs.append((pred_rows[i], gt_rows[j]))
    up = [r for i, r in enumerate(pred_rows) if i not in up_used]
    ug = [r for j, r in enumerate(gt_rows) if j not in ug_used]
    return pairs, up, ug
