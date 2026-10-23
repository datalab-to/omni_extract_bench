#!/usr/bin/env python3
"""Row matching: the maximum-weight pairing of predicted rows to gold rows, exact when it fits in memory
and greedy (and reported as approximate) when it doesn't."""

import numpy as _np
import scipy.sparse as _sp
from scipy import optimize
from scipy.sparse import csgraph

MAX_EXACT = 20000
MAX_CELLS = 250 * 10**6
MAX_CELLS_DENSE = 500 * 10**6
EXACT_BYTES_PER_CELL = 8
GREEDY_BYTES_PER_PAIR = 72


def _exact_ok(n, m, positives=None):
    """Whether a block of `n` by `m` rows is solved exactly rather than greedily. Below `MAX_CELLS` it is
    exact; above `MAX_CELLS_DENSE` it is greedy, so memory stays bounded; in between it is whichever is
    cheaper, judged from `positives`, an upper bound on the pairs worth anything (None: as if it weren't
    given)."""
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
    """The exact maximum-weight pairing from a sparse weight matrix, without building a dense one.

    The sparse solver assigns every row of the smaller side, but this metric must be free to leave a row
    unpaired. So each row gets a private escape column worth less than any real pair: real weights are
    scaled by `n + m + 1` and escapes are worth 1, so no set of escapes outweighs one real unit."""
    n, m = weights.shape
    W = weights.tocsr()
    W.eliminate_zeros()
    if W.nnz == 0:
        return [], list(pred_rows), list(gt_rows)
    escape = _sp.csr_matrix((_np.ones(n), (_np.arange(n), _np.arange(n))), shape=(n, n))
    aug = _sp.hstack([W.astype(_np.float64) * (n + m + 1), escape], format="csr")
    rows, cols = csgraph.min_weight_full_bipartite_matching(aug, maximize=True)

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
    """The maximum-weight pairing of `pred_rows` to `gt_rows`: `(pairs, unmatched_pred, unmatched_gt)`.

    `weight(p, g)` is the number of matching leaves. Only pairs worth more than 0 are kept, since
    pairing two rows that share nothing would turn two misses into one bad pair. `weights` is an
    optional precomputed matrix of the same numbers (see `pairing._pair_weights`); `weight` still
    decides which solved pairs are kept."""
    if not pred_rows or not gt_rows:
        return [], list(pred_rows), list(gt_rows)
    if weights is not None and _sp.issparse(weights):
        return sparse_pairs(pred_rows, gt_rows, weights)
    transposed = len(pred_rows) > len(gt_rows)
    A, B = (gt_rows, pred_rows) if transposed else (pred_rows, gt_rows)
    if weights is not None:
        cost = _np.negative(weights.T if transposed else weights)
    else:
        cost = _np.empty((len(A), len(B)), dtype=float)
        for ia, a in enumerate(A):
            row = cost[ia]
            for jb, b in enumerate(B):
                row[jb] = -(weight(a, b) if not transposed else weight(b, a))
    rows, cols = optimize.linear_sum_assignment(cost)
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
            continue
        pairs.append((p_row, g_row))
        used_b.add(jb)
    matched_p = {id(p) for p, _ in pairs}
    matched_g = {id(g) for _, g in pairs}
    up = [r for r in pred_rows if id(r) not in matched_p]
    ug = [r for r in gt_rows if id(r) not in matched_g]
    return pairs, up, ug


def match_rows(pred_rows, gt_rows, weight, positives=None, weights=None):
    """The best pairing over every candidate pair: `(pairs, unmatched_pred, unmatched_gt, exact)`.

    Every predicted row is priced against every gold row, with no partition or pre-filter, since a
    partition could only remove pairings the scorer would accept, without saying so. `positives` and
    `weights` are as in `_exact_ok` and `optimal_pairs`; the greedy path builds no matrix."""
    if not pred_rows or not gt_rows:
        return [], list(pred_rows), list(gt_rows), True
    if weights is not None and _sp.issparse(weights):
        p2, u2, g2 = sparse_pairs(pred_rows, gt_rows, weights)
        return p2, u2, g2, True
    if not _exact_ok(len(pred_rows), len(gt_rows), positives):
        p2, u2, g2 = _greedy(pred_rows, gt_rows, weight)
        return p2, u2, g2, False
    p2, u2, g2 = optimal_pairs(pred_rows, gt_rows, weight, weights)
    return p2, u2, g2, True


def _greedy(pred_rows, gt_rows, weight):
    """The fallback for blocks too large to solve exactly: best pairs first."""
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
