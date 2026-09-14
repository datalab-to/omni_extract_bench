#!/usr/bin/env python3
"""Provably-optimal array row matching for the benchmark grader.

WHY
---
Scoring an array means deciding which predicted row corresponds to which ground-truth row.
The previous grader inferred a "row key" and bucketed on it — a heuristic. It is
demonstrably suboptimal: an oracle best-match rescored one 10-Q at 79.8 vs the grader's
77.2 (2.6 points lost to matching alone), and 28% of array-cell misses on the contextual
subset were the *right value on the wrong row*.

Row matching is exactly **maximum-weight bipartite matching**: predicted rows on one side,
GT rows on the other, edge weight = number of leaves that would match if paired. Solving it
optimally makes the score a well-defined maximum instead of an artifact of key inference,
and it helps every vendor equally (it removes noise that punished whoever ordered rows
differently).

OPTIMALITY
----------
`hungarian()` implements the Jonker-Volgenant style O(n^3) shortest-augmenting-path
algorithm for rectangular cost matrices. It returns a minimum-cost perfect matching on the
smaller side, which (negating weights) is the maximum-weight matching. No heuristic can
score higher: the returned assignment maximises total matched leaves by construction.

EFFICIENCY
----------
O(n^3) is fine for typical arrays but not for micro1's giant tables (thousands of rows).
`match_rows()` therefore BLOCKS first: rows are partitioned by an exact-match "hard key"
(dimension fields whose values are directly comparable). Two rows in different blocks can
never be paired anyway, so solving each block optimally is *globally* optimal.

SOLVER
------
`scipy.optimize.linear_sum_assignment` -- Jonker-Volgenant, compiled -- and nothing else. A
pure-Python Hungarian implementation lived here as a fallback for when scipy was absent; scipy
is now a declared dependency, so a second implementation of it was untested-in-practice
duplication of the exact kind that produced this repository's worst bugs. Four separate
notions of "are these two values equal?" that were required to agree and silently did not is
how `canon_key` came to exist; two solvers for one assignment problem is the same shape.

While both existed they were measured against each other, and that measurement earned its
keep: they returned different equal-weight assignments and therefore DIFFERENT SCORES for one
input (50.60 against 51.85), because the objective maximised matched leaves while the score
divided by a union denominator. That is fixed at the source -- both scorers now maximise
matched leaves and then shared addresses, so every assignment achieving the maximum yields the
same score -- and the fix is asserted where it belongs, in the scorers' own property suites.

WHAT STILL BOUNDS EXACTNESS
---------------------------
Not the solve. The COST MATRIX, which is O(n*m) whatever solves it: the largest gold array
here is 6881 rows, 47 million cells, 0.38 GB as float64. Past the ceiling below the choice is
greedy or an out-of-memory, so greedy stays -- and reports itself (`exact=False`) rather than
pretending optimality.
"""
from __future__ import annotations

import numpy as _np
from scipy.optimize import linear_sum_assignment as _lsa
#: Ceilings on solving one block exactly. Both bound the COST MATRIX rather than the solve,
#: because with a compiled solver the matrix is what costs: 250 million cells is ~2 GB as
#: float64. A dimension cap as well as a cell cap, so a wildly rectangular problem cannot slip
#: through on cells alone. Past either, `match_rows` falls back to greedy and says so.
#:
#: The cell cap was 60 million while the scorer also held a Python dict of every priced pair,
#: which cost ~200 bytes a pair against the matrix's 8 -- so the matrix was never what ran the
#: machine out of memory, and a ceiling set for it was really a ceiling for the dict. That
#: dict is gone. The largest gold array in this corpus is 6881 x 6881, 47 million cells, which
#: sat at 79% of the old cap: one table half again as large would have dropped a whole
#: document to greedy. At 250 million the same cliff is past 15,000 rows.
MAX_EXACT = 20000
MAX_CELLS = 250 * 10**6

#: Absolute ceiling on the exact path, in cells. `MAX_CELLS` is the size below which exact is
#: taken without further thought; this is the size above which it is refused however sparse the
#: alternative looks. 500 million cells is ~4 GB as float64.
MAX_CELLS_DENSE = 500 * 10**6

#: What each path costs per unit, used to compare them between the two ceilings above.
#:
#: Exact holds one float64 per CELL, whether or not the pair is worth anything. Greedy holds a
#: Python 3-tuple per POSITIVE pair -- 64 bytes for the tuple and 8 for the list slot; the
#: integers inside are shared across pairs and amortise away. So greedy is cheaper only on a
#: sparse block, and on a dense one it costs ~9x MORE than the matrix it exists to avoid.
#:
#: That is the same failure the `MAX_CELLS` note above describes and `_best_pairing` removed:
#: a per-pair Python object dwarfing the matrix. It was fixed there and missed here, which is
#: why a fallback taken FOR memory reasons could run the machine out of it. Until `_greedy`
#: stores pairs compactly, the honest fix is to stop sending dense blocks down it.
EXACT_BYTES_PER_CELL = 8
GREEDY_BYTES_PER_PAIR = 72


def force_approximate():
    """Test hook: shrink the exactness budget so the greedy fallback engages.

    Returns:
        A callable that restores the real budget.

    The greedy path is unreachable on any realistic input -- the ceilings sit above the largest
    array this corpus contains -- so without a hook the property that an approximate score
    announces itself would pass without ever exercising the path it describes. Building an
    array big enough instead costs minutes per test for no extra coverage.
    """
    global MAX_EXACT, MAX_CELLS, MAX_CELLS_DENSE
    saved = (MAX_EXACT, MAX_CELLS, MAX_CELLS_DENSE)
    # MAX_CELLS_DENSE has to come down too, or the density comparison would route the
    # block back to exact and the hook would stop forcing anything.
    MAX_EXACT, MAX_CELLS, MAX_CELLS_DENSE = 8, 64, 64

    def restore():
        global MAX_EXACT, MAX_CELLS, MAX_CELLS_DENSE
        MAX_EXACT, MAX_CELLS, MAX_CELLS_DENSE = saved

    return restore


def _exact_ok(n, m, positives=None):
    """True when one block should be solved exactly rather than greedily.

    Args:
        n, m: the two sides' element counts, in either order.
        positives: an UPPER bound on the pairs that can carry positive weight, or None if the
            caller did not compute one. Only consulted between `MAX_CELLS` and
            `MAX_CELLS_DENSE`, where the two paths have to be compared rather than assumed.

    Returns:
        Whether `optimal_pairs` may be called directly. `match_rows` falls back to greedy when
        this is False, and records that the grade is approximate.

    Three bands. Below `MAX_CELLS` exact is taken outright. Above `MAX_CELLS_DENSE` it is
    refused outright, so the exact path's memory is bounded whatever the data does. Between
    them the question is which path is actually cheaper, and that depends on density: greedy
    stores nothing for a pair worth nothing, but ~9x more than a matrix cell for a pair worth
    something.

    Without `positives` this returns the pre-existing answer, so a caller that cannot estimate
    density loses nothing. An over-estimate biases towards exact, whose cost is known exactly
    before it runs; that is the safe direction to err in.
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


def optimal_pairs(pred_rows, gt_rows, weight):
    """Maximum-weight pairing of pred_rows to gt_rows.

    weight(p, g) -> number of matching leaves (higher is better).
    Returns (pairs, unmatched_pred, unmatched_gt) with pairs = [(p_row, g_row), ...].
    Only positive-weight pairs are kept: pairing two rows that share nothing is not a
    "match", it would just relabel two misses as one bad pair.
    """
    if not pred_rows or not gt_rows:
        return [], list(pred_rows), list(gt_rows)
    transposed = len(pred_rows) > len(gt_rows)
    A, B = (gt_rows, pred_rows) if transposed else (pred_rows, gt_rows)
    # cost = -weight, because both solvers minimise
    # The matrix is filled IN a numpy array rather than built as a Python list and converted:
    # at the largest real array (6881 x 6881, 47 million cells) a list of lists costs ~1.5 GB
    # of float objects and list slots before scipy sees any of it, against 0.38 GB for the
    # array itself. The O(n*m) matrix is what bounds exactness once the solve is compiled, so
    # it is worth not doubling it.
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


def _hard_key(row, keys, key_fn):
    """Block identity for a row.

    `key_fn` MUST be the same equality the scorer uses. Blocking asserts "these rows can never
    pair", so a stricter notion here silently forbids pairings the scorer would have accepted.
    This used to be bare `str()`, which meant rows differing only in the CASING of the blocking
    field landed in different blocks and could never match -- scoring 0.0 with recall 0.00 on
    content the comparator considers identical. Blocking is active on ~46% of the benchmark's
    arrays, so that was not a corner case.
    """
    return tuple(key_fn(row.get(k)) for k in keys)


def match_rows(pred_rows, gt_rows, weight, block_keys=(), key_fn=str,
               positives=None):
    """Optimal matching with safe blocking.

    block_keys: dimension fields on which rows must agree. Rows disagreeing on a block key can
    never be paired, so per-block optimal == global optimal.
    key_fn: how a block field is compared. Pass the SCORER's equality, or blocking will forbid
    pairings the scorer would accept. Returns (pairs, up, ug, exact_flag).
    positives: upper bound on pairs that can carry positive weight, for the exact-vs-greedy
    memory comparison in `_exact_ok`. Ignored when `block_keys` splits the problem, since the
    bound describes the whole and the decision is then made per block.
    """
    if not pred_rows or not gt_rows:
        return [], list(pred_rows), list(gt_rows), True
    if block_keys:
        blocks = {}
        for r in pred_rows:
            blocks.setdefault(_hard_key(r, block_keys, key_fn), ([], []))[0].append(r)
        for r in gt_rows:
            blocks.setdefault(_hard_key(r, block_keys, key_fn), ([], []))[1].append(r)
        pairs, up, ug, exact = [], [], [], True
        for k, (ps, gs) in blocks.items():
            if not _exact_ok(len(ps), len(gs)):
                p2, u2, g2 = _greedy(ps, gs, weight)
                exact = False
            else:
                p2, u2, g2 = optimal_pairs(ps, gs, weight)
            pairs += p2; up += u2; ug += g2
        return pairs, up, ug, exact
    if not _exact_ok(len(pred_rows), len(gt_rows), positives):
        p2, u2, g2 = _greedy(pred_rows, gt_rows, weight)
        return p2, u2, g2, False
    p2, u2, g2 = optimal_pairs(pred_rows, gt_rows, weight)
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


if __name__ == "__main__":
    # optimality demo: a case where greedy-by-key loses and Hungarian wins
    P = [{"k": "a", "v": 1}, {"k": "a", "v": 2}]
    G = [{"k": "a", "v": 2}, {"k": "a", "v": 1}]
    w = lambda p, g: sum(1 for f in ("k", "v") if p.get(f) == g.get(f))
    pairs, up, ug, exact = match_rows(P, G, w)
    total = sum(w(p, g) for p, g in pairs)
    print(f"optimal total weight = {total} (max possible 4), exact={exact}, pairs={len(pairs)}")
    assert total == 4, "must find the crossing assignment"
    # blocking equals global optimum
    P2 = P + [{"k": "b", "v": 9}]; G2 = G + [{"k": "b", "v": 9}]
    pb, _, _, ex2 = match_rows(P2, G2, w, block_keys=("k",))
    pg, _, _, _ = match_rows(P2, G2, w)
    print(f"blocked total {sum(w(p,g) for p,g in pb)} == global {sum(w(p,g) for p,g in pg)}")
    assert sum(w(p, g) for p, g in pb) == sum(w(p, g) for p, g in pg)
    print("optimal_match self-tests pass")
