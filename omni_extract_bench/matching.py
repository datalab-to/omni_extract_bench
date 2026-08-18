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
never be paired anyway, so solving each block optimally is *globally* optimal, at
O(sum b_i^3) << O(n^3). If no safe hard key exists and the array is larger than
`MAX_EXACT`, we fall back to greedy and SAY SO (`exact=False`) rather than silently
pretending optimality.

Pure stdlib on purpose — a scorer with no numeric dependencies is reproducible anywhere.
"""
from __future__ import annotations

INF = float("inf")
# The exact solver is O(n^2 * m) with n the SMALLER dimension (the caller transposes so n <= m).
# Gating on the larger dimension was therefore the wrong gate in both directions: it forced
# greedy on cheap RECTANGULAR cases (a provider returning 44 rows against 349 gold costs
# 44^2 * 349 ~ 7e5 operations, trivially exact) while a square case at the same limit costs
# ~100x more. Since a truncating provider produces exactly the rectangular shape, the old gate
# pushed the cases that most need accurate scoring onto the approximate path.
#
# So the budget is on WORK, not row count. Calibrated on this hardware: 900x900 (7.3e8 units)
# took 11.2s, i.e. ~6.5e7 units/sec. A 2e8 budget keeps any single array under ~3s while
# admitting every rectangular case the benchmark actually contains.
MAX_WORK = 2 * 10**8

# Hard ceiling on the smaller dimension, independent of the work budget: memory for the cost
# matrix is O(n*m) and the pure-Python inner loop degrades badly past this.
MAX_EXACT = 1500


def _exact_ok(n, m):
    """True when this block can be solved exactly within the work budget."""
    lo, hi = (n, m) if n <= m else (m, n)
    return lo <= MAX_EXACT and (lo * lo * hi) <= MAX_WORK


def hungarian(cost):
    """Minimum-cost assignment for a rectangular matrix.

    cost: list of rows (len n) each of len m, n <= m enforced by caller-side transpose.
    Returns: list `assign` of length n, assign[i] = column matched to row i (or -1).
    Classic O(n^2 m) shortest augmenting path with potentials (JV/Hungarian).
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if m == 0:
        return [-1] * n
    # potentials and column->row assignment (1-indexed internal arrays)
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)          # p[j] = row assigned to column j
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 == -1:
                break
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assign = [-1] * n
    for j in range(1, m + 1):
        if 1 <= p[j] <= n:
            assign[p[j] - 1] = j - 1
    return assign


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
    # cost = -weight (minimisation)
    cost = [[-float(weight(a, b) if not transposed else weight(b, a)) for b in B] for a in A]
    assign = hungarian(cost)
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


def match_rows(pred_rows, gt_rows, weight, block_keys=(), key_fn=str):
    """Optimal matching with safe blocking.

    block_keys: dimension fields on which rows must agree. Rows disagreeing on a block key can
    never be paired, so per-block optimal == global optimal.
    key_fn: how a block field is compared. Pass the SCORER's equality, or blocking will forbid
    pairings the scorer would accept. Returns (pairs, up, ug, exact_flag).
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
    if not _exact_ok(len(pred_rows), len(gt_rows)):
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
