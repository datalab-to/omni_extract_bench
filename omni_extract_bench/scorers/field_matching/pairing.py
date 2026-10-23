"""Pairing rows: which predicted row goes with which gold row, so every value has the same address on both
sides.

- `align`: renumbers the prediction's rows to match the gold's;
- `_best_pairing`: chooses the pairs, with the solver in `matching.py`; and
- `_worth_if_paired`: prices one candidate pair, pairing again one level down for nested arrays."""
from __future__ import annotations

from collections.abc import Hashable
from typing import NamedTuple

from . import matching as OM
from . import values
from .documents import INDEX, KEY, Address, Leaves, _find_arrays, node_key
from .documents import flatten, show  # noqa: F401 -- the examples in the docstrings use them


class Row(NamedTuple):
    """One element of an array, split by how its values are reached.

    - `named`: values reached by key (`issuer`), which the schema names the same on both sides, so they
      compare by lookup; and
    - `arrays`: nested arrays, whose elements are reached by position, so they must be paired first
      (`_worth_if_paired`).

    An array the caller calls `ordered` goes in `named` with its indices, since its positions already
    mean the same thing on both sides."""

    named: Leaves
    arrays: dict[Address, dict[Hashable, "Row"]]
    key: tuple = ()

    def is_object(self) -> bool:
        """True if the element is an object (has named children), not a plain value or an array. Only
        objects count as rows for `recall` and `precision`."""
        return (any(a and a[0][0] == KEY for a in self.named)
                or any(a and a[0][0] == KEY for a in self.arrays))

    @classmethod
    def at(cls, leaves: Leaves, address: Address, ordered: frozenset) -> "Row":
        """The row at `address`, with each value sorted into `named` or `arrays` and canonicalised once
        here."""
        named: Leaves = {}
        grouped: dict[Address, dict[Hashable, Leaves]] = {}
        for a, v in leaves.items():
            cut = next((i for i, s in enumerate(a) if s[0] == INDEX), None)
            if cut is None or (ordered and node_key(address + a[:cut]) in ordered):
                named[a] = values.canon_key(v)
            else:
                grouped.setdefault(a[:cut], {}).setdefault(a[cut][1], {})[a[cut + 1:]] = v
        arrays = {array_path: {i: cls.at(sub, address + array_path + ((INDEX, i),), ordered)
                               for i, sub in rows.items()}
                  for array_path, rows in grouped.items()}
        return cls(named, arrays, _content_key(named, arrays))


def _content_key(named: Leaves, arrays: dict) -> tuple:
    """What the row contains, as a sortable key read off content, never position, so renumbering rows can't
    change it. Nested arrays contribute their elements' sorted keys; an `ordered` array is in `named`
    with its indices."""
    def addr(a: Address) -> tuple:
        return tuple((kind, str(name)) for kind, name in a)

    return (tuple(sorted((addr(a), str(v)) for a, v in named.items())),
            tuple(sorted((addr(path), tuple(sorted(r.key for r in kids.values())))
                         for path, kids in arrays.items())))


def _extract_rows_at(leaves: Leaves, prefix: Address, ordered: frozenset) -> dict[Hashable, Row]:
    """The elements of the array at `prefix`, keyed by the document's own indices.

    >>> leaves = flatten({"r": [{"name": "Cloud", "q": [1.5, 2.5]}]})
    >>> rows = _extract_rows_at(leaves, ((KEY, "r"),), frozenset())
    >>> {show(a): v for a, v in rows[0].named.items()}
    {'name': 'cloud'}
    >>> {show(a): sorted(g) for a, g in rows[0].arrays.items()}
    {'q': [0, 1]}
    """
    d = len(prefix)
    grouped: dict[Hashable, Leaves] = {}
    for a, v in leaves.items():
        if len(a) > d and a[:d] == prefix and a[d][0] == INDEX:
            grouped.setdefault(a[d][1], {})[a[d + 1:]] = v
    return {i: Row.at(sub, prefix + ((INDEX, i),), ordered) for i, sub in grouped.items()}


def _worth_if_paired(pred: Row, gold: Row, scale: int,
                    inexact: list | None = None) -> tuple[int, int]:
    """What pairing two rows is worth: `(matched, shared)`.

    - `matched`: how many values would agree; and
    - `shared`: how many addresses both rows have, agreeing or not. It breaks ties, since the score
      divides by the addresses either document uses.

    It is exact: named values are a lookup, and each nested array adds its own best pairing, which can
    be added up because JSON is a tree (nested arrays don't affect each other).

    >>> as_row = lambda d: _extract_rows_at(flatten({"r": [d]}), ((KEY, "r"),), frozenset())[0]
    >>> _worth_if_paired(as_row({"issuer": "Acme", "value": 999.0}),
    ...        as_row({"issuer": "Acme", "value": 125.0}), 99)
    (1, 2)
    """
    matched = shared = 0
    small, big = ((pred.named, gold.named) if len(pred.named) <= len(gold.named)
                  else (gold.named, pred.named))
    for a, v in small.items():
        if a in big:
            shared += 1
            matched += big[a] == v
    for array_path in set(pred.arrays) | set(gold.arrays):
        for _i, _j, m, sh in _best_pairing(pred.arrays.get(array_path, {}),
                                           gold.arrays.get(array_path, {}), scale, inexact):
            matched += m
            shared += sh
    return matched, shared


def _positive_pair_bound(pred: dict, gold: dict) -> int | None:
    """Roughly how many row pairs can be worth anything, for the exact-vs-greedy choice."""
    n, m = len(pred), len(gold)
    if not n or not m:
        return None
    if min(n, m) > OM.MAX_EXACT or n * m <= OM.MAX_CELLS or n * m > OM.MAX_CELLS_DENSE:
        return None
    counts: dict = {}
    for row in pred.values():
        for item in row.named.items():
            counts.setdefault(item, [0, 0])[0] += 1
    for row in gold.values():
        for item in row.named.items():
            counts.setdefault(item, [0, 0])[1] += 1
    return min(sum(p * g for p, g in counts.values()), n * m)


MIN_VECTOR_CELLS = 16 * 16

MAX_VECTOR_CELLS = 64 * 10**6


def _pair_weights(pred: dict[Hashable, Row], gold: dict[Hashable, Row],
                  pi: list, gi: list, scale: int, inexact: list | None):
    """Every pair's weight at once, as two sparse matrix products, or None to price pairs one at a time.

    The numbers are the same as `_worth_if_paired`'s, not an approximation. Pairs where either row has a
    nested array are still priced by `_worth_if_paired`, since they need a recursive pairing. Returns
    None when the block is too small to be worth it, too large to hold, or holds a value that can't be
    hashed."""
    n, m = len(pi), len(gi)
    if n * m < MIN_VECTOR_CELLS:
        return None

    import numpy as np
    import scipy.sparse as sp

    values: dict = {}
    addresses: dict = {}

    def columns(keys, rows):
        """Column indices per row, for the (address, value) and address vocabularies."""
        vptr, vidx, aptr, aidx = [0], [], [0], []
        for k in keys:
            for a, v in rows[k].named.items():
                item = (a, v)
                j = values.get(item)
                if j is None:
                    j = values[item] = len(values)
                vidx.append(j)
                j = addresses.get(a)
                if j is None:
                    j = addresses[a] = len(addresses)
                aidx.append(j)
            vptr.append(len(vidx))
            aptr.append(len(aidx))
        return (vptr, vidx), (aptr, aidx)

    try:
        pv, pa = columns(pi, pred)
        gv, ga = columns(gi, gold)
    except TypeError:
        return None

    def csr(built, rows, width):
        ptr, idx = built
        return sp.csr_matrix((np.ones(len(idx), np.int32), np.array(idx, np.int32),
                              np.array(ptr, np.int64)), shape=(rows, width))

    matched_sp = (csr(pv, n, len(values)) @ csr(gv, m, len(values)).T).tocsr()
    shared_sp = (csr(pa, n, len(addresses)) @ csr(ga, m, len(addresses)).T).tocsr()
    matched_sp.eliminate_zeros()

    deep_p = {i: {a for a, sub in pred[k].arrays.items() if sub} for i, k in enumerate(pi)}
    deep_g = {j: {a for a, sub in gold[k].arrays.items() if sub} for j, k in enumerate(gi)}
    shared_paths = set().union(*deep_p.values()) & set().union(*deep_g.values()) \
        if deep_p and deep_g else set()

    if n * m > MAX_VECTOR_CELLS:
        if shared_paths:
            return None
        mask = matched_sp.copy()
        mask.data[:] = 1
        return (matched_sp * scale + shared_sp.multiply(mask)).tocsr()

    weights = matched_sp.toarray()
    shared = shared_sp.toarray()
    nonzero = weights != 0
    weights = weights.astype(np.float64)
    weights *= scale
    weights += shared
    weights[~nonzero] = 0.0
    del shared, nonzero

    if shared_paths:
        left = [i for i, paths in deep_p.items() if paths & shared_paths]
        right = [j for j, paths in deep_g.items() if paths & shared_paths]
        for i in left:
            for j in right:
                mt, sh = _worth_if_paired(pred[pi[i]], gold[gi[j]], scale, inexact)
                weights[i, j] = mt * scale + sh if mt else 0
    return weights


def _best_pairing(pred: dict[Hashable, Row], gold: dict[Hashable, Row], scale: int,
                inexact: list | None = None) -> list[tuple[Hashable, Hashable, int, int]]:
    """Which predicted rows go with which gold rows: one `(predicted index, gold index, matched, shared)`
    per pair.

    A pair is kept only if at least one value matches. A kept row's wrong fields score as wrong; a row
    that isn't kept is charged twice, as a gold row not found and as a row invented, so neither omission
    nor invention is free."""
    if not pred or not gold:
        return []

    def cost(i, j):
        matched, shared = _worth_if_paired(pred[i], gold[j], scale, inexact)
        return matched * scale + shared if matched else 0

    pi = sorted(pred, key=lambda i: pred[i].key)
    gi = sorted(gold, key=lambda j: gold[j].key)
    weights = _pair_weights(pred, gold, pi, gi, scale, inexact)
    pairs, _up, _ug, exact = OM.match_rows(pi, gi, cost, _positive_pair_bound(pred, gold),
                                           weights)
    if not exact and inexact is not None:
        inexact.append(max(len(pi), len(gi)))
    out = []
    for i, j in pairs:
        matched, shared = _worth_if_paired(pred[i], gold[j], scale, inexact)
        if matched:
            out.append((i, j, matched, shared))
    return out


def align(gold: Leaves, pred: Leaves, ordered: frozenset = frozenset()) -> tuple[Leaves, list]:
    """Renumbers the prediction's rows to match the gold's. Returns the rewritten addresses, and the sizes
    of any array too big to pair exactly.

    Object keys mean the same thing on both sides (the schema names them), but array indices don't: a
    system's row 0 may be the gold's row 2. Once rows are paired and renumbered, scoring compares two
    sets of addresses. Pricing a pair of rows can need a pairing of their nested arrays:

        _best_pairing(gold books, predicted books)
            _worth_if_paired(book A, book B)             are these the same book?
                _best_pairing(A.chapters, B.chapters)    only the chapters can say

    It works outside in, one depth at a time, and lists the arrays again after each depth, because
    renumbering an outer array changes every address inside it. An `ordered` array is left as it is.

    >>> g = flatten({"r": [{"a": 1}, {"a": 2}]})
    >>> p = flatten({"r": [{"a": 2}, {"a": 1}]})       # right answer, rows swapped
    >>> align(g, p)[0] == g
    True
    """
    inexact: list = []
    scale = 1 + len(pred)
    done: set[Address] = set()
    while True:
        todo = [n for n in _find_arrays(list(gold) + list(pred)) if n not in done]
        if not todo:
            return pred, inexact
        depth = len(todo[0])
        for node in [n for n in todo if len(n) == depth]:
            done.add(node)
            if node_key(node) in ordered:
                continue
            pred = _align_one(gold, pred, node, scale, inexact, ordered)


def _align_one(gold: Leaves, pred: Leaves, prefix: Address, scale: int,
               inexact: list, ordered: frozenset) -> Leaves:
    """Renumbers one array: a paired predicted row takes its partner's index, and an unpaired one gets its
    own label (`p2`), so an invented row is counted against the system rather than passing as a wrong
    real row or being dropped for free."""
    gold_rows = _extract_rows_at(gold, prefix, ordered)
    pred_rows = _extract_rows_at(pred, prefix, ordered)
    if not gold_rows or not pred_rows:
        return pred
    remap = {i: j for i, j, _m, _sh in _best_pairing(pred_rows, gold_rows, scale, inexact)}

    d = len(prefix)
    out: Leaves = {}
    for a, v in pred.items():
        if len(a) > d and a[:d] == prefix and a[d][0] == INDEX:
            i = a[d][1]
            fresh = i if isinstance(i, str) else f"p{i}"
            out[a[:d] + ((INDEX, remap.get(i, fresh)),) + a[d + 1:]] = v
        else:
            out[a] = v
    return out
