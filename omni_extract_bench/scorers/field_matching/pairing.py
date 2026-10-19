"""Pairing rows: which predicted row goes with which gold row, so every value has the same address on both
sides.

`align` renumbers the prediction's rows to match the gold's. It uses `_best_pairing`, which chooses the
pairs, and `_worth_if_paired`, which prices one candidate pair (and, for nested arrays, pairs again one level
down). `matching.py` holds the assignment solver they use.
"""
from __future__ import annotations

from collections.abc import Hashable
from typing import NamedTuple

from . import matching as OM
from . import values
from .documents import INDEX, KEY, Address, Leaves, _find_arrays, node_key
from .documents import flatten, show  # noqa: F401 -- the examples in the docstrings use them


class Row(NamedTuple):
    """One element of an array, split into two parts.

    Take a row like `{"issuer": "Acme", "prices": [1.5, 2.5]}`.

    `issuer` goes in `named`. You get to it by its key. The predicted row and the
    ground-truth row both call it `issuer`, because the schema does. So comparing it is
    easy: look up `issuer` in the other row and see if the values agree.

    `prices` goes in `arrays`. You get to its contents by position. The predicted row's
    `prices[0]` is not necessarily the ground-truth row's `prices[0]`, so there is nothing to
    look up yet. Those have to be matched first, which is what `_worth_if_paired` does when it
    descends.

    That split is the only distinction in this file. Everything else follows from it.

    One exception. If the caller said an array is `ordered`, its contents go in `named` too.
    Calling an array ordered means position 0 really does mean position 0, so there is
    nothing left to match. This has to happen here and not only in `align`, or
    `_worth_if_paired` would judge the array ignoring order while the final score judged it by
    position, and the two would disagree.
    """

    named: Leaves
    arrays: dict[Address, dict[Hashable, "Row"]]
    key: tuple = ()

    def is_object(self) -> bool:
        """True if this element is an object, meaning it has children with names.

        An array can hold three kinds of thing. `{"a": 1}` is an object. `"hello"` is a
        plain value. `[1, 2]` is another array. Only the first counts as a row for the
        `recall` and `precision` numbers, which count rows rather than values.
        """
        return (any(a and a[0][0] == KEY for a in self.named)
                or any(a and a[0][0] == KEY for a in self.arrays))

    @classmethod
    def at(cls, leaves: Leaves, address: Address, ordered: frozenset) -> "Row":
        """Build the row sitting at `address`, from that element's values.

        Sorts each value into `named` or `arrays`, and canonicalises it on the way past so
        it is done once here rather than on every comparison later.
        """
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
    """What this row contains, in a form that sorts.

    Two rows holding the same values get the same key, whatever order anything arrived in.
    That is the point: it is read off content, never off position, so relabelling rows cannot
    change it. Nested arrays contribute the sorted keys of their own elements, which makes the
    key blind to how those elements are numbered -- and an `ordered` array is already in
    `named` with its indices intact, so there the numbering is content and is kept.

    Values are canonical by the time they get here, so two spellings of one value key alike
    and sort together.

    Built once per row, from children that already carry theirs, so the whole tree costs one
    pass rather than one pass per comparison.
    """
    def addr(a: Address) -> tuple:
        return tuple((kind, str(name)) for kind, name in a)

    return (tuple(sorted((addr(a), str(v)) for a, v in named.items())),
            tuple(sorted((addr(path), tuple(sorted(r.key for r in kids.values())))
                         for path, kids in arrays.items())))


def _extract_rows_at(leaves: Leaves, prefix: Address, ordered: frozenset) -> dict[Hashable, Row]:
    """Pull one array's elements out of a document's flat address map.

    Give it the address of an array, get back its elements keyed by index. The indices are
    whatever the document used. Making them line up with the other document is `align`'s job.

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
    """What would it be worth to pair these two rows?

    Returns two numbers.

    `matched` is how many values would agree. That is the one you care about.

    `shared` is how many addresses the two rows have in common, whether or not the values
    agree. This one breaks ties. The final score divides by the number of addresses used by
    either document, so a pair that shares addresses keeps that number down. Between two
    pairings that match the same number of values, the one sharing more addresses scores
    better.

    The answer is exact, not a guess. Named values are a direct lookup. For each nested
    array, we work out the best its own pairing could do and add that. Arrays inside a row
    do not affect each other, so adding up their best cases gives the row's best case. That
    holds because JSON is a tree. If documents could share nodes by reference it would not.

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
    """Every pair's weight at once, or None to price them one at a time.

    `_worth_if_paired` counts two intersections per pair: the addresses both rows carry, and
    the ones where the values also agree. Counting shared features across every pair of two
    collections of sets is a boolean matrix product, so the whole matrix is two compiled
    multiplications rather than n*m interpreted calls. On the corpus's 3,437-row block that is
    236 ms against 20.6 s, and 74% of a heavy document's time is in those calls.

    It computes the same numbers, not an approximation of them. No pairing is forbidden and
    nothing is bucketed, so `matching_exact` keeps meaning what it means.

    Nested rows are not vectorised at all. A row carrying its own array is worth what its
    sub-pairings are worth, which is a recursive solve and not an intersection. Those pairs are
    handed to `_worth_if_paired` exactly as before -- same function, same arguments -- so the
    recursive descent, the `inexact` list it threads, and the `approximated` sizes it reports
    are untouched by this path. Only pairs where neither row has an array are read off the
    product, and a block where every row has one falls back entirely.

    Returns None rather than raising when the block is too small to be worth it, too large to
    hold, or holds a value that cannot be hashed -- the caller then prices pairs the old way,
    so this can decline any input it does not like.
    """
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
    """Decide which predicted rows go with which ground-truth rows.

    Returns one entry per pair: (predicted index, ground-truth index, matched, shared).

    The bar for keeping a pair is one matching value, and it decides the denominator. A row
    that clears it is charged once, and its wrong fields score as wrong. A row that fails it
    is charged twice: once as a gold row nobody found, once as a row the model made up. So
    neither omission nor invention is free.
    """
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
    """Renumber the prediction's rows to match the ground truth's.

    Returns the rewritten addresses, and the sizes of any array too big to match exactly.

    Why it is needed. Object keys work as addresses: the model was given the schema, so `invoice.total` means
    the same thing in both documents. Array indices don't: a model writes rows in the order it read them, so
    its row 0 might be the gold's row 2. Once the predicted rows are paired with gold rows and renumbered,
    every value has an address that means the same thing on both sides, and scoring compares two sets of
    addresses. Pricing a pair of rows can need a pairing of their own nested arrays, the same problem one level
    down:

        _best_pairing(gold books, predicted books)
            _worth_if_paired(book A, book B)             are these the same book?
                _best_pairing(A.chapters, B.chapters)    only the chapters can say

    Works outside in, one depth at a time. It has to: renumbering an outer array changes the
    addresses of everything inside it. `rows[2].prices` can become `rows[p2].prices`. So the
    list of arrays is rebuilt after each pass, because a list made up front goes stale.

    An array the caller marked order-dependent is left alone. Its index already means
    something, so there is nothing to renumber.

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
    """Renumber one array.

    A predicted row that was paired takes its partner's index. A predicted row that was not
    paired gets a made-up label like `p2`.

    The made-up label matters. If an invented row borrowed a spare ground-truth index it
    would look like a real row that happened to be wrong. If we dropped it instead, inventing
    rows would be free. Giving it its own address means it is counted, and counted against.
    """
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
