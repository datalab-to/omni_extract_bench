#!/usr/bin/env python3
"""Score a predicted JSON document against a ground-truth one.

The idea is to give every value an address, then compare addresses.

Object keys work as addresses. The model was given the schema, so it uses the same key
names the ground truth uses. `invoice.total` means the same thing in both documents.

Array indices do not work as addresses. The model emits rows in whatever order it read
them off the page. Its row 0 might be the ground truth's row 2.

So there is one real problem to solve: work out which predicted row goes with which
ground-truth row. Once that is settled, renumber the predicted rows to match. Now every
value has an address that means the same thing in both documents, and scoring is just
comparing two sets of addresses.

Three functions do that work.

    _worth_if_paired   what would this one pair of rows be worth?
    _best_pairing      which predicted row goes with which ground-truth row?
    align              renumber the predicted rows once that is settled

Choosing a pairing needs a price for every candidate pair. Pricing a candidate pair often
needs a "sub"-pairing" when we encounter nested arrays which means we encounter the 
same problem one level down:

    _best_pairing(gold books, predicted books)
        _worth_if_paired(book A, book B)             are these the same book?
            _best_pairing(A.chapters, B.chapters)    only the chapters can say
"""
__all__ = ["grade", "explain", "Verdict",
           "node_key", "show", "format_node", "KEY", "INDEX"]

import collections
from collections.abc import Hashable, Iterable
from typing import Any, Literal, NamedTuple

from . import matching as OM
from . import normalize as N
from .values import canon_key, cmp_leaf, drop_empty_gt_rows, states_nothing, unwrap_schema

# An address is a tuple of steps. Each step is tagged, because a document may contain the key
# "0" and ("k", "0") must not join ("i", 0).
Kind = Literal["k", "i"]
KEY: Kind = "k"
INDEX: Kind = "i"

Address = tuple[tuple[Kind, Hashable], ...]
Leaves = dict[Address, Any]


def node_key(prefix: Address) -> Address:
    """Name an array by where it sits, ignoring which particular index it sat at.

    >>> node_key(((KEY, "books"), (INDEX, 0), (KEY, "chapters")))
    (('k', 'books'), ('i', None), ('k', 'chapters'))
    >>> node_key(((KEY, "books"), (INDEX, 7), (KEY, "chapters"))) == _
    True
    """
    return tuple((KEY, name) if kind == KEY else (INDEX, None) for kind, name in prefix)


def show(address: Address) -> str:
    """Render an address the way you'd write it by hand.

    >>> show(((KEY, "holdings"), (INDEX, 0), (KEY, "issuer")))
    'holdings[0].issuer'
    """
    if not address:
        return "<root>"
    out = ""
    for kind, name in address:
        out += (f".{name}" if out else f"{name}") if kind == KEY else f"[{name}]"
    return out


def format_node(prefix: Address) -> str:
    """Name an array the way a person writes it: `books[*].chapters`.

    >>> format_node(((KEY, "books"), (INDEX, 0), (KEY, "chapters")))
    'books[*].chapters'
    >>> format_node(((KEY, "a.b"),))
    "['a.b']"
    """
    out = ""
    for kind, name in prefix:
        if kind != KEY:
            out += "[*]"
        elif isinstance(name, str) and any(c in name for c in ".[]"):
            out += f"[{name!r}]"
        else:
            out += f".{name}" if out else f"{name}"
    return out


def _reading_order(address: Address) -> tuple:
    """Sort addresses the way a person would read them.

    Plain sorting compares the tuples as text, which puts `rows[p2]` before `rows[0]` and
    `rows[10]` between `rows[1]` and `rows[2]`. This keeps real indices first and in
    numeric order, with the invented `p` labels after them.
    """
    return tuple((k, isinstance(n, str), n if isinstance(n, int) else 0, str(n))
                 for k, n in address)


def flatten(node: Any, schema: Any = None, prefix: Address = (),
            skipped: list[Address] | None = None) -> Leaves:
    """Give every value in one document an address.

    Only one document. This never looks at the other one.

    A value that states nothing gets no address, so it counts neither for nor against anyone.
    `null` and `""` both qualify, and for the same reason -- see `values.states_nothing`,
    which is where that judgement lives so both documents and both scorers share it.

    An `additionalProperties` object is skipped entirely, and its address is added to
    `skipped`. Those objects ask the model to invent the key names by reading headings off
    the page. We do not grade that. The schema we send to strict vendors drops the keyword
    anyway, so the model was never actually asked.

    >>> for a, v in flatten({"date": "2024-03-31", "rows": [{"x": 1}],
    ...                      "note": None, "memo": ""}).items():
    ...     print(f"{show(a):14} = {v!r}")
    date           = '2024-03-31'
    rows[0].x      = 1
    """
    schema = unwrap_schema(schema) if isinstance(schema, dict) else {}
    if N.is_open_map(schema):
        if skipped is not None:
            skipped.append(prefix)
        return {}
    out: Leaves = {}
    if isinstance(node, dict):
        props = schema.get("properties") or {}
        for key, value in node.items():
            out.update(flatten(value, props.get(key), prefix + ((KEY, key),), skipped))
    elif isinstance(node, list):
        item = schema.get("items")
        for i, value in enumerate(node):
            out.update(flatten(value, item, prefix + ((INDEX, i),), skipped))
    elif not states_nothing(node):
        out[prefix] = node
    return out


def _schema_arrays(schema: Any, prefix: Address = ()) -> list[Address]:
    """Every array a schema declares, as node keys.

    The document is not enough on its own. A table that happens to be empty in this
    document has no rows, so no address, so nothing to find -- but naming it in
    `order_matters` is still correct, and should not be an error.

    >>> [format_node(k) for k in _schema_arrays(
    ...     {"properties": {"rows": {"type": "array",
    ...                              "items": {"properties": {"tags": {"type": "array"}}}}}})]
    ['rows', 'rows[*].tags']
    """
    schema = unwrap_schema(schema) if isinstance(schema, dict) else {}
    out = []
    if schema.get("type") == "array" or "items" in schema:
        out.append(prefix)
        out += _schema_arrays(schema.get("items"), prefix + ((INDEX, None),))
    for key, sub in (schema.get("properties") or {}).items():
        out += _schema_arrays(sub, prefix + ((KEY, key),))
    return out


def _document_arrays(node: Any, prefix: Address = ()) -> list[Address]:
    """Every array in one document, as node keys.

    Walks the document itself rather than its addresses, because `_find_arrays` reads arrays
    off the leaves beneath them and an empty array has no leaves. `{"lines": []}` holds an
    array named `lines`, and naming it in `order_matters` is correct even though there is
    nothing in it to order.

    >>> [format_node(k) for k in _document_arrays({"lines": [], "b": [[{"c": 1}]]})]
    ['lines', 'b', 'b[*]']
    """
    out = []
    if isinstance(node, dict):
        for key, value in node.items():
            out += _document_arrays(value, prefix + ((KEY, key),))
    elif isinstance(node, list):
        out.append(prefix)
        for value in node:
            out += _document_arrays(value, prefix + ((INDEX, None),))
    return out


def _schema_leaves(schema: Any, prefix: Address = ()) -> set[Address]:
    """Every value the schema asks for, as node keys. The slots on offer.

    Needed to tell a value the model made up from one it invented a name for. If the schema
    offered the slot and the document is silent, filling it in is a fabricated fact. If the
    schema never mentioned the name, it is an invented field. Those are different bugs.

    Gold's own `null`s cannot answer this. A gold field written `null` and a gold field left
    out entirely mean the same thing (§10), so keying off gold would sort two identical
    documents into different buckets. The schema is the only authority that does not move.

    Nothing inside an `additionalProperties` object counts, because that subtree is not
    graded at all -- a slot nobody was asked for cannot be filled in wrongly.

    >>> [format_node(k) for k in sorted(_schema_leaves(
    ...     {"properties": {"n": {"type": "string"},
    ...                     "rows": {"type": "array",
    ...                              "items": {"properties": {"q": {"type": "number"}}}}}}),
    ...     key=format_node)]
    ['n', 'rows[*].q']
    """
    schema = unwrap_schema(schema) if isinstance(schema, dict) else {}
    if N.is_open_map(schema):
        return set()
    props, item = schema.get("properties") or {}, schema.get("items")
    if not props and item is None:
        return {prefix}
    out: set[Address] = set()
    for key, sub in props.items():
        out |= _schema_leaves(sub, prefix + ((KEY, key),))
    if item is not None:
        out |= _schema_leaves(item, prefix + ((INDEX, None),))
    return out


def _classify_extra(address: Address, slots: set[Address]) -> str:
    """Why does the prediction have this address when the ground truth does not?

    Order matters. An array element that paired with nothing carries a label like `p2` in
    place of an index, and every value under it is part of that one extra element rather than
    a set of separately filled fields -- a hallucinated twelve-column row would otherwise
    look exactly like twelve invented field names.

    "Item" rather than "row" because it covers both: an extra object in an array of objects,
    and an extra scalar in an array of scalars.
    """
    if any(kind == INDEX and isinstance(value, str) for kind, value in address):
        return "invented_item"
    if node_key(address) in slots:
        return "fabricated"
    return "invented_field"


def _find_arrays(addresses: Iterable[Address]) -> list[Address]:
    """Find every array, given a document's addresses.

    An array shows up as an index step in an address. `rows[0].name` tells you there is an
    array at `rows`. So: scan the addresses, and wherever there is an index, record the part
    of the address in front of it.

    Sorted shortest first, which puts outer arrays before the arrays inside them. `align`
    needs that order.

    >>> [show(a) for a in _find_arrays(flatten({"a": [[1]], "b": 2}))]
    ['a', 'a[0]']
    """
    seen = {a[:d] for a in addresses for d, step in enumerate(a) if step[0] == INDEX}
    return sorted(seen, key=len)


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

    named: Leaves                                # address -> canonical value
    arrays: dict[Address, dict[Hashable, "Row"]]
    key: tuple = ()                              # canonical content key; see `_content_key`

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
                named[a] = canon_key(v)
            else:
                grouped.setdefault(a[:cut], {}).setdefault(a[cut][1], {})[a[cut + 1:]] = v
        arrays = {array_path: {i: cls.at(sub, address + array_path + ((INDEX, i),), ordered)
                               for i, sub in rows.items()}
                  for array_path, rows in grouped.items()}
        return cls(named, arrays, _content_key(named, arrays))


def _content_key(named: Leaves, arrays: dict) -> tuple:
    """What this row CONTAINS, in a form that sorts.

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
        # Structure, not `show()`. `show` renders both `{"a.b": 1}` and `{"a": {"b": 1}}` as
        # "a.b", so rendering would let two different rows key alike and hand the tie back to
        # arrival order -- the bug this key exists to remove. The step kind separates a field
        # named "0" from index 0; `str` only makes the names orderable.
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
    """Roughly how many row pairs can be worth anything, for the exact-vs-greedy choice.

    Two rows are worth pairing only if they share at least one canonical value, so the pairs
    that can carry positive weight are exactly those co-occurring in an inverted index over
    `(address, value)`. Counting them needs only the POSTING LENGTHS -- for each item, the
    number of predicted rows carrying it times the number of gold rows -- so nothing is
    enumerated and nothing the size of the answer is allocated. The index itself is one entry
    per value, the same order as the flattened document: a few megabytes on the largest
    document in this corpus, against 45 GB for the pair list it is used to avoid building.

    The result OVER-counts, because a pair sharing three items is counted three times, and it
    ignores the nested-array contribution to `_worth_if_paired`, which can make a pair positive
    with no named value in common. Neither is a correctness problem for the caller: it is
    deciding which path is cheaper, an over-estimate sends it to the exact path whose cost is
    known before it runs, and `MAX_CELLS_DENSE` bounds that path regardless.

    Returns None when the answer cannot change the decision -- the common case by far -- so the
    index is built only for the handful of blocks big enough to be at risk.
    """
    n, m = len(pred), len(gold)
    if not n or not m:
        return None
    if min(n, m) > OM.MAX_EXACT or n * m <= OM.MAX_CELLS or n * m > OM.MAX_CELLS_DENSE:
        return None                      # decided by the ceilings; density cannot move it
    counts: dict = {}
    for row in pred.values():
        for item in row.named.items():
            counts.setdefault(item, [0, 0])[0] += 1
    for row in gold.values():
        for item in row.named.items():
            counts.setdefault(item, [0, 0])[1] += 1
    return min(sum(p * g for p, g in counts.values()), n * m)


#: Cells below which the vectorised weights are not worth building. Two sparse products and
#: their index arrays cost more than the Python loop they replace on a small block, and almost
#: every block in a corpus is small -- a median document's longest array is 6 rows.
MIN_VECTOR_CELLS = 16 * 16

#: Cells above which they are not built at all. Peak here is ~16 bytes a cell -- two int32
#: products and the float64 result -- against the 8 the matrix `optimal_pairs` needs anyway,
#: so this sits well inside the 250 million cells `matching.MAX_CELLS` already budgets for.
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

    **Nested rows are not vectorised at all.** A row carrying its own array is worth what its
    sub-pairings are worth, which is a recursive solve and not an intersection. Those pairs are
    handed to `_worth_if_paired` exactly as before -- same function, same arguments -- so the
    recursive descent, the `inexact` list it threads, and the `approximated` sizes it reports
    are untouched by this path. Only pairs where NEITHER row has an array are read off the
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
        return None                      # an unhashable value; the scalar path handles it

    def csr(built, rows, width):
        ptr, idx = built
        return sp.csr_matrix((np.ones(len(idx), np.int32), np.array(idx, np.int32),
                              np.array(ptr, np.int64)), shape=(rows, width))

    matched_sp = (csr(pv, n, len(values)) @ csr(gv, m, len(values)).T).tocsr()
    shared_sp = (csr(pa, n, len(addresses)) @ csr(ga, m, len(addresses)).T).tocsr()
    matched_sp.eliminate_zeros()

    # Rows carrying their own arrays are worth what their sub-pairings are worth. Price those
    # pairs with the function that knows how, and leave the rest read off the product.
    #
    # A pair needs that only where BOTH sides populate an array at the SAME path.
    # `_worth_if_paired` walks `set(pred.arrays) | set(gold.arrays)` and descends into
    # `_best_pairing`, which returns immediately when either side is empty -- so a predicted
    # row nesting something the gold never nested contributes nothing, and the product already
    # has its answer. The distinction is not pedantry: on the corpus's three largest arrays the
    # prediction nests 25,052 rows and the gold nests none, so asking only "does this row have
    # arrays?" would send every one of 704 million pairs down the slow path to add zero.
    deep_p = {i: {a for a, sub in pred[k].arrays.items() if sub} for i, k in enumerate(pi)}
    deep_g = {j: {a for a, sub in gold[k].arrays.items() if sub} for j, k in enumerate(gi)}
    shared_paths = set().union(*deep_p.values()) & set().union(*deep_g.values()) \
        if deep_p and deep_g else set()

    # Past the dense ceiling the matrix cannot be held at all, and only the sparse form is
    # possible. Corrections write individual cells, which a sparse matrix cannot absorb
    # cheaply and which may need cells the product does not have -- so the sparse form is
    # offered only when no pair needs one. Declining sends the block where it went before.
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
    weights[~nonzero] = 0.0              # the same bar `cost` applies: no match, no pair
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

    One consequence. If every row repeats a value -- a currency, a fiscal year -- then no
    pair is ever worth zero, so two entirely wrong rows pair on the currency alone and take
    partial credit rather than reading as a miss.

    We pick the pairing that matches the most values. If two pairings match the same number,
    we take the one sharing more addresses. `scale` makes that ordering work: it is bigger
    than any possible `shared` total, so a single extra matched value always wins.

    Sharing more addresses is the pairing that scores HIGHER -- the denominator is
    `|G| + |P| - shared` -- so where agreement cannot decide, the ambiguity is resolved in
    the model's favour rather than arbitrarily. The order matters and is not symmetric:
    agreement decides, and the score only breaks ties. Reversed, the matcher would pair rows
    that merely share field NAMES in order to shrink the denominator, and would call that
    correspondence.

    Because `scale` bounds `shared`, the maximum pins down the PAIR (matched, shared), so
    every assignment achieving it yields the same accuracy, precision, recall and f1. It does
    not pin down the assignment itself.

    It does NOT pin down the assignment itself, and that is visible: two equally optimal
    assignments can leave DIFFERENT rows unpaired, so the same document reported a different
    `fabricated`/`invented_item` split depending on the order its rows arrived in -- 35 of 600
    random row-documents under a permuted prediction, 11 under a permuted gold. Hence the
    canonical sort above. What was violated was not determinism (the same input always scored
    the same) but invariance under RELABELLING, which is the property this whole alignment
    stage exists to provide, so an index-based tie-break would have been exactly wrong: it is
    deterministic and still moves under permutation.

    Not resolved with a third weight term, and the reason is worth keeping. It could not have
    endangered the score -- (matched, shared) is already pinned, so any third term is
    score-neutral -- but the total leftover is `|P| - shared` and therefore fixed too. Only
    its SPLIT between "an invented row" and "invented fields" varies, so a weight term would
    not be weighing evidence; it would be declaring the answer to the question the split
    reports. `tests/test_pairing_determinism.py` holds the property at depth 3.


    TODO(paul): scalar arrays pay this matcher's price for a question that is a multiset
    intersection. A scalar element's `named` is `{(): value}`, so every candidate pair shares
    that one address -- `shared` is 1 throughout and `matched` is 0 or 1 -- and the matrix is
    0/(scale+1). Max-weight assignment on a 0/1 matrix is maximum bipartite matching on
    equality, which is exactly `Counter(gold) & Counter(pred)`. Confirmed equal at n = 200, 800
    and 2000; the 2000 takes 2.5 seconds, where the `Counter` is microseconds.

    Time is not the worst of it. Past `MAX_EXACT` such an array falls to greedy and the grade
    reports `matching_exact: false`, so a flag meaning "we could not afford the truth here"
    fires on the one case where the truth is O(n). P15 stays true and stops being informative.

    The fix is a fast path INSIDE this function: every row on both sides scalar (`named` is
    `{(): v}`, `arrays` empty), pair by `Counter` and return before `OM.match_rows`. Inside,
    because `_worth_if_paired` and `align` share this function and a path above it is a path
    they can drift apart on. Mixed arrays like `[1, {"a": 2}]` fail the test and fall through
    to the general path. Score-neutral by the argument above, so P2, P8 and P14 are untouched
    and the existing suite is the check.

    Not read off the schema, which cannot answer it: `items` may be absent or `anyOf`,
    `dialects.to_strict_dialect` rewrites the schema before the vendor sees it, and a model
    returning scalars where objects were promised is a finding to be scored rather than a case
    to branch on. What arrived is the only authority on what arrived.

    `_worth_if_paired` and `align` both call this. That is what keeps them in step.
    `_worth_if_paired` uses the numbers to judge a pairing; `align` uses the pairs to renumber
    rows. If they decided separately they could disagree, and then the score would not be the
    thing the matcher was aiming at.

    Every pair is priced, including the many that turn out to be worth nothing. Prices are
    not kept: holding them costs one dict entry per pair, which is far more memory than the
    matrix itself, and buys back only the handful of re-prices below.
    """
    if not pred or not gold:
        return []

    def cost(i, j):
        matched, shared = _worth_if_paired(pred[i], gold[j], scale, inexact)
        return matched * scale + shared if matched else 0

    # Canonical order, so the solver sees the SAME problem however the rows arrived. 
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
        if matched:                      # the same test `cost` makes: no match, no pair
            out.append((i, j, matched, shared))
    return out


def align(gold: Leaves, pred: Leaves, ordered: frozenset = frozenset()) -> tuple[Leaves, list]:
    """Renumber the prediction's rows to match the ground truth's.

    Returns the rewritten addresses, and the sizes of any array too big to match exactly.

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
    # One bound on shared for the whole document: any pairing shares at most this many.
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
            fresh = i if isinstance(i, str) else f"p{i}"   # already relabelled? leave it
            out[a[:d] + ((INDEX, remap.get(i, fresh)),) + a[d + 1:]] = v
        else:
            out[a] = v
    return out


def _has_ref(node: Any) -> bool:
    """Is there a `$ref` left anywhere in this schema?

    The scorer reads a schema for one purpose: finding `additionalProperties` objects, which
    it does not grade. A `$ref` hides that keyword behind a pointer this module does not
    follow, so an open map written as a ref would be graded while the same map written inline
    is skipped -- the same document scored two ways depending on how its schema was spelled.

    Rather than resolve refs here, say so. `dialects.resolve_refs` already does it, and it is
    what the harness sends to vendors anyway.
    """
    if isinstance(node, dict):
        return "$ref" in node or any(_has_ref(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_ref(v) for v in node)
    return False


def _resolve_order(order_matters: Iterable[str], schema: Any,
                   documents: Iterable[Any]) -> frozenset:
    """Turn the caller's array names into the keys the scorer matches against.

    A name that fits no array is an error, not a no-op. Getting it wrong used to be silent:
    a typo left the array order-free, which is the default, so the run finished and reported
    a number that looked entirely fine.

    A name counts if the schema declares it, or if any of `documents` contains it. Both are
    needed, and so is being generous about which documents. The schema covers a table that
    happens to be empty here. The documents cover an array the schema forgot to declare. And
    passing the documents both before and after preparation covers a gold row dropped for
    being all-null, which can take the last instance of a nested array with it -- the name is
    still right, and rejecting it would be a false alarm.
    """
    if isinstance(order_matters, str):
        raise TypeError(
            f"order_matters takes a list of array names, not one string. Did you mean "
            f"[{order_matters!r}]? A bare string iterates as characters."
        )
    known: dict[str, set] = {}
    found = list(_schema_arrays(schema))
    for doc in documents:
        found += _document_arrays(doc)
    for key in found:
        known.setdefault(format_node(key), set()).add(key)

    clashing = sorted(n for n, keys in known.items() if len(keys) > 1)
    if clashing:
        raise ValueError(
            f"two different arrays are both written {clashing[0]!r}, so naming one in "
            f"order_matters would silently mean the other. This should not be reachable -- "
            f"keys holding a dot or a bracket are quoted to prevent it. Please report it."
        )

    out = set()
    for name in order_matters:
        if not isinstance(name, str):
            raise TypeError(
                f"order_matters takes array names as strings, got {name!r}. Write the name "
                f"the way it is printed: 'invoice.lines', or 'matrix[*]' for the lists "
                f"inside matrix."
            )
        if name not in known:
            near = sorted(known) or ["<this document has no arrays>"]
            raise ValueError(
                f"order_matters names {name!r}, which is not an array here. "
                f"Arrays available: {', '.join(repr(n) for n in near[:12])}"
                + (f" (and {len(near) - 12} more)" if len(near) > 12 else "")
            )
        out |= known[name]
    return frozenset(out)


def _both(pred: Any, gt: Any, schema: Any,
          order_matters: Iterable[str]) -> tuple[Leaves, Leaves, list, list]:
    """Address both documents, then renumber the prediction to line up with the ground truth.

    Everything `grade` and `explain` need. They share it so they cannot drift apart.
    """
    if gt is not None and not isinstance(gt, dict):
        raise TypeError(
            f"ground truth must be a JSON object, got {type(gt).__name__}. A schema whose root "
            f"is an array cannot be sent to the vendors this benchmark measures -- OpenAI's "
            f"structured outputs rejects one outright -- so a document shaped like this was "
            f"never asked for. Wrap it in an object, on both sides."
        )
    if not isinstance(schema, dict) or not schema:
        raise TypeError(
            "a schema is required. Without it an additionalProperties object cannot be "
            "found, so an ungraded subtree would be graded silently, and the slots the "
            "model was offered are unknown, so a fabricated value cannot be told from an "
            "invented one. Pass the schema the prediction was generated against."
        )
    if _has_ref(schema):
        raise ValueError(
            "schema still contains $ref. Resolve it first with dialects.resolve_refs: an "
            "additionalProperties object behind a ref would be graded, while the same object "
            "written inline is skipped."
        )
    raw_gt, raw_pred = gt or {}, pred or {}
    gt = drop_empty_gt_rows(N.prep_ground_truth(raw_gt))
    pred = drop_empty_gt_rows(N.prep_prediction(raw_pred))
    ordered = _resolve_order(order_matters, schema, (raw_gt, raw_pred, gt, pred))
    skipped: list[Address] = []
    gold_leaves = flatten(gt, schema, skipped=skipped)
    pred_leaves, inexact = align(gold_leaves, flatten(pred, schema, skipped=skipped), ordered)
    return gold_leaves, pred_leaves, inexact, sorted(set(skipped), key=_reading_order)


def grade(pred: Any, gt: Any, schema: Any,
          order_matters: Iterable[str] = (), verdicts: bool = False) -> dict:
    """Score one prediction against one ground truth.

    `schema` is only used to find `additionalProperties` objects, which are not graded.
    Nothing else reads it. Comparing values never needed it: `canon_key` decides how strict
    to be from the value itself, so an ID-like integer stays exact whether the schema calls
    it an integer or a number.

    `order_matters` lists arrays whose order is part of the answer, written the way
    `explain` prints them: `["steps"]`, or `["books[*].chapters"]` for the chapters inside
    each book. `[*]` means "each element of", so for `{"matrix": [[1, 2], [3, 4]]}` the name
    `"matrix"` is the outer list and `"matrix[*]"` are the lists inside it.

    By default no array is ordered, because the order rows appear in a document is usually
    an accident of layout and a model should not be punished for it. A name that fits no
    array raises, rather than quietly leaving that array unordered.

    What comes back:

    `accuracy` is the score, 0 to 100. `matched` over `total` is where it comes from.

    `found` counts addresses, not values: of every address either document used, the share
    both used. An address counts even if the value sitting there is wrong; whether it is
    right is `read_right`.

    `precision`, `recall` and `f1` treat a keypath-and-value as one detection, so a value
    read wrongly is charged on both sides. Report `f1` next to `accuracy`: accuracy cannot
    tell a useful guess from a hopeless one, because a wrong value at a gold address costs
    exactly what leaving it blank costs. `f1` charges the guess, so filling in fields the
    model cannot read stops being free.

    `found` and `read_right` multiply to give the accuracy. `found` is the share of
    addresses that appear in both documents: did the model pick out the right cells?
    `read_right` is the share of those whose values agree: did it read them correctly? A
    model that returns half the table perfectly and one that returns the whole table with
    half the values wrong get similar accuracies. These two tell them apart.


    `matching_exact`, `approximated` and `skipped_open_maps` say where the scorer fell short:
    an array too big to match exactly, and anything not graded at all.

    >>> schema = {"properties": {"segments": {"type": "array", "items": {"properties": {
    ...     "name": {"type": "string"},
    ...     "q": {"type": "array", "items": {"type": "number"}}}}}}}
    >>> gold = {"segments": [{"name": "Cloud", "q": [10.5, 12.0]}]}
    >>> grade({"segments": [{"name": "Cloud", "q": [12.0, 10.5]}]}, gold, schema)["accuracy"]
    100.0
    >>> r = grade({"segments": [{"name": "Cloud", "q": [10.5, 99.9]}]}, gold, schema)
    >>> r["matched"], r["total"], round(r["accuracy"], 1)
    (2, 4, 50.0)
    >>> round(r["found"] * r["read_right"] * 100, 6)
    50.0
    >>> r["misread"], r["unfound"], r["invented_item"]
    (0, 1, 1)

    Pass `verdicts=True` to get the per-address view back in the same result, under
    `"verdicts"`. Do that rather than calling `explain` separately: aligning the two documents
    is nearly all of the work, so asking twice does it twice.

    >>> r = grade({"a": 1}, {"a": 2}, {"properties": {"a": {"type": "number"}}},
    ...           verdicts=True)
    >>> [(show(v.address), v.gold, v.pred, v.verdict) for v in r["verdicts"]]
    [('a', 2, 1, 'misread')]

    It is off by default because it is not free in memory: one `Verdict` per address, and the
    largest document in the benchmark corpus has 410,012 of them.

    That last line is worth reading twice. The wrong value is in a scalar array, whose
    elements have no identity to be wrong about -- they compare as a multiset. So it is not
    one value misread; it is one gold element nobody produced and one element the model
    produced that is not there. Charged on both sides, which costs more than the same error
    in a named field: 50.0 here against 66.7 for a wrong `name`.
    """
    gold, pred_addr, inexact, skipped = _both(pred, gt, schema, order_matters)
    shared = set(gold) & set(pred_addr)
    union = set(gold) | set(pred_addr)
    matched = sum(1 for a in shared if cmp_leaf(pred_addr[a], gold[a]) >= 1.0)
    total = len(union)

    # Rows at EVERY depth, not just the top. A table nested one level down is still a table,
    # and counting only the outer array counts the wrapper object instead of its contents.
    gt_rows = pred_rows = paired = 0
    for prefix in _find_arrays(union):
        g = _extract_rows_at(gold, prefix, frozenset())
        p = _extract_rows_at(pred_addr, prefix, frozenset())
        grow = {i for i, r in g.items() if r.is_object()}
        prow = {i for i, r in p.items() if r.is_object()}
        gt_rows += len(grow)
        pred_rows += len(prow)
        paired += len(grow & prow)

    # Why the prediction is wrong, split so the three failures have different names.
    # misread: the document has this value and the model read it wrongly.
    # fabricated: the schema offered the slot, the document is silent, the model filled it.
    # invented item / field: structure the schema never asked for.
    slots = _schema_leaves(schema)
    misread = len(shared) - matched
    unfound = len(gold) - len(shared)
    extra = collections.Counter(_classify_extra(a, slots)
                                for a in pred_addr if a not in gold)

    recall = matched / len(gold) if gold else 0.0
    precision = matched / len(pred_addr) if pred_addr else 0.0
    return {
        "accuracy": (100 * matched / total) if total else 0.0,
        "matched": matched,
        "total": total,
        "asserted": len(pred_addr),
        "misread": misread,
        "unfound": unfound,
        "fabricated": extra["fabricated"],
        "invented_item": extra["invented_item"],
        "invented_field": extra["invented_field"],
        "found": len(shared) / total if total else 0.0,
        "read_right": matched / len(shared) if shared else 0.0,
        "gt_rows": gt_rows,
        "pred_rows": pred_rows,
        "matched_rows": paired,
        "recall": recall,
        "precision": precision,
        "f1": (2 * precision * recall / (precision + recall)
               if precision + recall else 0.0),
        "matching_exact": not inexact,
        "approximated": sorted(set(inexact), reverse=True),
        "skipped_open_maps": [show(a) for a in skipped],
        **({"verdicts": _verdicts(gold, pred_addr, skipped, slots)} if verdicts else {}),
    }
    

class Verdict(NamedTuple):
    """What happened at one address. Returned by `explain`, one per address."""

    address: Address
    gold: Any
    pred: Any
    verdict: str        # matched | misread | unfound | fabricated |
                        # invented_item | invented_field | skipped_open_map
                        # -- the same words the summary counts them under, so
                        # count(verdict='misread') == summary['misread']
                        # fabricated | invented item | invented field


def explain(pred: Any, gt: Any, schema: Any,
            order_matters: Iterable[str] = ()) -> list[Verdict]:
    """List what happened at every address, so you can see why a score is what it is.

    `grade(..., verdicts=True)` is the same work and also gives you the score; this is the
    shorthand for when the score is not what you are after.

    One `Verdict` per address, in reading order. Anything skipped shows up too, as its own
    line, rather than quietly not appearing.

    A value the prediction has and the ground truth does not is labelled by WHY: `fabricated`
    if the schema offered that slot, `invented field` if it never declared the name, and
    `invented item` for a value under an array element that paired with nothing. See section
    4 of the spec.

    >>> schema = {"properties": {"a": {"type": "number"}, "b": {"type": "number"},
    ...                          "c": {"type": "number"}}}
    >>> for v in explain({"a": 1, "c": 3}, {"a": 2, "b": 9}, schema):
    ...     print(f"{show(v.address):4} {str(v.gold):5} {str(v.pred):5} {v.verdict}")
    a    2     1     wrong value
    b    9     None  missing
    c    None  3     fabricated

    Both documents are reported in ONE address space. A predicted row that arrived in a
    different position has already been renumbered onto the gold row it matched, so
    `lines[0]` means the same line on both sides. A predicted row that matched nothing keeps
    an index of its own, written `lines[p2]`, so it can never be read as a gold row.

    That is what lets you roll these up into a stricter view than the score gives. To count a
    row as correct only when every value under it is correct, group by the address up to the
    last index step:

    >>> sch = {"properties": {"lines": {"type": "array", "items": {"properties": {
    ...     "sku": {"type": "string"}, "qty": {"type": "number"}}}}}}
    >>> gold = {"lines": [{"sku": "x", "qty": 2}, {"sku": "y", "qty": 7}]}
    >>> pred = {"lines": [{"sku": "y", "qty": 7}, {"sku": "x", "qty": 9}]}
    >>> rows: dict = {}
    >>> for v in explain(pred, gold, sch):
    ...     cut = max(i for i, (kind, _) in enumerate(v.address) if kind == INDEX)
    ...     rows.setdefault(show(v.address[:cut + 1]), set()).add(v.verdict)
    >>> {row: kinds == {"matched"} for row, kinds in sorted(rows.items())}
    {'lines[0]': False, 'lines[1]': True}
    >>> round(grade(pred, gold, sch)["accuracy"], 1)   # the leaf view is kinder: 3 of 4
    75.0
    """
    return grade(pred, gt, schema, order_matters, verdicts=True)["verdicts"]


def _verdicts(gold, pred_addr, skipped, slots) -> list:
    """Build the per-address view from an already-aligned pair.

    Split out so `grade` can return it from the pass it has already done. Aligning the two
    documents is essentially all of the cost -- 44.85s against 44.08s on an 89,000-leaf
    document -- so a caller that wanted a score and an explanation used to pay for the
    matching twice.
    """
    out = [Verdict(a, None, None, "skipped_open_map") for a in skipped]
    for a in sorted(set(gold) | set(pred_addr), key=_reading_order):
        if a in gold and a in pred_addr:
            verdict = "matched" if cmp_leaf(pred_addr[a], gold[a]) >= 1.0 else "misread"
        elif a in gold:
            verdict = "unfound"
        else:
            verdict = _classify_extra(a, slots)
        out.append(Verdict(a, gold.get(a), pred_addr.get(a), verdict))
    return out


if __name__ == "__main__":
    import doctest

    failures, tests = doctest.testmod()
    print(f"doctests: {tests - failures}/{tests} pass")

    GOLD = {"filing_date": "2024-03-31",
            "holdings": [{"issuer": "Acme", "value": 125.0},
                         {"issuer": "Beta", "value": 310.5},
                         {"issuer": "Cygnus", "value": 88.25}]}
    PRED = {"filing_date": "03/31/2024",
            "holdings": [{"issuer": "Cygnus", "value": 88.25},
                         {"issuer": "Acme", "value": 999.0},
                         {"issuer": "Delta", "value": 42.0}]}
    SCHEMA = {"properties": {
        "filing_date": {"type": "string"},
        "holdings": {"type": "array", "items": {"properties": {
            "issuer": {"type": "string"}, "value": {"type": "number"}}}}}}
    r = grade(PRED, GOLD, SCHEMA)
    print(f"\naccuracy {r['accuracy']:.2f}  "
          f"= found {r['found']:.4f} x read_right {r['read_right']:.4f}")
    for v in explain(PRED, GOLD, SCHEMA):
        print(f"  {show(v.address):24} gold={v.gold!r:12} pred={v.pred!r:12} {v.verdict}")
    assert failures == 0
    print("\nok")
