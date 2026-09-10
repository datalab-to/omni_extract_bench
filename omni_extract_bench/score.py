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

Three functions do that work, and two of them call each other.

    _worth_if_paired   what would this one pair of rows be worth?
    _best_pairing      which predicted row goes with which ground-truth row?
    align              renumber the predicted rows once that is settled

Choosing a pairing needs a price for every candidate pair. Pricing a candidate pair often
needs a pairing, because two rows can look alike at the top and differ only in a list nested
inside them. Comparing two such lists is the same problem, one level down:

    _best_pairing(gold books, predicted books)
        _worth_if_paired(book A, book B)             are these the same book?
            _best_pairing(A.chapters, B.chapters)    only the chapters can say
"""
__all__ = ["grade", "explain", "Verdict",
           "node_key", "show", "format_node", "KEY", "INDEX"]

from collections.abc import Hashable, Iterable
from typing import Any, Literal, NamedTuple

from . import matching as OM
from . import normalize as N
from .grading import _drop_empty_gt_rows, _unwrap_schema, canon_key, cmp_leaf

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

    A `null` gets no address. It states nothing, so it should count neither for nor against
    anyone.

    An `additionalProperties` object is skipped entirely, and its address is added to
    `skipped`. Those objects ask the model to invent the key names by reading headings off
    the page. We do not grade that. The schema we send to strict vendors drops the keyword
    anyway, so the model was never actually asked.

    >>> for a, v in flatten({"date": "2024-03-31", "rows": [{"x": 1}], "note": None}).items():
    ...     print(f"{show(a):14} = {v!r}")
    date           = '2024-03-31'
    rows[0].x      = 1
    """
    schema = _unwrap_schema(schema) if isinstance(schema, dict) else {}
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
    elif node is not None:
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
    schema = _unwrap_schema(schema) if isinstance(schema, dict) else {}
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
        return cls(named, {array_path: {i: cls.at(sub, address + array_path
                                                  + ((INDEX, i),), ordered)
                                        for i, sub in rows.items()}
                           for array_path, rows in grouped.items()})


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

    pi, gi = list(pred), list(gold)
    pairs, _up, _ug, exact = OM.match_rows(pi, gi, cost)
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
    if _has_ref(schema):
        raise ValueError(
            "schema still contains $ref. Resolve it first with dialects.resolve_refs: an "
            "additionalProperties object behind a ref would be graded, while the same object "
            "written inline is skipped."
        )
    raw_gt, raw_pred = gt or {}, pred or {}
    gt = _drop_empty_gt_rows(N.prep_ground_truth(raw_gt))
    pred = _drop_empty_gt_rows(N.prep_prediction(raw_pred))
    ordered = _resolve_order(order_matters, schema, (raw_gt, raw_pred, gt, pred))
    skipped: list[Address] = []
    gold_leaves = flatten(gt, schema, skipped=skipped)
    pred_leaves, inexact = align(gold_leaves, flatten(pred, schema, skipped=skipped), ordered)
    return gold_leaves, pred_leaves, inexact, sorted(set(skipped), key=_reading_order)


def grade(pred: Any, gt: Any, schema: Any = None,
          order_matters: Iterable[str] = ()) -> dict:
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

    `recall` and `precision` count rows rather than values.

    `matching_exact`, `approximated` and `skipped_open_maps` say where the scorer fell short:
    an array too big to match exactly, and anything not graded at all.

    >>> gold = {"segments": [{"name": "Cloud", "q": [10.5, 12.0]}]}
    >>> grade({"segments": [{"name": "Cloud", "q": [12.0, 10.5]}]}, gold)["accuracy"]
    100.0
    >>> r = grade({"segments": [{"name": "Cloud", "q": [10.5, 99.9]}]}, gold)
    >>> r["matched"], r["total"], round(r["accuracy"], 1)
    (2, 4, 50.0)
    >>> round(r["found"] * r["read_right"] * 100, 6)
    50.0
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

    recall = matched / len(gold) if gold else 0.0
    precision = matched / len(pred_addr) if pred_addr else 0.0
    return {
        "accuracy": (100 * matched / total) if total else 0.0,
        "matched": matched,
        "total": total,
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
    }
    

class Verdict(NamedTuple):
    """What happened at one address. Returned by `explain`, one per address."""

    address: Address
    gold: Any
    pred: Any
    verdict: str        # match | wrong value | missing | spurious | skipped (open map)


def explain(pred: Any, gt: Any, schema: Any = None,
            order_matters: Iterable[str] = ()) -> list[Verdict]:
    """List what happened at every address, so you can see why a score is what it is.

    One `Verdict` per address, in reading order. Anything skipped shows up too, as its own
    line, rather than quietly not appearing.

    >>> for v in explain({"a": 1, "c": 3}, {"a": 2, "b": 9}):
    ...     print(f"{show(v.address):4} {str(v.gold):5} {str(v.pred):5} {v.verdict}")
    a    2     1     wrong value
    b    9     None  missing
    c    None  3     spurious

    Both documents are reported in ONE address space. A predicted row that arrived in a
    different position has already been renumbered onto the gold row it matched, so
    `lines[0]` means the same line on both sides. A predicted row that matched nothing keeps
    an index of its own, written `lines[p2]`, so it can never be read as a gold row.

    That is what lets you roll these up into a stricter view than the score gives. To count a
    row as correct only when every value under it is correct, group by the address up to the
    last index step:

    >>> gold = {"lines": [{"sku": "x", "qty": 2}, {"sku": "y", "qty": 7}]}
    >>> pred = {"lines": [{"sku": "y", "qty": 7}, {"sku": "x", "qty": 9}]}
    >>> rows: dict = {}
    >>> for v in explain(pred, gold):
    ...     cut = max(i for i, (kind, _) in enumerate(v.address) if kind == INDEX)
    ...     rows.setdefault(show(v.address[:cut + 1]), set()).add(v.verdict)
    >>> {row: kinds == {"match"} for row, kinds in sorted(rows.items())}
    {'lines[0]': False, 'lines[1]': True}
    >>> round(grade(pred, gold)["accuracy"], 1)     # the leaf view is kinder: 3 of 4 values
    75.0
    """
    gold, pred_addr, _inexact, skipped = _both(pred, gt, schema, order_matters)
    out = [Verdict(a, None, None, "skipped (open map)") for a in skipped]
    for a in sorted(set(gold) | set(pred_addr), key=_reading_order):
        if a in gold and a in pred_addr:
            verdict = "match" if cmp_leaf(pred_addr[a], gold[a]) >= 1.0 else "wrong value"
        else:
            verdict = "missing" if a in gold else "spurious"
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
    r = grade(PRED, GOLD)
    print(f"\naccuracy {r['accuracy']:.2f}  "
          f"= found {r['found']:.4f} x read_right {r['read_right']:.4f}")
    for v in explain(PRED, GOLD):
        print(f"  {show(v.address):24} gold={v.gold!r:12} pred={v.pred!r:12} {v.verdict}")
    assert failures == 0
    print("\nok")
