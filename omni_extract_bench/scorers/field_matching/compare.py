"""One document compared with its gold: every address paired, given a verdict, and counted into the
document's numbers (`score`). The documents are flattened by `documents.py`, and rows paired by `pairing.py`.
"""
from __future__ import annotations

import collections
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

from omni_bench_core import errors

from . import values
from .documents import (Address, _classify_extra, _document_arrays, _find_arrays, _has_ref, _reading_order,
                        _schema_arrays, _schema_nodes, flatten, format_node, prep_ground_truth, prep_prediction,
                        resolve_refs, show)
from .pairing import _extract_rows_at, align


def _resolve_order(order_matters: bool | Iterable[str], schema: Any,
                   documents: Iterable[Any]) -> frozenset:
    """Turn the caller's array names into the keys the scorer matches against.

    A name that fits no array is an error, not a no-op. Silence would be worse than useless
    here: a typo leaves the array order-free, which is the default, so the run finishes and
    reports a number that looks entirely fine.

    A name counts if the schema declares it, or if any of `documents` contains it. Both are
    needed, and so is being generous about which documents. The schema covers a table that
    happens to be empty here. The documents cover an array the schema forgot to declare. And
    passing the documents both before and after preparation covers a gold row dropped for
    being all-null, which can take the last instance of a nested array with it -- the name is
    still right, and rejecting it would be a false alarm.

    `True` names every array found that way, `False` none.
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
    if isinstance(order_matters, bool):
        return frozenset(found) if order_matters else frozenset()

    out = set()
    for name in order_matters:
        if not isinstance(name, str):
            raise TypeError(
                f"order_matters takes array names as strings, got {name!r}. Write the name "
                f"the way it is printed: 'invoice.lines', or 'matrix[*]' for the lists "
                f"inside matrix."
            )
        if name not in known:
            near = sorted(known)
            raise errors.UsageError(
                f"order_matters names {name!r}, which is not an array here. "
                + (f"Arrays available: {', '.join(repr(n) for n in near[:12])}"
                   + (f" (and {len(near) - 12} more)" if len(near) > 12 else "")
                   if near else "This document has no arrays.")
            )
        out |= known[name]
    return frozenset(out)


def score(pred: Any, gold: Any, schema: Any,
          order_matters: bool | Iterable[str] = (), verdicts: bool = False,
          absent: Callable[[Any], bool] = values.states_nothing) -> dict:
    """Score one prediction against one ground truth.

    The result holds, among others:

    - `accuracy`: from 0 to 1, the headline number;
    - `f1`: reported next to accuracy, because it charges a hopeless guess;
    - `recall`: matched addresses over gold addresses;
    - `precision`: matched addresses over asserted addresses; and
    - `matching_exact`: False when an array was too large to pair exactly.

    An address is a key path, so precision and recall treat a key path and its value as one detection: a
    value read wrongly is charged on both sides. Row counts are reported as `gt_rows`, `pred_rows` and
    `matched_rows`. With `verdicts=True`, the result also holds one `Verdict` per address, saying what
    happened there; it costs almost nothing, because aligning the two documents is nearly all of the work.

    `absent` decides which values state nothing, and so get no address on either side. The
    default drops `null`, `""`, `[]` and `{}`; `lambda v: False` makes each one a value the
    prediction must reproduce. Scores under different `absent`s are not comparable.
    """
    if pred is not None and not isinstance(pred, dict):
        raise TypeError(f"the prediction must be a JSON object, the extraction, or None for no answer; "
                        f"got a {type(pred).__name__}")
    if gold is not None and not isinstance(gold, dict):
        raise TypeError(
            f"ground truth must be a JSON object, got {type(gold).__name__}. A schema whose root "
            f"is an array cannot be sent to the vendors this benchmark measures -- OpenAI's "
            f"structured outputs rejects one outright -- so a document shaped like this was "
            f"never asked for. Wrap it in an object, on both sides."
        )
    if schema is not None and not isinstance(schema, dict):
        raise TypeError(f"schema must be a JSON object, got {type(schema).__name__}.")
    if not schema:
        raise TypeError(
            "a schema is required. Without it an additionalProperties object cannot be "
            "found, so an ungraded subtree would be graded silently, and the slots the "
            "model was offered are unknown, so a fabricated value cannot be told from an "
            "invented one. Pass the schema the prediction was generated against."
        )
    schema = resolve_refs(schema)
    if _has_ref(schema):
        raise ValueError(
            "schema still contains $ref after inlining, which means it is recursive beyond "
            "the resolver's depth limit. An additionalProperties object behind a surviving "
            "ref would be graded, while the same object written inline is skipped -- so this "
            "schema cannot be scored consistently. Flatten the recursion first."
        )
    raw_gt, pred_raw = gold or {}, pred or {}
    gold = prep_ground_truth(raw_gt)
    pred = prep_prediction(pred_raw)
    ordered = _resolve_order(order_matters, schema, (raw_gt, pred_raw, gold, pred))
    found_open: list[Address] = []
    gold_leaves = flatten(gold, schema, skipped=found_open, absent=absent)
    pred_leaves, inexact = align(
        gold_leaves, flatten(pred, schema, skipped=found_open, absent=absent), ordered)
    skipped = sorted(set(found_open), key=_reading_order)
    shared = set(gold_leaves) & set(pred_leaves)
    only_gold = set(gold_leaves) - shared
    only_pred = set(pred_leaves) - shared
    union = shared | only_gold | only_pred
    matched = sum(1 for a in shared if values.cmp_leaf(pred_leaves[a], gold_leaves[a]) >= 1.0)
    misread = len(shared) - matched
    unfound = len(only_gold)
    slots = _schema_nodes(schema)
    extra_kind = {a: _classify_extra(a, slots) for a in only_pred}
    extra = collections.Counter(extra_kind.values())
    gt_rows = pred_rows = paired = 0
    for prefix in _find_arrays(union):
        g = _extract_rows_at(gold_leaves, prefix, frozenset())
        p = _extract_rows_at(pred_leaves, prefix, frozenset())
        grow = {i for i, r in g.items() if r.is_object()}
        prow = {i for i, r in p.items() if r.is_object()}
        gt_rows += len(grow)
        pred_rows += len(prow)
        paired += len(grow & prow)

    total = len(union)               
    asserted = len(pred_leaves)
    accuracy = (matched / total) if total else 0.0
    found = len(shared) / total if total else 0.0
    read_right = matched / len(shared) if shared else 0.0
    recall = matched / len(gold_leaves) if gold_leaves else 0.0
    precision = matched / asserted if asserted else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    per_address: list[Verdict] = []
    if verdicts:
        per_address = [Verdict(a, None, None, None, None, result="skipped_open_map")
                       for a in skipped]
        for a in sorted(union, key=_reading_order):
            gold_canon = values.canon_key(gold_leaves[a]) if a in gold_leaves else None
            pred_canon = values.canon_key(pred_leaves[a]) if a in pred_leaves else None
            if a in shared:
                result = "matched" if gold_canon == pred_canon else "misread"
            elif a in only_gold:
                result = "unfound"
            else:
                result = extra_kind[a]
            per_address.append(Verdict(a, gold_leaves.get(a), pred_leaves.get(a),
                                       gold_canon=gold_canon, pred_canon=pred_canon,
                                       result=result))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
        "matched": matched,
        "misread": misread,
        "unfound": unfound,
        "fabricated": extra["fabricated"],
        "invented_item": extra["invented_item"],
        "invented_field": extra["invented_field"],
        "asserted": asserted,
        "addresses_found": found,
        "addresses_read_right": read_right,
        "gt_rows": gt_rows,
        "pred_rows": pred_rows,
        "matched_rows": paired,
        "matching_exact": not inexact,
        "approximated": sorted(set(inexact), reverse=True),
        "skipped_open_maps": [show(a) for a in skipped],
        **({"verdicts": per_address} if verdicts else {}),
    }


class Verdict(NamedTuple):
    """What happened at one address. One per address, from `score(..., verdicts=True)`."""
    address: Address
    gold_raw: Any   
    pred_raw: Any   
    gold_canon: str 
    pred_canon: str 
    result: str     # matched, misread, unfound, an invented kind, or skipped_open_map


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
    r = score(PRED, GOLD, SCHEMA)
    print(f"\naccuracy {r['accuracy']:.4f}  "
          f"= addresses_found {r['addresses_found']:.4f}"
          f" x addresses_read_right {r['addresses_read_right']:.4f}")
    assert failures == 0
    print("\nok")
