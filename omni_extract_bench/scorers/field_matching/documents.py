"""What field matching compares: a document with its vendor wrappers and sidecar fields stripped, the schema
as it was asked (`$ref`s resolved, nullable unions unwrapped, open maps recognised), and each document
flattened into addresses, one per value, by key path.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import Any, Literal

from . import values

Json = Any

_SIDECAR_SUFFIXES = ("_citations", "_meta")

_ENVELOPE_METADATA = frozenset({
    "citations", "citation", "confidence", "score", "reasoning", "evidence", "provenance",
    "source", "sources", "page", "pages", "bbox", "span", "spans", "offset", "offsets",
    "extraction_status", "status", "meta",
})

Kind = Literal["k", "i"]
KEY: Kind = "k"
INDEX: Kind = "i"

Address = tuple[tuple[Kind, Hashable], ...]
Leaves = dict[Address, Any]


def unwrap(o: Json) -> Json:
    """Strip a {value, metadata} envelope down to the value, recursively.

    An envelope has a "value" key, at least one metadata key, and nothing else. The last
    clause matters: "value" is also an ordinary field name, and rows like
    {"value": 12.4, "unit": "USD"} must survive intact -- 7,130 of them in the corpus.
    Matching on which keys are present rather than how many is what distinguishes
    {"value": x, "citations": [...]} from {"value": x, "citations": [...], "unit": "USD"}.

    Fires zero times on the present corpus; kept because vendors do emit envelopes.
    """
    if isinstance(o, dict):
        others = set(o) - {"value"}
        if "value" in o and others and others <= _ENVELOPE_METADATA:
            return unwrap(o["value"])
        return {k: unwrap(v) for k, v in o.items()}
    if isinstance(o, list):
        return [unwrap(x) for x in o]
    return o


def _drop_field_sidecars(obj: Json) -> Json:
    """Drop per-field provenance siblings so they are not charged as predicted values.

    A key is dropped only if it ends in a sidecar suffix AND its base name is a sibling in the
    same object. That protects a real field that merely ends the same way: `regulatory_citations`
    survives unless `regulatory` sits beside it.

    Applied to every vendor identically; the rule is about shape, not origin.
    """
    if isinstance(obj, dict):
        out: dict[str, Json] = {}
        for k, v in obj.items():
            sidecar = any(k.endswith(suf) and k[: -len(suf)] in obj
                          for suf in _SIDECAR_SUFFIXES)
            if not sidecar:
                out[k] = _drop_field_sidecars(v)
        return out
    if isinstance(obj, list):
        return [_drop_field_sidecars(x) for x in obj]
    return obj


def prep_prediction(obj):
    """Unwrap common response envelopes and drop per-field metadata sidecars.

    Some extractors decorate each field `X` with sibling keys `X_citations` (provenance) and
    `X_meta` (status/reasoning). That metadata is absent from the schema and from ground truth,
    so it is ignored when scoring rather than counted as extra predicted leaves -- otherwise a
    provider would be penalised for returning provenance.
    """
    obj = unwrap(obj)
    if isinstance(obj, dict):
        obj = _drop_field_sidecars(obj)
    return obj if isinstance(obj, dict) else {}


def prep_ground_truth(obj):
    """Unwrap common envelopes around ground truth."""
    obj = unwrap(obj)
    return obj if isinstance(obj, dict) else {}


def is_open_map(node) -> bool:
    """True when a schema node declares an object whose keys come from the document.

    ``additionalProperties`` asks the extractor to invent the property names by reading them
    off the page. This benchmark does not evaluate that shape: extraction APIs are built around
    a schema that names its fields, and extend's `STRICT_ALLOWED_KEYS` does not even forward
    the keyword, so a strict vendor receives a bare ``{"type": "object"}`` and has nothing to
    answer with. Grading such a node would score a request the harness never delivered.

    Detection reads explicit presence as intent rather than JSON Schema semantics, under which
    ``additionalProperties`` defaults to true and every object would qualify.
    """
    if not isinstance(node, dict):
        return False
    for branch in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(branch, []) or []:
            if is_open_map(sub):
                return True
    extra = node.get("additionalProperties")
    return extra is True or isinstance(extra, dict)


def unwrap_schema(node):
    """Resolve anyOf/oneOf to the non-null branch."""
    if not isinstance(node, dict):
        return {}
    for br in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(br, []) or []:
            if isinstance(sub, dict) and sub.get("type") != "null":
                return sub
    return node


def resolve_refs(schema, root=None, depth=0, max_depth=12):
    """Inline local ``$ref`` so a schema is self-describing.

    Some vendors resolve ``$defs`` themselves; others reject a bare ``$ref`` because it declares
    no type. Inlining changes nothing semantically. Recursive definitions stop at ``max_depth``
    rather than expanding forever.
    """
    # TODO(paul): `depth` counts nesting levels (each dict and list), not $ref expansions, so a
    # non-recursive $ref nested 7 or more object levels deep survives inlining, and `score` reports it
    # as recursive; a dangling $ref (`#/defs/nope`) gets the same "recursive" message. Count only
    # expansions toward `max_depth`, and name an unresolvable ref.
    if root is None:
        root = schema
    if depth > max_depth or not isinstance(schema, (dict, list)):
        return schema
    if isinstance(schema, list):
        return [resolve_refs(x, root, depth + 1, max_depth) for x in schema]
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        cur = root
        for part in ref[2:].split("/"):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, dict):
            merged = {k: v for k, v in schema.items() if k != "$ref"}
            merged.update({k: v for k, v in cur.items() if k not in merged})
            return resolve_refs(merged, root, depth + 1, max_depth)
    return {k: (resolve_refs(v, root, depth + 1, max_depth) if k != "$defs" else v)
            for k, v in schema.items() if k != "$defs"}


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
            skipped: list[Address] | None = None,
            absent: Callable[[Any], bool] = values.states_nothing) -> Leaves:
    """Give every value in one document an address.

    Only one document. This never looks at the other one.

    A value that states nothing gets no address, so it counts neither for nor against anyone.
    `null` and `""` both qualify, and for the same reason -- see `values.states_nothing`,
    which is where that judgement lives so both documents and both scorers share it. `absent`
    replaces that judgement; an empty `[]` or `{}` is asked about like any other value.

    An `additionalProperties` object is skipped entirely, and its address is added to
    `skipped`. Those objects ask the model to invent the key names by reading headings off
    the page. We do not score that. The schema we send to strict vendors drops the keyword
    anyway, so the model was never actually asked.

    >>> for a, v in flatten({"date": "2024-03-31", "rows": [{"x": 1}],
    ...                      "note": None, "memo": ""}).items():
    ...     print(f"{show(a):14} = {v!r}")
    date           = '2024-03-31'
    rows[0].x      = 1
    """
    schema = unwrap_schema(schema) if isinstance(schema, dict) else {}
    if is_open_map(schema):
        if skipped is not None:
            skipped.append(prefix)
        return {}
    out: Leaves = {}
    if isinstance(node, dict) and node:
        props = schema.get("properties") or {}
        for key, value in node.items():
            out.update(flatten(value, props.get(key), prefix + ((KEY, key),), skipped, absent))
    elif isinstance(node, list) and node:
        item = schema.get("items")
        for i, value in enumerate(node):
            out.update(flatten(value, item, prefix + ((INDEX, i),), skipped, absent))
    elif prefix and not absent(node):  # the document itself is never a value
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


def _schema_nodes(schema: Any, prefix: Address = ()) -> set[Address]:
    """Every name the schema declares, as node keys. The slots on offer.

    Needed to tell a value the model made up from one it invented a name for. If the schema
    offered the slot and the document is silent, filling it in is a fabricated fact. If the
    schema never mentioned the name, it is an invented field. Those are different bugs.

    Gold's own `null`s cannot answer this. A gold field written `null` and a gold field left
    out entirely mean the same thing (§5), so keying off gold would sort two identical
    documents into different buckets. The schema is the only authority that does not move.

    Nothing inside an `additionalProperties` object counts, because that subtree is not
    graded at all -- a slot nobody was asked for cannot be filled in wrongly.

    >>> [format_node(k) for k in sorted(_schema_nodes(
    ...     {"properties": {"n": {"type": "string"},
    ...                     "rows": {"type": "array",
    ...                              "items": {"properties": {"q": {"type": "number"}}}}}}),
    ...     key=format_node)]
    ['n', 'rows', 'rows[*]', 'rows[*].q']
    """
    schema = unwrap_schema(schema) if isinstance(schema, dict) else {}
    if is_open_map(schema):
        return set()
    props, item = schema.get("properties") or {}, schema.get("items")
    out: set[Address] = {prefix} if prefix else set()
    for key, sub in props.items():
        out |= _schema_nodes(sub, prefix + ((KEY, key),))
    if item is not None:
        out |= _schema_nodes(item, prefix + ((INDEX, None),))
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


def _has_ref(node: Any) -> bool:
    """Is there a `$ref` left anywhere in this schema?

    It is asked after `resolve_refs`, so a hit means the schema is recursive beyond the resolver's
    depth limit rather than merely unprepared -- which is the case that cannot be scored
    consistently, and the one worth refusing.
    """
    if isinstance(node, dict):
        return "$ref" in node or any(_has_ref(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_ref(v) for v in node)
    return False
