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
    """A `{value, metadata}` envelope stripped to its value, recursively.

    An envelope has a "value" key, at least one metadata key, and nothing else, so an ordinary row like
    `{"value": 12.4, "unit": "USD"}` is kept as it is."""
    if isinstance(o, dict):
        others = set(o) - {"value"}
        if "value" in o and others and others <= _ENVELOPE_METADATA:
            return unwrap(o["value"])
        return {k: unwrap(v) for k, v in o.items()}
    if isinstance(o, list):
        return [unwrap(x) for x in o]
    return o


def _drop_field_sidecars(obj: Json) -> Json:
    """Drops per-field provenance siblings, so they aren't charged as predicted values. A key is dropped
    only when it ends in a sidecar suffix and its base name is a sibling in the same object
    (`regulatory_citations` stays unless `regulatory` sits beside it). The rule is the same for every
    vendor."""
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
    """A prediction with envelopes unwrapped and per-field metadata dropped (`X_citations`, `X_meta`),
    which the schema and the gold don't have, so returning provenance costs nothing."""
    obj = unwrap(obj)
    if isinstance(obj, dict):
        obj = _drop_field_sidecars(obj)
    return obj if isinstance(obj, dict) else {}


def prep_ground_truth(obj):
    """A ground truth with envelopes unwrapped."""
    obj = unwrap(obj)
    return obj if isinstance(obj, dict) else {}


def is_open_map(node) -> bool:
    """True when a schema node is an object whose keys come from the document (`additionalProperties`
    given).

    Such a node isn't graded: extraction APIs work from named fields, and strict vendors aren't sent the
    keyword at all. Presence is read as intent, not by JSON Schema's rule that `additionalProperties`
    defaults to true."""
    if not isinstance(node, dict):
        return False
    for branch in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(branch, []) or []:
            if is_open_map(sub):
                return True
    extra = node.get("additionalProperties")
    return extra is True or isinstance(extra, dict)


def unwrap_schema(node):
    """A schema node with anyOf/oneOf resolved to the non-null branch."""
    if not isinstance(node, dict):
        return {}
    for br in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(br, []) or []:
            if isinstance(sub, dict) and sub.get("type") != "null":
                return sub
    return node


def resolve_refs(schema, root=None, depth=0, max_depth=12):
    """The schema with local `$ref`s inlined, since some vendors reject a bare `$ref`. Recursive
    definitions stop at `max_depth`."""
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
    """An array's name: where it sits, without which index it sat at.

    >>> node_key(((KEY, "books"), (INDEX, 0), (KEY, "chapters")))
    (('k', 'books'), ('i', None), ('k', 'chapters'))
    >>> node_key(((KEY, "books"), (INDEX, 7), (KEY, "chapters"))) == _
    True
    """
    return tuple((KEY, name) if kind == KEY else (INDEX, None) for kind, name in prefix)


def show(address: Address) -> str:
    """An address as you'd write it by hand.

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
    """An array's name as a person writes it: `books[*].chapters`.

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
    """The sort key that reads addresses as a person would: real indices first, in numeric order (`rows[2]`
    before `rows[10]`), then the invented `p` labels."""
    return tuple((k, isinstance(n, str), n if isinstance(n, int) else 0, str(n))
                 for k, n in address)


def flatten(node: Any, schema: Any = None, prefix: Address = (),
            skipped: list[Address] | None = None,
            absent: Callable[[Any], bool] = values.states_nothing) -> Leaves:
    """Every value of one document, by address.

    A value that states nothing (`null`, `""`; see `values.states_nothing`) gets no address, so it
    counts for and against no one; `absent` replaces that judgement. An `additionalProperties` object
    isn't graded: it is skipped, and its address added to `skipped`.

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
    """Every array a schema declares, as node keys, so `order_matters` may name an array that is empty in
    this document.

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
    """Every array in one document, as node keys, read off the document itself, since an empty array has no
    addresses.

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
    """Every name the schema declares, as node keys: the slots on offer.

    A predicted value in an offered slot the gold leaves empty is a fabricated value; one under a name
    the schema never offered is an invented field. Only the schema can tell them apart, since a gold
    `null` and a missing gold field mean the same (§5). Nothing inside an `additionalProperties` object
    counts, since it isn't graded.

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
    """Why the prediction has an address the gold doesn't. An unpaired array element (a `p` label) is
    checked first, so an invented twelve-column row counts as one invented item, not twelve invented
    fields."""
    if any(kind == INDEX and isinstance(value, str) for kind, value in address):
        return "invented_item"
    if node_key(address) in slots:
        return "fabricated"
    return "invented_field"


def _find_arrays(addresses: Iterable[Address]) -> list[Address]:
    """Every array in a document's addresses: the part of an address before each index step. Shortest
    first, so outer arrays come before the arrays inside them, as `align` needs.

    >>> [show(a) for a in _find_arrays(flatten({"a": [[1]], "b": 2}))]
    ['a', 'a[0]']
    """
    seen = {a[:d] for a in addresses for d, step in enumerate(a) if step[0] == INDEX}
    return sorted(seen, key=len)


def _has_ref(node: Any) -> bool:
    """True if a `$ref` is left anywhere in the schema. Asked after `resolve_refs`, so a hit means
    recursion deeper than the resolver's limit, which can't be scored consistently."""
    if isinstance(node, dict):
        return "$ref" in node or any(_has_ref(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_ref(v) for v in node)
    return False
