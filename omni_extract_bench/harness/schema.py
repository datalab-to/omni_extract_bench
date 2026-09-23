"""The schema the vendors are sent: what every one of them is asked, and the pieces more than
one adapter needs to re-encode it.

THE UNIVERSAL LAYER -- `predict` applies both to every vendor, in this order:

  STRIP    remove annotations the BENCHMARK added (`evaluation_config`, `default`). Sending
           our own grader metadata as part of the task is simply a bug; one vendor validates
           strictly and rejected 8 of 40 documents over them.
  STATE    write a gold convention into the field descriptions every vendor sees, rather than
           loosening the comparator -- which would credit a vendor that dumps a paragraph.

Neither ever edits the benchmark's dataset files: the published corpus stays unmodified and
the overlay is auditable on its own.

SHARED RE-ENCODING -- below the conventions. Every vendor accepts a slightly different subset
of JSON Schema, and the ones that validate strictly reject what the permissive ones accept
silently; send one shape to everyone and the strict vendors score zero on documents they could
have handled, which is a fact about the harness reported as a fact about the vendor. In one run
a vendor came 8th of 9 with 19 of 40 documents failed, all of it schema delivery, and placed
4th once fixed.

A transform whose only caller is one adapter lives in that adapter, next to the
`prepare_schema` that composes it -- `to_strict_dialect` is Extend's, `to_typed_enum_dialect`
and `drop_schema_metadata` are LlamaExtract's. What is here is what more than one of them
needs. The rule for all of them: a transform may change how a constraint is ENCODED, never
what is ASKED FOR. Resolving a $ref, collapsing a nullable union, or moving a null from an
enum to the field's optionality are encodings. Removing a field is not.
"""
from __future__ import annotations
import copy

# Re-exported, so adapters resolve refs exactly as the scorer does.
from ..metric import resolve_refs  # noqa: F401

MAX_REF_DEPTH = 200


BENCHMARK_ONLY_KEYS = ("evaluation_config", "default")


def strip_benchmark_keys(node, keys=BENCHMARK_ONLY_KEYS):
    """Remove harness-added annotations. Safe for every vendor; changes no field or type."""
    if isinstance(node, list):
        return [strip_benchmark_keys(x, keys) for x in node]
    if not isinstance(node, dict):
        return node
    return {k: strip_benchmark_keys(v, keys) for k, v in node.items() if k not in keys}


CONVENTIONS = [
    {
        "id": "monetary_sign_magnitude",
        "match": ("capex", "capital_expenditure", "dividends_paid", "share_repurchase",
                  "repurchases", "interest_expense", "cost_of_revenue", "operating_expense",
                  "income_tax_expense", "payments_of", "purchases_of", "distributions_paid"),
        "exclude": ("percent", "pct", "rate", "ratio", "days", "count", "number", "shares_"),
        "text": ("Report the magnitude as a positive number. Amounts shown in parentheses in "
                 "the source are outflows presented in accounting style; the field name "
                 "already indicates direction, so do not negate them."),
    },
    {
        "id": "affiliation_org_only",
        "match": ("affiliation",),
        "exclude": (),
        "text": ("Record the organisation name only. Omit street address, city, postal code "
                 "and department qualifiers that follow it."),
    },
    {
        "id": "author_name_as_printed",
        "match": ("author", "authors"),
        "exclude": ("count", "number"),
        "text": ("Record the author name exactly as printed on the document, including "
                 "initials."),
    },
]



def _matches(name: str, conv) -> bool:
    n = name.lower()
    if any(x in n for x in conv["exclude"]):
        return False
    return any(x in n for x in conv["match"])


def apply_overlay(schema: dict, _name: str = "") -> dict:
    """Return a copy of `schema` with convention sentences appended to matching descriptions."""
    if not isinstance(schema, dict):
        return schema
    out = copy.deepcopy(schema)

    def walk(node, name=""):
        if not isinstance(node, dict):
            return
        for conv in CONVENTIONS:
            if name and _matches(name, conv):
                d = node.get("description") or ""
                if conv["text"] not in d:
                    node["description"] = (d + " " + conv["text"]).strip()
                break
        for k, v in (node.get("properties") or {}).items():
            walk(v, k)
        it = node.get("items")
        if isinstance(it, dict):
            walk(it, name)
        for br in ("anyOf", "oneOf", "allOf"):
            for sub in node.get(br, []) or []:
                walk(sub, name)
        for k, v in (node.get("$defs") or {}).items():
            walk(v, k)

    walk(out)

    return out
