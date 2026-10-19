"""The schema every vendor is sent, and the reshaping more than one adapter needs.

Before any vendor sees a schema, two steps apply to every vendor alike: `strip_benchmark_keys` removes the
benchmark's own annotations, and `apply_overlay` states the gold conventions in the field descriptions.
Neither edits the dataset's files. A transform that only one adapter needs lives in that adapter.
"""
from __future__ import annotations
import copy

MAX_REF_DEPTH = 200


BENCHMARK_ONLY_KEYS = ("evaluation_config", "default")


def strip_benchmark_keys(node, keys=BENCHMARK_ONLY_KEYS):
    """`node` without the annotations the benchmark added (`evaluation_config`, `default`). It changes no field
    or type, so it is safe for every vendor.

    Sending our own grader metadata as part of the task is a bug: one vendor validates strictly and rejected
    8 of 40 documents over it."""
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
    """A copy of `schema` with each gold convention (`CONVENTIONS`) stated in the descriptions of the fields it
    applies to.

    A convention is stated to every vendor rather than allowed for in the comparison, because a looser
    comparison would credit a vendor that answers with a paragraph.

    Every reshaping of a schema, here or in an adapter, follows one rule: it may change how a constraint is
    written, never what is asked for. Resolving a $ref, collapsing a nullable union, or moving a null from an
    enum to the field's optionality changes how it is written; removing a field changes what is asked. One
    shape sent to every vendor would make the strict ones fail documents they could handle: in one run a vendor
    came 8th of 9 with 19 of 40 documents failed by schema delivery alone, and 4th once that was fixed."""
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
