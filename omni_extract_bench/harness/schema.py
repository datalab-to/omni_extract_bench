"""The schema every vendor is sent. Two steps apply to every vendor alike, and neither edits the dataset's
files:

- `strip_benchmark_keys`: removes the benchmark's own annotations; and
- `apply_overlay`: states the gold conventions in the field descriptions.

A reshaping only one adapter needs lives in that adapter."""
from __future__ import annotations
import copy

MAX_REF_DEPTH = 200


BENCHMARK_ONLY_KEYS = ("evaluation_config", "default")


def strip_benchmark_keys(node, keys=BENCHMARK_ONLY_KEYS):
    """`node` without the annotations the benchmark added (`evaluation_config`, `default`), which are
    grader metadata, not part of the task; a strict vendor rejects them. No field or type changes."""
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
    """A copy of `schema` with each gold convention (`CONVENTIONS`) stated in the descriptions of the
    fields it applies to. Stating a convention to every vendor, rather than loosening the comparison,
    keeps a vendor that answers with a paragraph from being credited.

    Every reshaping, here or in an adapter, may change how a constraint is written (resolving a `$ref`,
    collapsing a nullable union), never what is asked for (removing a field)."""
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
