#!/usr/bin/env python3
"""Non-destructive schema overlay — writes benchmark conventions into field descriptions.

Rationale: several benchmark fields are scored against a convention the schema never states,
so every vendor is penalised for guessing differently. The fix is to SAY the convention, not
to loosen the comparator (a substring-accept would credit a vendor that dumps a paragraph).

This overlay is applied when a schema is handed to a vendor. It never edits the benchmark's
own dataset files — those stay pristine so the published corpus is unmodified and the
overlay is auditable on its own.

CONVENTION 1 — SIGN OF MONETARY FLOWS: **magnitude**
    Chosen on evidence, not taste:
      * ground truth already stores magnitudes (+98.2 for capital expenditures)
      * 14 of 21 vendor readings with a signal already emit positive; only 7 emit negative
      * the field name already carries direction (`capex`, `dividends_paid`,
        `share_repurchases` are outflows by definition), so a sign adds nothing
      * it needs zero GT edits, so it introduces no risk of corrupting the corpus
    A parenthesised figure in the source is a PRESENTATION of an outflow, not a negative
    quantity. Vendors that emit -98.2 are not wrong about the world — they're answering an
    unasked question, and now the schema asks it explicitly.

    Note this is complementary to the grader's sign-NOTATION tolerance: "(98.2)", "-98.2"
    and "−98.2" are all read as the same signed number, and then the convention decides
    which sign the field wants.

Usage:  from schema_overlay import apply_overlay;  schema = apply_overlay(schema)
        python3 schema_overlay.py --preview     # show what would change
"""
from __future__ import annotations
import copy

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


STRUCTURAL_CONVENTIONS = []


def _resolve(node, root):
    """Follow a local $ref one hop, so structural rules can see through $defs."""
    if not isinstance(node, dict):
        return {}
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return node
    cur = root
    for part in ref[2:].split("/"):
        cur = (cur or {}).get(part) if isinstance(cur, dict) else None
    return cur if isinstance(cur, dict) else {}


def _row_properties(array_node, root):
    """Property names of an array's row type, resolving $ref."""
    items = array_node.get("items")
    if not isinstance(items, dict):
        return set()
    return set((_resolve(items, root).get("properties") or {}).keys())


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

    def walk_structural(node):
        if not isinstance(node, dict):
            return
        for _k, v in (node.get("properties") or {}).items():
            if isinstance(v, dict) and v.get("type") == "array":
                rowprops = _row_properties(v, out)
                for conv in STRUCTURAL_CONVENTIONS:
                    if all(r in rowprops for r in conv["row_has"]):
                        d = v.get("description") or ""
                        if conv["text"] not in d:
                            v["description"] = (d + " " + conv["text"]).strip()
            walk_structural(v)
        it = node.get("items")
        if isinstance(it, dict):
            walk_structural(it)
        for br in ("anyOf", "oneOf", "allOf"):
            for sub in node.get(br, []) or []:
                walk_structural(sub)
        for _k, v in (node.get("$defs") or {}).items():
            walk_structural(v)

    walk_structural(out)
    return out
