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
import copy, json, re, sys

# field-name markers -> convention sentence appended to that field's description
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


# Structural conventions match on the SHAPE of a field rather than its name. The mechanism is
# kept because it is the right shape for a convention that cannot key off a field name; nothing
# currently uses it.
#
# WITHDRAWN -- `enumerate_scoped_rows`. Segment-scoped fact tables (10-Qs) are scored against an
# expectation the schema never states: gold enumerates per-segment rows while the field
# descriptions say only "Total revenue". Stating it looked like a fairness fix in the same
# family as the sign convention.
#
# It was withdrawn because no wording of it behaved:
#   v1 "emit one entry for every COMBINATION of period and segment"
#        -> read as fill-the-grid; 43% under-emission became 145-220% over-emission
#   v2 "extract every figure the document states ... do not infer pairs it does not state"
#        -> worse, not better: 117-282%, over-emitting in 5 of 6 provider/document cases
#
# It also contaminated the headline. Those 3 documents of 40 produced 94-98% of the measured
# "over-extraction" for every top provider; removing them takes datalab from 2.9% to 0.2% and
# reducto from 3.7% to 0.1%. A convention meant to remove an unfair penalty had become the
# dominant source of one, and it was briefly reported as a product finding.
#
# The underlying ambiguity is real and stays DOCUMENTED-NOT-PATCHED: whether a 10-Q metric field
# wants consolidated totals only or every per-segment figure is a question for the corpus author,
# not something to keep re-wording against live scores. Fixing a benchmark by iterating prompt
# text against the numbers it produces is how a benchmark stops being neutral.
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
            walk(it, name)          # array items inherit the field's name for matching
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


if __name__ == "__main__":
    import glob, os
    CB = os.environ.get("OEB_DATA_ROOT", ".")
    print("Preview — fields that would receive a convention sentence:\n")
    for f in sorted(glob.glob(f"{CB}/extract-bench/dataset/*/*/*schema*.json")):
        s = json.load(open(f))
        s = s.get("schema_definition", s)
        before = json.dumps(s)
        after = json.dumps(apply_overlay(s))
        if before == after:
            continue
        dom = f.split("/dataset/")[1].split("/")[1]
        # count touched fields per convention
        counts = {}
        for conv in CONVENTIONS:
            counts[conv["id"]] = after.count(conv["text"])
        hit = {k: v for k, v in counts.items() if v}
        print(f"  {dom:20} {hit}")
    print("\nApplied at request time by run_provider; benchmark dataset files are never edited.")
