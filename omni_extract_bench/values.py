#!/usr/bin/env python3
"""When are two values the same value?

One question, one answer, one place. Everything that needs to know whether two values are
equal -- scoring an address, weighting a candidate row pairing -- calls `canon_key`. That is
not tidiness: the benchmark once had two implementations of this that were required to agree,
and they silently diverged twice. Once when the pairing weight demanded literal equality while
scoring accepted date formats, so a correct extraction scored 50.0. Again when a numeric fix
put integers and decimals in different namespaces, so `5` and `5.0` stopped pairing and one
provider's recall went to 0.000 across two subsets. With a single function a disagreement is
not a bug to be caught by testing -- it cannot be written down.

This module also holds the two preparation steps that both scorers share: resolving a nullable
schema union, and dropping ground-truth rows that assert nothing.

It depends on nothing in this package except `normalize`, so a scorer can be built on it
without pulling in a grader.
"""
from __future__ import annotations

import re
from datetime import datetime

from . import normalize as N

# ── string / value normalizers ───────────────────────────────────────────────────
_FRAC = {"½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3", "⅛": "1/8",
         "⅜": "3/8", "⅝": "5/8", "⅞": "7/8", "⅕": "1/5", "⅙": "1/6", "⅐": "1/7"}
def _defrac(s: str) -> str:
    for u, a in _FRAC.items():
        s = s.replace(u, a)
    return s

_DATEFMTS = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y",
             "%d %B %Y", "%d-%b-%Y", "%d%b%Y", "%m/%d/%y", "%Y/%m/%d", "%d.%m.%Y",
             "%m.%d.%Y", "%b %d %Y", "%B %d %Y"]
# A timestamp is often just how a vendor spells a date: asked for `filing_date`, a model
# returns `2024-10-31T00:00:00Z`. That is the same fact and must not score as wrong.
#
# But a timestamp is NOT always a date. A field that genuinely carries a time -- when a
# transaction cleared -- would lose its time-of-day if every timestamp folded to its day,
# and two different times would start matching. So the fold is conditional: midnight means
# "this is a date wearing a timestamp's clothes", and any other time is kept.
#
# `%z` accepts `Z` as well as `+00:00` from Python 3.7, so both spellings parse. The offset
# is then dropped rather than converted: extraction reads what is printed on the page, so the
# wall clock as written is the fact, and shifting it would invent one.
_DATETIMEFMTS = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                 "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z",
                 "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f",
                 "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S.%f%z",
                 "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"]


def _asdate(v):
    """The calendar date this value denotes, or None.

    Returns a plain date for anything date-shaped, including a timestamp at midnight.
    A timestamp with a real time returns None here and is handled by `_astime`.
    """
    s = str(v).strip()
    if not re.search(r"\d", s) or len(s) > 34:
        return None
    for f in _DATEFMTS:
        try:
            return datetime.strptime(s, f).date().isoformat()
        except ValueError:
            pass
    dt = _parse_datetime(s)
    if dt is not None and (dt.hour, dt.minute, dt.second, dt.microsecond) == (0, 0, 0, 0):
        return dt.date().isoformat()
    return None


def _parse_datetime(s: str):
    for f in _DATETIMEFMTS:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    return None


def _astime(v):
    """The wall-clock instant this value denotes, or None. Midnight belongs to `_asdate`.

    Normalised so that two spellings of one time agree: `09:00:00Z`, `09:00:00+00:00` and
    `09:00:00.000` are the same instant written three ways.
    """
    s = str(v).strip()
    if not re.search(r"\d", s) or len(s) > 34:
        return None
    dt = _parse_datetime(s)
    if dt is None or (dt.hour, dt.minute, dt.second, dt.microsecond) == (0, 0, 0, 0):
        return None
    return dt.replace(tzinfo=None).isoformat(timespec="microseconds")

# Sign NOTATION varies by convention; sign SEMANTICS must not. Accounting parentheses,
# the unicode minus, and a trailing minus all mean "negative" — those are spellings of the
# same number and are folded. A disagreement about whether the value IS negative is a real
# error and stays a mismatch: -98.2 never equals +98.2.
_NEG_WRAP = re.compile(r"^\((.*)\)$")


def _sign_normalize(s: str):
    """Return (unsigned_text, sign) with sign in {1,-1}, folding notation variants."""
    s = s.strip()
    sign = 1
    m = _NEG_WRAP.match(s)          # (98.2) -> accounting negative
    if m:
        sign, s = -1, m.group(1).strip()
    s = s.replace("\u2212", "-").replace("\u2013", "-")   # unicode minus / en-dash
    if s.endswith("-"):             # trailing minus (some ERP exports)
        sign, s = -sign, s[:-1].strip()
    while s.startswith(("-", "+")):
        if s[0] == "-":
            sign = -sign
        s = s[1:].strip()
    return s, sign


def _asfloat(v):
    """Parse a float ONLY if it looks decimal (has a '.') — integers/IDs stay exact.

    Sign notation is normalized first, so `(98.2)`, `-98.2` and `\u221298.2` all parse to
    -98.2 — but the resulting SIGN is preserved and compared.
    """
    body, sign = _sign_normalize(str(v))
    body = body.replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    if "." not in body:
        return None
    try:
        return sign * float(body)
    except ValueError:
        return None

# ── leaf comparators -> 1.0 (match) or 0.0 ───────────────────────────────────────
def _canon(v):
    try:
        return N.canonical(v)
    except Exception:
        return str(v).lower()

def canon_key(v):
    """THE canonical form of a value. Two values are equal iff their keys are equal.

    This is the whole comparison rule. There is exactly one of these functions, and both jobs
    that need "are these equal?" -- scoring a leaf, and weighting a candidate row pairing --
    call it. That is not a stylistic preference: the benchmark previously had two independent
    implementations that were required to agree, and they silently diverged twice. Once when
    the pairing weight demanded literal equality while scoring accepted date formats (a correct
    extraction scored 50.0), and again when a numeric fix put integers and decimals in
    different namespaces so `5` and `5.0` stopped pairing (one provider's recall went to 0.000
    across two entire subsets). With a single function, a pairing/scoring disagreement is not a
    bug that testing has to catch -- it is unrepresentable.

    Four cases, in this order:
      1. booleans   -- canonical form, never coerced to a number
      2. numbers    -- rounded to 7 significant digits (matching the previous 1e-6 relative
                       tolerance) and canonicalised, so 5, 5.0 and "5.00" agree. Only decimals
                       parse, so ID-like integers stay exact.
      3. dates      -- ISO form, so 2024-01-15 and 01/15/2024 agree, and so does a
                       timestamp at midnight. `_asdate` is strict: "1/2", "Q1" and "2-3-13"
                       are not dates, so this cannot swallow values that mean something else.
      3b. timestamps -- a real time of day is KEPT, normalised, so two spellings of one
                       instant agree while two different times still differ.
      4. otherwise  -- unicode fractions expanded, then canonicalised
    """
    if isinstance(v, bool):
        return _canon(v)
    f = _asfloat(v)
    if f is not None:
        try:
            return _canon(float(f"{f:.7g}"))
        except (ValueError, OverflowError):
            return _canon(v)
    d = _asdate(v)
    if d:
        return f"#d{d}"
    t = _astime(v)
    if t:
        return f"#t{t}"
    return _canon(_defrac(str(v)))


def cmp_leaf(pred, gold) -> float:
    """1.0 if the two values are equal under `canon_key`, else 0.0."""
    return 1.0 if canon_key(pred) == canon_key(gold) else 0.0


# ── schema helpers ───────────────────────────────────────────────────────────────
def unwrap_schema(node):
    """Resolve anyOf/oneOf to the non-null branch."""
    if not isinstance(node, dict):
        return {}
    for br in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(br, []) or []:
            if isinstance(sub, dict) and sub.get("type") != "null":
                merged = dict(sub)
                if "evaluation_config" in node and "evaluation_config" not in merged:
                    merged["evaluation_config"] = node["evaluation_config"]
                return merged
    return node


# ── ground-truth rows that assert nothing ────────────────────────────────────────
# Dropping these is uniform (it applies to every subset), deterministic, and favours no
# provider. Payload = any field that is not a repeated scoping/dimension field.
_DIMENSION_HINTS = ("period", "segment", "type", "name", "id", "label", "category", "unit",
                    "scale", "date", "quarter", "year")


def _row_asserts_nothing(row):
    if not isinstance(row, dict):
        return False
    payload = [k for k in row
               if not k.endswith(("_citations", "_meta"))
               and not any(h in k.lower() for h in _DIMENSION_HINTS)]
    if not payload:
        return False  # all-dimension row: can't tell, keep it
    return all(row.get(k) is None for k in payload)


def drop_empty_gt_rows(node):
    """Recursively remove GT array rows whose payload is entirely null."""
    if isinstance(node, dict):
        return {k: drop_empty_gt_rows(v) for k, v in node.items()}
    if isinstance(node, list):
        kept = [x for x in node if not _row_asserts_nothing(x)]
        return [drop_empty_gt_rows(x) for x in kept]
    return node
