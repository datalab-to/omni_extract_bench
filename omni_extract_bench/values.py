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

# Nearly every value handed to `_asdate` is not a date, and the obvious loop learns that by
# raising fifteen times -- which cost 40% of grading time on a table-heavy document. The
# escape is that `strptime` does not really parse character by character: it compiles each
# format to a regex, matches, and raises if the match fails or stops short of the end. So we
# can ask that same regex first, and only call `strptime` for a format that can actually
# match.
#
# The guard is BUILT FROM the parser rather than reasoning about it. An earlier version of
# this worked out by hand which literals a format demands, and got it wrong: `strptime`
# compiles whitespace in a format to `\s+`, so "January\t15\t2024" parses, while a
# hand-rolled rule looking for a literal " " rejected it. Deriving the guard from
# `_TimeRE_cache` makes that class of mistake unrepresentable -- the filter and the parser
# are the same pattern.
#
# `_strptime` is private, so every use of it is guarded: if a future Python moves these, we
# fall back to the plain loop and lose speed, never correctness.
try:
    import _strptime as _sp

    def _fmt_regex(fmt):
        """The regex `strptime` itself will use for `fmt`.

        Goes through `_strptime`'s own cache, which that module clears when the locale
        changes -- so the guard cannot be left describing a locale the parser has moved on
        from.
        """
        return _sp._TimeRE_cache.compile(fmt)

    #: One alternation over every format, to reject a non-date in a single regex call.
    #: The per-format patterns all name their groups the same way, and duplicate group names
    #: are illegal in an alternation, so they are stripped -- the prefilter only ever needs
    #: to match, never to capture. Rebuilt when the locale changes, for the same reason
    #: `_TimeRE_cache` is.
    _STRIP_GROUP_NAMES = re.compile(r"\(\?P<\w+>")
    _union_cache: tuple[object, "re.Pattern"] | None = None

    def _union_regex():
        global _union_cache
        lang = _sp._getlang()
        if _union_cache is None or _union_cache[0] != lang:
            pattern = "|".join(
                "(?:" + _STRIP_GROUP_NAMES.sub("(?:", _fmt_regex(f).pattern) + ")"
                for f in _DATEFMTS)
            _union_cache = (lang, re.compile(pattern, re.IGNORECASE))
        return _union_cache[1]

    def _candidate_formats(s):
        if not _union_regex().match(s):
            return                      # no format can match; skip all fifteen
        for f in _DATEFMTS:
            m = _fmt_regex(f).match(s)
            # Exactly the two failures `strptime` reports as a format mismatch: no match,
            # and a match that leaves unconverted data behind.
            if m is not None and m.end() == len(s):
                yield f

except Exception:                       # pragma: no cover - stdlib internals moved
    def _candidate_formats(s):
        return iter(_DATEFMTS)


def _asdate(v):
    s = str(v).strip()
    if not re.search(r"\d", s) or len(s) > 24:
        return None
    for f in _candidate_formats(s):
        try:
            return datetime.strptime(s, f).date().isoformat()
        except ValueError:
            pass                        # a real date error, e.g. February 30
    return None

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
      3. dates      -- ISO form, so 2024-01-15 and 01/15/2024 agree. `_asdate` is strict:
                       "1/2", "Q1" and "2-3-13" are not dates, so this cannot swallow values
                       that mean something else.
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
