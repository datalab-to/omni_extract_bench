#!/usr/bin/env python3
"""When two values are the same value.

`canon_key` is the one answer, used by both leaf scoring and row pairing; never add a second. In reading
order:

- `FOLDS`: the named normalisation rules, run by `run_folds`;
- `canon_key`: a value's canonical form, and `canon_trace`, the same run explained;
- `cmp_leaf`: whether two values are equal; and
- `states_nothing`: whether a value makes no claim at all."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, NamedTuple

from . import recognise

Json = Any

_DASH = re.compile(r"-+")


class Change(NamedTuple):
    """One normalisation step that altered a value, as `canon_trace` returns it."""

    step: str
    why: str
    before: str
    after: str


class Fold(NamedTuple):
    """One normalisation step. `why` names the rule in terms a reader disagreeing with a match would use."""

    name: str
    why: str
    run: Callable[[str], str]


class _Final(str):
    """A step's result that decides the key outright, ending the run of folds."""


def _f_case(s: str) -> str:
    return s.strip().lower()


def _f_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s
                if not unicodedata.combining(c) and unicodedata.category(c) != "Cf")
    return s.replace("\u03bc", "u")


_FOOTNOTE = re.compile(r"\s*\[\s*(?:\d{1,2}|[a-z])\s*\]")


def _f_footnote(s: str) -> str:
    out = _FOOTNOTE.sub("", s)
    return out if out.strip() else s


_TYPOGRAPHY = (("\u2019", "'"), ("\u2018", "'"),
               ("\u201c", '"'), ("\u201d", '"'),
               ("\u201a", ","), ("\u201e", '"'),
               ("\u2032", "'"), ("\u2033", '"'),
               ("\u2013", "-"), ("\u2014", "-"),
               ("\u2010", "-"), ("\u2011", "-"),
               ("\u2212", "-"), ("\u2015", "-"),
               ("\u2044", "/"),
               )


def _f_typography(s: str) -> str:
    for a, b in _TYPOGRAPHY:
        s = s.replace(a, b)
    return s


_LIST_MARKER = re.compile(r"^[\u2022\u2023\u25e6\u25aa\u25b6\u2219\u25cf\u25a0](?=\s|$)")


def _f_list_marker(s: str) -> str:
    return _LIST_MARKER.sub("", s)


_ZERO_PADDED = re.compile(r"-?0\d")
_PLAIN_NUM = re.compile(r"[+-]?\d+(?:\.\d+)?")


def _f_number(s: str) -> str:
    num = s.replace(",", "").replace("$", "").replace("%", "").strip()
    if not _PLAIN_NUM.fullmatch(num):
        return s
    if _ZERO_PADDED.match(num):
        return s
    f = float(num)
    return _Final(str(int(f)) if f == int(f) else str(f))


def _f_dash(s: str) -> str:
    return _Final("#ph_dash") if _DASH.fullmatch(re.sub(r"\s+", "", s)) else s


_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")


def _f_punctuation(s: str) -> str:
    if len(_DECIMAL_POINT.findall(s)) == 1:
        s = _DECIMAL_POINT.sub("\x00", s)
    s = re.sub(r"[\s,\-.]", "", s)
    return s.replace("\x00", ".")


def _f_brackets(s: str) -> str:
    return s.strip("\"'").replace("(", "").replace(")", "").replace("/", "")


FOLDS: tuple[Fold, ...] = (
    Fold("case",
         "Capitalisation and surrounding whitespace are not part of a value: ACME CORP, "
         "Acme Corp and ' Acme Corp ' are one answer.", _f_case),
    Fold("accents",
         "Unicode is reduced to its plainest form: accents drop (M\u00fcller reads as Muller), "
         "invisible characters like a zero-width space are removed, and anything Unicode "
         "considers a styled form of plain text is unstyled -- \u00b5g reads as ug, \ufb01le as file, "
         "\uff46\uff55\uff4c\uff4c as full, \u216b as XII, \u00bd as 1/2 and 10\u00b2 as 102. The last of "
         "those is the one to know about: a superscript is treated as an ordinary digit.",
         _f_accents),
    Fold("footnote marker",
         "A short [1] or [a] reference marker attached to a value is dropped -- but never when "
         "it is the whole value.", _f_footnote),
    Fold("typography",
         "Characters that are an ASCII character in disguise are replaced by it. Every quote "
         "form becomes ' or \", every dash form becomes -, and that includes the ones easily "
         "mistaken for punctuation: the prime in 5\u2032, the true MINUS SIGN in 180 \u2212 121, and the "
         "FRACTION SLASH that \u00bd decomposes into.",
         _f_typography),
    Fold("list marker",
         "A bullet at the start of a value is the page's formatting, not the value: "
         "\u2022 Maintain a safe work environment is the same answer as Maintain a safe work "
         "environment. A bullet anywhere else is kept, in case it separates two things.",
         _f_list_marker),
    Fold("number",
         "Read as a number, so 1,000 = 1000, $5 = 5 and 12.90 = 12.9. Zero-padded integers are "
         "exempt: 02000 is an identifier, not the number 2000.", _f_number),
    Fold("dash mark", "A run of dashes is one mark, so - and -- agree. It is not an empty cell.",
         _f_dash),
    Fold("punctuation",
         "Whitespace, commas and hyphens are removed from anywhere, so PO BOX 125 matches "
         "P.O. BOX 125 and 31-1440073 matches 311440073. Periods go too, except a single one "
         "between digits, which is a decimal point and is kept: 1.5 mg is not 15 mg. Two or "
         "more are separators rather than decimal points, so 1.000.000 matches 1,000,000 and "
         "512.784.7407 matches 512-784-7407.", _f_punctuation),
    Fold("quotes and brackets",
         "Quotes around a value are removed, and parentheses and slashes are removed from "
         "anywhere: A(B)C reads as ABC, and/or as andor, N/A as NA.",
         _f_brackets),
)


def run_folds(v: Json, record: list | None = None) -> str:
    """Folds away cosmetic noise in one value, never content, by running `FOLDS` in order until one returns
    `_Final`. With `record`, appends a `Change` for each step that altered the value, so explaining a
    value runs the same code as scoring it."""
    s = str(v)
    for fold in FOLDS:
        out = fold.run(s)
        if record is not None and out != s:
            record.append(Change(fold.name, fold.why, s, str(out)))
        s = out
        if isinstance(out, _Final):
            break
    return str(s)

def _canon(v, record=None):
    """`run_folds` that always returns a key: a value it can't fold (`Infinity`, a huge number, a fold
    raising on unexpected input) keys as itself, truncated, rather than failing the document."""
    try:
        return run_folds(v, record)
    except Exception:
        return " ".join(str(v).split()).lower()[:64]

def _canon_key_unguarded(v, trace=None):
    """A value's key, routed as the first of boolean, number, date, time or text that fits. Only text with
    a "." is parsed as a number, so IDs stay exact; dates and times are strict, so "1/2" and "Q1" stay
    text (METRIC_SPEC section 2)."""
    if isinstance(v, bool):
        if trace is not None:
            trace["route"] = "boolean"
        return _canon(v, trace and trace["changes"])
    d = recognise._asdecimal(v)
    if d is not None:
        q = recognise._round_fraction(d)
        if trace is not None:
            trace["route"] = "number"
        if q == q.to_integral_value():
            return _canon(int(q), trace and trace["changes"])
        return _canon(f"{q.normalize():f}", trace and trace["changes"])
    d = recognise._asdate(v)
    if d:
        if trace is not None:
            trace["route"] = "date"
        return f"#d{d}"
    t = recognise._astime(v)
    if t:
        if trace is not None:
            trace["route"] = "time"
        return f"#t{t}"
    if trace is not None:
        trace["route"] = "text"
    return _canon(str(v), trace and trace["changes"])


def canon_key(v):
    """The canonical form of a value: two values are equal if and only if their keys are equal.

    A fold may trim a value but never consume one: `canon_key(v) == ""` only when `states_nothing(v)`,
    and a value the folds would empty keys as itself without whitespace. Structurally empty values
    aren't scored by default; with `score(..., absent=...)` each kind keys as its own marker (`#NULL`,
    `#EMPTY_STRING`, `#EMPTY_LIST`, `#EMPTY_OBJECT`)."""
    return _canon_key_guarded(v)


def _canon_key_guarded(v, trace=None):
    """`canon_key`, with an optional recorder, so a trace describes the comparison the scorer made."""
    if states_nothing(v):
        if trace is not None:
            trace["route"] = "absent"
        # Note: uppercase, because every other key is lowercased, so no value can collide.
        return ("#NULL" if v is None else "#EMPTY_STRING" if isinstance(v, str)
                else "#EMPTY_LIST" if isinstance(v, list) else "#EMPTY_OBJECT")
    k = _canon_key_unguarded(v, trace)
    if k == "":
        restored = re.sub(r"\s+", "", str(v).strip().lower())
        if trace is not None:
            trace["changes"].append(Change(
                "value restored",
                "No fold may consume a value. This one folded away to nothing, so the value "
                "is compared as itself with whitespace removed.",
                k, restored))
        return restored
    return k


class Trace(NamedTuple):
    """Why one value compares the way it does, for a reader; scoring never uses it."""

    raw: Any
    canon: str
    route: str
    changes: list

    def __str__(self) -> str:
        if not self.changes:
            return f"{self.raw!r} compared as {self.canon!r} (unchanged, {self.route})"
        arrows = " -> ".join([repr(self.changes[0].before)]
                             + [repr(c.after) for c in self.changes])
        return (f"{arrows}   [{', '.join(c.step for c in self.changes)}]")


def canon_trace(v) -> Trace:
    """`canon_key`, plus each step that changed the value and a sentence about it:

    >>> t = canon_trace("31-1440073")
    >>> t.canon
    '311440073'
    >>> [c.step for c in t.changes]
    ['punctuation']
    >>> print(t.changes[0].why)
    Whitespace, commas and hyphens are removed from anywhere, so PO BOX 125 matches P.O. BOX 125 and 31-1440073 matches 311440073. Periods go too, except a single one between digits, which is a decimal point and is kept: 1.5 mg is not 15 mg. Two or more are separators rather than decimal points, so 1.000.000 matches 1,000,000 and 512.784.7407 matches 512-784-7407.

    A value can take a non-text route, in which case no fold runs:

    >>> canon_trace("01/15/2024").route
    'date'
    >>> canon_trace("01/15/2024").canon
    '#d2024-01-15'
    """
    trace: dict = {"route": "text", "changes": []}
    canon = _canon_key_guarded(v, trace)
    return Trace(v, canon, trace["route"], trace["changes"])


def cmp_leaf(pred, gold) -> float:
    """1.0 if the two values are equal under `canon_key`, else 0.0."""
    return 1.0 if canon_key(pred) == canon_key(gold) else 0.0


def states_nothing(value) -> bool:
    """True when a value makes no claim about the document: `null`, `""` (a blank cell in JSON),
    whitespace, or an empty list or object. Placeholder words like "n/a" are ink on the page, so they
    make a claim. Applied to both documents, so a vendor's house style can't move its score (METRIC_SPEC
    section 5).

    >>> [states_nothing(v) for v in (None, "", "   ", [], {}, 0, False, "n/a", "0")]
    [True, True, True, True, True, False, False, False, False]
    """
    return (value is None or value == [] or value == {}
            or (isinstance(value, str) and not value.strip()))
