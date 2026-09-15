#!/usr/bin/env python3
"""When are two values the same value.

`canon_key` is the single answer. Both callers that need it -- leaf scoring and row pairing --
go through it. Do not add a second: the two implementations this replaced diverged twice in
production, once scoring a correct extraction 50.0 and once taking a provider's recall to
0.000 across two subsets.

Reading order: FOLDS (the named rules) -> run_folds (driver) -> canon_key (entry point) ->
canon_trace (the same run, narrated) -> cmp_leaf -> states_nothing.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, NamedTuple

from .recognise import _asdate, _asdecimal, _astime, _round_fraction

Json = Any  # parsed-JSON value: dict / list / scalar

# A dash run is one mark: `-` and `--` share a key, but it is not absence and not `N/A` or
# `None`. Tagged, because the punctuation strip below would otherwise erase it to "".
_DASH = re.compile(r"-+")


class Change(NamedTuple):
    """One normalisation step that actually altered a value. Returned by `canon_trace`."""

    step: str
    why: str
    before: str
    after: str


class Fold(NamedTuple):
    """One normalisation step. `why` is written for a reader disagreeing with a match, so it
    should name the rule in their terms, not ours."""

    name: str
    why: str
    run: Callable[[str], str]


class _Final(str):
    """A step's result when it decides the key outright and the pipeline should stop."""


def _f_case(s: str) -> str:
    return s.strip().lower()


def _f_accents(s: str) -> str:
    # Cf is the whole invisible-formatting category: soft hyphen, zero-width space, ZWNJ/ZWJ,
    # word joiner, BOM. They render as nothing, so two identical-looking values must not differ
    # because one carries a stray one.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s
                if not unicodedata.combining(c) and unicodedata.category(c) != "Cf")
    return s.replace("\u03bc", "u")   # NFKD has already mapped U+00B5 to this


_FOOTNOTE = re.compile(r"\s*\[\s*(?:\d{1,2}|[a-z])\s*\]")


def _f_footnote(s: str) -> str:
    # 1-2 digits or a single letter only, so `[2024]` and real codes survive. Returns the
    # original when stripping would leave nothing: the marker can BE the value, as in the
    # `[1]`..`[14]` reference numbers that otherwise all key alike.
    out = _FOOTNOTE.sub("", s)
    return out if out.strip() else s


# Unicode spellings of ASCII characters. NFKD leaves these alone, so they are listed by hand;
# 994 corpus values carry one.
_TYPOGRAPHY = (("\u2019", "'"), ("\u2018", "'"),           # curly single quotes
               ("\u201c", '"'), ("\u201d", '"'),           # curly double quotes
               ("\u201a", ","), ("\u201e", '"'),           # low-9 quotes (German)
               ("\u2032", "'"), ("\u2033", '"'),           # prime, double prime (feet/inches)
               ("\u2013", "-"), ("\u2014", "-"),           # en dash, em dash
               ("\u2010", "-"), ("\u2011", "-"),           # HYPHEN, non-breaking hyphen
               ("\u2212", "-"), ("\u2015", "-"),           # MINUS SIGN, horizontal bar
               ("\u2044", "/"),                            # FRACTION SLASH -- what NFKD
                                                            # decomposes \u00bd \u00be \u2157 ... into, so
                                                            # this one line covers all twenty
               )


def _f_typography(s: str) -> str:
    for a, b in _TYPOGRAPHY:
        s = s.replace(a, b)
    return s


# A bullet is furniture only in first position; elsewhere it may separate two things, so
# `MITTAL COURT \u2219 NARIMAN POINT` keeps its dot. Requiring whitespace or end after it keeps
# `\u25a0\u25a0\u25a0-\u25a0\u25a0-\u25a0\u25a0\u25a0\u25a0` (a redacted SSN) intact.
#
# Excluded on corpus evidence: \u00b7 separates units (N\u00b7m is a newton-metre, nm a nanometre) and
# \u00bb is a quotation mark (\u00ab\u2026\u00bb around auditor names in the Greek filings).
_LIST_MARKER = re.compile(r"^[\u2022\u2023\u25e6\u25aa\u25b6\u2219\u25cf\u25a0](?=\s|$)")


def _f_list_marker(s: str) -> str:
    return _LIST_MARKER.sub("", s)


_ZERO_PADDED = re.compile(r"-?0\d")
#: A plain numeral. `float()` is far more permissive -- it reads scientific notation, so it
#: turned the CUSIP `46138E62` into 4.6138e66 and keyed a security identifier as a 67-digit
#: number. 1,014 CUSIPs in the corpus were being read as floats. Exponents are the only thing
#: excluded: a decimal point must still be accepted here, because `canon_key` hands the folds
#: the normalised text of a Decimal and this step is what stops `punctuation` eating its sign.
_PLAIN_NUM = re.compile(r"[+-]?\d+(?:\.\d+)?")


def _f_number(s: str) -> str:
    num = s.replace(",", "").replace("$", "").replace("%", "").strip()
    if not _PLAIN_NUM.fullmatch(num):
        return s
    # A zero-padded integer is an identifier, not a number: 02000 and 2000 are different
    # postal codes. Guard needs a leading zero followed by a digit, so 0 and 0.5 still parse.
    if _ZERO_PADDED.match(num):
        return s
    f = float(num)
    return _Final(str(int(f)) if f == int(f) else str(f))


def _f_dash(s: str) -> str:
    return _Final("#ph_dash") if _DASH.fullmatch(re.sub(r"\s+", "", s)) else s


#: A lone period between two digits is a decimal point. Several are separators -- a phone
#: number, a section id, or a European thousands grouping, since a number cannot have two
#: decimal points. That is why `1.000.000` already equals `1,000,000` and only the SINGLE
#: dotted group `1.000` is ambiguous (one-with-three-decimals, or one thousand).
_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")


def _f_punctuation(s: str) -> str:
    # Deliberately lenient, and it does merge things it should not: 5.2.1.5 == 5215. Kept
    # because the corpus says the trade is one-sided -- it recovers 1,607 matches, of which
    # 1,532 differ by punctuation alone, and credits no wrong value as right. Keeping hyphens
    # instead cost 526 matches in one Schedule I return. Price listed in
    # tests/test_canon_properties.py ACCEPTED_LENIENCY; policy in METRIC_SPEC 5.3.
    #
    # A DECIMAL POINT IS THE EXCEPTION, because deleting it does not fold a spelling, it
    # changes the number: `1.5 mg` became `15mg`, so a 1.5 mg and a 15 mg dose arm were one
    # key. `_f_number` only protects a value that is ENTIRELY a numeral, so anything carrying
    # a unit -- mg, kg, mL, mg/day -- was exposed. A lone dot between digits is kept; two or
    # more are separators (`512.784.7407`, `5.2.1.5`) and still fold.
    #
    # It picks a side rather than resolving the ambiguity: `1.250` is read as a decimal, not
    # as European thousands. That is the safe side -- a false merge credits a wrong answer, a
    # false split only withholds a right one. Cost is measured in TO_LOOK_AT item 31; the knob
    # for reading dotted groups as thousands instead is the `== 1` test below.
    if len(_DECIMAL_POINT.findall(s)) == 1:
        s = _DECIMAL_POINT.sub("\x00", s)
    s = re.sub(r"[\s,\-.]", "", s)
    return s.replace("\x00", ".")


def _f_brackets(s: str) -> str:
    return s.strip("\"'").replace("(", "").replace(")", "").replace("/", "")


#: THE normalisation pipeline, in order. Adding a step here is the only way to change how
#: values compare, and `canon_trace` will report it by name without further work.
FOLDS: tuple[Fold, ...] = (
    Fold("case",
         "Capitalisation and surrounding whitespace are not part of a value: ACME CORP, "
         "Acme Corp and ' Acme Corp ' are one answer.", _f_case),
    Fold("accents",
         "Unicode is reduced to its plainest form: accents drop (M\u00fcller reads as Muller), "
         "invisible characters like a zero-width space are removed, and anything Unicode "
         "considers a styled form of plain text is unstyled -- \u00b5g reads as ug, \ufb01le as file, "
         "\uff46\uff55\uff4c\uff4c as full, \u216b as XII, \u00bd as 1/2 and 10\u00b2 as 102. The last of those is the one "
         "to know about: a superscript is treated as an ordinary digit.",
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
         "A bullet at the START of a value is the page's formatting, not the value: "
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
         "P.O. BOX 125 and 31-1440073 matches 311440073. Periods go too, EXCEPT a single one "
         "between digits, which is a decimal point and is kept: 1.5 mg is not 15 mg. Two or "
         "more are separators rather than decimal points, so 1.000.000 matches 1,000,000 and "
         "512.784.7407 matches 512-784-7407.", _f_punctuation),
    Fold("quotes and brackets",
         "Quotes around a value are removed, and parentheses and slashes are removed from "
         "anywhere: A(B)C reads as ABC, and/or as andor, N/A as NA.",
         _f_brackets),
)


def run_folds(v: Json, record: list | None = None) -> str:
    """Fold cosmetic noise in ONE value -- never real content.

    The driver, and nothing else: the rules live in `FOLDS`, one named function each, so they
    can be listed, explained and pointed at from a UI rather than read out of one long
    function. A step returning `_Final` decides the key outright and ends the run.

    With `record`, appends a `Change` for every step that altered the value. That is the only
    difference between scoring a value and explaining one -- there is no second implementation
    for `canon_trace` to drift from.
    """
    s = str(v)
    for fold in FOLDS:
        out = fold.run(s)
        if record is not None and out != s:
            record.append(Change(fold.name, fold.why, s, str(out)))
        s = out
        if isinstance(out, _Final):
            break
    return str(s)

# ── leaf comparators -> 1.0 (match) or 0.0 ───────────────────────────────────────
def _canon(v, record=None):
    """`run_folds`, made total. A model can emit `Infinity`, `NaN`, or a number too large
    for `int(float(...))`, and a fold can raise on input nobody anticipated. Either way this
    must return a key rather than propagate: one unparseable value would otherwise fail a whole
    document. The truncation bounds a pathological value without merging it with anything."""
    try:
        return run_folds(v, record)
    except Exception:
        # Still fold what is safe to fold. Returning the raw text here made `Infinity` and
        # `infinity` different keys -- a value that trips this path should not also lose case
        # folding. Truncated so a pathological value is bounded, not merged.
        return " ".join(str(v).split()).lower()[:64]

def _canon_key_unguarded(v, trace=None):
    """Route a value and return its key. Tried in order: boolean, number, date, time, text.

    Do not add a second copy of this. Leaf scoring and row pairing both call it, and the two
    independent implementations it replaced diverged twice in production -- once scoring a
    correct extraction 50.0, once taking a provider's recall to 0.000 on two subsets.

    Numbers: only text containing a "." is parsed, which is what keeps IDs exact. Dates and
    times are strict, so "1/2" and "Q1" stay text. METRIC_SPEC section 2 has the rules.
    """
    if isinstance(v, bool):
        if trace is not None:
            trace["route"] = "boolean"
        return _canon(v, trace and trace["changes"])
    d = _asdecimal(v)
    if d is not None:
        q = _round_fraction(d)
        # An integral value is an integer and must key as one, whether it arrived as `5.0`
        # or rounded up to it. Only text containing a "." is parsed at all, which keeps an ID
        # exact -- but a vendor can emit that same ID as a JSON float, and without this branch
        # `8303911426.0` keys as `8303911000`: a correct account number scores wrong, and two
        # IDs differing in their last three digits score as equal.
        if trace is not None:
            trace["route"] = "number"
        if q == q.to_integral_value():
            return _canon(int(q), trace and trace["changes"])
        # Fixed-point text rather than scientific, so the value handed to the folds reads
        # the way the document printed it. `_f_number` will
        # re-parse this as a float, which is where the last of the precision goes: two keys
        # agreeing past the seventeenth significant digit collapse together. That needs an
        # integer part of ten digits AND a seven-digit fraction to reach, and it is shared
        # with every other numeric key in the benchmark, not introduced here.
        return _canon(f"{q.normalize():f}", trace and trace["changes"])
    d = _asdate(v)
    if d:
        if trace is not None:
            trace["route"] = "date"
        return f"#d{d}"
    t = _astime(v)
    if t:
        if trace is not None:
            trace["route"] = "time"
        return f"#t{t}"
    if trace is not None:
        trace["route"] = "text"
    return _canon(str(v), trace and trace["changes"])


def canon_key(v):
    """The canonical form of a value. Two values are equal iff their keys are equal.

    Enforces one invariant on top of the routing, asserted in test_canon_properties P1b:

        canon_key(v) == ""   implies   states_nothing(v)

    A fold may trim a value; it may never consume one. Keying as "" is not the same as being
    dropped -- a dropped value has no address, while an emptied one still scores and matches
    everything else some fold emptied. Structurally empty JSON is absence; anything a model
    chose to write asserts, or `()` in every unreadable field would be free. Values the folds
    would empty fall back to themselves with whitespace removed.
    """
    # Structural emptiness is handled HERE, before anything stringifies it. `str(None)` is
    # the four characters "None", so without this a raw null keyed as `none` -- the same key
    # as the printed word `None`, which is a real answer on an adverse-event form. `flatten`
    # gates on `states_nothing` and so never hands a null to this function, which is the only
    # reason that was invisible rather than wrong.
    return _canon_key_guarded(v)


def _canon_key_guarded(v, trace=None):
    """`canon_key`, with an optional recorder. ONE implementation, so a trace can never
    describe a comparison the scorer did not make."""
    if v is None or v == [] or v == {} or (isinstance(v, str) and not v.strip()):
        if trace is not None:
            trace["route"] = "absent"
        return ""
    k = _canon_key_unguarded(v, trace)
    if k == "":
        return re.sub(r"\s+", "", str(v).strip().lower())
    return k


class Trace(NamedTuple):
    """Why one value compares the way it does. For a UI, not for scoring."""

    raw: Any                # exactly what was in the document / the prediction
    canon: str              # what it was compared AS -- `canon_key(raw)`
    route: str              # absent | boolean | number | date | time | text
    changes: list           # list[Change], in order, only the steps that altered it

    def __str__(self) -> str:
        if not self.changes:
            return f"{self.raw!r} compared as {self.canon!r} (unchanged, {self.route})"
        arrows = " -> ".join([repr(self.changes[0].before)]
                             + [repr(c.after) for c in self.changes])
        return (f"{arrows}   [{', '.join(c.step for c in self.changes)}]")


def canon_trace(v) -> Trace:
    """`canon_key`, plus every step that changed the value and a sentence about each.

    The scorer never calls this -- `canon_key` is the hot path and stays a single pass. This
    exists so a reader can be shown WHY two values matched, in terms they can disagree with:

    >>> t = canon_trace("31-1440073")
    >>> t.canon
    '311440073'
    >>> [c.step for c in t.changes]
    ['punctuation']
    >>> print(t.changes[0].why)
    Whitespace, commas, hyphens and periods are removed from anywhere, so PO BOX 125 matches P.O. BOX 125 and 31-1440073 matches 311440073.

    A value can also take a non-text route, in which case no fold ran at all:

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

# ── values that assert nothing ───────────────────────────────────────────────────


def states_nothing(value) -> bool:
    """Does this value make no claim about the document?

    A page can print "N/A" but cannot print emptiness, so "" is what a blank cell becomes in
    JSON -- the same thing `null` means. Placeholder WORDS are not included: they are ink, and
    appear as real gold values 24,980 times. METRIC_SPEC section 5.

    Applied to both documents, so a vendor's house style cannot move its score.

    >>> [states_nothing(v) for v in (None, "", "   ", 0, False, "n/a", "0")]
    [True, True, True, False, False, False, False]
    """
    return value is None or (isinstance(value, str) and not value.strip())
