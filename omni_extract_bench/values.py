#!/usr/bin/env python3
"""When are two values the same value, and what is a value at all?

One question, one answer, one place. Everything that needs to know whether two values are
equal -- scoring an address, weighting a candidate row pairing -- calls `canon_key`. That is
not tidiness: the benchmark once had two implementations of this that were required to agree,
and they silently diverged twice. Once when the pairing weight demanded literal equality while
scoring accepted date formats, so a correct extraction scored 50.0. Again when a numeric fix
put integers and decimals in different namespaces, so `5` and `5.0` stopped pairing and one
provider's recall went to 0.000 across two subsets. With a single function a disagreement is
not a bug to be caught by testing -- it cannot be written down.

Both halves of every concern sit together: comparing values, preparing a document to be
compared, and the two schema questions the scorer asks.

This module depends on nothing else in the package, so a scorer can be built on it without
pulling in a grader.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, NamedTuple

from .recognise import _asdate, _asdecimal, _astime, _round_fraction

Json = Any  # parsed-JSON value: dict / list / scalar


# ── the fold pipeline ────────────────────────────────────────────────────────────
# `canon_key` is the only thing in this package anyone calls to compare two values, and this
# file is the answer to one question: are these two the same fact? It reads in that order --
#
#   FOLDS           the named, ordered string rules, one small function each
#   run_folds       the driver, which also records a Change per step when asked
#   canon_key       the entry point: route the value, run the folds, guard the result
#   canon_trace     the same run, narrated, for a reader who wants to disagree with it
#   cmp_leaf        the yes/no the scorer actually calls
#   states_nothing  the OTHER half of the absence rule -- `flatten` gates on it first, so
#                   `canon_key` is never handed a value that asserts nothing
#
# Two neighbours hold what this deliberately does not:
#   recognise.py    is this text a date, a time, a number, and what is it exactly
#   prepare.py      shaping a document and reading a schema, before any value is compared

#: A run of dashes is one printed answer: the clerk's mark for "nothing in this cell". `-` and
#: `--` are the same mark, so they share a key -- but they are NOT absence, and they are not
#: each other's neighbours `N/A` or `None` either. It returns a TAGGED token because the
#: punctuation strip further down would otherwise erase `-` to `""` and silently restore the
#: very folding this exists to prevent.
_DASH = re.compile(r"-+")


class Change(NamedTuple):
    """One normalisation step that actually altered a value. Returned by `canon_trace`."""

    step: str
    why: str
    before: str
    after: str


class Fold(NamedTuple):
    """One normalisation step, named so that a reader can object to it BY NAME.

    The point of naming them is downstream. A customer looking at a match and thinking "you
    should not have folded that" needs to know which rule to argue with; `31-1440073` folding
    to `311440073` is not an answer, "the punctuation rule removed the hyphen" is. `why` is
    written for that reader, not for us.
    """

    name: str
    why: str
    run: Callable[[str], str]


class _Final(str):
    """A step's result when it decides the key outright and the pipeline should stop."""


def _f_case(s: str) -> str:
    return s.strip().lower()


def _f_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("\u00b5", "u").replace("\u03bc", "u")


_FOOTNOTE = re.compile(r"\s*\[\s*(?:\d{1,2}|[a-z])\s*\]")


def _f_footnote(s: str) -> str:
    # Only 1-2 digit or single-letter brackets, so real bracketed content (`[2024]`, codes)
    # survives -- and only when something is LEFT. The marker can BE the value: gold
    # `bibliography_entries[].ref_number` in `internal/...eu_einvoice_standard_160p__s5` is
    # literally `[1]` through `[14]`. Strip those and all fourteen become one key, which makes
    # reversing every reference number score 100. A fold may remove an annotation, never the
    # value.
    out = _FOOTNOTE.sub("", s)
    return out if out.strip() else s


_FRAC = {"½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3", "⅛": "1/8",
         "⅜": "3/8", "⅝": "5/8", "⅞": "7/8", "⅕": "1/5", "⅙": "1/6", "⅐": "1/7"}
def _defrac(s: str) -> str:
    for u, a in _FRAC.items():
        s = s.replace(u, a)
    return s


#: Characters that ARE an ASCII character, spelled in Unicode. Folding them is not a policy
#: choice, it is finishing the job `unicodedata` starts -- NFKD leaves all of these alone.
#: The four dash forms below were missing and it showed: `Pascual\u2010Montano` did not fold like
#: `Pascual-Montano`, and `180,476,646.08 \u2212 121,058,316.40` kept a minus sign the arithmetic
#: fields elsewhere write as `-`. 994 values in the corpus carry one.
_TYPOGRAPHY = (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
               ("\u2013", "-"),        # en dash
               ("\u2014", "-"),        # em dash
               ("\u2010", "-"),        # HYPHEN -- the Unicode one, not hyphen-minus
               ("\u2011", "-"),        # non-breaking hyphen
               ("\u2212", "-"),        # MINUS SIGN
               ("\u00ad", ""),         # soft hyphen: an invisible line-break hint, not ink
               )


def _f_typography(s: str) -> str:
    for a, b in _TYPOGRAPHY:
        s = s.replace(a, b)
    return s


#: A list marker is furniture the page prints to start a bullet, not part of the value. It is
#: stripped only as the FIRST character, which is the only position where it is unambiguously
#: furniture -- an interior one may be a separator that carries meaning, so
#: `MITTAL COURT \u2219 NARIMAN POINT` keeps its dot. The corpus is why this is a position rule
#: rather than a plain character class:
#:
#:   \u25a0\u25a0\u25a0-\u25a0\u25a0-\u25a0\u25a0\u25a0\u25a0   a redacted SSN. Stripping \u25a0 everywhere makes every redaction
#:                  block the same key, and `---` then folds to the dash mark on top.
#:   \u25cf            appears ALONE as a value, a filled checkbox meaning "yes". Stripping empties
#:                  it, and `canon_key`'s guard restores it, so it survives either way.
#:
#: Two characters that look like they belong here are excluded, with the reason recorded:
#:   \u00b7  MIDDLE DOT is a unit separator. `N\u00b7m` is a newton-metre; folding it gives `nm`,
#:       which is a nanometre. Different quantity, same key -- exactly a P1 false merge.
#:   \u00bb  is a quotation mark, not a bullet. The Greek filings use \u00ab\u2026\u00bb around auditor names.
_LIST_MARKER = re.compile(r"^[\u2022\u2023\u25e6\u25aa\u25b6\u2219\u25cf\u25a0](?=\s|$)")


def _f_list_marker(s: str) -> str:
    return _LIST_MARKER.sub("", s)


_ZERO_PADDED = re.compile(r"-?0\d")


def _f_number(s: str) -> str:
    # A zero-PADDED integer is an identifier, not a number: `02000` and `2000` are different
    # postal codes, `06` and `6` different state codes, and padding is how a document says
    # which. Reading them numerically also merges `INV-007` with `INV-7`,
    # `arXiv:2405.06211v3` with `arXiv:2405.6211v3`, and `wenqifan03@gmail.com` with
    # `wenqifan3@gmail.com`. `0` and `0.5` are unaffected -- the guard needs a leading zero
    # followed by another digit.
    num = s.replace(",", "").replace("$", "").replace("%", "")
    if _ZERO_PADDED.match(num.strip()):
        return s
    try:
        f = float(num)
    except ValueError:
        return s
    return _Final(str(int(f)) if f == int(f) else str(f))


def _f_dash(s: str) -> str:
    return _Final("#ph_dash") if _DASH.fullmatch(re.sub(r"\s+", "", s)) else s


def _f_punctuation(s: str) -> str:
    # LENIENT, AND ON PURPOSE. Whitespace, commas, hyphens and periods fold from anywhere in
    # the value. It is the wrong rule in principle -- `5.2.1.5` and `5215` are two different
    # section identifiers and this makes them one -- and the right rule in practice, and the
    # corpus is what decided it.
    #
    # Measured over all 660 documents and nine vendors: folding periods recovers 1,607 value
    # matches, 1,532 of which differ by punctuation ALONE (`PO BOX 125` / `P.O. BOX 125`,
    # `FT LAUDERDALE` / `FT. LAUDERDALE`). The remaining 75 are bibliography entries where the
    # model kept the `[4] ` citation number -- the same reference either way. Not one of the
    # 1,607 is a wrong value credited as right. Of the gold values this merges within a single
    # field, ZERO change the digit string: all 215 are the same fact spelled twice. Hyphens
    # tell it more sharply -- keeping them cost 526 matches in a single Schedule I return,
    # where gold writes EINs and ZIP+4s bare (`311440073`) and every model hyphenates.
    #
    # What it costs is not hidden: `5.2.1.5` == `5215`, `1.1%w/w` == `11%w/w`, `RR-2` == `RR2`,
    # `#30-2` == `#302`, `90-94` milled to `9094`. Each is in ACCEPTED_LENIENCY, and
    # METRIC_SPEC 5.3 states the policy. Note how little P1 it gives up: `1:00.50` != `1:50`,
    # `0.11%w/w` != `11%w/w`, `COM PAR $.001` != `COM PAR $.01` and `arXiv:2405.06211v3` !=
    # `arXiv:2405.6211v3` all still hold -- protected by `number` keeping leading zeros, not
    # by punctuation, which is why that divergence is worth keeping and these are not.
    return re.sub(r"[\s,\-.]", "", s)


def _f_brackets(s: str) -> str:
    return s.strip("\"'").replace("(", "").replace(")", "").replace("/", "")


#: THE normalisation pipeline, in order. Adding a step here is the only way to change how
#: values compare, and `canon_trace` will report it by name without further work.
FOLDS: tuple[Fold, ...] = (
    # FIRST, and it has to be: `accents` runs NFKD, which decomposes \u00bd into `1\u20442` -- a form
    # the fraction table no longer recognises. Order in this list is behaviour.
    Fold("fractions", "A single-character fraction is written out, so \u00bd reads as 1/2.",
         _defrac),
    Fold("case", "Capitalisation is not part of a value: ACME CORP is Acme Corp.", _f_case),
    Fold("accents",
         "Accents and the micro sign fold to ASCII, so Muller matches M\u00fcller and ug matches \u00b5g.",
         _f_accents),
    Fold("footnote marker",
         "A short [1] or [a] reference marker attached to a value is dropped -- but never when "
         "it is the whole value.", _f_footnote),
    Fold("typography", "Curly quotes and en/em dashes become their plain ASCII forms.",
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
         "Whitespace, commas, hyphens and periods are removed from anywhere, so PO BOX 125 "
         "matches P.O. BOX 125 and 31-1440073 matches 311440073.", _f_punctuation),
    Fold("quotes and brackets", "Enclosing quotes, parentheses and slashes are removed.",
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
    except (OverflowError, ValueError):
        return str(v)[:64]
    except Exception:
        return str(v).lower()

def _canon_key_unguarded(v, trace=None):
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
      2. numbers    -- parsed and canonicalised, so 5, 5.0 and "5.00" agree, and so do
                       "$1,234.56" and 1234.56. Only text containing a "." is parsed at all,
                       so an ID-like integer is never touched; and an integral float keys as
                       its integer, so 8303911426.0 agrees with 8303911426 rather than being
                       rounded away from it.

                       The integer part is compared exactly. The FRACTION is rounded to
                       7 significant digits of its own, so 33.33333333 and 33.3333333 are
                       one printed rate at two precisions and agree, while 123456.78 and
                       123456.79 are one cent apart and differ. See `_FRACTION_DIGITS`
                       for why precision belongs to the fraction and not to the number,
                       and docs/METRIC_SPEC.md section 2 for what it costs.
      3. dates      -- ISO form, so 2024-01-15 and 01/15/2024 agree, and so does a
                       timestamp at midnight. `_asdate` is strict: "1/2", "Q1" and "2-3-13"
                       are not dates, so this cannot swallow values that mean something else.
      3b. timestamps -- a real time of day is KEPT, normalised, so two spellings of one
                       instant agree while two different times still differ.
      4. otherwise  -- unicode fractions expanded, then canonicalised
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
    """`_canon_key_unguarded`, plus the one invariant every fold must respect.

    A FOLD MAY TRIM A VALUE; IT MAY NEVER CONSUME ONE. Keying as `""` is not the same as being
    thrown out. A thrown-out value has no address at all -- `flatten` gates on `states_nothing`
    before this function is ever called, so `null`, `""`, whitespace and empty containers never
    reach it. A value that keys as `""` DOES have an address, is scored, and equals every other
    value some fold happened to empty -- `()`, `,`, `/`, `"`, `. . .`, `..`, `null` and every
    `[1]`-shaped value would otherwise share one key with each other.

    Which is right, and why it is not a judgement call. `null` and `""` are both STRUCTURALLY
    empty JSON, and the harness itself causes vendors to disagree about which to send -- a
    strict dialect must emit the key with `null` where a permissive one omits it -- so they
    have to score alike (METRIC_SPEC 5). `()` and the string `"null"` are content a model
    chose to emit; nothing in the harness induces them. Treating them as absence would make
    them free, and a model could write `()` in every field it could not read and pay nothing --
    the hole just closed for `-`. So: structurally empty is absence, everything else asserts.

    The test is exact rather than heuristic, which is what makes this safe. Run the folds and
    compare: if the result is empty and the input had non-whitespace content, the folds ate it.
    No field is consulted, so P4 holds. The fallback removes whitespace and nothing else --
    whitespace being the one fold that cannot destroy content -- so `( )` still agrees with
    `()` and `. . .` with `...`.

    The invariant, asserted in tests/test_canon_properties.py P1b:

        canon_key(v) == ""   implies   states_nothing(v)
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

    `null` is the obvious case. The empty string is the same case wearing a different
    serialization: a document can *print* "N/A", but it cannot print emptiness, so `""` is
    what a blank cell becomes on the way into JSON -- exactly what `null` means.

    The distinction is worth stating because it does not extend to the placeholder WORDS.
    "N/A", "None" and "-" are ink on the page, and the corpus uses them as real gold values
    24,980 times, with 67 documents holding both those strings and `null` in the same file.
    Folding those would delete real answers. Folding `""` cannot: every one of the 1,369
    gold empty strings in the corpus is a blank cell, 1,196 of them one empty column in a
    check register.

    Called on both documents, so the rule is symmetric by construction: a prediction that
    writes `""` scores exactly as one that writes `null` or omits the key, and gold spelled
    either way asks for the same thing. Without it the score moved with a vendor's house
    style -- one provider's `""` convention cost it 7.92 points on `longarray` alone.

    >>> [states_nothing(v) for v in (None, "", "   ", 0, False, "n/a", "0")]
    [True, True, True, False, False, False, False]
    """
    return value is None or (isinstance(value, str) and not value.strip())
