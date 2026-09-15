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
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Callable, NamedTuple

Json = Any  # parsed-JSON value: dict / list / scalar


# ── the fold pipeline ────────────────────────────────────────────────────────────
# `canon_key` is the only thing here anyone calls. Everything above it builds the answer to
# one question -- "are these two values the same fact?" -- and the order of this file is the
# order that answer is assembled:
#
#   FOLDS               the named, ordered string rules, one function each
#   unwrap / sidecars   envelope and metadata removal, before any value is looked at
#   _asdate / _asdecimal / _astime     the non-text routes a value can take instead
#   canon_key           the entry point: routes, runs the folds, guards the result
#   canon_trace         the same thing, narrated, for a reader who wants to disagree
#   states_nothing      the other half of the absence rule; `flatten` gates on it FIRST,
#                       so canon_key never sees a value that asserts nothing

#: A run of dashes is one printed answer: the clerk's mark for "nothing in this cell". `-` and
#: `--` are the same mark, so they share a key -- but they are NOT absence, and they are not
#: each other's neighbours `N/A` or `None` either. It returns a TAGGED token because the
#: punctuation strip further down would otherwise erase `-` to `""` and silently restore the
#: very folding this exists to prevent.
_DASH = re.compile(r"-+")
_SIDECAR_SUFFIXES = ("_citations", "_meta")


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


def _run_folds(s: str, record: list | None = None) -> str:
    """Run `FOLDS` in order. If `record` is given, append a `Change` per step that changed."""
    for fold in FOLDS:
        out = fold.run(s)
        if record is not None and out != s:
            record.append(Change(fold.name, fold.why, s, str(out)))
        s = out
        if isinstance(out, _Final):
            break
    return str(s)


def _fold_cosmetic(v: Json, record: list | None = None) -> str:
    """Fold cosmetic noise in ONE value -- never real content.

    This is only the driver: the rules live in `FOLDS`, one named function each, so they can be
    listed, explained, and pointed at from a UI instead of being read out of one long function.
    """
    if v is None or v == [] or v == {}:
        return ""
    return _run_folds(str(v), record)


# ── envelopes and sidecars: removed before any value is compared ─────────────────

#: Keys that can legitimately keep `value` company inside a METADATA ENVELOPE. An envelope is
#: a wrapper a vendor puts around an answer to attach provenance to it; the answer is
#: `["value"]` and everything else describes it.
_ENVELOPE_METADATA = frozenset({
    "citations", "citation", "confidence", "score", "reasoning", "evidence", "provenance",
    "source", "sources", "page", "pages", "bbox", "span", "spans", "offset", "offsets",
    "extraction_status", "status", "meta",
})


def unwrap(o: Json) -> Json:
    """Strip a metadata envelope so a cited field becomes its bare value, recursively.

    An object is an envelope when it has a `value` key, at least one recognised metadata key,
    and NOTHING ELSE. That last clause is the whole difficulty, because `value` is also an
    ordinary field name and a row that merely contains one must not be gutted:

        {"value": 12.4, "citations": ["p3"]}                      envelope -- unwrap to 12.4
        {"value": 12.4, "data_period": "FY25", "unit": "USD"}     a row -- 7,130 of these, keep
        {"value": 12.4, "label": "Revenue"}                       a row -- 21,792 of these, keep

    Naming the metadata is what makes that possible. Counting keys instead -- "a `value`, a
    `citations`, and not too many others" -- cannot tell these apart:

        {"value": 12.4, "citations": ["p3"]}                 envelope
        {"value": 12.4, "citations": ["p3"], "unit": "USD"}  a row that cites its source

    and unwrapping the second discards `unit` without trace. What the other keys ARE is the
    question; how many there are never was.

    This fires zero times on the present corpus: no vendor here emits the shape. It is kept
    because vendors do emit it, and a benchmark that silently scored an envelope object as a
    wrong answer would be reporting a fact about the response format as a fact about reading.
    """
    if isinstance(o, dict):
        others = set(o) - {"value"}
        if "value" in o and others and others <= _ENVELOPE_METADATA:
            return unwrap(o["value"])
        return {k: unwrap(v) for k, v in o.items()}
    if isinstance(o, list):
        return [unwrap(x) for x in o]
    return o


def _drop_field_sidecars(obj: Json) -> Json:
    """Drop per-field metadata siblings, so provenance is never charged as a predicted value.

    Some extractors decorate each field `X` with siblings `X_citations` (provenance block ids)
    and `X_meta` (extraction status, reasoning, verification). That metadata is in neither the
    schema nor the ground truth, so counting it would penalise a provider for being informative.

    A key is dropped ONLY when it ends in a sidecar suffix AND its base name is also a sibling
    in the SAME object. That pairing is what identifies a sidecar, and it protects a real field
    that happens to end the same way: a genuine `regulatory_citations` whose base `regulatory`
    is absent survives, while `regulatory_citations_citations` (whose base IS present) does not.

    Applied to every vendor's output identically -- the rule is about the SHAPE, not about who
    produced it, even though today only one provider emits it.
    """
    if isinstance(obj, dict):
        out: dict[str, Json] = {}
        for k, v in obj.items():
            sidecar = any(k.endswith(suf) and k[: -len(suf)] in obj
                          for suf in _SIDECAR_SUFFIXES)
            if not sidecar:
                out[k] = _drop_field_sidecars(v)
        return out
    if isinstance(obj, list):
        return [_drop_field_sidecars(x) for x in obj]
    return obj


def prep_prediction(obj):
    """Unwrap common response envelopes and drop per-field metadata sidecars.

    Some extractors decorate each field `X` with sibling keys `X_citations` (provenance) and
    `X_meta` (status/reasoning). That metadata is absent from the schema and from ground truth,
    so it is ignored when scoring rather than counted as extra predicted leaves -- otherwise a
    provider would be penalised for returning provenance.
    """
    obj = unwrap(obj)
    if isinstance(obj, dict):
        obj = _drop_field_sidecars(obj)
    return obj if isinstance(obj, dict) else {}


def prep_ground_truth(obj):
    """Unwrap common envelopes around ground truth."""
    obj = unwrap(obj)
    return obj if isinstance(obj, dict) else {}


def is_open_map(node) -> bool:
    """True when a schema node declares an object whose KEYS come from the document.

    ``additionalProperties`` asks the extractor to invent the property names by reading them
    off the page. This benchmark does not evaluate that shape: extraction APIs are built around
    a schema that names its fields, and `dialects.STRICT_ALLOWED_KEYS` does not even forward
    the keyword, so a strict vendor receives a bare ``{"type": "object"}`` and has nothing to
    answer with. Grading such a node would score a request the harness never delivered.

    Detection reads EXPLICIT presence as intent rather than JSON Schema semantics, under which
    ``additionalProperties`` defaults to true and every object would qualify.
    """
    if not isinstance(node, dict):
        return False
    for branch in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(branch, []) or []:
            if is_open_map(sub):
                return True
    extra = node.get("additionalProperties")
    return extra is True or isinstance(extra, dict)


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

# Nearly every value handed to these is not a date, and the obvious loop learns that by
# raising once per format -- twenty-five of them between the two lists, which cost 40% of
# grading time on a table-heavy document. The escape is that `strptime` does not really
# parse character by character: it compiles each format to a regex, matches, and raises if
# the match fails or stops short of the end. So we can ask that same regex first, and only
# call `strptime` for a format that can actually match.
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

    #: One alternation per format list, to reject a non-match in a single regex call.
    #: The per-format patterns all name their groups the same way, and duplicate group names
    #: are illegal in an alternation, so they are stripped -- the prefilter only ever needs
    #: to match, never to capture. Keyed on the locale as well as the list, for the same
    #: reason `_TimeRE_cache` is cleared when the locale changes.
    _STRIP_GROUP_NAMES = re.compile(r"\(\?P<\w+>")
    _union_cache: dict = {}

    def _union_regex(fmts):
        key = (_sp._getlang(), fmts)
        rx = _union_cache.get(key)
        if rx is None:
            rx = re.compile("|".join(
                "(?:" + _STRIP_GROUP_NAMES.sub("(?:", _fmt_regex(f).pattern) + ")"
                for f in fmts), re.IGNORECASE)
            _union_cache[key] = rx
        return rx

    def _candidate_formats(s, fmts):
        if not _union_regex(tuple(fmts)).match(s):
            return                      # nothing in this list can match; skip all of them
        for f in fmts:
            m = _fmt_regex(f).match(s)
            # Exactly the two failures `strptime` reports as a format mismatch: no match,
            # and a match that leaves unconverted data behind.
            if m is not None and m.end() == len(s):
                yield f

except Exception:                       # pragma: no cover - stdlib internals moved
    def _candidate_formats(s, fmts):
        return iter(fmts)


def _asdate(v):
    """The calendar date this value denotes, or None.

    Returns a plain date for anything date-shaped, including a timestamp at midnight.
    A timestamp with a real time returns None here and is handled by `_astime`.
    """
    s = str(v).strip()
    if not re.search(r"\d", s) or len(s) > 34:
        return None
    for f in _candidate_formats(s, _DATEFMTS):
        try:
            return datetime.strptime(s, f).date().isoformat()
        except ValueError:
            pass                        # a real date error, e.g. February 30
    dt = _parse_datetime(s)
    if dt is not None and (dt.hour, dt.minute, dt.second, dt.microsecond) == (0, 0, 0, 0):
        return dt.date().isoformat()
    return None


def _parse_datetime(s: str):
    for f in _candidate_formats(s, _DATETIMEFMTS):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass                        # e.g. a 25th hour
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


def _numeric_text(v):
    """The signed numeric text of `v`, or None if `v` is not a decimal number.

    ONLY text containing a '.' is numeric here — that is the whole of the ID protection.
    Sign notation is normalized first, so `(98.2)`, `-98.2` and `\u221298.2` all become
    "-98.2" — but the resulting SIGN is preserved and compared.
    """
    body, sign = _sign_normalize(str(v))
    body = body.replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    if "." not in body:
        return None
    return ("-" if sign < 0 else "") + body


def _asfloat(v):
    """Parse a float ONLY if it looks decimal (has a '.') — integers/IDs stay exact."""
    t = _numeric_text(v)
    if t is None:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _asdecimal(v):
    """Same parse as `_asfloat`, exact. Comparison uses this; a float cannot round honestly.

    `123456.78` as a float is 123456.78000000000174623, and rounding THAT is how a rule
    about decimal places starts disagreeing with the decimal places the document printed.
    """
    t = _numeric_text(v)
    if t is None:
        return None
    try:
        d = Decimal(t)
    except InvalidOperation:
        return None
    # NaN and infinity are not numbers to compare, and an absurd magnitude is not a value a
    # document printed. Both fall through to text comparison, which is exact and cheap.
    #
    # The guard is not cosmetic. `Decimal` has an unbounded exponent where `float` saturates,
    # so `1.0e1000000000` parses happily and then `int()` of it tries to materialise a
    # billion digits -- it hangs rather than failing. The float path this replaced had the
    # matching bug in the other direction: it overflowed to `inf` and `int(inf)` raised
    # OverflowError out of `canon_key`, so one absurd value in one field crashed the scorer.
    if not d.is_finite() or not -_MAX_EXPONENT <= d.adjusted() <= _MAX_EXPONENT:
        return None
    return d


# Precision is a property of the FRACTION, not of the number. An amount and a rate want
# opposite things from a rounding rule -- cents must survive at any magnitude, while a
# re-derived rate wants its trailing digits forgiven -- and both are the same field type to
# a scorer. Rounding to significant digits of the WHOLE number cannot serve both: it spends
# its budget on the integer part first, so 123456.78 and 123456.79 came out equal while
# 0.12345678 and 0.12345679 also came out equal. Only the second of those is wanted.
#
# So the integer part is never rounded, and the fraction keeps seven significant digits of
# its own. Cents are then compared at every magnitude, and two rates agreeing to seven
# figures still agree however small they are.
#
# Why round at all, given rounding is the one rule here that folds VALUES rather than
# spellings? Because nothing gentler can be a key. The rule you would rather have is "equal
# at the precision of the less precise value" -- it calls 33.33333333 and 33.3333333 equal
# and 123456.78 and 123456.79 different, which is exactly right. It is also not transitive:
# 1.25 ~ 1.2 and 1.2 ~ 1.24, but 1.25 != 1.24. A scorer doing set arithmetic on addresses
# needs a function from value to key, and a pairwise comparison cannot be one. So the choice
# is round or be exact, with nothing in between.
#
# TODO(paul): the rounding is now inert unless a fraction carries MORE than seven significant
# digits, so whether it is needed at all is an empirical question about the gold corpus, not
# a judgement call: grep it for values with an eight-digit-or-longer fraction. If there are
# none, delete `_round_fraction` and compare numbers exactly -- the leniency exists only to
# forgive ground truth that was re-derived at a different precision than the page printed.
_FRACTION_DIGITS = 7

# Beyond this many digits either side of the point, a value is not a number a document
# printed and is compared as text instead. See `_asdecimal` for why the bound must exist.
_MAX_EXPONENT = 100


def _fraction_places(d: Decimal) -> int:
    """How many decimal places `d` keeps, so that its FRACTION carries seven significant
    digits — however wide the integer part is, and however many zeros follow the point.

    Read off the digit tuple rather than computed, so there is no arithmetic to lose
    precision in. `Decimal("0.0025")` is `(2,5)` at exponent -4: two zeros follow the point,
    so the seven significant digits start after them and the value keeps nine places.
    `9825.000082185` keeps eleven, because its fraction also opens with zeros — counting
    decimal places instead would have spent the budget on `9825` and rounded a fraction
    that was only five significant digits long.
    """
    _sign, digits, exp = d.as_tuple()
    if not isinstance(exp, int) or exp >= 0:
        return 0                                   # no fractional digits at all
    width = -exp                                   # digits printed after the point
    fraction = ((0,) * (width - len(digits)) + digits)[-width:]
    leading_zeros = 0
    for digit in fraction:
        if digit:
            break
        leading_zeros += 1
    if leading_zeros == width:
        return 0                                   # "5.00" — an integral value
    return _FRACTION_DIGITS + leading_zeros


def _round_fraction(d: Decimal) -> Decimal:
    """`d` with its fractional part rounded; the integer part is left exactly alone."""
    places = _fraction_places(d)
    if not places:
        return d
    with localcontext() as ctx:
        ctx.prec = places + max(d.adjusted(), 0) + 2
        return d.quantize(Decimal(1).scaleb(-places))

# ── leaf comparators -> 1.0 (match) or 0.0 ───────────────────────────────────────
def _canon(v):
    """`_fold_cosmetic`, made total. A model can emit `Infinity`, `NaN`, or a number too large
    for `int(float(...))`, and a fold can raise on input nobody anticipated. Either way this
    must return a key rather than propagate: one unparseable value would otherwise fail a whole
    document. The truncation bounds a pathological value without merging it with anything."""
    try:
        return _fold_cosmetic(v)
    except (OverflowError, ValueError):
        return str(v)[:64]
    except Exception:
        return str(v).lower()

def _canon_key_unguarded(v):
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
        return _canon(v)
    d = _asdecimal(v)
    if d is not None:
        q = _round_fraction(d)
        # An integral value is an integer and must key as one, whether it arrived as `5.0`
        # or rounded up to it. Only text containing a "." is parsed at all, which keeps an ID
        # exact -- but a vendor can emit that same ID as a JSON float, and without this branch
        # `8303911426.0` keys as `8303911000`: a correct account number scores wrong, and two
        # IDs differing in their last three digits score as equal.
        if q == q.to_integral_value():
            return _canon(int(q))
        # Fixed-point text rather than scientific, so the value handed to the folds reads
        # the way the document printed it. `_f_number` will
        # re-parse this as a float, which is where the last of the precision goes: two keys
        # agreeing past the seventeenth significant digit collapse together. That needs an
        # integer part of ten digits AND a seven-digit fraction to reach, and it is shared
        # with every other numeric key in the benchmark, not introduced here.
        return _canon(f"{q.normalize():f}")
    d = _asdate(v)
    if d:
        return f"#d{d}"
    t = _astime(v)
    if t:
        return f"#t{t}"
    return _canon(_defrac(str(v)))


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
    if v is None or v == [] or v == {} or (isinstance(v, str) and not v.strip()):
        return ""
    k = _canon_key_unguarded(v)
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
    canon = canon_key(v)
    if v is None or v == [] or v == {} or (isinstance(v, str) and not v.strip()):
        return Trace(v, canon, "absent", [])
    if isinstance(v, bool):
        return Trace(v, canon, "boolean", [])
    if _asdecimal(v) is not None:
        route = "number"
    elif _asdate(v):
        return Trace(v, canon, "date", [])
    elif _astime(v):
        return Trace(v, canon, "time", [])
    else:
        route = "text"
    # Re-run the folds over whatever the numeric or text route hands them, recording.
    changes: list = []
    _fold_cosmetic(_defrac(str(_canon_source(v))), changes)
    return Trace(v, canon, route, changes)


def _canon_source(v):
    """The string the folds actually see -- `canon_key` normalises numbers before folding."""
    d = _asdecimal(v)
    if d is None:
        return v
    q = _round_fraction(d)
    return int(q) if q == q.to_integral_value() else f"{q.normalize():f}"


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
