"""Is this text actually a date, a time, or a number -- and if so, what is it exactly?

Three recognisers, and they are deliberately strict. A value only leaves the text route if it
is unmistakably one of these, because the alternative is worse in both directions: a loose date
parser swallows `1/2` and `Q1`, and a loose number parser turns an account number into a float
and rounds its last digits away. When a recogniser declines, the value falls through to the
folds in `values.py`, which is the conservative path.

Nothing here knows about folding, scoring, or documents. It answers one question about one
string.
"""
from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

Json = Any  # parsed-JSON value: dict / list / scalar


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


def _asdecimal(v):
    """Parse a value as an exact Decimal. Comparison uses this; a float cannot round honestly.

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

