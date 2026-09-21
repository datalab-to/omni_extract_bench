"""Date, time and number recognisers.

Each answers "is this text really a X?" and returns the canonical form, or None. They are
strict on purpose: a value that is not obviously one of these falls through to the string
folds in values.py, which is the conservative path. See METRIC_SPEC section 2.
"""
from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

Json = Any


_DATEFMTS = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y",
             "%d %B %Y", "%d-%b-%Y", "%d%b%Y", "%m/%d/%y", "%Y/%m/%d", "%d.%m.%Y",
             "%m.%d.%Y", "%b %d %Y", "%B %d %Y"]
_DATETIMEFMTS = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                 "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z",
                 "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f",
                 "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S.%f%z",
                 "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"]

try:
    import _strptime as _sp

    def _fmt_regex(fmt):
        """The regex strptime itself uses for fmt. Cache is locale-aware, as strptime's is."""
        return _sp._TimeRE_cache.compile(fmt)

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
            return
        for f in fmts:
            m = _fmt_regex(f).match(s)
            if m is not None and m.end() == len(s):
                yield f

except Exception:                       # pragma: no cover - stdlib internals moved
    def _candidate_formats(s, fmts):
        return iter(fmts)


def _asdate(v):
    """ISO date this value denotes, or None. A timestamp at midnight counts; any other
    time is left to _astime."""
    s = str(v).strip()
    if not re.search(r"\d", s) or len(s) > 34:
        return None
    for f in _candidate_formats(s, _DATEFMTS):
        try:
            return datetime.strptime(s, f).date().isoformat()
        except ValueError:
            pass
    dt = _parse_datetime(s)
    if dt is not None and (dt.hour, dt.minute, dt.second, dt.microsecond) == (0, 0, 0, 0):
        return dt.date().isoformat()
    return None


def _parse_datetime(s: str):
    for f in _candidate_formats(s, _DATETIMEFMTS):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    return None


def _astime(v):
    """ISO wall-clock instant, or None. Midnight belongs to _asdate. Normalised so
    09:00:00Z, 09:00:00+00:00 and 09:00:00.000 agree."""
    s = str(v).strip()
    if not re.search(r"\d", s) or len(s) > 34:
        return None
    dt = _parse_datetime(s)
    if dt is None or (dt.hour, dt.minute, dt.second, dt.microsecond) == (0, 0, 0, 0):
        return None
    return dt.replace(tzinfo=None).isoformat(timespec="microseconds")

_NEG_WRAP = re.compile(r"^\((.*)\)$")


def _sign_normalize(s: str):
    """Return (unsigned_text, sign) with sign in {1,-1}, folding notation variants."""
    s = s.strip()
    sign = 1
    m = _NEG_WRAP.match(s)
    if m:
        sign, s = -1, m.group(1).strip()
    s = s.replace("\u2212", "-").replace("\u2013", "-")
    if s.endswith("-"):
        sign, s = -sign, s[:-1].strip()
    while s.startswith(("-", "+")):
        if s[0] == "-":
            sign = -sign
        s = s[1:].strip()
    return s, sign


def _numeric_text(v):
    """Signed numeric text of v, or None if v is not a decimal number.

    Only text containing a '.' counts as numeric. That is the whole ID protection:
    8303911426 is never parsed, so it cannot fuzzy-match a neighbour.
    """
    body, sign = _sign_normalize(str(v))
    body = body.replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    if "." not in body:
        return None
    return ("-" if sign < 0 else "") + body


def _asdecimal(v):
    """Exact Decimal, or None. Decimal not float: 123456.78 as a float is
    123456.78000000000174623, and rounding that disagrees with the printed decimals."""
    t = _numeric_text(v)
    if t is None:
        return None
    try:
        d = Decimal(t)
    except InvalidOperation:
        return None
    if not d.is_finite() or not -_MAX_EXPONENT <= d.adjusted() <= _MAX_EXPONENT:
        return None
    return d


_FRACTION_DIGITS = 7

_MAX_EXPONENT = 100


def _fraction_places(d: Decimal) -> int:
    """Decimal places to keep so the fraction carries 7 significant digits.

    Leading zeros after the point do not count against the budget: Decimal("0.0025") keeps
    nine places, not seven. Read off the digit tuple so no arithmetic can lose precision.
    """
    _sign, digits, exp = d.as_tuple()
    if not isinstance(exp, int) or exp >= 0:
        return 0
    width = -exp
    fraction = ((0,) * (width - len(digits)) + digits)[-width:]
    leading_zeros = 0
    for digit in fraction:
        if digit:
            break
        leading_zeros += 1
    if leading_zeros == width:
        return 0
    return _FRACTION_DIGITS + leading_zeros


def _round_fraction(d: Decimal) -> Decimal:
    """`d` with its fractional part rounded; the integer part is left exactly alone."""
    places = _fraction_places(d)
    if not places:
        return d
    with localcontext() as ctx:
        ctx.prec = places + max(d.adjusted(), 0) + 2
        return d.quantize(Decimal(1).scaleb(-places))

