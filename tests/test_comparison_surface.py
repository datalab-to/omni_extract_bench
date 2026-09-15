#!/usr/bin/env python3
"""Exactly which value differences are free, and which are not.

METRIC_SPEC section 2 prints two lists: format differences the comparator folds, and ones
that look like formatting to a person but score as disagreements. Both lists are asserted
here, so the spec cannot promise a fold the code does not do -- which it did, until this file
existed. It claimed "any disagreement the grader reports is a real one", and six plausible
format differences say otherwise.

The second list is the interesting one: it is a list of candidate SCORER gaps. If a provider's
misses look like formatting, look here before blaming the vendor.

Run: python3 tests/test_comparison_surface.py
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench.values import canon_key, cmp_leaf                  # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


FOLDED = [
    ("1,234", 1234, "thousands comma"),
    ("$1,234.56", 1234.56, "currency symbol"),
    ("1 234.56", 1234.56, "space as thousands separator"),
    ("12.34%", 12.34, "percent sign"),
    ("5.00", 5, "trailing zeros"),
    ("1.0", 1, "decimal spelling of an integer"),
    ("(1,234.56)", -1234.56, "accounting negative"),
    ("1234.56-", -1234.56, "trailing minus"),
    ("−1234.56", -1234.56, "unicode minus"),
    ("31/10/2024", "2024-10-31", "day-first date"),
    ("10/31/24", "2024-10-31", "two-digit year"),
    ("Oct 31, 2024", "2024-10-31", "abbreviated month"),
    ("ACME CORP", "Acme Corp", "case"),
    ("Acme  Corp.", "Acme Corp", "double space and trailing period"),
    ("TRUE", True, "boolean spelling"),
    ("2024-10-31T00:00:00Z", "2024-10-31", "ISO timestamp at midnight, against a date"),
    ("2024-10-31 00:00:00", "2024-10-31", "datetime at midnight, against a date"),
    ("2024-10-31T00:00:00+00:00", "2024-10-31", "explicit UTC offset at midnight"),
    ("2024-10-31T00:00:00.000Z", "2024-10-31", "milliseconds at midnight"),
    ("2024-10-31T09:00:00Z", "2024-10-31 09:00:00", "two spellings of one instant"),
    ("2024-10-31T09:00:00Z", "2024-10-31T09:00:00+00:00", "Z against an explicit offset"),
]
CHARGED = [
    ("USD 1234.56", 1234.56, "currency code prefix"),
    ("1234.56 USD", 1234.56, "currency code suffix"),
    ("1.234,56", 1234.56, "European decimal notation"),
    ("2024-10-31T09:00:00Z", "2024-10-31", "a timestamp with a real time is not a date"),
    ("2024-10-31T09:00:00Z", "2024-10-31T17:00:00Z", "two different times on one date"),
    ("yes", True, "'yes' as a boolean"),
]

print("\nFORMAT DIFFERENCES THE COMPARATOR FOLDS")
for a, b, why in FOLDED:
    report(f"{why}: {a!r} == {b!r}", cmp_leaf(a, b) >= 1.0)

print("\nAND THE ONES IT DOES NOT -- CANDIDATE SCORER GAPS, NOT VENDOR ERRORS")
for a, b, why in CHARGED:
    report(f"{why}: {a!r} != {b!r}", cmp_leaf(a, b) < 1.0,
           "this now folds -- update METRIC_SPEC section 2, it lists this as charged")
note("if one of these starts folding, the spec's second list is stale, not this test")

print("\nA TIMESTAMP AT MIDNIGHT IS A DATE; ANY OTHER TIME IS A TIME")
report("midnight folds to the date, so a vendor spelling a date as a timestamp is not charged",
       cmp_leaf("2024-10-31T00:00:00Z", "2024-10-31") >= 1.0)
report("a real time of day is kept, so two different times still differ",
       cmp_leaf("2024-10-31T09:00:00Z", "2024-10-31T17:00:00Z") < 1.0)
report("...and a timed value does not match a bare date",
       cmp_leaf("2024-10-31T09:00:00Z", "2024-10-31") < 1.0)
report("two spellings of one instant agree",
       cmp_leaf("2024-10-31T09:00:00Z", "2024-10-31 09:00:00") >= 1.0
       and cmp_leaf("2024-10-31T09:00:00Z", "2024-10-31T09:00:00.000Z") >= 1.0)
report("nothing that merely contains digits is swallowed as a date",
       all(not str(canon_key(x)).startswith(("#d", "#t"))
           for x in ("Q1", "1/2", "2-3-13", "12:00", "2024", "1.2.3")),
       str([canon_key(x) for x in ("Q1", "1/2", "2-3-13", "12:00", "2024", "1.2.3")]))
note("the offset is dropped rather than converted: extraction reads the printed wall clock")

print("\nPRECISION BELONGS TO THE FRACTION, NOT TO THE WHOLE NUMBER")
# Rounding the whole number to 7 significant digits spent its budget on the integer part
# first, so it got these two cases backwards: cents merged once an amount reached six
# figures, while a small rate got no more leniency than a large one. The integer part is now
# compared exactly and the FRACTION keeps 7 significant digits of its own.
report("the same printed rate at two precisions matches -- the case leniency is FOR",
       cmp_leaf(33.33333333, 33.3333333) >= 1.0)
report("...and a fraction agreeing to 7 figures matches however small it is",
       cmp_leaf(0.12345678, 0.12345679) >= 1.0 and cmp_leaf(0.052500001, 0.0525) >= 1.0)
report("cents are compared at EVERY magnitude",
       all(cmp_leaf(a, b) < 1.0 for a, b in
           ((12345.67, 12345.68), (123456.78, 123456.79),
            (1234567.89, 1234567.91), ("12222222.78", "12222222.79"))),
       "the last three of these used to be false matches")
report("...while the fraction stays lenient at those same magnitudes",
       cmp_leaf("1.7800000001", "1.7800000002") >= 1.0
       and cmp_leaf("12222222.7800000001", "12222222.7800000002") >= 1.0,
       "this is the pairing a whole-number rule could not express")
report("a ten-millionth apart is a difference, not a reformatting a model would make",
       cmp_leaf(1.0, 1.0000001) < 1.0)
report("rounding the fraction CAN carry into the integer part -- that is what rounding is",
       cmp_leaf("1.9999999999", 2) >= 1.0, f"1.9999999999 keys as {canon_key('1.9999999999')!r}")
report("...but the integer part is never a rounding target, so it cannot be eroded",
       cmp_leaf("0.99999994", "1.00000004") < 1.0,
       f"{canon_key('0.99999994')!r} vs {canon_key('1.00000004')!r}")
report("every FORMATTING difference folds by PARSING, independently of the rounding",
       all(cmp_leaf(a, b) >= 1.0 for a, b in
           ((5, "5.00"), (1, "1.0"), (1234.5, "1234.50"),
            (1234.56, "$1,234.56"), (12.34, "12.34%"), (-1234.56, "(1,234.56)"))))
report("and no collision is introduced across the decimal point",
       cmp_leaf(1234.5, 12345) < 1.0 and cmp_leaf(0.5, 5) < 1.0,
       f"{canon_key(1234.5)!r} vs {canon_key(12345)!r}")
note("rounding is a bucket, not a tolerance: a comparison cannot be written as a key")

print("\nSIGN VALUE IS NEVER FREE, HOWEVER THE SIGN IS SPELLED")
report("every negative spelling folds to the same key",
       len({canon_key(x) for x in ("(98.2)", "-98.2", "−98.2", "98.2-")}) == 1)
report("...and a disagreement about the sign is a mismatch",
       cmp_leaf(98.2, -98.2) < 1.0)

print("\nAN ID KEYS THE SAME WHETHER IT ARRIVES AS AN INTEGER OR A FLOAT")
# This was a real defect. Only values containing a "." are parsed as numbers, which is what
# keeps an ID exact -- but a vendor emitting the same ID as a JSON float had it rounded to 7
# significant digits: 8303911426.0 keyed as 8303911000. So a CORRECT account number scored as
# wrong, and two IDs differing in their last three digits scored as equal. An integral float
# now keys as its integer.
report("the same ID as an int and as a float agree",
       cmp_leaf(8303911426, 8303911426.0) >= 1.0
       and cmp_leaf(8303911426, "8303911426.0") >= 1.0,
       f"keys {canon_key(8303911426)!r} vs {canon_key(8303911426.0)!r}")
report("two IDs differing only in their last three digits still differ, as floats",
       cmp_leaf(8303911426.0, 8303911400.0) < 1.0,
       f"keys {canon_key(8303911426.0)!r} vs {canon_key(8303911400.0)!r}")
report("...and the integral-float rule does not disturb decimals",
       cmp_leaf(5, "5.00") >= 1.0 and cmp_leaf(1, "1.0") >= 1.0
       and cmp_leaf(1234.5, "1234.50") >= 1.0 and cmp_leaf(1234.5, 1234.6) < 1.0)
note("only values containing a '.' are parsed at all -- that is the whole ID protection")

print("\nID-LIKE INTEGERS STAY EXACT")
report("one digit different in an account number is a mismatch",
       cmp_leaf("8303911426", "8303511426") < 1.0)
report("...and an integer is never rounded into a neighbour",
       cmp_leaf(1234567891, 1234567892) < 1.0)
note("only values containing a '.' are parsed as decimals, so IDs cannot be fuzzed")

print(f"\n{'COMPARISON SURFACE IS AS DOCUMENTED' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\nA VERDICT CARRIES BOTH FORMS, AND `None` MEANS ONE THING")
# The surface a UI reads. `gold_raw`/`pred_raw` are what the document and the model wrote;
# `gold_canon`/`pred_canon` are what they were compared AS. A match between two visibly
# different strings is then self-explaining, and `values.canon_trace(raw)` names the step.
from omni_extract_bench.score import explain, show, Verdict          # noqa: E402
from omni_extract_bench.values import states_nothing, canon_trace, canon_key    # noqa: E402

_SCH = {"type": "object", "properties": {
    "a": {"type": "string"}, "b": {"type": "string"}, "c": {"type": "string"},
    "g": {"type": "object", "additionalProperties": {"type": "string"}}}}
_V = {v.verdict: v for v in explain({"a": "X", "c": "Z", "g": {"k": "v"}},
                                    {"a": "X", "b": "Y", "g": {"k": "v"}}, _SCH)}

report("Verdict carries raw and canon for each side",
       Verdict._fields == ("address", "gold_raw", "pred_raw", "verdict",
                           "gold_canon", "pred_canon"), f"{Verdict._fields}")
report("a missing address has no pred side",
       _V["missing"].pred_raw is None and _V["missing"].pred_canon is None
       and _V["missing"].gold_raw == "Y")
report("a fabricated address has no gold side",
       _V["fabricated"].gold_raw is None and _V["fabricated"].gold_canon is None
       and _V["fabricated"].pred_raw == "Z")
report("a skipped open map carries neither side",
       all(x is None for x in (_V["skipped (open map)"].gold_raw,
                               _V["skipped (open map)"].pred_raw,
                               _V["skipped (open map)"].gold_canon,
                               _V["skipped (open map)"].pred_canon)))

# The property that lets a caller test `is None` and be done: nothing that states nothing ever
# reaches a Verdict, so None cannot mean "an empty value" -- only "no value here".
_cases = [({"a": "X", "c": "Z"}, {"a": "X", "b": "Y"}),
          ({"a": None, "b": ""}, {"a": "Y", "b": "Z"}),
          ({"a": "  ", "b": []}, {"a": "Y", "b": {}}),
          ({}, {"a": "Y"}), ({"a": "Y"}, {})]
_bad = []
for _p, _g in _cases:
    for v in explain(_p, _g, _SCH):
        if (v.gold_raw is None) != (v.gold_canon is None): _bad.append(("gold disagree", v))
        if (v.pred_raw is None) != (v.pred_canon is None): _bad.append(("pred disagree", v))
        if v.gold_raw is not None and states_nothing(v.gold_raw): _bad.append(("gold blank", v))
        if v.pred_raw is not None and states_nothing(v.pred_raw): _bad.append(("pred blank", v))
        if v.gold_canon == "" or v.pred_canon == "": _bad.append(("canon empty", v))
report("raw and canon are None together, and a set value is never blank", not _bad,
       f"{_bad[:3]}")
note("`flatten` gates on states_nothing before an address exists, so None on a Verdict means")
note("'no value at this address' -- never 'a value that happened to be empty'.")

# THE CARRY-THROUGH ITSELF: the canon a Verdict reports must be the canon the scorer used.
# A UI that showed a different one would be explaining a comparison that never happened.
_carry = []
for _p, _g in _cases + [
        ({"a": "31-1440073", "b": "\u2022 x", "c": "03/31/2024"},
         {"a": "311440073", "b": "x", "c": "2024-03-31"}),
        ({"a": "NIKE, Inc.", "b": "12.90", "c": "\u00bd"},
         {"a": "Nike", "b": "12.9", "c": "1/2"})]:
    for v in explain(_p, _g, _SCH):
        if v.gold_raw is not None and v.gold_canon != canon_key(v.gold_raw):
            _carry.append(("gold", v.gold_raw, v.gold_canon, canon_key(v.gold_raw)))
        if v.pred_raw is not None and v.pred_canon != canon_key(v.pred_raw):
            _carry.append(("pred", v.pred_raw, v.pred_canon, canon_key(v.pred_raw)))
        # and the verdict must follow from the two canons, not from anything else
        if v.gold_raw is not None and v.pred_raw is not None:
            want = "match" if v.gold_canon == v.pred_canon else "wrong value"
            if v.verdict != want:
                _carry.append(("verdict", v.address, v.verdict, want))
report("gold_canon/pred_canon are exactly canon_key of the raw values", not _carry,
       f"{_carry[:3]}")
report("...and the verdict follows from comparing them", not [c for c in _carry if c[0] == "verdict"])

_m = _V["match"]
report("canon_trace explains a match between two raw forms",
       canon_trace(_m.pred_raw).canon == _m.pred_canon == _m.gold_canon,
       f"{canon_trace(_m.pred_raw)}")
report("canon_trace agrees with the Verdict on every raw value it carries",
       all(canon_trace(r).canon == c
           for _p, _g in _cases for v in explain(_p, _g, _SCH)
           for r, c in ((v.gold_raw, v.gold_canon), (v.pred_raw, v.pred_canon))
           if r is not None))



# ═══════════════════════════════════════════════════════════════════════════════
print("\nA NUMBER TOO ABSURD TO BE PRINTED IS COMPARED AS TEXT")
# `Decimal` has an unbounded exponent where `float` saturates, so `1.0e1000000000` parses
# happily and `int()` of it then tries to materialise a billion digits -- it HANGS rather than
# failing. The magnitude guard in `_asdecimal` is what stops that, and nothing tested it:
# mutation-testing the recognisers removed the guard and the whole suite still passed.
from omni_extract_bench.recognise import _asdecimal, _fraction_places   # noqa: E402
from decimal import Decimal                                             # noqa: E402

for _v in ("1.0e1000000000", "-1.0e1000000000", "1e400", "nan", "inf", "-inf", "NaN"):
    report(f"{_v!r} is not a number to compare", _asdecimal(_v) is None,
           f"_asdecimal returned {_asdecimal(_v)!r}")
report("...and such a value still produces a short key, promptly",
       all(isinstance(canon_key(v), str) and len(canon_key(v)) < 100
           for v in ("1.0e1000000000", "nan", "inf")),
       "a key of a billion digits means the guard stopped firing")
note("they fall through to text comparison, which is exact and cheap. The failure this")
note("prevents is a hang, not a wrong answer, so it cannot be caught by a score check.")

print("\nPRECISION IS A PROPERTY OF THE FRACTION, NOT OF THE NUMBER")
# Leading zeros after the point do not spend the 7-significant-digit budget. Documented in
# `_fraction_places` and, until now, asserted nowhere.
for _v, _want, _why in (("0.5", 7, "no leading zeros, 7 places"),
                        ("0.0025", 9, "two leading zeros, so 7 significant digits start later"),
                        ("9825.000082185", 11, "a wide integer part spends none of the budget"),
                        ("123.4567890123", 7, "no leading zeros"),
                        ("5.00", 0, "integral, nothing to round")):
    report(f"{_v} keeps {_want} decimal places ({_why})",
           _fraction_places(Decimal(_v)) == _want,
           f"got {_fraction_places(Decimal(_v))}")
report("so cents survive at any magnitude",
       canon_key("123456.78") != canon_key("123456.79"))
report("...while a re-derived rate agreeing to 7 figures still agrees",
       canon_key("33.33333333") == canon_key("33.3333333"))


_sys.exit(1 if FAILS else 0)
