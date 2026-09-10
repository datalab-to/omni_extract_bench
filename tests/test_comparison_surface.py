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

print("\nSEVEN SIGNIFICANT DIGITS IS A BUCKET, NOT A TOLERANCE")
report("the same printed rate at two precisions matches -- the case this is FOR",
       cmp_leaf(33.33333333, 33.3333333) >= 1.0 and cmp_leaf(1.0, 1.0000001) >= 1.0)
# What the bucket costs, in both directions. Neither is an oversight: no single rule gets
# both re-derived precision and cents at seven figures, and precision is the common case.
report("...paid for by cents merging once an amount reaches six figures",
       cmp_leaf(123456.78, 123456.79) >= 1.0 and cmp_leaf(1234567.89, 1234567.91) >= 1.0,
       "a documented cost, not a bug")
report("...but cents below six figures are compared normally",
       cmp_leaf(12345.67, 12345.68) < 1.0)
report("...and two values straddling a bucket boundary differ, however close",
       cmp_leaf(0.99999994, 1.00000004) < 1.0,
       "0.99999994 vs 1.00000004 differ by 1e-7")
report("every FORMATTING difference folds by PARSING, independently of the bucket",
       all(cmp_leaf(a, b) >= 1.0 for a, b in
           ((5, "5.00"), (1, "1.0"), (1234.5, "1234.50"),
            (1234.56, "$1,234.56"), (12.34, "12.34%"), (-1234.56, "(1,234.56)"))))
report("and no collision is introduced across the decimal point",
       cmp_leaf(1234.5, 12345) < 1.0 and cmp_leaf(0.5, 5) < 1.0,
       f"{canon_key(1234.5)!r} vs {canon_key(12345)!r}")
note("a tolerance cannot be written as a key, and keys are what make the set arithmetic work")

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
_sys.exit(1 if FAILS else 0)
