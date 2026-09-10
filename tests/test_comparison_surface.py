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
]
CHARGED = [
    ("USD 1234.56", 1234.56, "currency code prefix"),
    ("1234.56 USD", 1234.56, "currency code suffix"),
    ("1.234,56", 1234.56, "European decimal notation"),
    ("2024-10-31T00:00:00Z", "2024-10-31", "ISO timestamp against a date"),
    ("2024-10-31 00:00:00", "2024-10-31", "datetime against a date"),
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

print("\nSEVEN SIGNIFICANT DIGITS IS A BUCKET, NOT A TOLERANCE")
report("values agreeing to 7 significant digits match",
       cmp_leaf(1.0000001, 1.0) >= 1.0 and cmp_leaf(99999995.0, 100000005.0) >= 1.0)
report("...but two that straddle a bucket boundary do not, however close",
       cmp_leaf(0.99999994, 1.00000004) < 1.0,
       "0.99999994 vs 1.00000004 differ by 1e-7")
report("the spec does NOT claim a relative tolerance, which would disagree here",
       True)
note("a tolerance cannot be written as a key, and keys are what make the set arithmetic work")

print("\nSIGN VALUE IS NEVER FREE, HOWEVER THE SIGN IS SPELLED")
report("every negative spelling folds to the same key",
       len({canon_key(x) for x in ("(98.2)", "-98.2", "−98.2", "98.2-")}) == 1)
report("...and a disagreement about the sign is a mismatch",
       cmp_leaf(98.2, -98.2) < 1.0)

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
