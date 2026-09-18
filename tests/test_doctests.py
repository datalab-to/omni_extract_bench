#!/usr/bin/env python3
"""Every doctest in the package, run.

The scorer's docstrings carry worked examples -- what a fold does, what an address looks like,
why two different strings compare equal -- and those examples are the explanation someone reads
before trusting a number. An example that no longer matches the code is worse than none: it
documents behaviour the scorer does not have.

Nothing was running them. `metric.py`'s 19 doctests only executed under
`python -m omni_extract_bench.metric`, which CI does not do, and `values.py`'s were not executed
at all -- so `canon_trace`'s example sat stale, still describing periods as always removed after
the rule had grown a decimal-point exception.

Run: python3 tests/test_doctests.py
"""
import doctest
import importlib
import pkgutil
import sys

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import omni_extract_bench  # noqa: E402

#: The vendor adapters import SDKs that only the `harness` extra installs, and a machine that
#: only scores has none of them. Their docstrings carry usage lines, not worked examples, so
#: there is nothing here to lose by leaving them out.
SKIP = ("omni_extract_bench.harness",)

FAILS = []
total = 0

for module in pkgutil.walk_packages(omni_extract_bench.__path__,
                                    prefix="omni_extract_bench."):
    name = module.name
    if any(name.startswith(s) for s in SKIP):
        continue
    try:
        mod = importlib.import_module(name)
    except ImportError as exc:                       # an optional extra is not installed
        print(f"  SKIP  {name} -- {exc}")
        continue
    result = doctest.testmod(mod, verbose=False, report=True)
    total += result.attempted
    mark = "PASS" if not result.failed else "FAIL"
    if result.attempted:
        print(f"  {mark}  {name}: {result.attempted - result.failed}/{result.attempted}")
    if result.failed:
        FAILS.append(f"{name} ({result.failed} of {result.attempted})")

print(f"\n{total} doctests run")
print("ALL DOCTESTS PASS" if not FAILS else "FAILURES:")
for f in FAILS:
    print(f"   {f}")
sys.exit(1 if FAILS else 0)
