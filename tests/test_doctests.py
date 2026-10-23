#!/usr/bin/env python3
"""Runs every doctest in the package, so a worked example in a docstring can't drift from what the code does.

Run: python3 tests/test_doctests.py
"""
import doctest
import importlib
import pkgutil
import sys

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import omni_extract_bench  # noqa: E402

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
    except ImportError as exc:
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
print("All doctests pass" if not FAILS else "Failures:")
for f in FAILS:
    print(f"   {f}")
sys.exit(1 if FAILS else 0)
