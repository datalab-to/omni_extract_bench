#!/usr/bin/env python3
"""A rate limit while polling must not buy the extraction twice.

The poll loops raised `VendorError` on any non-200, so one 429 on one poll escaped to
`predict` -- which reads the status as transient, correctly, and retries the document FROM THE
UPLOAD. The job already running at the vendor was abandoned and a second one submitted, for
$1.55 on datalab, silently, because the retry returns a perfectly good answer.

It is not rare: a 259-second document polls ~51 times, and at `--predict-workers 25` that is
five requests a second for hours. Raising concurrency makes it likelier, not rarer.

Run: python3 tests/test_poll_retry.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from omni_extract_bench.harness.extraction import (                            # noqa: E402
    Budget, PollRetry, TRANSIENT_STATUSES, VendorError)

FAILS = []


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


def retry(limit=60, seconds=30.0):
    return PollRetry(Budget(seconds), limit=limit)


print("\nWHAT IS WORTH ASKING AGAIN, AND WHAT IS NOT")
# The same rule `predict` applies one level up, and for the same reason: read off the status,
# never matched in a message.
for status in sorted(TRANSIENT_STATUSES):
    report(f"{status} is polled again", retry().again(status) is True)
for status in (400, 401, 403, 404, 422):
    report(f"{status} gives up at once", retry().again(status) is False)
report("a transport failure carries no status and is polled again",
       retry().again(None) is True, "it says nothing about the job either")

print("\nIT GIVES UP EVENTUALLY, AND COUNTS CONSECUTIVE FAILURES ONLY")
r = retry(limit=3)
report("under the limit, keep polling", [r.again(429) for _ in range(3)] == [True] * 3)
report("...past it, stop", r.again(429) is False)

r = retry(limit=3)
r.again(429); r.again(429)
r.ok()                                   # a poll came back: the run of failures is over
report("a successful poll resets the count", [r.again(429) for _ in range(3)] == [True] * 3,
       "otherwise a long job accumulates unrelated blips and gives up on a healthy vendor")

print("\nIT NEVER WAITS PAST THE DOCUMENT'S DEADLINE")
# Backing off beyond the budget would spend the document's time doing nothing and then call a
# vendor with none left. Whether the budget then ran out is the loop's business: it re-checks
# and raises VendorTimeout, which is the honest answer while the vendor is still working.
import time                                                                    # noqa: E402
spent = Budget(0.05)
r = PollRetry(spent)
t0 = time.monotonic()
for _ in range(6):                       # backoff would reach 2s, 4s, 8s... unbounded
    r.again(503)
report("the whole backoff fits inside what is left of the budget",
       time.monotonic() - t0 < 1.0, f"{time.monotonic() - t0:.2f}s")

print("\nAND EVERY POLLING ADAPTER USES IT")
# The point of one implementation is that none of them can drift back. reducto had this from
# the start; the other four raised on any non-200 and paid twice.
import inspect                                                                 # noqa: E402
import importlib                                                               # noqa: E402

for name in ("datalab", "reducto", "extend", "llamaextract", "azure_cu"):
    src = inspect.getsource(importlib.import_module(
        f"omni_extract_bench.harness.providers.{name}"))
    report(f"{name} polls through PollRetry", "PollRetry(" in src)

print(f"\n{'A FAILED POLL NO LONGER BUYS A SECOND JOB' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
