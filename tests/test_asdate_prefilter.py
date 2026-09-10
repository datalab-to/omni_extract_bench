#!/usr/bin/env python3
"""`_asdate`'s prefilter must never change what counts as a date.

The prefilter exists only to skip `strptime` calls that were going to fail. If it ever
skips one that would have SUCCEEDED, a date silently stops being a date, two values that
used to compare equal stop comparing equal, and a provider's score moves for a reason
nobody can see in the output.

So every check below is the same property, approached from a different direction:
filtered and unfiltered agree. The property is worth this much machinery because the
obvious way to test it does not work. An earlier hand-rolled prefilter -- one that worked
out by hand which literal characters each format demands -- passed 265,056 real values
drawn from the corpus and was still wrong. It rejected "January<TAB>15<TAB>2024", because
`strptime` compiles whitespace in a format to `\\s+` and matches any whitespace, while a
rule looking for a literal " " does not. No document in the corpus separates a date with
tabs, so the corpus had nothing to say about it.

That is the lesson these tests encode: a differential against real data only proves you
did not break the data you already have. The generated cases below are what actually holds
the property down.

Run: python3 tests/test_asdate_prefilter.py
"""
import os as _os
import random
import re
import sys as _sys
from datetime import date, datetime

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench import values as V                              # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


def unfiltered(v):
    """What `_asdate` did before the prefilter: try every format, in order.

    Kept here rather than imported so the comparison survives the day someone deletes the
    slow path entirely.
    """
    s = str(v).strip()
    if not re.search(r"\d", s) or len(s) > 24:
        return None
    for f in V._DATEFMTS:
        try:
            return datetime.strptime(s, f).date().isoformat()
        except ValueError:
            pass
    return None


DAYS = [date(y, m, d)
        for y in (1999, 2000, 2024, 2025)
        for m in range(1, 13)
        for d in (1, 9, 10, 28)]


print("\nASDATE PREFILTER\n")

# ── the prefilter is actually installed ───────────────────────────────────────────────
# `values` falls back to the plain loop if the stdlib internals move, which keeps the
# answers right and quietly loses the speed this code exists for. Say so out loud.
report("the prefilter is installed, not silently fallen back to the slow loop",
       hasattr(V, "_union_regex") and bool(V._union_regex().match("2024-01-15")),
       "no _union_regex: _strptime internals moved and values.py fell back")

# ── whitespace ────────────────────────────────────────────────────────────────────────
# The case that broke the hand-rolled filter, and its neighbours.
ws_cases = ["January\t15\t2024", "Jan\t15, 2024", "Jan\n15 2024",
            "15\xa0January 2024", "Jan  15,  2024"]
bad = [s for s in ws_cases if V._asdate(s) != unfiltered(s) or V._asdate(s) is None]
report("whitespace separators still parse, because strptime matches them with \\s+",
       not bad, f"disagreed or stopped parsing: {bad!r}")

# ── round trip ────────────────────────────────────────────────────────────────────────
# Anything a format can render, that same format must still read back.
checked = 0
lost = []
for fmt in V._DATEFMTS:
    for dt in DAYS:
        rendered = dt.strftime(fmt)
        # strptime accepts unpadded numbers that strftime always pads.
        for s in {rendered, re.sub(r"\b0(\d)", r"\1", rendered)}:
            if len(s) > 24 or not re.search(r"\d", s):
                continue
            try:
                datetime.strptime(s, fmt)
            except ValueError:
                continue                        # not parseable anyway
            checked += 1
            if V._asdate(s) != unfiltered(s):
                lost.append((fmt, s))
report("every format can still read back everything it can write, padded or not",
       not lost, f"lost {lost[:5]!r}")
note(f"{checked:,} rendered date strings checked across {len(V._DATEFMTS)} formats")

# ── values the spec says are not dates ────────────────────────────────────────────────
not_dates = ["1/2", "Q1", "2-3-13", "5", "T", "", "hello", "2024", "-98.2"]
bad = [s for s in not_dates if V._asdate(s) != unfiltered(s)]
report("values the spec refuses as dates are refused exactly as before",
       not bad, f"changed: {bad!r}")

# ── fuzz ──────────────────────────────────────────────────────────────────────────────
# Random strings and mutated dates: shapes no corpus is guaranteed to contain, which is
# the whole point.
rnd = random.Random(0)
alphabet = "0123456789/-. ,\t\xa0JanFebMarchXQ"
pool = [d.strftime(f) for f in V._DATEFMTS for d in DAYS[:20]]
positives = 0
disagreements = []
N = 50_000
for _ in range(N):
    if rnd.random() < 0.5:
        s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 20)))
    else:
        chars = list(rnd.choice(pool))
        for _ in range(rnd.randint(1, 3)):
            i = rnd.randrange(len(chars))
            op = rnd.random()
            if op < 0.4:
                chars[i] = rnd.choice(alphabet)
            elif op < 0.7:
                chars.insert(i, rnd.choice(alphabet))
            else:
                chars.pop(i)
            if not chars:
                break
        s = "".join(chars)
    expected = unfiltered(s)
    positives += expected is not None
    if V._asdate(s) != expected and len(disagreements) < 5:
        disagreements.append((s, expected, V._asdate(s)))

report("filtered and unfiltered agree on random and mutated input",
       not disagreements, f"{disagreements!r}")
# A fuzz run that produced no dates would only be exercising the reject path.
report("the fuzz actually generated dates, so it tests the accept path too",
       positives > 500, f"only {positives} real dates in {N:,} inputs")
note(f"{N:,} fuzz inputs, {positives:,} of them real dates")

print(f"\n{'ASDATE PREFILTER HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
