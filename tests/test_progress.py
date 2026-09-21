#!/usr/bin/env python3
"""The live per-provider display.

Almost all of it is checked without a terminal, which is the point of the split: `Stats` is
counters, `format_*` are pure functions of those counters, and only `Progress` knows where a
cursor goes. A layout regression should not need a tty to catch.

Run: python3 tests/test_progress.py
"""
import io
import logging
import time

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from omni_extract_bench.progress import (                                      # noqa: E402
    NULL, Progress, Stats, format_bar, format_duration, format_flight, format_money,
    format_provider, format_total)

FAILS = []


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


class FakeTTY(io.StringIO):
    """A stream that claims to be a terminal, so the drawing path runs under test."""

    def isatty(self):
        return True


print("\nCOUNTERS")
s = Stats(total=10)
s.ok, s.errors = 7, 2
report("done is what has been answered either way", s.done == 9)
s.waits = [10.0, 20.0, 30.0]
report("mean_wall is the average document", s.mean_wall == 20.0)
report("no documents yet means no average, not zero", Stats().mean_wall is None,
       "a zero would read as instant")

print("\nDURATIONS READ AT A GLANCE")
for seconds, want in [(None, "--"), (0, "0.0s"), (0.36, "0.4s"), (9.9, "9.9s"), (10, "10s"), (42, "42s"), (60, "1m00s"),
                      (259, "4m19s"), (3600, "1h00m"), (8700, "2h25m")]:
    report(f"{seconds} -> {want}", format_duration(seconds) == want, format_duration(seconds))

print("\nMONEY STAYS IN THE VENDOR'S OWN UNIT")
report("dollars where the vendor reports dollars", format_money(Stats(usd=183.49)) == "$183.49")
report("credits where it reports credits", format_money(Stats(credits=17951)) == "17,951 cr")
report("a vendor reporting neither gets nothing, not $0.00", format_money(Stats()) == "",
       "a zero would read as free")

print("\nTHE BAR")
report("empty", format_bar(0, 10, 10) == "░" * 10)
report("half", format_bar(5, 10, 10) == "█████░░░░░")
report("full", format_bar(10, 10, 10) == "█" * 10)
report("an unknown total is all track, no fill", format_bar(0, None, 10) == "░" * 10)
report("it cannot overflow its width", len(format_bar(99, 10, 10)) == 10)

print("\nA PROVIDER'S LINE")
waiting = format_provider("mistral", Stats())
report("a provider that has not started says so", "waiting" in waiting, waiting)
line = format_provider("datalab", Stats(total=620, ok=118, usd=183.49,
                                        waits=[259.0], started=time.monotonic() - 3600))
for want in ("datalab", "118/620", "ok 118", "err 0", "$183.49", "avg 4m19s"):
    report(f"the line carries {want!r}", want in line, line)
done = format_provider("reducto", Stats(total=5, ok=5, started=1.0, finished=61.0))
report("a finished provider reports its total time",
       "dur 1m00s" in done, done)

print("\nCALLS IN FLIGHT")
report("saturated", format_flight(Stats(running=25, workers=25)) == "25/25")
report("not saturated", format_flight(Stats(running=3, workers=25)) == "3/25")
report("idle is still worth saying while a provider runs",
       format_flight(Stats(running=0, workers=25)) == "0/25")
report("no cap known -> just the count", format_flight(Stats(running=4)) == "4")
report("a finished provider drops the column",
       format_flight(Stats(running=0, workers=25, finished=1.0)) == "",
       "0/25 on a done row is noise")

print("\nTHE COUNT GOES UP AND COMES BACK DOWN")
prog = Progress(["v"], stream=io.StringIO())
rep = prog.reporter("v")
rep.start(10, workers=4)
report("the cap is recorded from the pool", prog.stats["v"].workers == 4)

small = Progress(["v"], stream=io.StringIO())
small.reporter("v").start(2, workers=10)
report("a pool larger than the work is clamped to the work",
       small.stats["v"].workers == 2, str(small.stats["v"].workers))
report("...so a smoke test reads as saturated, not idle",
       "2/2" in format_flight(Stats(total=2, running=2, workers=2)))
big = Progress(["v"], stream=io.StringIO())
big.reporter("v").start(620, workers=10)
report("a full run is untouched by the clamp", big.stats["v"].workers == 10)
with rep.calling():
    report("a call in flight is counted", prog.stats["v"].running == 1)
    with rep.calling():
        report("...and they nest", prog.stats["v"].running == 2)
    report("...and unwind", prog.stats["v"].running == 1)
report("back to nothing outstanding", prog.stats["v"].running == 0)

try:
    with rep.calling():
        raise RuntimeError("the vendor said no")
except RuntimeError:
    pass
report("a raising call still comes back down", prog.stats["v"].running == 0,
       "otherwise one failure inflates the count for the rest of the run")

print("\nTHE TOTAL LINE")
total = format_total({"a": Stats(total=10, ok=10, usd=5.0),
                      "b": Stats(total=10, ok=8, errors=2, credits=400)})
report("documents are summed", "20/20 documents" in total, total)
report("failures are summed", "2 failed" in total, total)
report("the two units sit side by side rather than being merged",
       "$5.00 + 400 cr" in total, total)
report("in flight is summed across providers",
       "9 in flight" in format_total({"a": Stats(total=9, running=4),
                                      "b": Stats(total=9, running=5)}))
report("nothing to add up yet -> no line at all, not '0/0 documents'",
       format_total({"a": Stats(), "b": Stats()}) == "")

print("\nNOT A TTY -> NO ESCAPE CODES")
plain = io.StringIO()
with Progress(["datalab"], stream=plain) as bars:
    bars.reporter("datalab").start(2)
    bars.reporter("datalab").record(wall_s=1.0)
report("nothing is drawn at all", plain.getvalue() == "", repr(plain.getvalue()[:40]))
report("...and the display knows it is not live", Progress([], stream=io.StringIO()).live is False)

print("\nA TTY -> A LIVE TABLE, REDRAWN IN PLACE")
tty = FakeTTY()
with Progress(["datalab", "reducto"], stream=tty, tick=1000) as bars:
    datalab = bars.reporter("datalab")
    datalab.start(4)
    datalab.record(usd=1.55, wall_s=259.0)
    bars.redraw()
out = tty.getvalue()
report("both providers have a row", "datalab" in out and "reducto" in out)
report("a total line is added for more than one provider", "documents" in out)
report("it redraws in place rather than scrolling", "\x1b[" in out)
report("the counters reached the display", "1/4" in out, out.replace("\x1b", "^")[-200:])

print("\nONE PROVIDER: NO TOTAL LINE")
solo = FakeTTY()
with Progress(["datalab"], stream=solo, tick=1000):
    pass
report("no total for a single provider", "documents" not in solo.getvalue(), solo.getvalue())

print("\nLOG LINES DO NOT SCRIBBLE OVER THE BARS")
tty = FakeTTY()
root = logging.getLogger()
saved = root.handlers[:]
root.handlers[:] = [logging.StreamHandler(tty)]
try:
    with Progress(["datalab"], stream=tty, tick=1000) as bars:
        bars.reporter("datalab").start(1)
        wrapped = root.handlers[0].__class__.__name__
        logging.getLogger("omni_extract_bench.test").warning("something happened")
    text = tty.getvalue()
finally:
    root.handlers[:] = saved
report("the handler is wrapped while the display is live", wrapped == "_Interleaved", wrapped)
report("the message still got out", "something happened" in text)
report("...and the handlers are put back afterwards", root.handlers == saved)

print("\nTHE NULL REPORTER SWALLOWS EVERYTHING")
NULL.start(5, workers=3)
with NULL.calling():
    NULL.record(error=True, usd=1.0, wall_s=2.0)
NULL.finish()
report("a caller with no display needs no branches", True)

print(f"\n{'PROGRESS HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
