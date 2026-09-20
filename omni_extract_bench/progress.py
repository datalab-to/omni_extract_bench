"""A live line per provider, while the benchmark runs.

    datalab       ██████░░░░░░░░   118/620  ok 118  err 0    $183.49  avg 4m19s
    reducto       ███░░░░░░░░░░░    64/620  ok  63  err 1   17951 cr  avg 3m12s
    mistral       ░░░░░░░░░░░░░░         waiting

A run takes hours and the vendors now go at once, so "how far along is each one, and what is
it costing" is the question the terminal should be answering the whole time. It was answered
by a log line every ten documents per provider, which interleaves into noise once more than
one vendor is writing it.

THREE LAYERS, AND ONLY THE LAST ONE KNOWS ABOUT A TERMINAL:

    Stats        plain counters. No formatting, no output, no lock.
    format_*     Stats -> one string. Pure functions, so the layout is testable without a tty.
    Progress     the terminal: where the cursor goes, what a redraw costs, when to tick.

`Progress` is also the only thing that needs a lock, because the providers run concurrently
and each has a pool of its own behind it. `Stats` is touched under that lock and nowhere else.

NOT A TTY -> NOT A BAR. Piped to a file or running in CI, there is no cursor to move and ANSI
would be litter in a log, so the same counters come out as a periodic line per provider. The
caller does not choose: `Progress` reads the stream and decides.
"""
from __future__ import annotations

import contextlib
import logging
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: How often the bar redraws. Documents here take minutes, so nothing moves faster than this;
#: it is a refresh rate for the clock and the spinner, not for the counts.
TICK_S = 0.5

#: Counts printed per provider between redraws when there is no tty to redraw on.
PLAIN_EVERY = 10


# ── 1. the counters ──────────────────────────────────────────────────────────────────────
@dataclass
class Stats:
    """What one provider has done so far.

    `total` is None until the provider has worked out its own workload -- on a resume that is
    the documents still missing, not the size of the corpus, so it cannot be known up front.
    """

    total: int | None = None
    workers: int | None = None        # how many this provider may have in flight at once
    running: int = 0                  # how many it has in flight right now
    ok: int = 0
    errors: int = 0
    usd: float = 0.0
    credits: float = 0.0
    waits: list[float] = field(default_factory=list)   # wall seconds, one per document
    started: float | None = None
    finished: float | None = None

    @property
    def done(self) -> int:
        return self.ok + self.errors

    @property
    def mean_wall(self) -> float | None:
        """Seconds a document takes at this vendor, as the harness saw it."""
        return sum(self.waits) / len(self.waits) if self.waits else None

    @property
    def elapsed(self) -> float | None:
        if self.started is None:
            return None
        return (self.finished or time.monotonic()) - self.started


# ── 2. the layout ────────────────────────────────────────────────────────────────────────
def format_duration(seconds: float | None) -> str:
    """A span a person can read at a glance: `42s`, `4m19s`, `2h24m`."""
    if seconds is None:
        return "--"
    if seconds < 10:
        return f"{seconds:.1f}s"          # a sub-second document would otherwise read as 0s
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def format_money(stats: Stats) -> str:
    """What the vendor has said this cost, in ITS OWN unit.

    Credits are never converted to dollars: the rate is contract-specific, so a dollar figure
    derived from one would be invented rather than measured. A vendor that reports neither
    gets an empty column rather than a zero, which would read as free.
    """
    if stats.usd:
        return f"${stats.usd:,.2f}"
    if stats.credits:
        return f"{stats.credits:,.0f} cr"
    return ""


def format_flight(stats: Stats) -> str:
    """Calls out at the vendor right now, against how many are allowed.

    The denominator is the point. `25/25` says the pool is saturated and more workers would
    buy more throughput; `3/25` says something else is the limit -- the vendor's own queue,
    or simply that there are only three documents left. Tuning `--predict-workers` without it
    is guesswork.
    """
    if not stats.running and stats.finished is not None:
        return ""
    return (f"{stats.running}/{stats.workers} in flight" if stats.workers
            else f"{stats.running} in flight")


def format_bar(done: int, total: int | None, width: int = 14) -> str:
    return "░" * width if not total else (
        "█" * round(width * min(done, total) / total)).ljust(width, "░")


def format_provider(name: str, stats: Stats, name_width: int = 12) -> str:
    """One provider's line. Pure: give it a `Stats` and it gives you the text."""
    head = f"{name:<{name_width}}  {format_bar(stats.done, stats.total)}"
    if stats.total is None:
        return f"{head}       waiting"
    width = len(str(stats.total))
    parts = [f"{head}  {stats.done:>{width}}/{stats.total}",
             f"ok {stats.ok:>{width}}",
             f"err {stats.errors}"]
    flight = format_flight(stats)
    if flight:
        parts.append(flight)
    money = format_money(stats)
    if money:
        parts.append(money)
    if stats.mean_wall is not None:
        parts.append(f"avg {format_duration(stats.mean_wall)}")
    if stats.finished is not None:
        parts.append(f"done in {format_duration(stats.elapsed)}")
    return "  ".join(parts)


def format_total(everything: dict[str, Stats]) -> str:
    """The bottom line: every provider added up, or "" before there is anything to add.

    Dollars and credits are added SEPARATELY and printed side by side, for the same reason
    `format_money` refuses to convert -- one number spanning both units would be a fiction.
    """
    totals = list(everything.values())
    if not any(s.total for s in totals):
        return ""                         # "0/0 documents" before the first provider starts
    done = sum(s.done for s in totals)
    total = sum(s.total or 0 for s in totals)
    errors = sum(s.errors for s in totals)
    usd = sum(s.usd for s in totals)
    credits = sum(s.credits for s in totals)
    started = [s.started for s in totals if s.started is not None]

    running = sum(s.running for s in totals)
    parts = [f"{done}/{total} documents", f"{errors} failed"]
    if running:
        parts.append(f"{running} in flight")
    money = " + ".join(x for x in (f"${usd:,.2f}" if usd else "",
                                   f"{credits:,.0f} cr" if credits else "") if x)
    if money:
        parts.append(money)
    if started:
        parts.append(f"{format_duration(time.monotonic() - min(started))} elapsed")
    return "  ".join(parts)


# ── 3. the terminal ──────────────────────────────────────────────────────────────────────
class Reporter:
    """One provider's handle on the display. This is all `predict_all` ever holds.

    Four things happen to a provider, and there is a method for each rather than one call
    with flags: it learns its workload, it puts a call in flight, that call comes back, and
    eventually it is finished.
    """

    def __init__(self, progress: "Progress", name: str):
        self._progress, self._name = progress, name

    def start(self, total: int, workers: int | None = None) -> None:
        """This provider now knows how many documents it runs, and how many at a time.

        `workers` is CLAMPED TO THE WORKLOAD, because the ceiling on concurrency is the
        smaller of the two: a pool of ten against two documents can never be more than two
        deep, so `2/10 in flight` would report as idle something that is in fact saturated --
        and the denominator exists precisely to answer "would more workers buy anything".

        On a full run the clamp does nothing (`min(10, 620)` is 10). It corrects the case a
        `--limit` smoke test creates, where every provider is capped by the corpus instead.
        Clamped here rather than by the caller, so the invariant holds for all of them.
        """
        self._progress._set(self._name, total=total,
                            workers=min(workers, total) if workers else None,
                            started=time.monotonic())

    @contextlib.contextmanager
    def calling(self):
        """Hold one call open. What is inside this block is what the vendor is working on.

        A context manager rather than a begin/end pair, because the count must come back down
        however the call leaves -- returned, raised, or cancelled -- and a missed decrement
        shows a vendor permanently busier than it is.
        """
        self._progress._flight(self._name, +1)
        try:
            yield
        finally:
            self._progress._flight(self._name, -1)

    def record(self, *, error: bool = False, usd: float | None = None,
               credits: float | None = None, wall_s: float | None = None) -> None:
        """One document finished, well or badly."""
        self._progress._done(self._name, error=error, usd=usd, credits=credits,
                             wall_s=wall_s)

    def finish(self) -> None:
        self._progress._set(self._name, finished=time.monotonic())


class NullReporter:
    """What a caller with no display gets, so nothing downstream needs an `if`."""

    def start(self, total: int, workers: int | None = None) -> None: ...
    def record(self, **_) -> None: ...
    def finish(self) -> None: ...

    @contextlib.contextmanager
    def calling(self):
        yield


NULL = NullReporter()


class Progress:
    """The live display. A context manager: it owns the cursor only between enter and exit.

        with Progress(["datalab", "reducto"]) as bars:
            predict_all(..., progress=bars.reporter("datalab"))

    Providers are registered UP FRONT and in order, so the lines keep their places and a
    vendor that has not started yet is visibly waiting rather than absent.
    """

    def __init__(self, names, stream=None, tick: float = TICK_S):
        self.stream = stream if stream is not None else sys.stderr
        self.live = bool(getattr(self.stream, "isatty", lambda: False)())
        self.stats: dict[str, Stats] = {name: Stats() for name in names}
        self.lock = threading.RLock()
        self._width = max((len(n) for n in self.stats), default=8)
        self._drawn = 0
        self._tick = tick
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved_handlers: list | None = None

    # -- the caller's side -----------------------------------------------------------
    def reporter(self, name: str) -> Reporter:
        with self.lock:
            self.stats.setdefault(name, Stats())
        return Reporter(self, name)

    def _set(self, name: str, **fields) -> None:
        """Facts about the provider rather than about a document: workload, start, finish."""
        with self.lock:
            s = self.stats[name]
            for key, value in fields.items():
                if value is not None:
                    setattr(s, key, value)
            # A provider's last word, for a log that has no bar to look at. Without it a
            # provider finishing at 617 of 620 never prints its final tally, because the
            # periodic line only lands on a multiple of `PLAIN_EVERY`.
            line = (format_provider(name, s, self._width)
                    if fields.get("finished") and not self.live else None)
        if line:
            log.info("%s", line)

    def _flight(self, name: str, delta: int) -> None:
        """A call went out, or came back. Called from the pool's threads, not the main one."""
        with self.lock:
            self.stats[name].running += delta

    def _done(self, name: str, *, error=False, usd=None, credits=None, wall_s=None) -> None:
        """One document is answered. The only thing that moves `ok`, `errors` or the money."""
        with self.lock:
            s = self.stats[name]
            s.errors += bool(error)
            s.ok += not error
            if usd is not None:
                s.usd += usd
            if credits is not None:
                s.credits += credits
            if wall_s is not None:
                s.waits.append(wall_s)
            # Rendered under the lock, LOGGED OUTSIDE IT. `_Interleaved.emit` takes this
            # lock from inside the logging module's own lock, so logging while holding it is
            # the opposite order and two threads can meet in the middle. The two cannot
            # overlap today -- this line needs `not self.live`, the wrapper only exists when
            # `self.live` -- but that is a coincidence of configuration, not a guarantee.
            line = (format_provider(name, s, self._width)
                    if not self.live and s.done % PLAIN_EVERY == 0 else None)
        if line:
            log.info("%s", line)

    # -- the terminal side -----------------------------------------------------------
    def _clear(self) -> None:
        if self.live and self._drawn:
            self.stream.write(f"\x1b[{self._drawn}A\x1b[J")
            self._drawn = 0

    def _draw(self) -> None:
        if not self.live:
            return
        columns = shutil.get_terminal_size((120, 24)).columns
        lines = [format_provider(n, s, self._width) for n, s in self.stats.items()]
        if len(self.stats) > 1:
            lines += [line for line in [format_total(self.stats)] if line]
        self.stream.write("".join(line[:columns] + "\n" for line in lines))
        self.stream.flush()
        self._drawn = len(lines)

    def redraw(self) -> None:
        with self.lock:
            self._clear()
            self._draw()

    def _loop(self) -> None:
        while not self._stop.wait(self._tick):
            self.redraw()

    # -- lifecycle -------------------------------------------------------------------
    def __enter__(self) -> "Progress":
        if self.live:
            # A log line written straight to the stream would scribble over the bars, so the
            # root handlers are wrapped for the duration: clear, let the real handler write,
            # draw again. Wrapped rather than replaced, so formatting and level are whatever
            # the CLI configured.
            root = logging.getLogger()
            self._saved_handlers = root.handlers[:]
            root.handlers[:] = [_Interleaved(self, h) for h in self._saved_handlers]
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self.redraw()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self._tick)
        if self._saved_handlers is not None:
            logging.getLogger().handlers[:] = self._saved_handlers
            self._saved_handlers = None
        if self.live:
            # Leave the finished state on screen rather than erasing it: it is the summary of
            # what just happened, and the JSON on stdout is a different audience.
            self.redraw()
            self._drawn = 0


class _Interleaved(logging.Handler):
    """A log handler that does not walk over the bars: clear, write, redraw."""

    def __init__(self, progress: Progress, inner: logging.Handler):
        super().__init__(level=inner.level)
        self.progress, self.inner = progress, inner

    def emit(self, record: logging.LogRecord) -> None:
        with self.progress.lock:
            self.progress._clear()
            self.inner.emit(record)
            self.progress._draw()
