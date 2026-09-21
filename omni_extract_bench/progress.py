"""A live table while the benchmark runs, and a log line per provider when there is no tty.

     run                                      done    ok   err   in flight        cost     avg     dur
    ─────────────────────────────────────────────────────────────────────────────────────────────────
    datalab-f46415c9   ━━━                118/620   118     0       10/10     $183.49   4m19s   1h12m
    reducto-e54d3a1d   ━╸                  64/620    63     1         3/3   17,951 cr   3m12s   1h12m
    mistral-44136fa3   waiting
    182/1240 documents  1 failed  13 in flight  $183.49 + 17,951 cr  1h12m elapsed

A run takes hours and the vendors go at once, so "how far along is each one, and what is it
costing" is the question the terminal should be answering the whole time. It was answered by a
log line every ten documents per provider, which interleaves into noise once more than one
vendor is writing it.

THREE LAYERS, AND ONLY THE LAST ONE KNOWS ABOUT A TERMINAL:

    Stats        plain counters. No formatting, no output, no lock.
    format_*     Stats -> one string, and `render` -> one table. Pure, so the layout is
                 testable without a tty.
    Progress     the terminal: a rich `Live`, and what a log line does to it.

`Progress` is also the only thing that needs a lock, because the providers run concurrently and
each has a pool of its own behind it. `Stats` is touched under that lock and nowhere else.

NOT A TTY -> NOT A TABLE. Piped to a file or running in CI there is nothing to redraw, and a
table every ten documents is worse in a log than a line, so the same counters come out as a
periodic line per provider. The caller does not choose: `Progress` reads the stream and decides.

Colour carries only what the numbers already say -- red where errors are not zero, green where
a provider is finished -- so a terminal that strips it loses nothing.
"""
from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from dataclasses import dataclass, field

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

log = logging.getLogger(__name__)

TICK_S = 0.5

PLAIN_EVERY = 10

BAR_WIDTH = 16


@dataclass
class Stats:
    """What one provider has done so far.

    `total` is None until the provider has worked out its own workload -- on a resume that is
    the documents still missing, not the size of the corpus, so it cannot be known up front.
    """

    total: int | None = None
    workers: int | None = None
    running: int = 0
    ok: int = 0
    errors: int = 0
    usd: float = 0.0
    credits: float = 0.0
    waits: list[float] = field(default_factory=list)
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


def format_duration(seconds: float | None) -> str:
    """A span a person can read at a glance: `42s`, `4m19s`, `2h24m`."""
    if seconds is None:
        return "--"
    if seconds < 10:
        return f"{seconds:.1f}s"
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
    buy more throughput; `3/25` says something else is the limit. Bare, because the table heads
    the column and the log line labels it -- said in both it read `10/10 in flight` under a
    heading of `in flight`.
    """
    if not stats.running and stats.finished is not None:
        return ""
    return f"{stats.running}/{stats.workers}" if stats.workers else str(stats.running)


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
        parts.append(f"{flight} in flight")
    money = format_money(stats)
    if money:
        parts.append(money)
    if stats.mean_wall is not None:
        parts.append(f"avg {format_duration(stats.mean_wall)}")
    if stats.elapsed is not None:
        parts.append(f"dur {format_duration(stats.elapsed)}")
    return "  ".join(parts)


def format_total(everything: dict[str, Stats]) -> str:
    """The bottom line: every provider added up, or "" before there is anything to add.

    Dollars and credits are added SEPARATELY and printed side by side, for the same reason
    `format_money` refuses to convert -- one number spanning both units would be a fiction.
    """
    totals = list(everything.values())
    if not any(s.total for s in totals):
        return ""
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


def render(stats: dict[str, Stats]) -> Group:
    """Every provider as one row. Pure, like the `format_*` above it: Stats in, a table out.

    Colour carries only what the numbers already say -- red where errors are not zero, green
    where a provider is finished -- so a terminal that strips it loses nothing.
    """
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="dim", expand=False)
    # `fold` so a long label wraps rather than being cut, `min_width` so it cannot be folded
    # away to nothing when the terminal is tight -- rich will shrink a foldable column to one
    # character before it drops a fixed one.
    table.add_column("run", overflow="fold", min_width=16)
    table.add_column("", width=BAR_WIDTH, no_wrap=True)
    table.add_column("done", justify="right", no_wrap=True)
    table.add_column("ok", justify="right", no_wrap=True)
    table.add_column("err", justify="right", no_wrap=True)
    table.add_column("in flight", justify="right", no_wrap=True)
    table.add_column("cost", justify="right", no_wrap=True)
    table.add_column("avg", justify="right", no_wrap=True)
    table.add_column("dur", justify="right", no_wrap=True)
    for name, s in stats.items():
        if s.total is None:
            # In the bar's column, not the last one: `waiting` under a heading of `dur` reads
            # as a duration, and it is the absence of one.
            table.add_row(Text(name, style="dim"), Text("waiting", style="dim"),
                          "", "", "", "", "", "", "")
            continue
        done = s.finished is not None
        table.add_row(
            Text(name, style="green" if done else ""),
            ProgressBar(total=max(s.total, 1), completed=min(s.done, s.total),
                        width=BAR_WIDTH, complete_style="green" if done else "cyan",
                        finished_style="green"),
            f"{s.done}/{s.total}",
            str(s.ok),
            Text(str(s.errors), style="red bold" if s.errors else "dim"),
            format_flight(s),
            format_money(s),
            format_duration(s.mean_wall) if s.mean_wall is not None else "",
            # How long this provider has been going, which stops where it finished. No
            # `done in` prefix and no branch: the column is headed, and a run that is still
            # going has an elapsed worth reading too.
            Text(format_duration(s.elapsed), style="dim" if done else ""),
        )
    total = format_total(stats) if len(stats) > 1 else ""
    return Group(table, Text(total, style="bold")) if total else Group(table)


class Reporter:
    """One provider's handle on the display. This is all a Run ever holds.

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
            ProviderRun(...).predict(docs, timeout)   # its Runs hold reporters

    Providers are registered UP FRONT and in order, so the lines keep their places and a
    vendor that has not started yet is visibly waiting rather than absent.
    """

    def __init__(self, names, stream=None, tick: float = TICK_S):
        self.stream = stream if stream is not None else sys.stderr
        self.live = bool(getattr(self.stream, "isatty", lambda: False)())
        self.stats: dict[str, Stats] = {name: Stats() for name in names}
        self.lock = threading.RLock()
        self._width = max((len(n) for n in self.stats), default=8)
        self._tick = tick
        self._console: Console | None = None
        self._live: Live | None = None
        self._saved_handlers: list | None = None

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
            line = (format_provider(name, s, self._width)
                    if not self.live and s.done % PLAIN_EVERY == 0 else None)
        if line:
            log.info("%s", line)

    def _renderable(self) -> Group:
        """What `Live` asks for on every refresh. Under the lock: the providers are writing."""
        with self.lock:
            return render(self.stats)

    def redraw(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def __enter__(self) -> "Progress":
        if self.live:
            self._console = Console(file=self.stream, highlight=False)
            # `get_renderable` rather than a ticker of our own: rich refreshes on its own
            # thread and asks for the table each time, so there is one clock instead of two.
            self._live = Live(get_renderable=self._renderable, console=self._console,
                              refresh_per_second=max(1, int(1 / self._tick)),
                              redirect_stdout=False, redirect_stderr=False)
            self._live.start()
            # A log line must not land inside the live region. Handlers hold the stream they
            # were built with, so rich's own redirect cannot catch them -- they are wrapped.
            root = logging.getLogger()
            self._saved_handlers = root.handlers[:]
            root.handlers[:] = [_Interleaved(self, h) for h in self._saved_handlers]
        return self

    def __exit__(self, *exc) -> None:
        if self._saved_handlers is not None:
            logging.getLogger().handlers[:] = self._saved_handlers
            self._saved_handlers = None
        if self._live is not None:
            self._live.stop()
            self._live = None


class _Interleaved(logging.Handler):
    """A log handler that does not walk over the bars.

    A handler holds the stream it was built with, so it writes straight past rich and its own
    redirect never sees it. Printing through the live console instead is what puts the line
    ABOVE the table rather than through it -- and what stops each line leaving a copy of the
    table behind, which stopping and restarting the live region did.
    """

    def __init__(self, progress: Progress, inner: logging.Handler):
        super().__init__(level=inner.level)
        self.progress, self.inner = progress, inner

    def emit(self, record: logging.LogRecord) -> None:
        console = self.progress._console
        # A handler writing somewhere else -- a file -- cannot walk over a terminal, so it is
        # left alone. One writing to OUR stream goes through rich's console instead, which
        # moves the live region down and prints above it.
        if console is None or getattr(self.inner, "stream", None) is not self.progress.stream:
            self.inner.emit(record)
            return
        with self.progress.lock:
            console.print(self.inner.format(record), highlight=False, markup=False)
