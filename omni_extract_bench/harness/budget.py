"""Spending one document's share of the clock.

Every provider gets the SAME budget, and the adapter enforces it on its own poll loop. The
first run of this benchmark gave one provider 1800s and the raw-model legs 300s; one vendor
lost 48 documents to "analysis timed out" under a cap another never reached. A harness
parameter must never decide a vendor's coverage.
"""
from __future__ import annotations

import time

from .errors import TRANSIENT_STATUSES, VendorTimeout

class Budget:
    """The uniform per-document deadline. ONE of these per `extract()` call, made first.

    Every provider gets the same number of seconds for the whole document -- upload, submit and
    poll together -- because a harness parameter must never decide a vendor's coverage. The
    first run of this benchmark gave one provider 1800s and the raw-model legs 300s, and one
    vendor lost 48 documents to a cap another never reached.

    A type rather than a convention, because the rule was once spelled seven ways and two
    were wrong in the same direction -- a fresh deadline per polling phase, a full budget per
    retry -- so those vendors got two to three times what the others did.

    `monotonic`, never `time.time()`: a wall clock can step backwards under NTP.
    """

    __slots__ = ("total", "_deadline")

    def __init__(self, seconds: float):
        self.total = seconds
        self._deadline = time.monotonic() + seconds

    def remaining(self) -> float:
        """Seconds left, never negative. What the NEXT call in this document may take."""
        return max(0.0, self._deadline - time.monotonic())

    def spent(self) -> float:
        return self.total - self.remaining()

    def expired(self) -> bool:
        return time.monotonic() >= self._deadline

    def check(self, what: str) -> None:
        """Raise if the budget is gone. `what` says what was still outstanding."""
        if self.expired():
            raise VendorTimeout(f"timed out at the uniform {self.total:.0f}s limit; {what}")


class PollRetry:
    """Survive a failed POLL without abandoning the job behind it.

    A POLL FAILING IS NOT THE JOB FAILING. The work is running and already billed. Letting a
    429 out of the loop hands the document back to `predict`, which reads it as transient and
    retries from the upload -- so the running job is abandoned and the extraction is bought
    twice, silently, because the second attempt answers perfectly well. A 259-second document
    polls ~51 times, so raising concurrency makes this likelier, not rarer.

    Only a TRANSIENT status is worth asking again; a 400 or 404 says the job is not there to
    poll. A transport failure carries no status and is retried too -- it says nothing about
    the job either.
    """

    __slots__ = ("budget", "limit", "count")

    def __init__(self, budget: Budget, limit: int = 60):
        self.budget, self.limit, self.count = budget, limit, 0

    def ok(self) -> None:
        """A poll came back. Consecutive failures start again from zero."""
        self.count = 0

    def again(self, status: int | None = None) -> bool:
        """A poll failed. True if the caller should poll again rather than give up.

        `status` is the HTTP status where there was one; None means the request never got far
        enough to have one. Sleeps the backoff itself, and never past the document's deadline:
        waiting beyond it would spend the budget doing nothing. Whether the budget then ran
        out is the loop's own business -- it re-checks and raises `VendorTimeout`, which is the
        honest answer when the vendor is still working.
        """
        if status is not None and status not in TRANSIENT_STATUSES:
            return False
        self.count += 1
        if self.count > self.limit:
            return False
        time.sleep(min(30.0, 2.0 ** min(self.count, 5), self.budget.remaining()))
        return True
