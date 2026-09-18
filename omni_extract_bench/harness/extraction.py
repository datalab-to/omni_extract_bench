"""What an adapter returns, and how it reports a failure.

A leaf module: it imports nothing from this package, so every adapter and the runner above
them can share these without a cycle.

THE CONTRACT. An adapter is a function and a `Config`:

    extract(pdf, schema, *, timeout, config: Config) -> Extraction

It makes the vendor call, parses the answer, and returns both.

`Config` is a frozen dataclass whose FIELDS are what the vendor can be asked -- one
declaration, read by `--options`, by `oeb providers`, by `run_cli` for the adapter's own
flags, and by `run_manifest.settings` for the record. Nothing infers an option from a
signature and nothing restates a default somewhere else.

A field must hold what the vendor is actually SENT. A `None` that `extract` later resolves
into a real value is a setting the record cannot state: it writes down `None` while the
vendor was handed 128000. Resolve it in `__post_init__`, where the Config still says it.
"""
from __future__ import annotations

import time
from typing import Any, NamedTuple

#: Statuses that say nothing about whether the vendor can extract this document. A 400 is an
#: answer -- usually our schema, sometimes a real vendor limit -- and retrying it only hides
#: the evidence, so it is deliberately absent.
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


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


class Cost(NamedTuple):
    """What the vendor said this document cost. `usd` is None where it does not say.

    `source` names the field it was read from, so a figure can be checked against the vendor's
    own response rather than trusted. Credits are never converted to dollars: the rate is
    contract-specific, so a credits figure is reported as itself or not at all. A guessed cost
    sitting in the same column as a measured one is how a cost comparison becomes fiction.
    """

    usd: float | None = None
    source: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    #: What the vendor billed in ITS OWN unit, where that is not dollars. Reducto and Extend
    #: both report credits and no dollar figure at all. Recorded as itself and never converted:
    #: the rate is contract-specific, so a dollar column derived from it would be fiction --
    #: but reporting nothing loses a number the vendor actually stated.
    credits: float | None = None

    @classmethod
    def reported(cls, value, source: str, *, cents: bool = False, **extra) -> "Cost":
        """A cost the vendor stated, or an empty `Cost` if it did not say.

        One definition of "is this a figure?", because three adapters had their own and they
        had already drifted: `isinstance(True, int)` is True in Python, so a JSON `true` in a
        cost field reads as a number, and only one of the copies excluded it.

        `cents=True` converts, once, here. Reporting cents as dollars overstates by 100x, and
        a cost table never recovers from that.
        """
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return cls(**extra)
        return cls(usd=round(float(value) / (100.0 if cents else 1.0), 6), source=source, **extra)


class Extraction(NamedTuple):
    """One document's answer, and the evidence for it.

    `raw` is the vendor's response as received, before our parsing. It is kept because the
    expensive mistake is a PARSER bug: with only the parsed value stored, fixing the parser
    means paying for every call again -- that cost 166 of them once. With `raw`, it means
    re-reading a file.

    Only the response that carried the answer, not every call made getting there. A tap that
    observed all of them existed because adapters were opaque subprocesses; an adapter that
    makes its own calls can simply say what it got, and count its own polls.
    """

    result: dict
    raw: Any = None                # the vendor's response, as received
    cost: Cost = Cost()
    job_id: str | None = None      # the vendor's handle, where it has one


def as_object(value) -> dict | None:
    """Normalise an adapter's parsed answer to the schema-shaped object, or None.

    One vendor returns a LIST -- one entry per extraction pass. A single-entry list is just the
    extraction; multi-entry lists are merged shallowly, first writer wins, so array fields from
    separate passes survive instead of the last pass overwriting them.

    This existed in the old runner and was lost in the rewrite, which turned every completed
    Reducto job into "completed with no extraction": the work was done and paid for, and the
    adapter threw the answer away for being a list.
    """
    if isinstance(value, dict):
        return value or None
    if isinstance(value, list):
        objects = [x for x in value if isinstance(x, dict)]
        if len(objects) == 1:
            return objects[0] or None
        if len(objects) > 1:
            merged: dict = {}
            for obj in objects:
                for k, v in obj.items():
                    if k not in merged or merged[k] in (None, [], {}):
                        merged[k] = v
            return merged or None
    return None


class VendorError(RuntimeError):
    """The call was made and did not produce an extraction.

    `status` is the HTTP status where there was one, and it is what decides whether a retry is
    worth anything -- rather than matching substrings in a message, which is what the harness
    did before and how two spellings of the same condition drifted into different behaviour.
    """

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body

    @property
    def transient(self) -> bool:
        return self.status in TRANSIENT_STATUSES


class VendorTimeout(VendorError):
    """The uniform budget ran out while the vendor was still working.

    Distinct from a `VendorError` because it is a different fact about the vendor, and the two
    were collapsed once: "timed out at the shared limit after 412 polls" and "returned
    something unparseable" both read as "no output". Never transient -- a retry spends another
    full budget on a document already known to exceed it.
    """

    @property
    def transient(self) -> bool:
        return False


class AccountFailure(RuntimeError):
    """The account cannot pay. NOT a fact about the document, so it stops the run.

    One vendor marked 174 documents "failed" in 60 seconds after hitting a credit ceiling,
    turning a recoverable pause into 174 stored zeros.
    """


class MissingCredential(RuntimeError):
    """A vendor's API key is not in the environment.

    Ours to fix and identical for every document, so it is raised past the runner rather than
    recorded -- exactly like a missing SDK. Recorded, it would be a settled failure that no
    resume re-attempts, and the vendor would read as 0% coverage over an unset variable.
    """


class MissingDependency(ImportError):
    """An adapter could not import its SDK. Ours to fix, and identical for every document."""
