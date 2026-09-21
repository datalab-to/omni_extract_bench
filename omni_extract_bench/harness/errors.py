"""How a call fails, and whether a retry can fix it.

Transience is read off the STATUS, not matched in a message: two spellings of the same
condition once drifted apart and left 91 predictions unretried while 18 identical ones were.

`AccountFailure`, `MissingCredential` and `MissingDependency` are RAISED, never returned.
None is a fact about a document -- each is identical for every document in the corpus -- so a
returned one would be written down as a settled answer that no resume re-attempts. One vendor
marked 174 documents "failed" in 60 seconds after a credit ceiling.
"""
from __future__ import annotations

TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


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


class DialectError(VendorError):
    """The schema could not be shaped into what this vendor accepts, so no call was made.

    A fact about the harness rather than the vendor -- but RETURNED, not raised, because it is
    a fact about ONE document's schema. A dialect that throws on one corpus schema would
    otherwise take the run down and lose the 619 documents either side of it, and the crash
    would name a `KeyError` rather than the schema that caused it.

    Never transient: the same schema shapes the same way every time, so a retry buys nothing.
    """

    @property
    def transient(self) -> bool:
        return False


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
