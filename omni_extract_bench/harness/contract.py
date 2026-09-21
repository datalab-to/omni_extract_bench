"""What an adapter returns, and how it reports a failure.

A leaf module: it imports nothing from this package, so every adapter and the runner above
them can share these without a cycle.

THE CONTRACT. An adapter is a MODULE with three names -- see `Adapter` below:

    Config          what the vendor can be asked
    prepare_schema  the JSON Schema -> whatever this vendor's API takes
    extract         (pdf, schema, *, timeout, config) -> Extraction

`extract` makes the vendor call, parses the answer, and returns both.

`Config` is a frozen dataclass whose FIELDS are what the vendor can be asked -- one
declaration, read by `--options`, by `oeb providers`, and by `run_manifest.settings` for the
record. Nothing infers an option from a
signature and nothing restates a default somewhere else.

A field must hold what the vendor is actually SENT. A `None` that `extract` later resolves
into a real value is a setting the record cannot state: it writes down `None` while the
vendor was handed 128000. Resolve it in `__post_init__`, where the Config still says it.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, NamedTuple, Protocol

class Adapter(Protocol):
    """What every module in `providers/` provides. `vendor.ADAPTERS` holds one of each.

    Three names, and nothing else is looked up on an adapter:

        Config          a frozen dataclass; its fields are what the vendor can be asked
        prepare_schema  the JSON Schema -> whatever this vendor's API takes
        extract         make the call, parse the answer, return it

    `prepare_schema` is separate from `extract` so that ONE value is both what `extract`
    receives and what the record stores: an adapter that reshaped privately inside `extract`
    left the record naming a schema the vendor never saw. Identity is a real answer here --
    three vendors take a JSON Schema as written and say so with a one-line function, rather
    than by omitting one and being defaulted.

    Structural, not inherited: an adapter is a module, and a module cannot subclass. Nothing
    enforces this at runtime -- it is the contract in one place, and `tests/` checks it.
    """

    Config: type

    def prepare_schema(self, schema: dict) -> dict: ...

    def extract(self, pdf: Path, schema: dict, *, timeout: float, config) -> Extraction: ...


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

    Only the response that carried the answer, not every call made getting there: an adapter
    makes its own calls, so it can say what it got and count its own polls.
    """

    result: dict
    raw: Any = None
    cost: Cost = Cost()
    job_id: str | None = None
