"""What an extraction adapter is:

- `Question`: what every system is sent, a PDF and the schema to extract against;
- `Extractor`: the base class an extraction adapter subclasses; and
- `unreachable` and `extraction`: helpers for an adapter's failures and its `read`."""
from __future__ import annotations

import abc
import dataclasses
from pathlib import Path
from typing import Any, NamedTuple

from omni_bench_core import adapter, errors

from . import responses


class Question(NamedTuple):
    """What every system is sent for one document: the PDF, and the schema with the benchmark's conventions
    applied and its own annotations removed (`Extract.input`)."""

    pdf: Path
    schema: dict


class DialectError(errors.SystemFailed):
    """The schema can't be written in what this vendor accepts, so no call was made: `prepare_schema`
    raises it. It is the document's answer, counted 0, and never transient, since the same schema
    reshapes the same way."""

    @property
    def transient(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class Extractor(adapter.Adapter[Question]):
    """An extraction vendor. Write `extract`; write `prepare_schema` and `read` when the vendor needs them;
    declare the env vars it needs in `keys`.

    `call` does, in order:

    1. reshapes the schema with `prepare_schema`. A `DialectError` is the document's answer; anything else
       is a bug and stops the run;
    2. calls `extract`. A failure it raises carries the reshaped schema, so the record says what was
       asked; and
    3. fills in the cost from the vendor's response when `extract` didn't (`responses.cost_from_response`).

    The schema as sent is kept in the record (`Call.sent`). A refused connection, an account failure and
    retries are omni-bench-core's to handle."""

    def prepare_schema(self, schema: dict) -> dict:
        """The schema in the form this vendor's API takes; by default the schema as it is. It may change
        how a constraint is written (resolving a `$ref`, collapsing a nullable union), never what is
        asked for."""
        return schema

    @abc.abstractmethod
    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """Sends the document and the reshaped schema, and returns the vendor's answer as received. Raise
        `errors.SystemFailed` (or one of its kinds) for a failure, never an empty answer."""

    def call(self, q: Question, *, timeout: float) -> adapter.Call:
        try:
            sent = self.prepare_schema(q.schema)
        except DialectError as exc:
            if exc.sent is None: exc.sent = q.schema
            raise
        try:
            got = self.extract(q.pdf, sent, timeout=timeout)
        except errors.SystemFailed as exc:
            if exc.sent is None: exc.sent = sent
            raise
        if got.cost.usd is None and got.cost.credits is None:
            usd, field = responses.cost_from_response(got.raw)
            if usd is not None: got = got._replace(cost=adapter.Cost(usd=usd, source=field))
        return got._replace(sent=sent)

    def read(self, raw: Any, q: Question) -> dict:
        """The extraction in the answer; by default `raw["extraction"]`, which must be a non-empty JSON
        object. Raise `errors.SystemFailed` (`extraction` does) for an answer with no usable extraction,
        and read the vendor's fields inside `adapter.reading`; anything else raised is a bug."""
        return extraction(raw.get("extraction") if isinstance(raw, dict) else None)


def unreachable(what: str, exc: BaseException) -> errors.SystemFailed:
    """The failure for a request that got no answer (`exc`, a transport error): `SystemTimeout` when the
    client timed out, `SystemUnreachable` otherwise. Raise it `from exc`, so a refused connection stays
    in the chain, where omni-bench-core finds it and stops the run."""
    import httpx  # deferred: the base install grades without the harness, which brings httpx

    reason = getattr(exc, "reason", None)   # urllib's URLError carries the socket's error as its reason
    timed_out = isinstance(exc, (TimeoutError, httpx.TimeoutException)) or isinstance(reason, TimeoutError)
    return (errors.SystemTimeout if timed_out else errors.SystemUnreachable)(f"{what}: {type(exc).__name__}: {exc}")


def extraction(x: Any) -> dict:
    """`x` when it is a non-empty JSON object; `errors.SystemFailed`, saying what it is, otherwise."""
    if isinstance(x, dict) and x: return x
    raise errors.SystemFailed(f"the answer holds no extraction, a non-empty JSON object; it holds {x}")
