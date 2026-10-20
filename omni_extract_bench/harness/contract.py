"""What an extraction adapter is: `Question`, what every system is sent, and `Extractor`, the base class.

An extraction adapter is an omni-bench-core adapter whose input is a `Question`, a PDF and the schema to extract
against. `Extractor`'s docstring says how to write one.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, NamedTuple

from omni_bench_core import adapter, errors

from . import responses


class Question(NamedTuple):
    """What every system is sent for one document: the PDF, and the schema after `Extract.input` has
    applied the benchmark's conventions and removed its own annotations."""

    pdf: Path
    schema: dict


class DialectError(errors.SystemFailed):
    """The schema can't be written in what this vendor accepts, so no call was made. `prepare_schema` raises it,
    on purpose, for a schema the vendor's dialect can't express.

    It is the document's answer, not a stop, because it is a fact about one document's schema. It is never
    transient: the same schema reshapes the same way every time."""

    @property
    def transient(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class Extractor(adapter.Adapter[Question]):
    """An extraction vendor. Write `extract`, and `prepare_schema` and `read` when the vendor needs them.

    `call` does three things, in order:

    1. reshapes the schema with `prepare_schema`. A `DialectError` raised here is recorded as the document's
       answer, with the schema it couldn't reshape. Anything else it raises is a bug in the reshaping, and
       stops the run before any call;
    2. calls `extract`. A failure it raises carries the reshaped schema, so the record says what the vendor
       was asked even when it refused. A failure caused by a refused connection stops the run instead
       (`refused`), since nothing listens at the address for any document; and
    3. fills in the cost from the vendor's response when `extract` didn't say (`responses.cost_from_response`).

    Before all three it refuses a missing key (`keys`, `missing_keys`). A run checks keys before any call
    (`Benchmark.plan`); this check is for a call made without a plan, such as `oeb predict`.

    The schema as sent is kept in the record (`Call.sent`), so the record says what the vendor was asked.
    Whether a failure is the account's is omni-bench-core's default (`Adapter.account_failure`), which knows the
    phrases these vendors use."""

    def prepare_schema(self, schema: dict) -> dict:
        """The schema in the form this vendor's API takes. By default the schema as it is.

        It may change how a constraint is written, never what is asked for: resolving a $ref or
        collapsing a nullable union is fine, removing a field is not."""
        return schema

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """Sends the document and the reshaped schema, and returns the vendor's answer as received.
        Raise a failure (`errors.SystemFailed` or one of its kinds) rather than returning an empty answer."""
        raise NotImplementedError(f"{type(self).__name__} has no extract(pdf, schema, *, timeout) -> adapter.Call")

    def call(self, q: Question, *, timeout: float) -> adapter.Call:
        if missing := self.missing_keys():
            raise errors.MissingCredential(f"{self.label} needs {', '.join(missing)}: set "
                                           + " and ".join(f"export {k}=..." for k in missing) + ", then run again")
        try:
            sent = self.prepare_schema(q.schema)
        except DialectError as exc:
            if exc.sent is None: exc.sent = q.schema
            raise
        try:
            got = self.extract(q.pdf, sent, timeout=timeout)
        except errors.SystemFailed as exc:
            # Note: an adapter reports a network fault as the document's failure (`SystemUnreachable`), but a
            # refused connection means nothing listens at the address, which is the same for every document.
            if errors.refused(exc): raise refused(self, exc) from None
            if exc.sent is None: exc.sent = sent
            raise
        if got.cost.usd is None and got.cost.credits is None:
            usd, field = responses.cost_from_response(got.raw)
            if usd is not None: got = got._replace(cost=adapter.Cost(usd=usd, source=field))
        return got._replace(sent=sent)

    def read(self, raw: Any, q: Question) -> dict:
        """The extraction in the answer. By default `raw["extraction"]`, which must be a non-empty JSON
        object; an adapter whose answer holds it elsewhere overrides this.

        Raise `errors.SystemFailed` (`extraction` does) for an answer that holds no usable extraction, and read
        the vendor's fields inside `adapter.reading`. Anything else `read` raises is a bug: the run stops, the
        answer is kept unread, and running the same command again after the fix reads it with no call."""
        return extraction(raw.get("extraction") if isinstance(raw, dict) else None)


def refused(a: adapter.Adapter, exc: BaseException) -> errors.Stop:
    """The `Stop` for a refused connection to the system `a` calls, worded as omni-bench-core words it in a run."""
    return errors.Stop(f"{a.label} couldn't connect: the connection was refused ({exc}). If it calls a server you "
                       "run, start it (or pass its base_url), then run the same command again")


def unreachable(what: str, exc: BaseException) -> errors.SystemFailed:
    """The failure for a request to the system that got no answer (`exc`, a transport error): `SystemTimeout`
    when the client timed out, `SystemUnreachable` otherwise. Raise it `from exc`, so that a refused connection
    stays in the chain, where `Extractor.call` and the core find it and stop the run."""
    import httpx  # deferred: the base install grades without the harness, which brings httpx

    reason = getattr(exc, "reason", None)   # urllib's URLError carries the socket's error as its reason
    timed_out = isinstance(exc, (TimeoutError, httpx.TimeoutException)) or isinstance(reason, TimeoutError)
    return (errors.SystemTimeout if timed_out else errors.SystemUnreachable)(f"{what}: {type(exc).__name__}: {exc}")


def extraction(x: Any) -> dict:
    """`x` when it is a non-empty JSON object, the extraction; `errors.SystemFailed`, saying what it is,
    otherwise."""
    if isinstance(x, dict) and x: return x
    raise errors.SystemFailed(f"the answer holds no extraction, a non-empty JSON object; it holds {x}")
