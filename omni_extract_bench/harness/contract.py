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


class DialectError(errors.VendorError):
    """The schema could not be reshaped into what this vendor accepts, so no call was made.

    It is the document's answer, not a stop, because it is a fact about one document's schema. It is never
    transient: the same schema reshapes the same way every time."""

    @property
    def transient(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class Extractor(adapter.Adapter[Question]):
    """An extraction vendor. Write `extract`, and `prepare_schema` and `read` when the vendor needs them.

    `call` does three things, in order:

    1. reshapes the schema with `prepare_schema`. A failure here is a `DialectError`, recorded as the
       document's answer with the schema it couldn't reshape;
    2. calls `extract`. A failure it raises carries the reshaped schema, so the record says what the vendor
       was asked even when it refused; and
    3. fills in the cost from the vendor's response when `extract` didn't say (`responses.cost_from_response`).

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
        Raise a failure (`errors.VendorError` or one of its kinds) rather than returning an empty answer."""
        raise NotImplementedError(f"{type(self).__name__} has no extract(pdf, schema, *, timeout) -> adapter.Call")

    def call(self, q: Question, *, timeout: float) -> adapter.Call:
        try:
            sent = self.prepare_schema(q.schema)
        except Exception as exc:  # noqa: BLE001 -- a reshaping may raise anything; it is this document's answer
            raise DialectError(f"{self.name} could not shape this schema: {type(exc).__name__}: {exc}"[:400],
                               sent=q.schema) from None
        try:
            got = self.extract(q.pdf, sent, timeout=timeout)
        except errors.VendorError as exc:
            if exc.sent is None: exc.sent = sent
            raise
        if got.cost.usd is None and got.cost.credits is None:
            usd, field = responses.cost_from_response(got.raw)
            if usd is not None: got = got._replace(cost=adapter.Cost(usd=usd, source=field))
        return got._replace(sent=sent)

    def read(self, raw: Any, q: Question) -> dict:
        """The extraction in the answer. By default `raw["extraction"]`, which must be a non-empty JSON
        object; an adapter whose answer holds it elsewhere overrides this."""
        return extraction(raw.get("extraction") if isinstance(raw, dict) else None)


def extraction(x: Any) -> dict:
    """`x` when it is a non-empty JSON object, the extraction; ValueError, saying what it is, otherwise."""
    if isinstance(x, dict) and x: return x
    raise ValueError(f"the answer holds no extraction, a non-empty JSON object; it holds {str(x)[:120]}")
