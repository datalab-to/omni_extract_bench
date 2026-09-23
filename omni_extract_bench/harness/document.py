"""One document, one vendor, under the rules that make a comparison fair.

    predict(provider, pdf, schema) -> dict

WHAT IS UNIFORM, AND WHY EACH RULE EXISTS

* ONE TIMEOUT for every provider, passed to the adapter, which enforces it on its own poll
  loop -- see `budget.py` for what that cost when it was not uniform.
* MAXIMUM TIER for every provider. Parity is "as much as the vendor will give", not one number
  for everyone, so each adapter's `Config` defaults ARE its top settings.
* THE SAME QUESTION. `schema.py` strips the benchmark's own annotations and states the gold
  conventions; the adapter's `prepare_schema` then re-encodes that for its vendor. What comes
  out is both what the vendor receives and what the record stores.
* TRANSIENT FAILURES RETRY, REAL ANSWERS DO NOT. A 429 or 5xx says nothing about whether a
  vendor can extract a document; a 400 is a real result and retrying it hides the evidence.
* ACCOUNT-LEVEL FAILURES STOP THE RUN. A 402 is not about the document: one vendor marked 174
  documents "failed" in 60 seconds after a credit ceiling, turning a recoverable pause into
  174 stored zeros.

Orchestration is absent on purpose. Which documents, in what order, and how many at once are
decisions about a corpus, not about a document. `omni_extract_bench/benchmark.py` is one such
loop; `registry.WORKERS` is advice for building one.
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path

from . import schema as SCHEMA
from .budget import Budget
from .contract import Cost, Extraction
from .errors import AccountFailure, DialectError, VendorError, VendorTimeout
from .registry import DEFAULT_TIMEOUT, MODEL_SEPARATOR, adapter, config_for
from .responses import cost_from_response

TRANSIENT_ATTEMPTS = 4
TRANSIENT_BACKOFF = (20, 60, 120)

ACCOUNT_MARKERS = ("exceeded the maximum number of credits", "subscription has expired",
                   "insufficient_quota")


def _is_account_failure(exc: VendorError) -> bool:
    blob = f"{exc} {exc.body or ''}".lower()
    return exc.status == 402 or any(m in blob for m in ACCOUNT_MARKERS)


def predict(provider: str, pdf, schema: dict, *, timeout: float = DEFAULT_TIMEOUT,
            overlay: bool = True, **options) -> dict:
    """Run one document through one provider and return the answer with its evidence.

    Retries only what a retry can fix. Raises `AccountFailure` and `MissingDependency` instead
    of returning them: neither is a fact about the document, both are identical for every one
    of them, and a returned failure is written down as a settled answer no resume re-attempts.
    """
    api = adapter(provider)
    pdf = Path(pdf)
    if not pdf.exists():
        raise FileNotFoundError(f"no document at {pdf}")
    if not isinstance(schema, dict):
        raise TypeError(f"schema must be a JSON object, got {type(schema).__name__}.")

    config = config_for(provider, options)
    budget = Budget(timeout)
    started = time.time()
    got, error, attempts = None, None, 0

    # `sent` is both what `extract` receives and what the record stores as `schema_sent`.
    overlaid = SCHEMA.apply_overlay(schema) if overlay else schema
    asked = SCHEMA.strip_benchmark_keys(overlaid)
    # Outside the guard: a missing `prepare_schema` is a harness bug, not a DialectError per
    # document.
    shape = api.prepare_schema
    try:
        sent = shape(asked)
    except Exception as exc:  # noqa: BLE001 -- a dialect may raise anything; see DialectError
        sent = asked
        error = DialectError(f"{provider} could not shape this schema: "
                             f"{type(exc).__name__}: {exc}"[:400])
    # The DialectError is this document's answer; `cost.attempts` stays 0, as nothing was sent.
    if error is None:
        for attempt in range(TRANSIENT_ATTEMPTS):
            attempts = attempt + 1
            try:
                got = api.extract(pdf, sent, timeout=budget.remaining(), config=config)
                error = None
                break
            except VendorError as exc:
                if _is_account_failure(exc):
                    raise AccountFailure(str(exc)[:200]) from None
                error = exc
                if not exc.transient or attempt == TRANSIENT_ATTEMPTS - 1:
                    break
                time.sleep(min(TRANSIENT_BACKOFF[min(attempt, len(TRANSIENT_BACKOFF) - 1)],
                               budget.remaining()))
                if budget.expired():
                    error = VendorTimeout(
                        f"the uniform {timeout:.0f}s budget was spent over {attempts} "
                        f"attempt(s); last failure: {exc}"[:400])
                    break

    elapsed = round(time.time() - started, 1)
    if got is not None and got.cost.usd is None:
        usd, field = cost_from_response(got.raw if isinstance(got.raw, dict) else {})
        if usd is not None:
            got = got._replace(cost=Cost(usd=usd, source=field))
    if error is not None:
        got = Extraction(result={"__error__": f"{type(error).__name__}: {error}"[:600]},
                         raw=error.body, cost=Cost())
    return {
        "result": got.result,
        "raw": got.raw,
        "error": None if error is None else {
            "type": type(error).__name__,
            "status": error.status,
            "transient": error.transient,
            "message": str(error)[:600],
        },
        "provider": provider,
        "cost": {**got.cost._asdict(), "wall_s": elapsed, "attempts": attempts,
                 "billed_out_of_band": got.cost.usd is None},
        "job_id": got.job_id,
        "schema_sent": sent,
        "run_manifest": {
            "timeout_s": timeout,
            "settings": dataclasses.asdict(config),
            "model": provider if MODEL_SEPARATOR in provider else None,
            "timed_out": isinstance(error, VendorTimeout),
            "conventions_applied": overlay and overlaid != schema,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
