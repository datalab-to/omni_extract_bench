"""One document, one vendor, under the rules that make a comparison fair.

    predict(provider, pdf, schema) -> dict

WHAT IS UNIFORM, AND WHY EACH RULE EXISTS
-----------------------------------------
* ONE TIMEOUT for every provider (default 1800s), passed to the adapter, which enforces it on
  its own poll loop. The first run of this benchmark gave one provider 1800s and the raw-model
  legs 300s; one vendor lost 48 documents to "analysis timed out" under a cap another never
  reached. A harness parameter must never decide a vendor's coverage.
* MAXIMUM TIER for every provider. Parity is "as much as the vendor will give", not one number
  for everyone, so `PROVIDER_TIER` names the top tier per vendor, and each adapter chooses its
  own maximum -- `llm_single_shot.MODEL_MAX_OUTPUT` holds the published output ceiling per
  model, beside the code that sends it, rather than a shared floor that truncates the big ones.
* THE SAME SCHEMA, with benchmark-only annotations removed (`evaluation_config`, `default`).
  Those tell a grader how to compare a value and tell a model nothing; one vendor validates
  strictly and rejected 8 of 40 documents over them.
* CONVENTIONS ARE STATED, NOT ASSUMED (`schema_overlay`). Where the gold follows a convention
  the schema never states, it is written into the field description for every vendor alike --
  rather than loosening the comparator, which would credit a vendor that dumps a paragraph.
* TRANSIENT FAILURES RETRY, REAL ANSWERS DO NOT. A 429 or 5xx says nothing about whether a
  vendor can extract a document; a 400 is a real result and retrying it hides the evidence.
  Transience is read off the STATUS, not matched in a message -- two spellings of the same
  condition once drifted apart and left 91 predictions unretried while 18 identical ones were.
* ACCOUNT-LEVEL FAILURES STOP THE RUN. A 402 is not about the document: one vendor marked 174
  documents "failed" in 60 seconds after a credit ceiling, converting a recoverable pause into
  174 stored zeros.

AN ADAPTER IS A FUNCTION. `providers/<name>.extract(pdf, schema, *, timeout, **opts)` makes the
call, parses the answer and returns an `Extraction`; it RAISES its failures, from where they
happen. There is no subprocess, no transport tap and no envelope: those existed to observe
adapters we treated as opaque programs, and cost a tempfile dance, a `sitecustomize` injection,
a cross-process log merge, and an error channel that was the last line of stderr.
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path

from . import schema_overlay as SO
from .dialects import cost_from_response, strip_benchmark_keys
from .extraction import (AccountFailure, Budget, Cost, Extraction, MissingDependency,
                         VendorError, VendorTimeout)

DEFAULT_TIMEOUT = 1800.0
#: An OpenRouter model id is recognised by its slash: no vendor name has one, every model id
#: does. So `openai/gpt-5.6-sol` routes to the single-shot LLM adapter and `datalab` does not,
#: with no alias table to keep current.
#:
#: There were aliases once -- `gpt`, `claude`, `gemini`, `gpt-pro` -- and they were ambiguous in
#: the one place it matters: a published row labelled "gpt" does not say which model produced
#: it, and the answer changed whenever the alias was repointed.
MODEL_SEPARATOR = "/"

#: provider -> (adapter module, the options that put it at its maximum tier).
#: The module is a NAME, imported on use: importing an adapter pulls its SDK, and a machine
#: that only scores has none of them.
ADAPTERS: dict[str, tuple[str, dict]] = {
    "datalab": ("datalab", {"mode": "balanced"}),
    "datalab-accurate": ("datalab", {"mode": "accurate"}),
    "mistral": ("mistral", {}),
    "reducto": ("reducto", {}),
    "extend": ("extend", {}),
    "llamaextract": ("llamaextract", {"tier": "agentic_plus"}),
    "azure-cu": ("azure_cu", {}),
}
#: The named vendors. Any `org/model` is also accepted; see `resolve`.
PROVIDERS = sorted(ADAPTERS)


def resolve(provider: str) -> tuple[str, dict]:
    """(adapter module, its options) for a provider name or a model id."""
    if MODEL_SEPARATOR in provider:
        return "llm_single_shot", {"model": provider}
    if provider not in ADAPTERS:
        raise ValueError(f"unknown provider {provider!r}. Known vendors: "
                         f"{', '.join(PROVIDERS)}. Any OpenRouter model id also works, "
                         f"e.g. openai/gpt-5.6-sol")
    return ADAPTERS[provider]


def out_name(provider: str) -> str:
    """A provider as a single directory name: `openai/gpt-5.6-sol` would otherwise nest."""
    return provider.replace(MODEL_SEPARATOR, "__")

PROVIDER_TIER = {
    "reducto": "super_agent", "llamaextract": "agentic_plus", "extend": "default processor",
    "mistral": "mistral-ocr-latest", "azure-cu": "gpt-4.1-mini", "datalab": "balanced",
    "datalab-accurate": "accurate",
}
#: A safe concurrency per vendor. Advisory: the caller owns the pool.
WORKERS = {"reducto": 3, "llamaextract": 3, "azure-cu": 3, "datalab": 10,
           "datalab-accurate": 10}

#: The adapters read the environment for CREDENTIALS only -- which change whether a call is
#: allowed, not what it asks. Everything that steers a vendor (mode, tier, array strategy, api
#: version, base url, completion model) is a keyword argument, reaching the adapter through
#: `options` and landing in `run_manifest.overrides`. That is the whole reason: an environment
#: variable steers a run without appearing in its record, so `LLAMAEXTRACT_TIER=cost_effective`
#: used to produce a run indistinguishable from a maxed-out one. `test_no_steering_env` holds
#: the line; credentials stay in the environment and out of the record.

TRANSIENT_ATTEMPTS = 4
TRANSIENT_BACKOFF = (20, 60, 120)

#: A 402, or a vendor saying in words that the account is out. Checked on the MESSAGE because
#: vendors disagree about the status for this, which is the one place a substring is the only
#: signal there is.
ACCOUNT_MARKERS = ("exceeded the maximum number of credits", "subscription has expired",
                   "insufficient_quota")


def _is_account_failure(exc: VendorError) -> bool:
    blob = f"{exc} {exc.body or ''}".lower()
    return exc.status == 402 or any(m in blob for m in ACCOUNT_MARKERS)


def adapter(provider: str):
    """The adapter function for a provider, imported on demand.

    A missing SDK surfaces HERE, as `MissingDependency`, rather than as a per-document result.
    It is our environment, it is identical for all 620 documents, and it is not transient -- so
    a recorded one is a settled failure no resume re-attempts, and a whole provider reads as 0%
    coverage over one missing install.
    """
    module, _ = resolve(provider)
    try:
        return importlib.import_module(f".providers.{module}", __package__).extract
    except ImportError as exc:
        raise MissingDependency(
            f"the {provider} adapter could not import what it needs:\n"
            f"    {exc}\n"
            f"    pip install 'omni-extract-bench[harness]'"
        ) from None


def predict(provider: str, pdf, schema: dict, *, timeout: float = DEFAULT_TIMEOUT,
            overlay: bool = True, **options) -> dict:
    """Run one document through one provider and return the answer with its evidence.

    Retries only what a retry can fix. Raises `AccountFailure` and `MissingDependency` instead
    of returning them: neither is a fact about the document, both are identical for every one
    of them, and a returned failure is written down as a settled answer no resume re-attempts.

    """
    extract = adapter(provider)
    pdf = Path(pdf)
    if not pdf.exists():
        raise FileNotFoundError(f"no document at {pdf}")

    stripped = strip_benchmark_keys(schema)
    sent = strip_benchmark_keys(SO.apply_overlay(schema)) if overlay else stripped
    defaults = resolve(provider)[1]
    opts = {**defaults, **options}
    # Only what the CALLER changed, not the maximum-tier defaults. A run that was not stock
    # has to say so on every document: the benchmark's claim is that each vendor ran at its
    # maximum, and a figure produced with that turned down is a different measurement.
    overrides = {k: v for k, v in options.items() if defaults.get(k) != v} or None

    # ONE budget for the document, shared by every attempt and by the waits between them.
    # Each attempt used to get the full `timeout`, so four of them plus backoff could spend
    # 7,400s on a document whose budget is documented as 1,800s "end to end". This is the same
    # mistake `Budget` was written to fix inside the adapters -- azure-cu taking it twice,
    # llm_single_shot three times -- sitting one level up, where nothing had checked for it.
    budget = Budget(timeout)
    started = time.time()
    got, error, attempts = None, None, 0
    for attempt in range(TRANSIENT_ATTEMPTS):
        attempts = attempt + 1
        try:
            got = extract(pdf, sent, timeout=budget.remaining(), **opts)
            error = None
            break
        except VendorError as exc:
            if _is_account_failure(exc):
                raise AccountFailure(str(exc)[:200]) from None
            error = exc
            if not exc.transient or attempt == TRANSIENT_ATTEMPTS - 1:
                break
            # The wait comes out of the same budget: backing off past the deadline would spend
            # the document's time doing nothing and then call the vendor with none left.
            time.sleep(min(TRANSIENT_BACKOFF[min(attempt, len(TRANSIENT_BACKOFF) - 1)],
                           budget.remaining()))
            if budget.expired():
                error = VendorTimeout(
                    f"the uniform {timeout:.0f}s budget was spent over {attempts} attempt(s); "
                    f"last failure: {exc}"[:400])
                break

    elapsed = round(time.time() - started, 1)
    if got is not None and got.cost.usd is None:
        # The adapter did not name a cost, so look for one in the response it returned, using
        # the field names vendors actually use. The runner this replaced scraped EVERY vendor's
        # body this way, so an adapter with no cost code still produced a figure -- and reading
        # "the adapter has no cost code" as "the vendor reports none" is how four providers
        # came to be recorded as billed out of band on no evidence at all.
        usd, field = cost_from_response(got.raw if isinstance(got.raw, dict) else {})
        if usd is not None:
            got = got._replace(cost=Cost(usd=usd, source=field))
    if error is not None:
        got = Extraction(result={"__error__": f"{type(error).__name__}: {error}"[:600]},
                         raw=error.body, cost=Cost())
    return {
        "result": got.result,
        "raw": got.raw,
        # Why it failed, as STRUCTURE. A resume has to decide whether re-running could help,
        # and reading that back out of a message is how two spellings of one condition drifted
        # into different behaviour. `transient` is decided once, here, by the status.
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
            "tier": PROVIDER_TIER.get(provider, provider),
            "model": provider if MODEL_SEPARATOR in provider else None,
            "timed_out": isinstance(error, VendorTimeout),
            "conventions_applied": overlay and sent != stripped,
            "overrides": overrides,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
