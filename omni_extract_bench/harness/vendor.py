"""One document, one vendor, under the rules that make a comparison fair.

    predict(provider, pdf, schema) -> dict

WHAT IS UNIFORM, AND WHY EACH RULE EXISTS
-----------------------------------------
* ONE TIMEOUT for every provider (default 1800s), passed to the adapter, which enforces it on
  its own poll loop. The first run of this benchmark gave one provider 1800s and the raw-model
  legs 300s; one vendor lost 48 documents to "analysis timed out" under a cap another never
  reached. A harness parameter must never decide a vendor's coverage.
* MAXIMUM TIER for every provider. Parity is "as much as the vendor will give", not one number
  for everyone, so each adapter's `Config` defaults ARE its top settings and
  `run_manifest.settings` records what was sent -- `llm_single_shot.MODEL_MAX_OUTPUT` holds the
  published output ceiling per model, beside the code that sends it, not a shared floor.
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

AN ADAPTER IS A FUNCTION AND A CONFIG. `providers/<name>.extract(pdf, schema, *, timeout,
config)` makes the call, parses the answer and returns an `Extraction`; it RAISES its failures,
from where they happen. Its `Config` is a frozen dataclass whose fields are exactly what the
vendor can be asked -- the one declaration `--options`, `oeb providers` and the adapter's own
command line all read. There is no subprocess, no transport tap and no envelope: those existed to observe
adapters we treated as opaque programs, and cost a tempfile dance, a `sitecustomize` injection,
a cross-process log merge, and an error channel that was the last line of stderr.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import re
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

#: provider -> adapter module. A NAME, imported on use: importing an adapter pulls its SDK,
#: and a machine that only scores has none of them.
#:
#: The maximum-tier settings used to be pinned here as well, and they are now the adapter's
#: own `Config` defaults -- one place, beside the code that sends them, rather than a pin here
#: that had to agree with a default there.
ADAPTERS: dict[str, str] = {
    "datalab": "datalab",
    "mistral": "mistral",
    "reducto": "reducto",
    "extend": "extend",
    "llamaextract": "llamaextract",
    "azure-cu": "azure_cu",
}
#: The named vendors. Any `org/model` is also accepted; see `resolve`.
PROVIDERS = sorted(ADAPTERS)


def resolve(provider: str) -> str:
    """The adapter module for a provider name or a model id."""
    if MODEL_SEPARATOR in provider:
        return "llm_single_shot"
    if provider not in ADAPTERS:
        raise ValueError(f"unknown provider {provider!r}. Known vendors: "
                         f"{', '.join(PROVIDERS)}. Any OpenRouter model id also works, "
                         f"e.g. openai/gpt-5.6-sol")
    return ADAPTERS[provider]


def out_name(provider: str, options: dict | None = None) -> str:
    """A run's directory name: the provider, and a digest of everything it was asked.

    THE NAME IS A FUNCTION OF THE SETTINGS, so a directory holds one configuration and two
    configurations never share one -- `runs/datalab-f46415c9` and `runs/datalab-01a72762` are
    the balanced and the accurate run, and neither can be handed the other's records.

    It named the DIFFERENCE from the adapter's defaults once, which fails twice. A stock run
    came out a bare `datalab`, silent about the tier it measured. And the day a vendor's
    maximum moves and a `Config` default follows it, the new stock run lands on that same bare
    `datalab`: `needs_run` finds records, skips every document, and two tiers are averaged into
    one published number.

    The digest is not readable, and nothing here tries to make it so. WHAT A RUN WAS ASKED IS
    WRITTEN INTO THE RUN, by `benchmark.run`, as `settings.json` -- at the start, so it is
    there for a run that is interrupted, and beside the answers rather than encoded in a path
    with a width budget. A name that carries a selected field instead is a name that can be
    read two ways.

    `openai/gpt-5.6-sol` would otherwise nest, and a `:batch` suffix is not a filename on every
    filesystem, so the provider half is spelled out and sanitised too. Two ids that sanitise
    alike still differ: the model is one of the settings the digest covers.
    """
    # `sort_keys` so the spelling does not depend on the order the options were written in, and
    # `sha256` rather than `hash()`, which is salted per process and would name the same run
    # differently tomorrow.
    spelled = json.dumps(settings_for(provider, options), sort_keys=True, default=str)
    digest = hashlib.sha256(spelled.encode()).hexdigest()[:8]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", provider.replace(MODEL_SEPARATOR, "__"))
    return f"{name}-{digest}"


#: A safe concurrency per vendor. Advisory: the caller owns the pool.
WORKERS = {"reducto": 3, "llamaextract": 3, "azure-cu": 3, "datalab": 10}

#: The adapters read the environment for CREDENTIALS only. Everything that steers a vendor is
#: a `Config` field, reaching the adapter through `--options` and landing in
#: `run_manifest.settings` -- an environment variable steers a run without appearing in its
#: record, so `LLAMAEXTRACT_TIER=cost_effective` once produced a run indistinguishable from a
#: maxed-out one. `test_no_steering_env` holds the line.

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
    try:
        return _module(provider).extract
    except ImportError as exc:
        raise MissingDependency(
            f"the {provider} adapter could not import what it needs:\n"
            f"    {exc}\n"
            f"    pip install 'omni-extract-bench[harness]'"
        ) from None


def _module(provider: str):
    return importlib.import_module(f".providers.{resolve(provider)}", __package__)


def config_for(provider: str, options: dict | None = None):
    """This provider's adapter `Config`, with the caller's options applied.

    THE CONFIG IS THE DECLARATION. Its fields are the options: what `--options` may set, what
    `oeb providers` lists, and what `run_cli` builds this adapter's flags from. Nothing infers
    them from a signature and nothing restates a default elsewhere.

    An option the adapter does not have is refused here, by name. Passed through to a
    `**options` catch-all, a typo was simply ignored and the run reported as stock.
    """
    import dataclasses

    config_type = _module(provider).Config
    options = options or {}
    # A MODEL ID IS THE MODEL. It comes from the provider name, so the directory, the summary
    # key and the record all name the model that ran. It used to be merely a default that
    # `options` then overrode, which is the one way that could stop being true: asking
    # `openai/gpt-5.6-sol` for `model=anthropic/claude-opus-5` ran Claude, and stored it, under
    # a directory spelled `openai__gpt-5.6-sol`. Refused rather than quietly overruled -- a
    # caller who wrote it meant something, and running a different model than they typed is
    # not it.
    if MODEL_SEPARATOR in provider:
        if "model" in options:
            raise ValueError(
                f"{provider}: `model` is the provider name, not an option -- otherwise the "
                f"run would be stored and published under a model it did not use. Run the "
                f"one you want: --providers {options['model']}")
        options = {**options, "model": provider}
    try:
        return config_type(**options)
    except TypeError as exc:
        known = ", ".join(f.name for f in dataclasses.fields(config_type)) or "(none)"
        raise ValueError(f"{provider}: {exc}. Its options are: {known}") from None


def settings_for(provider: str, options: dict | None = None) -> dict:
    """What the adapter will be sent, as a plain dict: the `Config` it is handed.

    The record states it, `out_name` digests it to name a run, and `oeb providers` prints it.
    There are no credentials in it -- every adapter reads its key from the environment -- so
    nothing secret reaches a record or a directory name.
    """
    import dataclasses

    return dataclasses.asdict(config_for(provider, options))


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
    # Resolved ONCE, here: the adapter is handed this object and the record states its fields,
    # so what was sent and what was written down cannot be two different resolutions. The whole
    # of it is recorded, not just what the caller changed -- the benchmark's claim is that each
    # vendor ran at its maximum, and a document has to say what that was on the day.
    config = config_for(provider, options)
    budget = Budget(timeout)
    started = time.time()
    got, error, attempts = None, None, 0
    for attempt in range(TRANSIENT_ATTEMPTS):
        attempts = attempt + 1
        try:
            got = extract(pdf, sent, timeout=budget.remaining(), config=config)
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
            "conventions_applied": overlay and sent != stripped,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
