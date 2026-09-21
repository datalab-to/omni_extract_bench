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
command line all read.
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
MODEL_SEPARATOR = "/"

ADAPTERS: dict[str, str] = {
    "datalab": "datalab",
    "mistral": "mistral",
    "reducto": "reducto",
    "extend": "extend",
    "llamaextract": "llamaextract",
    "azure-cu": "azure_cu",
}
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

    THE NAME IS A FUNCTION OF THE SETTINGS, ALL OF THEM, so a directory holds one
    configuration and two configurations never share one -- `runs/datalab-f46415c9` and
    `runs/datalab-01a72762` are the balanced and the accurate run, and `needs_run` cannot hand
    either the other's records. Naming what a run merely CHANGED about the defaults would tie
    the name to the defaults, and they move: the day a vendor's maximum tier moves with them,
    two tiers land in one directory and average into one published number.

    The digest is not readable, and nothing here tries to make it so. WHAT A RUN WAS ASKED IS
    WRITTEN INTO THE RUN, by `benchmark.run`, as `settings.json` -- beside the answers, where
    there is no width budget to select fields against.

    `openai/gpt-5.6-sol` would otherwise nest, and a `:batch` suffix is not a filename on every
    filesystem, so the provider half is spelled out and sanitised too. Two ids that sanitise
    alike still differ: the model is one of the settings the digest covers.
    """
    spelled = json.dumps(settings_for(provider, options), sort_keys=True, default=str)
    digest = hashlib.sha256(spelled.encode()).hexdigest()[:8]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", provider.replace(MODEL_SEPARATOR, "__"))
    return f"{name}-{digest}"


WORKERS = {
    "datalab": 10,
    "reducto": 3,
    "llamaextract": 3,
    "azure_cu": 3,
    "llm_single_shot": 5,
    "mistral": 5,
    "extend": 5,
}


TRANSIENT_ATTEMPTS = 4
TRANSIENT_BACKOFF = (20, 60, 120)

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

    An option the adapter does not have is REFUSED, by name, rather than ignored -- a typo
    that goes through changes nothing and the run reports as stock.
    """
    import dataclasses

    config_type = _module(provider).Config
    options = options or {}
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
            time.sleep(min(TRANSIENT_BACKOFF[min(attempt, len(TRANSIENT_BACKOFF) - 1)],
                           budget.remaining()))
            if budget.expired():
                error = VendorTimeout(
                    f"the uniform {timeout:.0f}s budget was spent over {attempts} attempt(s); "
                    f"last failure: {exc}"[:400])
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
            "conventions_applied": overlay and sent != stripped,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
