"""Everything that is not the scorer: producing predictions.

The package above this one is the metric. `metric.py` reaches exactly `matching`, `normalize`
and `values` and nothing else -- `tests/test_score_standalone.py` fails if that ever widens --
so a score cannot come to depend on a transport, a vendor dialect, or an envelope convention.

    vendor.py        `predict()` -- one document, one vendor, under the parity rules
    extraction.py    what an adapter returns, and how it reports a failure
    dialects.py      reshapes a JSON Schema into what a given vendor will accept
    schema_overlay.py writes a gold convention into the field descriptions every vendor sees
    providers/       one module per vendor, each an `extract()` function

    pip install 'omni-extract-bench[harness]'

AN ADAPTER IS A FUNCTION, not a program:

    extract(pdf, schema, *, timeout, **options) -> Extraction

It makes the call, parses the answer, returns both, and RAISES its failures from where they
happen. Adapters used to be subprocesses observed by a transport tap that monkey-patched httpx,
requests and urllib -- which cost a tempfile dance, a `sitecustomize` injection, a cross-process
log merge, and an error channel that was the last line of the child's stderr. All of it existed
to reconstruct facts the adapter already had: "timed out while the vendor was still working"
was recovered by reading an HTTP log, when the code that ran the poll loop knew it all along.

THE SURFACE
-----------
    predict(provider, pdf, schema)   the whole per-document procedure; returns the answer
                                     and the evidence for it in one dict
    Extraction, Cost                 what an adapter hands back
    VendorError, VendorTimeout       a failure of the call; `.transient` decides a retry
    AccountFailure                   raised, never returned: not a fact about a document
    MissingCredential                likewise: an unset API key
    MissingDependency                likewise: an adapter that could not import its SDK
    PROVIDERS, PROVIDER_TIER, WORKERS, DEFAULT_TIMEOUT      advisory, for building a loop

Orchestration is absent on purpose. Which documents, in what order, and how many at once are
decisions about a corpus, not about a document, and a library that made them would be deciding
the shape of every benchmark built on it. `omni_extract_bench/benchmark.py` is one such loop.

WHAT A RUN LEAVES BEHIND
------------------------
Every prediction is accompanied by the schema actually sent, the vendor's own response, the
cost it reported and a `run_manifest`. That is what makes a vendor's score checkable after the
fact: what it was asked, what it answered, and what the call cost, rather than the parsed
result alone -- a parser bug is then fixed by re-reading a file instead of re-paying for 166
calls, which is what it cost once.
"""

from .extraction import (AccountFailure, Cost, Extraction, MissingCredential,
                         MissingDependency, VendorError, VendorTimeout)

_LAZY = ("predict", "adapter", "PROVIDERS", "PROVIDER_TIER", "WORKERS", "DEFAULT_TIMEOUT")


def __getattr__(name):
    """PEP 562: the engine's names resolve on first use, so importing this package for
    an exception type does not pull in an adapter's SDK."""
    if name in _LAZY:
        from . import vendor
        return getattr(vendor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "predict", "adapter",
    "Extraction", "Cost",
    "VendorError", "VendorTimeout",
    "AccountFailure", "MissingCredential", "MissingDependency",
    "PROVIDERS", "PROVIDER_TIER", "WORKERS", "DEFAULT_TIMEOUT",
]
