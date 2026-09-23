"""Everything that is not the scorer: producing predictions.

The package above this one is the metric. `metric.py` reaches exactly `matching`, `prepare`,
`recognise` and `values` -- `tests/test_score_standalone.py` fails if that ever widens --
so a score cannot come to depend on a transport, a vendor dialect, or an envelope convention.

    contract.py      what an adapter IS: `Adapter`, and the `Extraction`/`Cost` it returns
    errors.py        how a call fails, and whether a retry can fix it
    budget.py        spending one document's share of the clock
    schema.py        the schema vendors are sent: the universal layer, and the shared
                     pieces adapters re-encode it with
    responses.py     making sense of what came back: fenced JSON, list replies, cost units
    registry.py      which adapter, at what settings, filed under what name
    document.py      run one document, return the answer and the evidence for it
    providers/       one module per vendor, each satisfying `Adapter`

    pip install 'omni-extract-bench[harness]'

AN ADAPTER IS A MODULE, not a program -- `contract.Adapter` states it:

    Config          what the vendor can be asked
    prepare_schema  the JSON Schema -> whatever this vendor's API takes
    extract         (pdf, schema, *, timeout, config) -> Extraction

`extract` makes the call, parses the answer, returns both, and RAISES its failures from where
they happen. Not a subprocess watched by a transport tap, which is the shape this replaced:
every fact that cost -- a tempfile dance, a `sitecustomize` injection, a cross-process log
merge, an error channel that was the last line of the child's stderr -- the adapter already
had. "Timed out while the vendor was still working" was being recovered by reading an HTTP
log, from the code that ran the poll loop.

THE SURFACE
-----------
    predict(provider, pdf, schema)   the whole per-document procedure; returns the answer
                                     and the evidence for it in one dict
    Extraction, Cost                 what an adapter hands back
    VendorError, VendorTimeout       a failure of the call; `.transient` decides a retry
    DialectError                     the schema would not shape for this vendor; no call made
    AccountFailure                   raised, never returned: not a fact about a document
    MissingCredential                likewise: an unset API key
    MissingDependency                likewise: an adapter that could not import its SDK
    PROVIDERS, DEFAULT_TIMEOUT       advisory, for building a loop

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

from .contract import Adapter, Cost, Extraction
from .errors import (AccountFailure, DialectError, MissingCredential, MissingDependency,
                     VendorError, VendorTimeout)

# Vendor SDKs are checked once, at import: one message naming the extra, not a failure hours
# into a run. The scorer never imports this package, so scoring needs none of them.
try:
    from .document import predict
    from .registry import DEFAULT_TIMEOUT, PROVIDERS, adapter, resolve, settings_for
except ImportError as exc:
    # `exc.name` tells an SDK that is missing or incompatible apart from an import of ours that
    # broke; only the first is fixed by installing the extra.
    if (exc.name or "").startswith(__name__.split(".")[0]):
        raise
    raise MissingDependency(
        f"the harness could not import what the vendor adapters need:\n"
        f"    {exc}\n"
        f"    pip install 'omni-extract-bench[harness]'"
    ) from None

__all__ = [
    "predict", "adapter", "Adapter",
    "Extraction", "Cost",
    "VendorError", "VendorTimeout", "DialectError",
    "AccountFailure", "MissingCredential", "MissingDependency",
    "PROVIDERS", "DEFAULT_TIMEOUT", "settings_for", "resolve",
]
