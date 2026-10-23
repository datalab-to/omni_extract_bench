"""Producing predictions: the vendor adapters, and how a provider name becomes one.

The loop that runs adapters over a dataset (resuming, retries, budgets, run folders) is omni-bench-core's
(`omni_bench_core.benchmark`, `omni_bench_core.adapter.predict`). What is particular to extraction is here:

    contract.py      `Question`, what every vendor is sent, and `Extractor`, the base class of an adapter
    schema.py        the conventions every vendor is asked under, and the shared pieces of schema reshaping
    responses.py     reading what came back: fenced JSON, list replies, cost units
    registry.py      which adapter a provider name means, made from a config's options
    providers/       one adapter class per vendor

`system(provider, options)` makes an adapter from a provider name (`registry.py`).

    pip install 'omni-extract-bench[harness]'
"""
from typing import Any

from omni_bench_core.adapter import predict

from . import contract

# Note: the registry imports every vendor SDK, so it is imported on first use, not here. The task imports
# `contract` and `schema`, and grading must work without the [harness] extra.
REGISTRY = ("ADAPTERS", "ENTRY_POINT_GROUP", "LLM", "PLUGINS", "PROVIDERS", "SHIPPED", "adapter_class", "system")

__all__ = ["ADAPTERS", "ENTRY_POINT_GROUP", "LLM", "PLUGINS", "PROVIDERS", "SHIPPED", "adapter_class", "contract",
           "predict", "system"]


def __getattr__(name: str) -> Any:
    if name not in REGISTRY: raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import registry

    return getattr(registry, name)
