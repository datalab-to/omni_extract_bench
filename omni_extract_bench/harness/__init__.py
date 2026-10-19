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
from . import contract
from .registry import ADAPTERS, ENTRY_POINT_GROUP, LLM, PLUGINS, PROVIDERS, SHIPPED, adapter_class, system

__all__ = ["ADAPTERS", "ENTRY_POINT_GROUP", "LLM", "PLUGINS", "PROVIDERS", "SHIPPED", "adapter_class", "contract",
           "system"]
