"""Which adapter a provider name means: a shipped adapter (`ADAPTERS`), one an installed package declares
under the entry-point group `omni_extract_bench.adapters` (`PLUGINS`), or, for an OpenRouter model id such as
`openai/gpt-5.6-sol`, the single-shot LLM adapter. It also checks, once, that the adapters' packages are
installed.
"""
from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
from typing import Any

from omni_bench_core import adapter, errors

from . import contract

# Vendor SDKs are checked once, at import: one message naming the extra, not a failure hours into a run.
try:
    from .providers import azure_cu, datalab, extend, llamaextract, llm_single_shot, mistral, reducto
except ImportError as exc:
    if (exc.name or "").startswith(__name__.split(".")[0]):
        raise
    raise errors.MissingDependency(f"the harness could not import what the vendor adapters need:\n    {exc}\n"
                                   "    pip install 'omni-extract-bench[harness]'") from None

ENTRY_POINT_GROUP = "omni_extract_bench.adapters"
SHIPPED: tuple[type[contract.Extractor], ...] = (
    datalab.Datalab, mistral.Mistral, reducto.Reducto, extend.Extend, llamaextract.LlamaExtract, azure_cu.AzureCU)
LLM = llm_single_shot.LLMSingleShot


ADAPTERS: dict[str, type[adapter.Adapter]] = {a.name: a for a in SHIPPED}


def plugins() -> dict[str, metadata.EntryPoint]:
    """The adapters installed packages declare under `ENTRY_POINT_GROUP`, by name, not yet loaded.

    A plugin is loaded only when its name is looked up (`adapter_class`), so a broken plugin package breaks
    only the runs that use it. A declared name that is taken, or that holds a "/" (a model id), is refused."""
    out: dict[str, metadata.EntryPoint] = {}
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in ADAPTERS or ep.name in out or "/" in ep.name:
            raise errors.UsageError(f"package {ep.dist.name if ep.dist else '?'} declares adapter {ep.name!r}, which "
                                    "is taken or holds a '/'; rename it in its pyproject.toml")
        out[ep.name] = ep
    return out


PLUGINS = plugins()
PROVIDERS = sorted({*ADAPTERS, *PLUGINS})


def adapter_class(provider: str) -> type[adapter.Adapter]:
    """The class a provider name means: the LLM adapter for a model id, a shipped adapter, or a plugin, loaded
    the first time it is asked for."""
    if "/" in provider: return LLM
    if provider not in ADAPTERS and (ep := PLUGINS.get(provider)):
        try:
            ADAPTERS[provider] = ep.load()
        except ImportError as exc:
            exc.add_note(f"while loading adapter {provider!r} from package {ep.dist.name if ep.dist else '?'}")
            raise
    if provider not in ADAPTERS:
        raise errors.UsageError(f"unknown provider {provider!r}. Known vendors: {', '.join(PROVIDERS)}. "
                                "Any OpenRouter model id also works, e.g. openai/gpt-5.6-sol")
    return ADAPTERS[provider]


def system(provider: str, options: Mapping[str, Any] | None = None) -> adapter.Adapter:
    """The adapter `provider` names, with `options`, checked.

    A provider with a "/" is an OpenRouter model id, run by the single-shot LLM adapter. Its model is the
    provider name, and a `model` option is refused, so a run can't be stored under a model it didn't use."""
    options = dict(options or {})
    if "/" in provider:
        if "model" in options:
            raise errors.UsageError(f"{provider}: `model` is the provider name, not an option; run the one you "
                                    f"want: --providers {options['model']}")
        return adapter.configure(LLM, {**options, "model": provider})
    return adapter.configure(adapter_class(provider), options)
