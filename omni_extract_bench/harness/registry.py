"""Which adapter a system name means: a shipped adapter (`ADAPTERS`), one an installed package declares
under the entry-point group `omni_extract_bench.adapters` (`plugins`), or, for an OpenRouter model id
such as `openai/gpt-5.6-sol`, the single-shot LLM adapter."""
from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
from typing import Any

from omni_bench_core import adapter

from . import contract

# Vendor SDKs are checked once, at import: one message naming the extra, not a failure hours into a run.
try:
    from .providers import azure_cu, datalab, extend, llamaextract, llm_single_shot, mistral, reducto
except ImportError as exc:
    if (exc.name or "").startswith(__name__.split(".")[0]):
        raise
    exc.add_note("the harness could not import what the vendor adapters need: pip install "
                 "'omni-extract-bench[harness]'")
    raise

ENTRY_POINT_GROUP = "omni_extract_bench.adapters"
SHIPPED: tuple[type[contract.Extractor], ...] = (
    datalab.Datalab, mistral.Mistral, reducto.Reducto, extend.Extend, llamaextract.LlamaExtract, azure_cu.AzureCU)
LLM = llm_single_shot.LLMSingleShot


ADAPTERS: dict[str, type[adapter.Adapter]] = {a.name: a for a in SHIPPED}


def plugins() -> dict[str, metadata.EntryPoint]:
    """The adapters installed packages declare under `ENTRY_POINT_GROUP`, by name, not yet loaded, so a
    broken plugin breaks only the runs that use it. A name already taken, or holding a "/", is refused."""
    out: dict[str, metadata.EntryPoint] = {}
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in ADAPTERS or ep.name in out or "/" in ep.name:
            raise ValueError(f"package {ep.dist.name if ep.dist else '?'} declares adapter {ep.name!r}, which "
                             "is taken or holds a '/'; rename it in its pyproject.toml")
        out[ep.name] = ep
    return out


PLUGINS = plugins()
PROVIDERS = sorted({*ADAPTERS, *PLUGINS})


def adapter_class(provider: str) -> type[adapter.Adapter]:
    """The class a system name means: the LLM adapter for a model id, a shipped adapter, or a plugin."""
    if "/" in provider: return LLM
    if provider not in ADAPTERS and (ep := PLUGINS.get(provider)):
        try:
            ADAPTERS[provider] = ep.load()
        except ImportError as exc:
            exc.add_note(f"while loading adapter {provider!r} from package {ep.dist.name if ep.dist else '?'}")
            raise
    if provider not in ADAPTERS:
        raise ValueError(f"unknown system {provider!r}. Known vendors: {', '.join(PROVIDERS)}. "
                         "Any OpenRouter model id also works, e.g. openai/gpt-5.6-sol")
    return ADAPTERS[provider]


def system(provider: str, options: Mapping[str, Any] | None = None) -> adapter.Adapter:
    """The adapter `provider` names, with `options`, checked. A name with a "/" is a model id for the
    single-shot LLM adapter, and a `model` option other than it is refused, so a run can't be stored under a
    model it didn't use."""
    options = dict(options or {})
    if "/" in provider:
        if options.get("model", provider) != provider:
            raise ValueError(f"{provider}: its `model` is its name, not {options['model']!r}; to run "
                             f"{options['model']}, write its config: oeb init {options['model']}")
        return adapter.configure(LLM, {**options, "model": provider})
    return adapter.configure(adapter_class(provider), options)
