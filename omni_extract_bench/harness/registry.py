"""Which adapter, at what settings, filed under what name.

An adapter is a MODULE with three names -- `contract.Adapter` states them. `ADAPTERS` maps a
provider name to one, and a model id routes to the single-shot LLM adapter, so a model id
needs no entry.

THE CONFIG IS THE DECLARATION. Its fields are the options: what `--options` may set and what
`oeb providers` lists. Nothing infers an option from a signature and nothing restates a
default elsewhere. An environment variable that steered a run without appearing in it is the
bug this shape exists to prevent -- `LLAMAEXTRACT_TIER=cost_effective` once produced a run
indistinguishable from the maxed-out one the benchmark claims to publish.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from importlib import metadata
from types import ModuleType

from .providers import (azure_cu, datalab, extend, llamaextract, llm_single_shot, mistral,
                        reducto)

DEFAULT_TIMEOUT = 1800.0
MODEL_SEPARATOR = "/"

ADAPTERS = {
    "datalab": datalab,
    "mistral": mistral,
    "reducto": reducto,
    "extend": extend,
    "llamaextract": llamaextract,
    "azure-cu": azure_cu,
}
ENTRY_POINT_GROUP = "omni_extract_bench.adapters"


def plugins() -> dict[str, metadata.EntryPoint]:
    out = {}
    # Note: walked per distribution because `metadata.entry_points` keeps one entry per name,
    # silently dropping a second package that claims it.
    for ep in (ep for d in metadata.distributions() for ep in d.entry_points):
        if ep.group != ENTRY_POINT_GROUP: continue
        if (ep.name in ADAPTERS or MODEL_SEPARATOR in ep.name
                or (ep.name in out and out[ep.name].dist.name != ep.dist.name)):
            raise ValueError(f"package {ep.dist.name!r} declares adapter {ep.name!r}, which is "
                             f"taken or contains {MODEL_SEPARATOR!r}. Rename it in its "
                             f"pyproject.toml")
        out[ep.name] = ep
    return out


PLUGINS = plugins()
PROVIDERS = sorted({*ADAPTERS, *PLUGINS})


def adapter(provider: str):
    """The module that talks to this vendor: its `Config`, `prepare_schema` and `extract`.

    Any OpenRouter model id is the single-shot LLM adapter, which is why model ids need no
    entry above -- there is one adapter for all of them, and the id is one of its settings.
    """
    if MODEL_SEPARATOR in provider:
        return llm_single_shot
    if provider not in ADAPTERS and (ep := PLUGINS.get(provider)):
        try:
            add_adapter(provider, ep.load(), replace=True)
        except ImportError as exc:
            exc.add_note(f"while loading adapter {provider!r} from package {ep.dist.name!r}")
            raise
    if provider not in ADAPTERS:
        raise ValueError(f"unknown provider {provider!r}. Known vendors: "
                         f"{', '.join(PROVIDERS)}. Any OpenRouter model id also works, "
                         f"e.g. openai/gpt-5.6-sol")
    return ADAPTERS[provider]


def add_adapter(provider: str, module: ModuleType, *, replace: bool = False) -> None:
    """File an adapter defined in code; a package declares one under `ENTRY_POINT_GROUP`."""
    taken = provider in ADAPTERS or provider in PLUGINS
    if MODEL_SEPARATOR in provider or (taken and not replace):
        raise ValueError(f"adapter name {provider!r} is taken or contains {MODEL_SEPARATOR!r}. "
                         f"Pick another, or pass replace=True to replace a taken one")
    if missing := [n for n in ("Config", "prepare_schema", "extract") if not hasattr(module, n)]:
        raise TypeError(f"adapter {provider!r} is missing {', '.join(missing)}; see contract.py")
    ADAPTERS[provider] = module
    # Note: updated in place because callers hold the list itself (`from . import PROVIDERS`).
    PROVIDERS[:] = sorted({*PROVIDERS, provider})


def resolve(provider: str) -> str:
    """This adapter's module name -- what `WORKERS` is keyed by, and what groups the runs of
    one adapter together however many model ids reached it."""
    name = adapter(provider).__name__
    # Note: only our own modules are shortened; two outside ones may end in the same word.
    return name.rsplit(".", 1)[-1] if name.startswith(f"{__package__}.providers.") else name


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


def config_for(provider: str, options: dict | None = None):
    """This provider's adapter `Config`, with the caller's options applied.

    THE CONFIG IS THE DECLARATION. Its fields are the options: what `--options` may set and
    what `oeb providers` lists. Nothing infers them from a signature and nothing restates a
    default elsewhere.

    An option the adapter does not have is REFUSED, by name, rather than ignored -- a typo
    that goes through changes nothing and the run reports as stock.
    """
    config_type = adapter(provider).Config
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

    The record states it and `out_name` digests it to name a run.
    There are no credentials in it -- every adapter reads its key from the environment -- so
    nothing secret reaches a record or a directory name.
    """
    return dataclasses.asdict(config_for(provider, options))
