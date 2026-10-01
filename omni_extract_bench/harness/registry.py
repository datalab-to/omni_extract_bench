"""Which adapter, at what settings, filed under what name.

An adapter is a MODULE with three names -- `contract.Adapter` states them. `ADAPTERS` maps a
provider name to one, and a model id routes to the single-shot LLM adapter, so a model id
needs no entry.

AN ADAPTER CAN LIVE IN ANOTHER PACKAGE. An installed distribution declares one under the
`omni_extract_bench.adapters` entry-point group, naming the module:

    [project.entry-points."omni_extract_bench.adapters"]
    my-agent = "my_package.oeb_adapter"

Its name is listed from the metadata alone; the module is imported the first time that name
is asked for, so a heavy agent costs nothing to a run that does not use it.

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
ADAPTER_NAMES = ("Config", "prepare_schema", "extract")


def plugins() -> dict[str, metadata.EntryPoint]:
    """The adapters other installed packages declare, by provider name. Reads metadata only."""
    # Note: walked per distribution rather than through `metadata.entry_points`, which keeps
    # one entry per name and so would drop the second of two packages claiming it, silently.
    eps = [ep for d in metadata.distributions() for ep in d.entry_points
           if ep.group == ENTRY_POINT_GROUP]
    out = {}
    for ep in eps:
        dist = ep.dist.name
        # Note: a clash is refused rather than resolved, because either winner would publish a
        # run under a name that does not say which adapter produced it.
        if ep.name in ADAPTERS or MODEL_SEPARATOR in ep.name:
            raise ValueError(f"package {dist!r} declares adapter {ep.name!r} under "
                             f"{ENTRY_POINT_GROUP!r}, which is a built-in provider or contains "
                             f"{MODEL_SEPARATOR!r}. Rename it in {dist}'s pyproject.toml")
        # The same distribution can be found twice when its directory is on `sys.path` twice.
        if ep.name in out and out[ep.name].dist.name != dist:
            raise ValueError(f"adapter {ep.name!r} is declared twice under "
                             f"{ENTRY_POINT_GROUP!r}: by {out[ep.name].dist.name!r} and {dist!r}. "
                             f"Rename one of them")
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
            module = ep.load()
        except ImportError as exc:
            raise ImportError(f"adapter {provider!r}, declared by package {ep.dist.name!r} as "
                              f"{ep.value!r}, failed to import:\n    {exc}") from exc
        add_adapter(provider, module)
    if provider not in ADAPTERS:
        raise ValueError(f"unknown provider {provider!r}. Known vendors: "
                         f"{', '.join(PROVIDERS)}. Any OpenRouter model id also works, "
                         f"e.g. openai/gpt-5.6-sol")
    return ADAPTERS[provider]


def add_adapter(provider: str, module: ModuleType) -> None:
    """File `module` under `provider`, once it has the names `contract.Adapter` requires."""
    if missing := [n for n in ADAPTER_NAMES if not hasattr(module, n)]:
        raise TypeError(f"adapter {provider!r} ({module.__name__}) is missing "
                        f"{', '.join(missing)}. An adapter module defines "
                        f"{', '.join(ADAPTER_NAMES)}; see harness/contract.py")
    ADAPTERS[provider] = module


def resolve(provider: str) -> str:
    """This adapter's module name -- what `WORKERS` is keyed by, and what groups the runs of
    one adapter together however many model ids reached it."""
    return adapter(provider).__name__.rsplit(".", 1)[-1]


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
