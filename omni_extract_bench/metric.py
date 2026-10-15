"""Deprecated: renamed `scorer`. Every name here is `scorer`'s, with a DeprecationWarning; the module
will be removed in a later release."""
from __future__ import annotations

import warnings
from typing import Any

from . import scorer as _scorer


def __getattr__(name: str) -> Any:
    warnings.warn(f"omni_extract_bench.metric.{name} is deprecated: use omni_extract_bench.scorer.{name}. "
                  "The metric module will be removed in a later release.", DeprecationWarning, stacklevel=2)
    return getattr(_scorer, name)
