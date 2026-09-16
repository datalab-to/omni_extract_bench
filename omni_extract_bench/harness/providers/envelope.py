"""The uniform provider output envelope, as used by the vendored micro1 adapters.

Every CLI adapter writes `{result, _meta}` so a run carries the extraction plus wall-clock
latency and the vendor's own usage block, in one shape, for every provider.

Derived from longextract_bench (MIT, (c) 2026 Micro1) -- see providers/LICENSE-micro1.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_output(out_path: Path, *, provider: str, result: Any, latency_s: float, usage: Any) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"result": result,
         "_meta": {"provider": provider, "latency_s": round(latency_s, 2), "usage": usage}},
        indent=2, default=str))
