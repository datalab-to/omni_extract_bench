"""Extend adapter: upload, submit an extract run with the schema inline, poll it to a terminal state.

Extend validates schemas strictly, so `prepare_schema` restates the schema in its dialect
(`to_strict_dialect`) and aliases the names it reserves (`RESERVED`); `read` restores them.

Auth: EXTEND_API_KEY, and EXTEND_WORKSPACE_ID when the key reaches several workspaces.

    oeb run --providers extend
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, ClassVar

import httpx
from omni_bench_core import adapter, errors

from ...scorers import field_matching
from .. import contract
from .. import schema as SCHEMA

BASE = "https://api.extend.ai"
API_VERSION = "2026-02-09"

ARRAY_STRATEGY = "large_array_max_context"
POLL_S = 5
DEFAULT_TIMEOUT_S = 1800.0
TERMINAL = {"PROCESSED", "COMPLETED", "FAILED", "CANCELLED", "ERROR"}


def headers(key: str, api_version: str = API_VERSION) -> dict:
    h = {"Authorization": f"Bearer {key}", "x-extend-api-version": api_version}
    ws = os.environ.get("EXTEND_WORKSPACE_ID")
    if ws:
        h["x-extend-workspace-id"] = ws
    return h


def upload(c: httpx.Client, pdf: Path, base_url: str = BASE) -> str:
    r = c.post(f"{base_url}/files/upload",
               files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")}, timeout=300)
    if r.status_code >= 400:
        raise RuntimeError(f"upload HTTP {r.status_code}: {r.text[:300]}")
    with adapter.reading("the uploaded file's id"):
        b = r.json()
        fid = (b.get("file") or {}).get("id") or b.get("id")
    if not fid:
        raise RuntimeError(f"upload: no file id in {r.text[:200]}")
    return fid


def submit(c: httpx.Client, file_id: str, schema: dict, base_url: str = BASE,
           array_strategy: str = ARRAY_STRATEGY) -> str:
    """Submit an extraction run against the current API.

    The schema goes inline in `config`, which is what the current API version is for. The old
    one needed a saved processor to hang a schema on, so this adapter fetched any extract
    processor from the workspace and passed its id -- a wasted round trip per document, and a
    hard failure on an account that happens to have no processor saved. `submit` had already
    stopped using it.

    `advancedOptions.arrayStrategy` selects the large-array, maximum-context strategy
    (`large_array_max_context`).
    """
    body = {"file": {"id": file_id},
            "config": {"schema": schema,
                       "advancedOptions": {"arrayStrategy": {"type": array_strategy}}}}
    r = c.post(f"{base_url}/extract_runs", json=body, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"extract_runs HTTP {r.status_code}: {r.text[:400]}")
    with adapter.reading("the run's id"):
        b = r.json()
        run = b.get("extractRun") or b.get("processorRun") or b.get("run") or b
        rid = run.get("id")
    if not rid:
        raise RuntimeError(f"extract_runs: no run id in {r.text[:250]}")
    return rid


def poll(c: httpx.Client, run_id: str, deadline, base_url: str = BASE) -> dict:
    polls = 0
    retry = adapter.PollRetry(deadline)
    while True:
        deadline.check(f"run {run_id} was still processing after {polls} polls")
        try:
            r = c.get(f"{base_url}/extract_runs/{run_id}", timeout=60)
        except httpx.TransportError as exc:
            if retry.again():
                continue
            raise errors.VendorUnreachable(f"polling failed: {exc}"[:300]) from None
        if r.status_code >= 400:
            if retry.again(r.status_code):
                continue
            raise errors.VendorError(f"poll HTTP {r.status_code}: {r.text[:250]}",
                                     status=r.status_code, body=r.text)
        retry.reset()
        polls += 1
        with adapter.reading("the run's status"):
            b = r.json()
            run = b.get("extractRun") or b.get("processorRun") or b.get("run") or b
            st = str(run.get("status", "")).upper()
        if st in TERMINAL:
            return run
        time.sleep(POLL_S)


def extraction_of(run: dict):
    """The extracted data, and the path it came from.

    `output.value` is the documented shape -- "extraction output splits into `output.value`
    (your data, shaped like the schema) and `output.metadata`" -- and a live run confirms it.
    This used to try five shapes in order and take the first that looked plausible, which is
    the wrong instinct in an extractor: if two could match, the wrong one is chosen silently
    and the wrong thing is scored. Better to read the documented field and fail loudly.
    """
    output = run.get("output")
    if isinstance(output, dict):
        value = output.get("value", output)
        if isinstance(value, dict) and value:
            return value, "output.value"
    return None, None


@dataclasses.dataclass(frozen=True)
class Extend(contract.Extractor):
    """Extend's extract runs, with its large-array, maximum-context strategy, because this corpus is
    array-heavy."""

    name: ClassVar[str] = "extend"
    workers: ClassVar[int] = 5
    array_strategy: str = dataclasses.field(
        default=ARRAY_STRATEGY, metadata={"help": "large-array, maximum-context extraction; this corpus is array-heavy"})
    api_version: str = API_VERSION
    base_url: str = BASE

    def prepare_schema(self, schema: dict) -> dict:
        """The schema Extend is sent: reserved names aliased, $refs inlined, strict dialect."""
        return to_strict_dialect(
            field_matching.resolve_refs(rename_reserved(schema), max_depth=SCHEMA.MAX_REF_DEPTH))

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """Upload, submit a run, poll it to a terminal state.

        Extend does not report a per-document cost, so `cost.usd` is None rather than a figure invented
        from a price list.
        """
        key = os.environ.get("EXTEND_API_KEY")
        if not key:
            raise errors.MissingCredential("EXTEND_API_KEY is not set")

        deadline = adapter.Budget(timeout)
        with httpx.Client(headers=headers(key, self.api_version), timeout=120) as c:
            try:
                file_id = upload(c, pdf, self.base_url)
                run_id = submit(c, file_id, schema, self.base_url, self.array_strategy)
            except httpx.HTTPStatusError as exc:
                raise errors.VendorError(f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
                                         status=exc.response.status_code, body=exc.response.text) from None
            except RuntimeError as exc:
                raise errors.VendorError(str(exc)[:300], status=200) from None
            run = poll(c, run_id, deadline, self.base_url)
        with adapter.reading("the run's cost"):
            usage = run.get("usage") or {}
            credits = usage.get("totalCredits", usage.get("credits"))
        return adapter.Call(raw=run,
                            cost=adapter.Cost(credits=credits if isinstance(credits, (int, float))
                                              and not isinstance(credits, bool) else None,
                                              source="usage.totalCredits" if credits is not None else None),
                            job_id=run_id)

    def read(self, raw: Any, q: contract.Question) -> dict:
        """The extraction in `output.value`, with the reserved names restored."""
        payload, _ = extraction_of(raw if isinstance(raw, dict) else {})
        found = restore_reserved(payload)
        if not isinstance(found, dict) or not found:
            status = str(raw.get("status", "")).upper() if isinstance(raw, dict) else "?"
            raise ValueError(f"run finished {status} with no extraction: {json.dumps(raw)[:300]}")
        return found


#: Extend reserves `id`. `rename_reserved` (in `prepare_schema`) and `restore_reserved` (in
#: `read`) both read this mapping; drop one and drop the other.
RESERVED = {"id": "id__"}


def rename_reserved(node):
    if isinstance(node, list):
        return [rename_reserved(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "properties" and isinstance(v, dict):
            out[k] = {RESERVED.get(pk, pk): rename_reserved(pv) for pk, pv in v.items()}
        elif k == "required" and isinstance(v, list):
            out[k] = [RESERVED.get(x, x) for x in v]
        else:
            out[k] = rename_reserved(v)
    return out


STRICT_ALLOWED_KEYS = ("type", "enum", "properties", "items", "required", "description")


def collapse_nullable_union(node):
    """Reduce ``anyOf: [{...}, {"type": "null"}]`` to its non-null branch, merging siblings.

    The nullable idiom declares no type of its own, and Extend's validator wants a type. The
    nullability is not lost by collapsing here: `to_strict_dialect` puts it back in the form
    Extend requires -- `["string", "null"]`, and `null` among an enum's members.
    """
    if not isinstance(node, dict):
        return node
    for branch_key in ("anyOf", "oneOf", "allOf"):
        branches = [b for b in (node.get(branch_key) or []) if isinstance(b, dict)]
        if not branches:
            continue
        pick = next((b for b in branches if b.get("type") != "null"), None)
        if pick is not None:
            merged = {k: v for k, v in node.items()
                      if k not in ("anyOf", "oneOf", "allOf")}
            for k, v in pick.items():
                merged.setdefault(k, v)
            return merged
    return node


def to_strict_dialect(node, in_items=False, allowed=STRICT_ALLOWED_KEYS):
    """Reshape for a vendor that validates strictly and requires nullable properties.

    Three rules, and they differ by position. That is the detail worth writing down, because applying
    nullability everywhere fixes properties and breaks array items at once:

      * keys: an allowlist (see ``STRICT_ALLOWED_KEYS``);
      * properties: must be nullable, ``["string", "null"]``, and enums must include ``null``; and
      * items: must be a bare type; a ``["string","null"]`` union is rejected here.

    Modelled on Extend's validator; useful for any vendor with the same shape of constraints.
    """
    if isinstance(node, list):
        return [to_strict_dialect(x, in_items, allowed) for x in node]
    if not isinstance(node, dict):
        return node

    if "type" not in node and "enum" not in node:
        collapsed = collapse_nullable_union(node)
        if collapsed is not node:
            return to_strict_dialect(collapsed, in_items, allowed)

    out = {}
    for key in allowed:
        if key not in node:
            continue
        value = node[key]
        if key == "properties" and isinstance(value, dict):
            out[key] = {k: to_strict_dialect(v, False, allowed) for k, v in value.items()}
        elif key == "items":
            out[key] = to_strict_dialect(value, True, allowed)
        else:
            out[key] = value

    declared = out.get("type")
    if isinstance(declared, list):
        out["type"] = next((t for t in declared if t != "null"), None)
    if "type" not in out and "enum" not in out and out.get("properties"):
        out["type"] = "object"

    if in_items:
        if isinstance(out.get("type"), str) and out["type"] in (
                "string", "number", "integer", "boolean"):
            return {"type": out["type"]}
        return {k: v for k, v in out.items()
                if k in ("type", "properties", "items", "required")}

    if isinstance(out.get("type"), str) and out["type"] in (
            "string", "number", "integer", "boolean"):
        out["type"] = [out["type"], "null"]
    if isinstance(out.get("enum"), list) and None not in out["enum"]:
        out["enum"] = list(out["enum"]) + [None]
    return out


def restore_reserved(obj):
    back = {v: k for k, v in RESERVED.items()}
    if isinstance(obj, list):
        return [restore_reserved(x) for x in obj]
    if isinstance(obj, dict):
        return {back.get(k, k): restore_reserved(v) for k, v in obj.items()}
    return obj

