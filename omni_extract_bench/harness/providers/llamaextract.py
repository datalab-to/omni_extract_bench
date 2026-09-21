#!/usr/bin/env python3
"""Run LlamaExtract (LlamaCloud) structured extraction against a PDF + JSON Schema.

Uploads the PDF file BYTES (never a URL — no source domain leaks to the vendor),
submits a stateless v2 extraction job, polls to completion, and writes the same
`{result, _meta}` envelope as the other providers.

Mode: tier="agentic_plus" — the highest extraction tier the v2 API offers. It was once
recorded as unavailable on a standard key (a 422 saying "Input should be 'cost_effective'
or 'agentic'"), and the default sat at `agentic` on that basis; re-probing the live endpoint
shows it now validates. Entitlements change — check a tier against the API, not a comment.

Auth: LLAMA_CLOUD_API_KEY (llx-...).

Usage:
    oeb predict --provider llamaextract --doc doc.pdf --schema schema.json

Adapted from longextract_bench (MIT, (c) 2026 Micro1) -- see providers/LICENSE-micro1.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from ..extraction import (Budget, Cost, Extraction, MissingCredential, PollRetry,
                          VendorError)

BASE = "https://api.cloud.llamaindex.ai"
TIER = "agentic_plus"
_TERMINAL = {"SUCCESS", "COMPLETED", "FAILED", "ERROR", "CANCELLED"}


@dataclasses.dataclass(frozen=True)
class Config:
    """What llamaextract can be asked, and what it is asked at its maximum tier."""

    tier: str = dataclasses.field(
        default=TIER,
        metadata={"choices": ["cost_effective", "agentic", "agentic_plus"],
                  "help": "extraction tier; entitlements change, so check one against the API"})
    poll_interval: int = 5


def prepare_schema(schema: dict, defs: dict | None = None) -> dict:
    """Inline $ref/$defs, drop $-prefixed metadata keys, and collapse type-lists to a
    single type. The latter is required: LlamaExtract turns `"type": ["array","null"]`
    into an anyOf and the array branch loses its `items` → 400 schema_validation. We
    drop the "null" so arrays/objects/scalars stay single-typed (items preserved). No
    field, description, enum, or type-category is changed — pure dialect cleanup."""
    if defs is None:
        defs = schema.get("$defs", {})
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        return prepare_schema(
            copy.deepcopy(defs.get(schema["$ref"].split("/")[-1], {})), defs
        )
    node = {k: v for k, v in schema.items() if not k.startswith("$")}

    for comb in ("anyOf", "oneOf", "allOf"):
        branches = [b for b in (node.get(comb) or []) if isinstance(b, dict)]
        if branches:
            pick = next((b for b in branches if b.get("type") != "null"), None)
            if pick is not None:
                merged = {k: v for k, v in node.items()
                          if k not in ("anyOf", "oneOf", "allOf")}
                for k, v in pick.items():
                    merged.setdefault(k, v)
                return prepare_schema(merged, defs)

    t = node.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        node["type"] = non_null[0] if non_null else "string"

    if "enum" in node and "type" not in node:
        vals = [v for v in node["enum"] if v is not None]
        kinds = {type(v) for v in vals}
        node["type"] = ({str: "string", bool: "boolean", int: "integer", float: "number"}
                        .get(kinds.pop()) if len(kinds) == 1 else "string")

    if isinstance(node.get("enum"), list) and node.get("type") in (
            "string", "boolean", "integer", "number"):
        node["enum"] = [v for v in node["enum"] if v is not None]

    ap = node.get("additionalProperties")
    if isinstance(ap, dict):
        node["additionalProperties"] = True

    if "properties" in node:
        node["properties"] = {
            k: prepare_schema(v, defs) for k, v in node["properties"].items()
        }
    if "items" in node:
        node["items"] = prepare_schema(node["items"], defs)
    return node


def _req(
    method: str,
    url: str,
    key: str,
    headers: dict | None = None,
    data: bytes | None = None,
) -> dict | list:
    h = {"Authorization": f"Bearer {key}", **(headers or {})}
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")[:600]
        except Exception:  # noqa: BLE001
            body = "<body unavailable>"
        raise VendorError(f"HTTP {e.code} {method} {url.split('?')[0]}: {body}",
                          status=e.code, body=body) from None


def _project_id(key: str) -> str:
    projs = _req("GET", f"{BASE}/api/v1/projects", key)
    if isinstance(projs, list) and projs:
        return projs[0]["id"]
    return (projs.get("projects") or [{}])[0].get("id")


def _upload(pdf: Path, key: str) -> str:
    boundary = "----llamaextract" + str(int(time.time()))
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="document.pdf"\r\nContent-Type: application/pdf\r\n\r\n'.encode()
        + pdf.read_bytes()
        + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\n'
        f"extract\r\n--{boundary}--\r\n".encode()
    )
    up = _req(
        "POST",
        f"{BASE}/api/v1/beta/files",
        key,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        data=body,
    )
    return up["id"]


def _dt(v: object) -> datetime | None:
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    return None


def extract(pdf: Path, schema: dict, *, timeout: float = 1800.0,
            config: Config = Config()) -> Extraction:
    """Upload, submit a v2 extract job at `tier`, poll it to a terminal state.

    `tier` is an argument rather than an environment variable because it is a PARITY decision:
    the harness runs every vendor at its maximum, and a setting that reads itself out of the
    environment is one a run cannot state in its own record.

    The job id comes back with the result, so a job that outlived its budget is named in the
    record rather than being anonymous.
    """
    budget = Budget(timeout)
    key = (os.environ.get("LLAMA_CLOUD_API_KEY")
           or os.environ.get("LLAMAPARSE_API_KEY"))
    if not key:
        raise MissingCredential("LLAMA_CLOUD_API_KEY (or LLAMAPARSE_API_KEY) is not set")

    project_id = _project_id(key)
    file_id = _upload(pdf, key)
    job = _req("POST", f"{BASE}/api/v2/extract?project_id={project_id}", key,
               headers={"Content-Type": "application/json"},
               data=json.dumps({
                   "file_input": file_id,
                   "configuration": {"tier": config.tier, "extraction_target": "per_doc",
                                     "data_schema": schema},
               }).encode())
    job_id = job.get("id") or job.get("job_id")

    polls = 0
    retry = PollRetry(budget)
    while True:
        budget.check(f"job {job_id} was still running server-side after {polls} polls")
        try:
            body = _req("GET", f"{BASE}/api/v2/extract/{job_id}"
                               f"?project_id={project_id}&expand=metadata", key)
        except VendorError as exc:
            if retry.again(exc.status):
                continue
            raise
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if retry.again():
                continue
            raise VendorError(f"job {job_id}: polling failed repeatedly: {exc}"[:300],
                              status=None) from None
        retry.ok()
        polls += 1
        status = body.get("status")
        if status in _TERMINAL:
            break
        time.sleep(config.poll_interval)

    if status in ("FAILED", "ERROR", "CANCELLED"):
        raise VendorError(f"LlamaExtract {status}: {body.get('error_message')}", status=200)

    result = body.get("extract_result") or body.get("data") or body.get("result")
    if not isinstance(result, dict) or not result:
        raise VendorError(f"{status} with no extraction: {json.dumps(body)[:300]}", status=200)

    start, end = _dt(body.get("created_at")), _dt(body.get("updated_at"))
    return Extraction(
        result=result,
        raw={**{k: v for k, v in body.items()
                if k not in ("extract_result", "data", "result")},
             "tier": config.tier,
             "server_latency_s": round((end - start).total_seconds(), 2) if start and end else None},
        cost=Cost(),
        job_id=job_id)

