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

import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from ..dialects import MAX_REF_DEPTH, collapse_nullable_union, resolve_refs
from ..budget import Budget, PollRetry
from ..contract import Cost, Extraction
from ..errors import MissingCredential, VendorError

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


def to_typed_enum_dialect(node):
    """Reshape for a vendor that requires every enum to declare a matching type.

    ``{"enum": ["MILD", "MODERATE", null]}`` is valid JSON Schema and rejected here twice over:
    once for having no ``type``, and then -- after a type is inferred -- for the ``null`` member
    not matching it. The type is inferred from the enum's own values rather than defaulted, and
    the null is removed, since nullability belongs to the field's optionality, not to the value
    set. Also reduces ``additionalProperties`` from a schema to a boolean, which some validators
    require; the map stays open, only the per-value constraint is lost.
    """
    if isinstance(node, list):
        return [to_typed_enum_dialect(x) for x in node]
    if not isinstance(node, dict):
        return node

    collapsed = collapse_nullable_union(node)
    if collapsed is not node:
        # RECURSE on the merged node, as extend's `to_strict_dialect` does: a union whose branch is
        # a union (`Optional[list[str] | dict]`) otherwise keeps the inner `anyOf`, and a
        # nested union is what the vendor rejected in the first place.
        return to_typed_enum_dialect(collapsed)
    out = dict(node)

    declared = out.get("type")
    if isinstance(declared, list):
        non_null = [t for t in declared if t != "null"]
        out["type"] = non_null[0] if non_null else "string"

    if "enum" in out and "type" not in out:
        values = [v for v in out["enum"] if v is not None]
        kinds = {type(v) for v in values}
        out["type"] = ({str: "string", bool: "boolean", int: "integer", float: "number"}
                       .get(kinds.pop()) if len(kinds) == 1 else "string")

    if isinstance(out.get("enum"), list) and out.get("type") in (
            "string", "boolean", "integer", "number"):
        out["enum"] = [v for v in out["enum"] if v is not None]

    if isinstance(out.get("additionalProperties"), dict):
        out["additionalProperties"] = True

    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: to_typed_enum_dialect(v) for k, v in out["properties"].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = to_typed_enum_dialect(out["items"])
    return out


def drop_schema_metadata(node):
    """Remove `$`-prefixed annotations -- `$schema`, `$id`, `$comment`.

    `resolve_refs` consumes `$ref` and `$defs`; these are what is left, and they describe the
    document rather than the data. Several validators reject them as unknown keys.
    """
    if isinstance(node, list):
        return [drop_schema_metadata(x) for x in node]
    if not isinstance(node, dict):
        return node
    return {k: drop_schema_metadata(v) for k, v in node.items() if not k.startswith("$")}


def prepare_schema(schema: dict) -> dict:
    """Inline $refs, drop $-prefixed metadata, and reduce to the subset the v2 API accepts.

    Three constraints, each learned from a rejection, and all three are what
    `to_typed_enum_dialect` already encodes -- this used to restate them:

      * a nullable ARRAY must not stay a union. LlamaExtract turns `["array","null"]` into an
        anyOf whose array branch loses its `items`, and answers with 400 schema_validation.
      * a typeless enum is rejected ("Invalid type for field"), and once a type is inferred the
        `null` member no longer matches it ("Input should be a valid string at ...enum.3").
      * `additionalProperties` as a SCHEMA is rejected ("Input should be a valid boolean").
    """
    return to_typed_enum_dialect(
        drop_schema_metadata(resolve_refs(schema, max_depth=MAX_REF_DEPTH)))


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

