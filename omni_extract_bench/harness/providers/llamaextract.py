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
    python -m omni_extract_bench.harness.providers.llamaextract \
        --pdf doc.pdf --schema schema.json --out /tmp/llamaextract.json

Adapted from longextract_bench (MIT, (c) 2026 Micro1) -- see providers/LICENSE-micro1.
"""

from __future__ import annotations

import copy
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from ..extraction import (Budget, Cost, Extraction, MissingCredential, PollRetry,
                          VendorError)
from ._cli import run_cli

BASE = "https://api.cloud.llamaindex.ai"
#: The v2 API enum is 'cost_effective' | 'agentic' | 'agentic_plus'.
#: The benchmark runs every vendor at its maximum, and `vendor.ADAPTERS` selects this one.
#: The default here MATCHES it, so reproducing a document by hand reproduces the
#: benchmark -- it used to default to `agentic`, one tier below, which meant a hand-run
#: silently answered a different question from the run it was meant to explain.
TIER = "agentic_plus"
_TERMINAL = {"SUCCESS", "COMPLETED", "FAILED", "ERROR", "CANCELLED"}


def _adapt_schema(schema: dict, defs: dict | None = None) -> dict:
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
        return _adapt_schema(
            copy.deepcopy(defs.get(schema["$ref"].split("/")[-1], {})), defs
        )
    node = {k: v for k, v in schema.items() if not k.startswith("$")}

    # Collapse a union to its non-null branch. Recursing into the branches while LEAVING the
    # anyOf in place produced `properties.skills.anyOf.anyOf.1...` -- a nested union the API
    # rejects. Same resolution the grader applies when scoring, so what is sent matches how the
    # answer is judged.
    for comb in ("anyOf", "oneOf", "allOf"):
        branches = [b for b in (node.get(comb) or []) if isinstance(b, dict)]
        if branches:
            pick = next((b for b in branches if b.get("type") != "null"), None)
            if pick is not None:
                merged = {k: v for k, v in node.items()
                          if k not in ("anyOf", "oneOf", "allOf")}
                for k, v in pick.items():
                    merged.setdefault(k, v)
                return _adapt_schema(merged, defs)

    t = node.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        node["type"] = non_null[0] if non_null else "string"

    # An enum with no `type` is valid JSON Schema, but the API reports "Invalid type for
    # field". Infer the type from the enum's own non-null values rather than defaulting.
    if "enum" in node and "type" not in node:
        vals = [v for v in node["enum"] if v is not None]
        kinds = {type(v) for v in vals}
        node["type"] = ({str: "string", bool: "boolean", int: "integer", float: "number"}
                        .get(kinds.pop()) if len(kinds) == 1 else "string")

    # Once a type is declared, every enum member must match it: leaving the `null` in
    # `["MILD","MODERATE","SEVERE",null]` alongside `type: string` fails with "Input should be
    # a valid string at ...enum.3". Nullability is carried by the field being optional, not by
    # a null enum member, so the null is removed rather than the type loosened.
    if isinstance(node.get("enum"), list) and node.get("type") in (
            "string", "boolean", "integer", "number"):
        node["enum"] = [v for v in node["enum"] if v is not None]

    # `additionalProperties` as a SCHEMA (an open map, e.g. skill-category -> list) is rejected:
    # "Input should be a valid boolean". Reduce it to the boolean the dialect allows; the map
    # stays open, only the per-value constraint is dropped.
    ap = node.get("additionalProperties")
    if isinstance(ap, dict):
        node["additionalProperties"] = True

    if "properties" in node:
        node["properties"] = {
            k: _adapt_schema(v, defs) for k, v in node["properties"].items()
        }
    if "items" in node:
        node["items"] = _adapt_schema(node["items"], defs)
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
        # urllib's HTTPError stringifies to just "HTTP Error 400: Bad Request" -- the response
        # body, which is where the API says WHAT was wrong, is on the exception object and is
        # lost unless read here. Two benchmark failures were unattributable for exactly this
        # reason: a status code with no cause. Read the body and put it in the message.
        try:
            body = e.read().decode("utf-8", "replace")[:600]
        except Exception:  # noqa: BLE001
            body = "<body unavailable>"
        # VendorError, not RuntimeError: `predict` catches VendorError and records the
        # document as failed. A bare RuntimeError escapes it, reaches `future.result()`
        # in the pool, and kills the whole run -- so one 400 on one schema would end a
        # 620-document job. Schema validation is this vendor's documented failure mode.
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


def extract(pdf: Path, schema: dict, *, timeout: float = 1800.0, tier: str = TIER,
            poll_interval: int = 5) -> Extraction:
    """Upload, submit a v2 extract job at `tier`, poll it to a terminal state.

    `tier` is an argument rather than an environment variable because it is a PARITY decision:
    the harness runs every vendor at its maximum, and a setting that reads itself out of the
    environment is one a run cannot state in its own record.

    The job id comes back with the result, so a job that outlived its budget is named in the
    record rather than being anonymous.
    """
    budget = Budget(timeout)          # before the upload: it is part of the document
    # LlamaCloud issues ONE key for the whole platform, and people export it under the
    # name of whichever product they reached first.
    key = (os.environ.get("LLAMA_CLOUD_API_KEY")
           or os.environ.get("LLAMAPARSE_API_KEY"))
    if not key:
        raise MissingCredential("LLAMA_CLOUD_API_KEY (or LLAMAPARSE_API_KEY) is not set")

    # No try/except here: `_req` already raises `VendorError` with the vendor's own message
    # and status, which is what `predict` records and what decides whether a retry is worth
    # anything. Catching and re-wrapping would only lose the status.
    project_id = _project_id(key)
    file_id = _upload(pdf, key)
    job = _req("POST", f"{BASE}/api/v2/extract?project_id={project_id}", key,
               headers={"Content-Type": "application/json"},
               data=json.dumps({
                   "file_input": file_id,
                   "configuration": {"tier": tier, "extraction_target": "per_doc",
                                     "data_schema": _adapt_schema(schema)},
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
            # A failed poll is not a failed job: it is still running, and already billed.
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
        time.sleep(poll_interval)

    if status in ("FAILED", "ERROR", "CANCELLED"):
        raise VendorError(f"LlamaExtract {status}: {body.get('error_message')}", status=200)

    result = body.get("extract_result") or body.get("data") or body.get("result")
    if not isinstance(result, dict) or not result:
        raise VendorError(f"{status} with no extraction: {json.dumps(body)[:300]}", status=200)

    # server-side processing span = updated_at - created_at (excludes our poll/upload)
    start, end = _dt(body.get("created_at")), _dt(body.get("updated_at"))
    return Extraction(
        result=result,
        # The whole job body, minus the extraction itself.
        raw={**{k: v for k, v in body.items()
                if k not in ("extract_result", "data", "result")},
             "tier": tier,
             "server_latency_s": round((end - start).total_seconds(), 2) if start and end else None},
        cost=Cost(),                         # LlamaCloud bills in credits, not per call
        job_id=job_id)


def main() -> None:
    run_cli(extract, "llamaextract",
            ("--poll-interval", {"type": int, "default": 5}),
            ("--tier", {"default": TIER}),
            description="LlamaExtract v2")


if __name__ == "__main__":
    main()
