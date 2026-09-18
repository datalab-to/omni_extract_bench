#!/usr/bin/env python3
"""Run LlamaExtract (LlamaCloud) structured extraction against a PDF + JSON Schema.

Uploads the PDF file BYTES (never a URL — no source domain leaks to the vendor),
submits a stateless v2 extraction job, polls to completion, and writes the same
`{result, _meta}` envelope as the other providers.

Mode: tier="agentic" — the highest extraction tier available on a standard
LlamaCloud key. NOTE: tier="agentic_plus" (the premium tier) is not available on a
standard key — the v2 extract API rejects it with 422 ("Input should be
'cost_effective' or 'agentic'"), so `agentic` is the maxed-out mode here.

Auth: LLAMA_CLOUD_API_KEY (llx-...).

Usage:
    python -m longextract_bench.providers.llamaextract \
        --pdf doc.pdf --schema schema.json --out /tmp/llamaextract.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from .envelope import write_output

BASE = "https://api.cloud.llamaindex.ai"
# Overridable so a caller can select the maximum tier without editing this module.
# The v2 API enum is 'cost_effective' | 'agentic' | 'agentic_plus'. `agentic_plus` was recorded
# as unavailable after a 422 on this key, and the default stayed at `agentic` on that basis --
# but re-probing the live endpoint shows it now validates, so that note was stale. Verify a
# tier against the API rather than trusting a comment; entitlements change.
TIER = os.environ.get("LLAMAEXTRACT_TIER", "agentic")
_TERMINAL = {"SUCCESS", "COMPLETED", "FAILED", "ERROR", "CANCELLED"}

#: JSON Schema type names for the Python values an enum member can be.
JSON_TYPE_NAMES = {str: "string", bool: "boolean", int: "integer", float: "number"}


def _enum_type(members: list) -> str | list[str]:
    """The `type` an enum needs: the type of each member, "null" included."""
    names = sorted({JSON_TYPE_NAMES[type(m)] for m in members if m is not None})
    if None in members:
        names.append("null")
    return names[0] if len(names) == 1 else names


def _adapt_schema(schema: dict, defs: dict | None = None) -> dict:
    """Inline $ref/$defs, drop $-prefixed metadata keys, and reduce the schema to the subset
    LlamaExtract's v2 API accepts"""
    if defs is None:
        defs = schema.get("$defs", {})
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        return _adapt_schema(
            copy.deepcopy(defs.get(schema["$ref"].split("/")[-1], {})), defs
        )
    node = {k: v for k, v in schema.items() if not k.startswith("$")}

    for comb in ("anyOf", "oneOf", "allOf"):
        if isinstance(node.get(comb), list):
            node[comb] = [_adapt_schema(b, defs) for b in node[comb]]

    # A typeless enum is rejected ("Invalid type for field"), so infer the type from the
    # enum's own values -- including `null`, or the field stops being nullable.
    if "enum" in node and "type" not in node:
        node["type"] = _enum_type(node["enum"])

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
        raise RuntimeError(f"HTTP {e.code} {method} {url.split('?')[0]}: {body}") from None


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


def run(pdf: Path, schema: dict, out: Path, key: str, poll_interval: int) -> None:
    side = out.with_suffix(".runid.json")
    resume = None
    if side.exists():
        try:
            resume = json.loads(side.read_text())
        except (OSError, json.JSONDecodeError):
            resume = None

    if resume and resume.get("job_id"):
        # Prior attempt already submitted — the job is persistent; poll it, never
        # re-submit (no double-billing).
        pid, job_id = resume["project_id"], resume["job_id"]
        print(f"Resuming existing job {job_id} (sidecar)…")
    else:
        pid = _project_id(key)
        print(f"project={pid}\nUploading {pdf.name} ({pdf.stat().st_size // 1024} KB)…")
        file_id = _upload(pdf, key)
        print(f"  → file_id: {file_id}\nSubmitting extract (tier={TIER})…")
        payload = {
            "file_input": file_id,
            "configuration": {
                "tier": TIER,
                "extraction_target": "per_doc",
                "data_schema": _adapt_schema(schema),
            },
        }
        job = _req(
            "POST",
            f"{BASE}/api/v2/extract?project_id={pid}",
            key,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode(),
        )
        job_id = job.get("id") or job.get("job_id")
        # Persist the job id immediately so a killed/timed-out job stays pollable.
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            side.write_text(
                json.dumps(
                    {"provider": "llamaextract", "job_id": job_id, "project_id": pid}
                )
            )
        except OSError as e:
            print(f"  (warn: could not write runid sidecar: {e})")
        print(f"  → job_id: {job_id}\nPolling…")

    transient = 0
    while True:
        try:
            body = _req(
                "GET",
                f"{BASE}/api/v2/extract/{job_id}?project_id={pid}&expand=metadata",
                key,
            )
        except (urllib.error.URLError, OSError, ValueError):
            transient += 1
            if transient > 60:
                raise
            time.sleep(min(30, 2 * transient))
            continue
        transient = 0
        status = body.get("status")
        print(f"  {status}", end="\r", flush=True)
        if status in _TERMINAL:
            print()
            break
        time.sleep(poll_interval)

    if status in ("FAILED", "ERROR", "CANCELLED"):
        raise RuntimeError(f"LlamaExtract {status}: {body.get('error_message')}")

    result = body.get("extract_result") or body.get("data") or body.get("result")
    meta = body.get("metadata") or {}
    # server-side processing span = updated_at - created_at (excludes our poll/upload)
    a, b = _dt(body.get("created_at")), _dt(body.get("updated_at"))
    latency = (b - a).total_seconds() if a and b else 0.0

    write_output(
        out,
        provider="llamaextract",
        result=result,
        latency_s=round(latency, 2),
        usage={"tier": TIER, "job_id": job_id, "metadata": meta},
    )
    print(f"Status: {status}  latency={latency:.0f}s\nSaved → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="LlamaExtract (tier=agentic)")
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--poll-interval", type=int, default=5)
    args = ap.parse_args()

    key = os.environ.get("LLAMA_CLOUD_API_KEY", "")
    if not key:
        sys.exit("LLAMA_CLOUD_API_KEY not set")

    schema = json.loads(args.schema.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run(args.pdf, schema, args.out, key, args.poll_interval)


if __name__ == "__main__":
    main()
