#!/usr/bin/env python3
"""LlamaExtract (LlamaCloud): uploads the PDF's bytes (never a URL, so no source domain reaches the
vendor), submits a v2 extraction job at its highest tier, `agentic_plus`, and polls it. Auth:
LLAMA_CLOUD_API_KEY.

    oeb run --system llamaextract

Adapted from longextract_bench (MIT, (c) 2026 Micro1); see providers/LICENSE-micro1."""

from __future__ import annotations

import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from omni_bench_core import adapter, errors

from ...scorers import field_matching
from .. import contract
from .. import schema as SCHEMA

BASE = "https://api.cloud.llamaindex.ai"
TIER = "agentic_plus"
_TERMINAL = {"SUCCESS", "COMPLETED", "FAILED", "ERROR", "CANCELLED"}


JSON_TYPE_NAMES = {str: "string", bool: "boolean", int: "integer", float: "number"}


def enum_type(members: list):
    """The `type` an enum needs: each member's type, `"null"` included, so a nullable enum stays nullable
    (`["MILD", null]` is `["string", "null"]`)."""
    names = sorted({JSON_TYPE_NAMES[type(m)] for m in members
                    if m is not None and type(m) in JSON_TYPE_NAMES})
    if not names:
        names = ["string"]
    if None in members:
        names.append("null")
    return names[0] if len(names) == 1 else names


def to_typed_enum_dialect(node):
    """The schema reshaped for a vendor that requires every enum to declare a type: the type is inferred
    from the members, `null` included, so the field stays as nullable as declared. Unions are recursed
    into, and an `additionalProperties` schema becomes `true` (the map stays open; only the per-value
    constraint is lost)."""
    if isinstance(node, list):
        return [to_typed_enum_dialect(x) for x in node]
    if not isinstance(node, dict):
        return node

    out = dict(node)
    for comb in ("anyOf", "oneOf", "allOf"):
        if isinstance(out.get(comb), list):
            out[comb] = [to_typed_enum_dialect(b) for b in out[comb]]

    if "enum" in out and "type" not in out:
        out["type"] = enum_type(out["enum"])

    if isinstance(out.get("additionalProperties"), dict):
        out["additionalProperties"] = True

    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: to_typed_enum_dialect(v) for k, v in out["properties"].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = to_typed_enum_dialect(out["items"])
    return out


def drop_schema_metadata(node):
    """The schema without `$`-prefixed annotations (`$schema`, `$id`, `$comment`), which validators reject."""
    if isinstance(node, list):
        return [drop_schema_metadata(x) for x in node]
    if not isinstance(node, dict):
        return node
    return {k: drop_schema_metadata(v) for k, v in node.items() if not k.startswith("$")}


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
            with adapter.reading(f"the answer to {method} {url.split('?')[0]}"):
                return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            body = "<body unavailable>"
        raise errors.SystemFailed(f"HTTP {exc.code} {method} {url.split('?')[0]}: {body}",
                                  status=exc.code, body=body) from exc
    except OSError as exc:   # a URLError, or a timeout reading the answer
        raise contract.unreachable(f"{method} {url.split('?')[0]}", exc) from exc


def _project_id(key: str) -> str:
    projs = _req("GET", f"{BASE}/api/v1/projects", key)
    with adapter.reading("the project's id"):
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
    with adapter.reading("the uploaded file's id"):
        return up["id"]


def _dt(v: object) -> datetime | None:
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    return None


@dataclasses.dataclass(frozen=True)
class LlamaExtract(contract.Extractor):
    """LlamaExtract's v2 stateless extraction, at its highest tier (`tier`)."""

    name: ClassVar[str] = "llamaextract"
    keys: ClassVar[tuple[str, ...]] = ("LLAMA_CLOUD_API_KEY",)
    workers: ClassVar[int] = 3
    tier: str = dataclasses.field(
        default=TIER,
        metadata={"choices": ["cost_effective", "agentic", "agentic_plus"],
                  "help": "extraction tier; entitlements change, so check one against the API"})
    poll_interval: int = 5

    def missing_keys(self) -> list[str]:
        """LLAMAPARSE_API_KEY, the key's older name, stands in for LLAMA_CLOUD_API_KEY."""
        return [] if os.environ.get("LLAMAPARSE_API_KEY") else super().missing_keys()

    def prepare_schema(self, schema: dict) -> dict:
        """The schema the v2 API accepts: `$ref`s inlined, `$` metadata dropped, enums typed and
        `additionalProperties` a boolean (`to_typed_enum_dialect`). What the schema declares nullable is
        sent nullable."""
        return to_typed_enum_dialect(
            drop_schema_metadata(field_matching.resolve_refs(schema, max_depth=SCHEMA.MAX_REF_DEPTH)))

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """Uploads, submits a v2 extract job at `tier`, and polls it."""
        deadline = adapter.Budget(timeout)
        key = os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ["LLAMAPARSE_API_KEY"]

        project_id = _project_id(key)
        file_id = _upload(pdf, key)
        job = _req("POST", f"{BASE}/api/v2/extract?project_id={project_id}", key,
                   headers={"Content-Type": "application/json"},
                   data=json.dumps({
                       "file_input": file_id,
                       "configuration": {"tier": self.tier, "extraction_target": "per_doc",
                                         "data_schema": schema},
                   }).encode())
        with adapter.reading("the job's id"):
            job_id = job.get("id") or job.get("job_id")

        polls = 0
        retry = adapter.PollRetry(deadline)
        while True:
            deadline.check(f"job {job_id} was still running server-side after {polls} polls")
            try:
                body = _req("GET", f"{BASE}/api/v2/extract/{job_id}?project_id={project_id}&expand=metadata", key)
            except errors.SystemFailed as exc:
                if retry.again(exc.status):
                    continue
                raise
            retry.reset()
            polls += 1
            with adapter.reading("the job's status"):
                status = body.get("status")
            if status in _TERMINAL:
                break
            time.sleep(self.poll_interval)

        with adapter.reading("the job's outcome and timing"):
            message = body.get("error_message")
            start, end = _dt(body.get("created_at")), _dt(body.get("updated_at"))
        if status in ("FAILED", "ERROR", "CANCELLED"):
            raise errors.SystemFailed(f"LlamaExtract {status}: {message}", status=200)
        return adapter.Call(
            raw={**body, "tier": self.tier,
                 "server_latency_s": round((end - start).total_seconds(), 2) if start and end else None},
            job_id=job_id)

    def read(self, raw: Any, q: contract.Question) -> dict:
        """The extraction in `extract_result`, `data` or `result`, whichever the job came back with."""
        r = raw if isinstance(raw, dict) else {}
        return contract.extraction(r.get("extract_result") or r.get("data") or r.get("result"))
