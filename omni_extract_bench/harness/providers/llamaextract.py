#!/usr/bin/env python3
"""LlamaExtract (LlamaCloud) adapter: uploads the PDF's bytes (never a URL, so no source domain reaches the
vendor), submits a v2 extraction job at its highest tier, `agentic_plus`, and polls it to the end.

Auth: LLAMA_CLOUD_API_KEY (llx-...).

Usage:
    oeb run --system llamaextract

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
    """The `type` an enum needs: the type of each member, `"null"` included.

    A list when the members are of more than one type, which is what keeps a nullable enum
    nullable -- `["MILD", "MODERATE", null]` is `["string", "null"]`, not `"string"` with the
    null quietly dropped.
    """
    names = sorted({JSON_TYPE_NAMES[type(m)] for m in members
                    if m is not None and type(m) in JSON_TYPE_NAMES})
    if not names:
        names = ["string"]
    if None in members:
        names.append("null")
    return names[0] if len(names) == 1 else names


def to_typed_enum_dialect(node):
    """Reshape for a vendor that requires every enum to declare a matching type.

    ``{"enum": ["mild", "moderate", null]}`` is valid JSON Schema and rejected here for having
    no ``type``. The type is inferred from the enum's own values, the null among them, so the
    field stays as nullable as it was declared: a benchmark that quietly forbids null asks this
    vendor a different question than the schema asks, and then scores the answer as if it had
    not. Unions are recursed into rather than collapsed, for the same reason. Also reduces
    ``additionalProperties`` from a schema to a boolean, which some validators require; the map
    stays open, only the per-value constraint is lost.
    """
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
    """Remove `$`-prefixed annotations -- `$schema`, `$id`, `$comment`.

    `resolve_refs` consumes `$ref` and `$defs`; these are what is left, and they describe the
    document rather than the data. Several validators reject them as unknown keys.
    """
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
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")[:600]
        except Exception:  # noqa: BLE001
            body = "<body unavailable>"
        raise errors.VendorError(f"HTTP {e.code} {method} {url.split('?')[0]}: {body}",
                                 status=e.code, body=body) from None


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
    """LlamaExtract's v2 stateless extraction, at its highest tier."""

    name: ClassVar[str] = "llamaextract"
    keys: ClassVar[tuple[str, ...]] = ("LLAMA_CLOUD_API_KEY",)
    workers: ClassVar[int] = 3
    tier: str = dataclasses.field(
        default=TIER,
        metadata={"choices": ["cost_effective", "agentic", "agentic_plus"],
                  "help": "extraction tier; entitlements change, so check one against the API"})
    poll_interval: int = 5

    def missing_keys(self) -> list[str]:
        """LLAMAPARSE_API_KEY, the older name of the same key, stands in for LLAMA_CLOUD_API_KEY."""
        return [] if os.environ.get("LLAMAPARSE_API_KEY") else super().missing_keys()

    def prepare_schema(self, schema: dict) -> dict:
        """Inline $refs, drop $-prefixed metadata, and reduce to the subset the v2 API accepts.

        Two constraints, each learned from a rejection, and both are what `to_typed_enum_dialect`
        encodes:

          * a typeless enum is rejected ("Invalid type for field"), so the type is inferred from
            the enum's own members -- `null` included, rather than stripped.
          * `additionalProperties` given as a schema rather than a boolean is rejected ("Input should be a
            valid boolean").

        What the schema declares as nullable is sent as nullable.
        """
        return to_typed_enum_dialect(
            drop_schema_metadata(field_matching.resolve_refs(schema, max_depth=SCHEMA.MAX_REF_DEPTH)))

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """Upload, submit a v2 extract job at `tier`, poll it to a terminal state.

        `tier` is an option rather than an environment variable because it is a parity decision:
        the harness runs every vendor at its maximum, and a setting that reads itself out of the
        environment is one a run cannot state in its own record.
        """
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
            except errors.VendorError as exc:
                if retry.again(exc.status):
                    continue
                raise
            except (urllib.error.URLError, OSError, ValueError) as exc:
                if retry.again():
                    continue
                raise errors.VendorError(f"job {job_id}: polling failed repeatedly: {exc}"[:300], status=None) from None
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
            raise errors.VendorError(f"LlamaExtract {status}: {message}", status=200)
        return adapter.Call(
            raw={**body, "tier": self.tier,
                 "server_latency_s": round((end - start).total_seconds(), 2) if start and end else None},
            job_id=job_id)

    def read(self, raw: Any, q: contract.Question) -> dict:
        """The extraction in `extract_result`, `data` or `result`, whichever the job came back with."""
        r = raw if isinstance(raw, dict) else {}
        return contract.extraction(r.get("extract_result") or r.get("data") or r.get("result"))
