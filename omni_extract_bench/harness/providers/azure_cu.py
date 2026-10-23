"""Azure AI Content Understanding: a JSON Schema becomes an analyzer, created once per schema, and each
document is analysed against it. Auth: AZURE_CU_KEY. `endpoint` is a required option:

    oeb init azure-cu > azure-cu.yaml
    oeb run --config azure-cu.yaml --set system.azure-cu.endpoint=https://<resource>.cognitiveservices.azure.com"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import httpx
from omni_bench_core import adapter, errors

from .. import contract

API_VERSION = "2025-05-01-preview"
DEFAULT_COMPLETION_MODEL = "gpt-4.1-mini"
_TERMINAL_OK = {"succeeded", "completed"}
_TERMINAL_BAD = {"failed", "cancelled"}


def _field(prop: dict) -> dict:
    """One JSON Schema property as a Content Understanding field definition."""
    prop = prop or {}
    t = prop.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    desc = prop.get("description") or ""
    if t == "object":
        return {"type": "object", "description": desc,
                "properties": {k: _field(v) for k, v in (prop.get("properties") or {}).items()}}
    if t == "array":
        items = prop.get("items") or {}
        it = items.get("type")
        if isinstance(it, list):
            it = next((x for x in it if x != "null"), "string")
        if it == "object":
            return {"type": "array", "description": desc,
                    "items": {"type": "object",
                              "properties": {k: _field(v) for k, v in (items.get("properties") or {}).items()}}}
        return {"type": "array", "description": desc, "items": {"type": "string"}}
    if t in ("number", "integer"):
        return {"type": "number", "description": desc}
    if t == "boolean":
        return {"type": "boolean", "description": desc}
    return {"type": "string", "description": desc}


def field_schema(schema: dict) -> dict:
    """The schema as Azure's `fieldSchema`, which isn't a JSON Schema."""
    return {"fields": {k: _field(v) for k, v in ((schema or {}).get("properties") or {}).items()}}


def _value(node):
    """One Content Understanding value node as a plain value."""
    if not isinstance(node, dict):
        return node
    for key in ("valueString", "valueNumber", "valueInteger", "valueBoolean", "valueDate"):
        if key in node:
            return node[key]
    if "valueArray" in node:
        return [_value(x) for x in node["valueArray"] or []]
    if "valueObject" in node:
        return {k: _value(v) for k, v in (node["valueObject"] or {}).items()}
    if "content" in node:
        return node["content"]
    return None


def fields_to_dict(fields: dict) -> dict:
    return {k: _value(v) for k, v in (fields or {}).items()}


_ANALYZERS: dict[tuple[str, str], str] = {}
_ANALYZER_LOCK = threading.Lock()


def analyzer_id(fields: dict, completion_model: str, api_version: str) -> str:
    """The analyzer's name: a digest of everything it is built from. Creating an analyzer that exists
    returns 409, taken as success, so anything that changes the analyzer must change its name."""
    spelled = json.dumps({"fieldSchema": fields, "completion": completion_model,
                          "api_version": api_version}, sort_keys=True)
    return f"oeb-{hashlib.sha256(spelled.encode()).hexdigest()[:16]}"


def _ensure_analyzer(client, analyzer: str, fields: dict, config: AzureCU, deadline) -> None:
    """Creates the analyzer once per process. The cache is keyed by endpoint and analyzer, since an
    analyzer lives in one Azure resource. The lock is held through create-and-wait, because a 409 says
    the analyzer exists, not that it is ready."""
    with _ANALYZER_LOCK:
        if (config.endpoint, analyzer) in _ANALYZERS:
            return
        try:
            r = client.put(
                f"{config.endpoint}/contentunderstanding/analyzers/{analyzer}"
                f"?api-version={config.api_version}",
                json={"baseAnalyzerId": "prebuilt-documentAnalyzer",
                      "config": {"returnDetails": False,
                                 "completion": config.completion_model},
                      "fieldSchema": fields})
        except httpx.TransportError as exc:
            raise contract.unreachable("creating the analyzer", exc) from exc
        if r.status_code != 409:
            if r.status_code >= 400:
                raise errors.SystemFailed(f"creating analyzer: HTTP {r.status_code}: {r.text}",
                                          status=r.status_code, body=r.text)
            if r.headers.get("Operation-Location"):
                _await(client, r.headers["Operation-Location"], deadline=deadline,
                       poll_interval=config.poll_interval, want_result=False)
        _ANALYZERS[(config.endpoint, analyzer)] = analyzer


def _await(client, op_url: str, *, deadline, poll_interval: float, want_result: bool):
    """Polls one operation within the document's budget, which covers both the analyzer and the analysis."""
    polls = 0
    retry = adapter.PollRetry(deadline)
    while True:
        deadline.check(f"still analysing after {polls} polls")
        try:
            r = client.get(op_url)
        except httpx.TransportError as exc:
            if retry.again():
                continue
            raise contract.unreachable("polling failed", exc) from exc
        if r.status_code >= 400:
            if retry.again(r.status_code):
                continue
            raise errors.SystemFailed(f"HTTP {r.status_code}: {r.text}",
                                      status=r.status_code, body=r.text)
        retry.reset()
        polls += 1
        with adapter.reading("the analysis's status"):
            body = r.json()
            status = str(body.get("status", "")).lower()
        if status in _TERMINAL_OK:
            return body if want_result else True
        if status in _TERMINAL_BAD:
            raise errors.SystemFailed(f"azure-cu {status}: {json.dumps(body.get('error') or {})}",
                                      status=200, body=r.text)
        time.sleep(poll_interval)


@dataclasses.dataclass(frozen=True)
class AzureCU(contract.Extractor):
    """Azure AI Content Understanding. Options:

    - `endpoint`: the Azure resource to analyse against (required; there is no public default);
    - `completion_model`: the deployment behind the analyzer, a different system per model, so publish
      it with the score; and
    - `api_version` and `poll_interval`.

    Its field types are fewer than JSON Schema's (nested objects, arrays of objects, and strings), so it
    is sent a less exact statement of the task than other vendors. Azure reports no per-call cost."""

    name: ClassVar[str] = "azure-cu"
    keys: ClassVar[tuple[str, ...]] = ("AZURE_CU_KEY",)
    workers: ClassVar[int] = 3
    endpoint: str = dataclasses.field(
        default="", metadata={"help": "(required) the Azure resource to analyse against, e.g. "
                                      "https://<resource>.cognitiveservices.azure.com"})
    completion_model: str = dataclasses.field(
        default=DEFAULT_COMPLETION_MODEL,
        metadata={"help": "the deployment behind the analyzer; publish it with the score"})
    api_version: str = API_VERSION
    poll_interval: float = 3.0

    def __post_init__(self):
        """Drops the endpoint's trailing slash, so the options hold what the URLs use. An empty one is refused at the
        first call, not here, so `oeb init azure-cu` writes a config to fill in."""
        object.__setattr__(self, "endpoint", self.endpoint.rstrip("/"))

    def prepare_schema(self, schema: dict) -> dict:
        return field_schema(schema)

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """Creates (or reuses) the analyzer for this field schema, then analyses the document."""
        endpoint = self.endpoint
        if not endpoint.strip():
            raise ValueError("azure-cu has no endpoint, the Azure resource to analyse against; give it: --set "
                             "system.azure-cu.endpoint=https://<resource>.cognitiveservices.azure.com")
        key = os.environ["AZURE_CU_KEY"]
        deadline = adapter.Budget(timeout)

        with httpx.Client(headers={"Ocp-Apim-Subscription-Key": key}, timeout=120) as client:
            analyzer = analyzer_id(schema, self.completion_model, self.api_version)
            _ensure_analyzer(client, analyzer, schema, self, deadline)
            try:
                r = client.post(f"{endpoint}/contentunderstanding/analyzers/{analyzer}:analyze"
                                f"?api-version={self.api_version}",
                                content=pdf.read_bytes(), headers={"Content-Type": "application/octet-stream"})
            except httpx.TransportError as exc:
                raise contract.unreachable("submitting the document", exc) from exc
            if r.status_code >= 400:
                raise errors.SystemFailed(f"HTTP {r.status_code}: {r.text}", status=r.status_code, body=r.text)
            op = r.headers.get("Operation-Location")
            if not op:
                raise errors.SystemFailed("no Operation-Location header on :analyze", status=r.status_code, body=r.text)
            body = _await(client, op, deadline=deadline, poll_interval=self.poll_interval, want_result=True)
        with adapter.reading("the analysis's contents"):
            done = bool(((body or {}).get("result") or {}).get("contents"))
        if not done:
            raise errors.SystemFailed("analysis returned no contents", status=200, body=json.dumps(body))
        return adapter.Call(raw=body, job_id=analyzer)

    def read(self, raw: Any, q: contract.Question) -> dict:
        """The first content's fields, as plain values."""
        with adapter.reading("the first content's fields"):
            contents = ((raw or {}).get("result") or {}).get("contents") or []
            if not contents:
                raise errors.SystemFailed("analysis returned no contents")
            return fields_to_dict(contents[0].get("fields") or {})
