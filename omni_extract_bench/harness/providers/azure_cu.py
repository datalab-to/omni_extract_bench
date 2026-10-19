"""Azure AI Content Understanding adapter. A JSON Schema becomes an analyzer, created once per schema, and
each document is analysed against it. `AzureCU` says what to know before comparing its numbers.

Auth: AZURE_CU_KEY. The `endpoint` is an option, not a credential.

    oeb run --providers azure-cu --options '{"azure-cu": {"endpoint": "https://<resource>.cognitiveservices.azure.com"}}'
"""
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
    """One JSON-Schema property as a Content Understanding field definition."""
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
    """Azure takes a `fieldSchema`, not a JSON Schema, so this vendor's payload is not a schema at all."""
    return {"fields": {k: _field(v) for k, v in ((schema or {}).get("properties") or {}).items()}}


def _value(node):
    """Unwrap one Content Understanding value node to a plain Python value."""
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
    """The analyzer's name: a digest of everything the analyzer is built from.

    An analyzer is an object on Azure, reused by name, and a 409 when it is created means it already exists,
    which this treats as success. So anything that changes what the analyzer is must change its name, or the
    409 hands back a different analyzer. Named after the field schema alone, a run with `gpt-4.1` once reused
    the `gpt-4.1-mini` analyzer, while its record said `gpt-4.1`.
    """
    spelled = json.dumps({"fieldSchema": fields, "completion": completion_model,
                          "api_version": api_version}, sort_keys=True)
    return f"oeb-{hashlib.sha256(spelled.encode()).hexdigest()[:16]}"


def _ensure_analyzer(client, analyzer: str, fields: dict, config: AzureCU, deadline) -> None:
    """Create the analyzer once per process, not once per thread.

    The cache is keyed by endpoint and analyzer, because the cache is shared by the whole process and an
    analyzer lives inside one Azure resource. Keyed by the analyzer alone, a second run on another endpoint
    skipped creating its own and then failed with a 404 on every document.

    The lock is held for the whole create-and-wait, not only the cache lookup. A 409 says the analyzer exists,
    not that it is ready, so a thread that lost the race would skip the wait below and analyse against an
    analyzer still being set up. More `--workers` make that more likely, not less.

    Threads that block here are waiting for something they need anyway, and the wait is bounded
    by the holding document's own budget.
    """
    with _ANALYZER_LOCK:
        if (config.endpoint, analyzer) in _ANALYZERS:
            return
        r = client.put(
            f"{config.endpoint}/contentunderstanding/analyzers/{analyzer}"
            f"?api-version={config.api_version}",
            json={"baseAnalyzerId": "prebuilt-documentAnalyzer",
                  "config": {"returnDetails": False,
                             "completion": config.completion_model},
                  "fieldSchema": fields})
        if r.status_code != 409:
            if r.status_code >= 400:
                raise errors.VendorError(f"creating analyzer: HTTP {r.status_code}: {r.text[:300]}",
                                         status=r.status_code, body=r.text)
            if r.headers.get("Operation-Location"):
                _await(client, r.headers["Operation-Location"], deadline=deadline,
                       poll_interval=config.poll_interval, want_result=False)
        _ANALYZERS[(config.endpoint, analyzer)] = analyzer


def _await(client, op_url: str, *, deadline, poll_interval: float, want_result: bool):
    """Polls one operation, within the document's budget rather than a fresh timeout. It is called twice per
    document, once for the analyzer and once for the analysis; with a timeout each, azure-cu got two full
    budgets where every other vendor got one."""
    polls = 0
    retry = adapter.PollRetry(deadline)
    while True:
        deadline.check(f"still analysing after {polls} polls")
        try:
            r = client.get(op_url)
        except httpx.TransportError as exc:
            if retry.again():
                continue
            raise errors.VendorUnreachable(f"polling failed: {exc}"[:300]) from None
        if r.status_code >= 400:
            if retry.again(r.status_code):
                continue
            raise errors.VendorError(f"HTTP {r.status_code}: {r.text[:300]}",
                                     status=r.status_code, body=r.text)
        retry.reset()
        polls += 1
        with adapter.reading("the analysis's status"):
            body = r.json()
            status = str(body.get("status", "")).lower()
        if status in _TERMINAL_OK:
            return body if want_result else True
        if status in _TERMINAL_BAD:
            raise errors.VendorError(f"azure-cu {status}: {json.dumps(body.get('error') or {})[:300]}",
                                     status=200, body=r.text)
        time.sleep(poll_interval)


@dataclasses.dataclass(frozen=True)
class AzureCU(contract.Extractor):
    """Azure AI Content Understanding.

    `completion_model` is a deployment choice, not a product tier: `gpt-4.1-mini` and `gpt-4.1` are different
    systems behind one API, so which one ran has to be published beside the score.

    Its field types are a smaller set than JSON Schema's: nested objects become `object`, arrays of objects
    become `array` of `object`, and anything else becomes `string`. So the schema it is sent is a less exact
    statement of the task than the one other vendors receive.

    `endpoint` is an option for the same reason: it selects an Azure resource and region, and
    two runs against different deployments used to produce records nothing could tell apart.
    There is no public Azure endpoint to default to, so the default is empty and `extract` says
    what to pass. The key stays in the environment: it is a credential and belongs in no record.
    """

    name: ClassVar[str] = "azure-cu"
    workers: ClassVar[int] = 3
    endpoint: str = dataclasses.field(
        default="", metadata={"help": "the Azure resource to analyse against, e.g. "
                                      "https://<resource>.cognitiveservices.azure.com"})
    completion_model: str = dataclasses.field(
        default=DEFAULT_COMPLETION_MODEL,
        metadata={"help": "the deployment behind the analyzer; publish it with the score"})
    api_version: str = API_VERSION
    poll_interval: float = 3.0

    def __post_init__(self):
        """Normalise here, so the options hold what the URLs are actually built from."""
        object.__setattr__(self, "endpoint", self.endpoint.rstrip("/"))

    def prepare_schema(self, schema: dict) -> dict:
        return field_schema(schema)

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """Create (or reuse) an analyzer for this field schema, analyse the document, poll for it.

        Azure does not report a per-call cost, so `cost.usd` is None rather than a figure invented
        from a price list.
        """
        endpoint = self.endpoint
        key = os.environ.get("AZURE_CU_KEY")
        if not endpoint or not key:
            raise errors.MissingCredential(
                "azure-cu needs AZURE_CU_KEY in the environment and an endpoint in its options:\n"
                "    --options '{\"azure-cu\": {\"endpoint\": "
                "\"https://<resource>.cognitiveservices.azure.com\"}}'")
        deadline = adapter.Budget(timeout)

        with httpx.Client(headers={"Ocp-Apim-Subscription-Key": key}, timeout=120) as client:
            analyzer = analyzer_id(schema, self.completion_model, self.api_version)
            _ensure_analyzer(client, analyzer, schema, self, deadline)
            r = client.post(f"{endpoint}/contentunderstanding/analyzers/{analyzer}:analyze"
                            f"?api-version={self.api_version}",
                            content=pdf.read_bytes(), headers={"Content-Type": "application/octet-stream"})
            if r.status_code >= 400:
                raise errors.VendorError(f"HTTP {r.status_code}: {r.text[:300]}", status=r.status_code, body=r.text)
            op = r.headers.get("Operation-Location")
            if not op:
                raise errors.VendorError("no Operation-Location header on :analyze", status=r.status_code, body=r.text)
            body = _await(client, op, deadline=deadline, poll_interval=self.poll_interval, want_result=True)
        with adapter.reading("the analysis's contents"):
            done = bool(((body or {}).get("result") or {}).get("contents"))
        if not done:
            raise errors.VendorError("analysis returned no contents", status=200, body=json.dumps(body)[:300])
        return adapter.Call(raw=body, job_id=analyzer)

    def read(self, raw: Any, q: contract.Question) -> dict:
        """The first content's fields, unwrapped to plain values."""
        contents = ((raw or {}).get("result") or {}).get("contents") or []
        if not contents:
            raise ValueError("analysis returned no contents")
        return fields_to_dict(contents[0].get("fields") or {})
