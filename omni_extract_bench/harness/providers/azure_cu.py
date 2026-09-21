"""Azure AI Content Understanding adapter.

Content Understanding is analyzer-based: a JSON Schema becomes a `fieldSchema`, the analyzer is
created once per schema (cached by schema hash, because creating one per document would both be
slow and litter the resource), then each document is analysed against it.

Two things worth knowing before comparing its numbers with anyone else's:

  * the `completion` model is a deployment choice, not a product tier. `gpt-4.1-mini` and
    `gpt-4.1` are different systems behind the same API, so which one was used has to be
    recorded and published beside the score -- see `--completion-model`.
  * its field types are a smaller set than JSON Schema's. Nested objects become `object`,
    arrays of objects become `array` of `object`, and anything else degrades to `string`.
    That is the vendor's surface, not a benchmark choice, but it means a schema this adapter
    sends is a lossier statement of the task than the one other vendors receive.

Auth: AZURE_CU_ENDPOINT + AZURE_CU_KEY.

    python -m omni_extract_bench.harness.providers.azure_cu --pdf doc.pdf --schema s.json --out out.json
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import httpx

from ..extraction import (Budget, Cost, Extraction, MissingCredential, PollRetry,
                          VendorError)
from ._cli import run_cli

API_VERSION = "2025-05-01-preview"
DEFAULT_COMPLETION_MODEL = "gpt-4.1-mini"
_TERMINAL_OK = {"succeeded", "completed"}
_TERMINAL_BAD = {"failed", "cancelled"}


@dataclasses.dataclass(frozen=True)
class Config:
    """What azure-cu can be asked.

    `completion_model` is a DEPLOYMENT CHOICE, not a product tier: `gpt-4.1-mini` and `gpt-4.1`
    are different systems behind one API, so which one ran has to be published beside the score.
    """

    completion_model: str = dataclasses.field(
        default=DEFAULT_COMPLETION_MODEL,
        metadata={"help": "the deployment behind the analyzer; publish it with the score"})
    api_version: str = API_VERSION
    poll_interval: float = 3.0


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


_ANALYZERS: dict[str, str] = {}
_ANALYZER_LOCK = threading.Lock()


def _ensure_analyzer(client, endpoint: str, api_version: str, digest: str, analyzer_id: str,
                     schema: dict, completion_model: str, budget, poll_interval: float) -> None:
    """Create the analyzer for this schema once per process, not once per thread.

    UNDER THE LOCK FOR THE WHOLE CREATE-AND-WAIT, not just the cache lookup. A 409 says the
    analyzer EXISTS, not that it is READY -- so a thread that lost the race would skip the
    readiness wait below and analyse against an analyzer still provisioning. Checking the cache
    without holding anything is what let several threads reach the PUT at once, and widening
    `--predict-workers` makes that likelier rather than rarer.

    Threads that block here are waiting for something they need anyway, and the wait is bounded
    by the holding document's own budget.
    """
    with _ANALYZER_LOCK:
        if digest in _ANALYZERS:
            return
        r = client.put(
            f"{endpoint}/contentunderstanding/analyzers/{analyzer_id}"
            f"?api-version={api_version}",
            json={"baseAnalyzerId": "prebuilt-documentAnalyzer",
                  "config": {"returnDetails": False, "completion": completion_model},
                  "fieldSchema": field_schema(schema)})
        if r.status_code != 409:
            if r.status_code >= 400:
                raise VendorError(f"creating analyzer: HTTP {r.status_code}: {r.text[:300]}",
                                  status=r.status_code, body=r.text)
            if r.headers.get("Operation-Location"):
                _await(client, r.headers["Operation-Location"], budget=budget,
                       poll_interval=poll_interval, want_result=False)
        _ANALYZERS[digest] = analyzer_id


def _await(client, op_url: str, *, budget, poll_interval: float, want_result: bool):
    """Poll one operation. Takes the DOCUMENT's budget, not a fresh timeout: this is
    called twice per document -- once for the analyzer, once for the analysis -- and with
    a timeout each it gave azure-cu two full budgets where every other vendor got one."""
    polls = 0
    retry = PollRetry(budget)
    while True:
        budget.check(f"still analysing after {polls} polls")
        try:
            r = client.get(op_url)
        except httpx.TransportError as exc:
            if retry.again():
                continue
            raise VendorError(f"polling failed: {exc}"[:300], status=None) from None
        if r.status_code >= 400:
            if retry.again(r.status_code):
                continue
            raise VendorError(f"HTTP {r.status_code}: {r.text[:300]}",
                              status=r.status_code, body=r.text)
        retry.ok()
        polls += 1
        body = r.json()
        status = str(body.get("status", "")).lower()
        if status in _TERMINAL_OK:
            return body if want_result else True
        if status in _TERMINAL_BAD:
            raise VendorError(f"azure-cu {status}: {json.dumps(body.get('error') or {})[:300]}",
                              status=200, body=r.text)
        time.sleep(poll_interval)


def extract(pdf: Path, schema: dict, *, timeout: float = 1800.0,
            config: Config = Config()) -> Extraction:
    """Create (or reuse) an analyzer for this schema, analyse the document, poll for the result.

    Azure does not report a per-call cost, so `cost.usd` is None and the record says
    `billed_out_of_band` -- rather than inventing a figure from a price list.
    """
    endpoint = os.environ.get("AZURE_CU_ENDPOINT")
    key = os.environ.get("AZURE_CU_KEY")
    if not endpoint or not key:
        raise MissingCredential("AZURE_CU_ENDPOINT and AZURE_CU_KEY must be set")
    endpoint = endpoint.rstrip("/")
    budget = Budget(timeout)

    with httpx.Client(headers={"Ocp-Apim-Subscription-Key": key}, timeout=120) as client:
        digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:16]
        analyzer_id = f"oeb-{digest}"
        _ensure_analyzer(client, endpoint, config.api_version, digest, analyzer_id, schema,
                         config.completion_model, budget, config.poll_interval)
        r = client.post(f"{endpoint}/contentunderstanding/analyzers/{analyzer_id}:analyze"
                        f"?api-version={config.api_version}",
                        content=pdf.read_bytes(),
                        headers={"Content-Type": "application/octet-stream"})
        if r.status_code >= 400:
            raise VendorError(f"HTTP {r.status_code}: {r.text[:300]}",
                              status=r.status_code, body=r.text)
        op = r.headers.get("Operation-Location")
        if not op:
            raise VendorError("no Operation-Location header on :analyze",
                              status=r.status_code, body=r.text)

        body = _await(client, op, budget=budget, poll_interval=config.poll_interval,
                      want_result=True)

    contents = ((body or {}).get("result") or {}).get("contents") or []
    if not contents:
        raise VendorError("analysis returned no contents", status=200,
                          body=json.dumps(body)[:300])
    return Extraction(result=fields_to_dict(contents[0].get("fields") or {}),
                      raw=body,
                      cost=Cost(),
                      job_id=analyzer_id)


def main() -> None:
    run_cli(extract, Config, "azure-cu")


if __name__ == "__main__":
    main()
