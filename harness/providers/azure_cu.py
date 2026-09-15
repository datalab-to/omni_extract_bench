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

    python -m harness.providers.azure_cu --pdf doc.pdf --schema s.json --out out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx

from .envelope import write_output

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


class AzureContentUnderstanding:
    def __init__(self, endpoint: str, key: str, *, completion_model: str = DEFAULT_COMPLETION_MODEL,
                 api_version: str = API_VERSION, poll_interval: float = 3, timeout: float = 1800):
        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.completion_model = completion_model
        self.api_version = api_version
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._analyzers: dict[str, str] = {}
        self.client = httpx.Client(headers={"Ocp-Apim-Subscription-Key": key}, timeout=120)

    # -- analyzers -----------------------------------------------------------------
    def analyzer_for(self, schema: dict) -> str:
        h = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:16]
        if h in self._analyzers:
            return self._analyzers[h]
        analyzer_id = f"oeb-{h}"
        url = f"{self.endpoint}/contentunderstanding/analyzers/{analyzer_id}?api-version={self.api_version}"
        body = {"baseAnalyzerId": "prebuilt-documentAnalyzer",
                "config": {"returnDetails": False, "completion": self.completion_model},
                "fieldSchema": field_schema(schema)}
        r = self.client.put(url, json=body)
        if r.status_code == 409:                       # already exists: reuse it
            self._analyzers[h] = analyzer_id
            return analyzer_id
        r.raise_for_status()
        op = r.headers.get("Operation-Location")
        if op:
            self._await(op, want_result=False)
        self._analyzers[h] = analyzer_id
        return analyzer_id

    # -- analysis ------------------------------------------------------------------
    def __call__(self, pdf: Path, schema: dict) -> dict:
        analyzer_id = self.analyzer_for(schema)
        url = (f"{self.endpoint}/contentunderstanding/analyzers/{analyzer_id}:analyze"
               f"?api-version={self.api_version}")
        r = self.client.post(url, content=pdf.read_bytes(),
                             headers={"Content-Type": "application/octet-stream"})
        if r.status_code >= 400:
            return {"__error__": f"HTTP {r.status_code}: {r.text[:300]}"}
        op = r.headers.get("Operation-Location")
        if not op:
            return {"__error__": "no Operation-Location header on :analyze"}
        body = self._await(op, want_result=True)
        if isinstance(body, dict) and "__error__" in body:
            return body
        contents = ((body or {}).get("result") or {}).get("contents") or []
        if not contents:
            return {"__error__": "analysis returned no contents"}
        return fields_to_dict(contents[0].get("fields") or {})

    def _await(self, op_url: str, *, want_result: bool):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            r = self.client.get(op_url)
            if r.status_code >= 400:
                return {"__error__": f"HTTP {r.status_code}: {r.text[:300]}"}
            body = r.json()
            status = str(body.get("status", "")).lower()
            if status in _TERMINAL_OK:
                return body if want_result else True
            if status in _TERMINAL_BAD:
                return {"__error__": f"azure-cu {status}: {json.dumps(body.get('error') or {})[:300]}"}
            time.sleep(self.poll_interval)
        return {"__error__": f"timeout after {self.timeout}s while polling"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--completion-model", default=os.environ.get("AZURE_CU_COMPLETION_MODEL", DEFAULT_COMPLETION_MODEL))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("OEB_TIMEOUT", 1800)))
    a = ap.parse_args()
    endpoint, key = os.environ.get("AZURE_CU_ENDPOINT"), os.environ.get("AZURE_CU_KEY")
    if not endpoint or not key:
        raise SystemExit("AZURE_CU_ENDPOINT and AZURE_CU_KEY must be set")
    p = AzureContentUnderstanding(endpoint, key, completion_model=a.completion_model, timeout=a.timeout)
    t0 = time.time()
    try:
        result = p(a.pdf, json.loads(a.schema.read_text()))
    except Exception as exc:  # noqa: BLE001 -- the runner records the message as the result
        result = {"__error__": f"{type(exc).__name__}: {exc}"[:600]}
    write_output(a.out, provider="azure-cu", result=result, latency_s=time.time() - t0,
                 usage={"completion_model": a.completion_model, "api_version": a.api_version})
    if isinstance(result, dict) and "__error__" in result:
        print(result["__error__"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
