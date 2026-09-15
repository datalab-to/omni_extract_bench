"""Mistral OCR adapter: one `/v1/ocr` call with a `document_annotation_format` schema.

The PDF goes inline as a base64 data URL and the extraction comes back as
`document_annotation`, a JSON string. Single call, no polling.

Auth: MISTRAL_API_KEY.

    python -m harness.providers.mistral --pdf doc.pdf --schema s.json --out out.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import httpx

from .envelope import write_output

MODEL = "mistral-ocr-latest"
URL = "https://api.mistral.ai/v1/ocr"


def extract(pdf: Path, schema: dict, *, api_key: str, timeout: float = 1800):
    b64 = base64.b64encode(pdf.read_bytes()).decode()
    body = {
        "model": MODEL,
        "document": {"type": "document_url",
                     "document_url": f"data:application/pdf;base64,{b64}"},
        "document_annotation_format": {
            "type": "json_schema",
            "json_schema": {"name": "extraction", "schema": schema, "strict": False}},
    }
    r = httpx.post(URL, headers={"Authorization": f"Bearer {api_key}",
                                 "Content-Type": "application/json"},
                   json=body, timeout=timeout)
    if r.status_code >= 400:
        return {"__error__": f"HTTP {r.status_code}: {r.text[:300]}"}, None
    payload = r.json()
    annotation = payload.get("document_annotation")
    usage = {k: v for k, v in payload.items() if k != "document_annotation"}
    if annotation is None:
        return {"__error__": "200 with no document_annotation"}, usage
    try:
        parsed = json.loads(annotation) if isinstance(annotation, str) else annotation
    except json.JSONDecodeError as e:
        return {"__error__": f"document_annotation is not JSON: {e}"}, usage
    return (parsed if isinstance(parsed, dict) and parsed
            else {"__error__": "document_annotation was empty"}), usage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("OEB_TIMEOUT", 1800)))
    a = ap.parse_args()
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit("MISTRAL_API_KEY must be set")
    t0 = time.time()
    result, usage = extract(a.pdf, json.loads(a.schema.read_text()), api_key=key, timeout=a.timeout)
    write_output(a.out, provider="mistral", result=result, latency_s=time.time() - t0,
                 usage=usage or {"model": MODEL})
    if isinstance(result, dict) and "__error__" in result:
        print(result["__error__"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
