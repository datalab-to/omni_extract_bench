"""Mistral OCR adapter: one `/v1/ocr` call with a `document_annotation_format` schema.

The PDF goes inline as a base64 data URL and the extraction comes back as
`document_annotation`, a JSON string. Single call, no polling.

Auth: MISTRAL_API_KEY.

    python -m omni_extract_bench.harness.providers.mistral --pdf doc.pdf --schema s.json --out out.json
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx

from ..extraction import Cost, Extraction, MissingCredential, VendorError
from ._cli import run_cli

MODEL = "mistral-ocr-latest"
URL = "https://api.mistral.ai/v1/ocr"


def extract(pdf: Path, schema: dict, *, timeout: float = 1800.0) -> Extraction:
    """One POST, one response, one parse.

    No polling, so the document's budget IS this request's timeout -- there is no second phase
    to share it with. Note that httpx applies it per socket operation rather than to total
    elapsed time, so a server trickling bytes could outlast it; for a single call that is the
    closest an HTTP client gets to a wall-clock cap.
    """
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise MissingCredential("MISTRAL_API_KEY is not set")

    b64 = base64.b64encode(pdf.read_bytes()).decode()
    body = {
        "model": MODEL,
        "document": {"type": "document_url",
                     "document_url": f"data:application/pdf;base64,{b64}"},
        "document_annotation_format": {
            "type": "json_schema",
            "json_schema": {"name": "extraction", "schema": schema, "strict": False}},
    }
    r = httpx.post(URL, headers={"Authorization": f"Bearer {key}",
                                 "Content-Type": "application/json"},
                   json=body, timeout=timeout)
    if r.status_code >= 400:
        raise VendorError(f"HTTP {r.status_code}: {r.text[:300]}",
                          status=r.status_code, body=r.text)

    payload = r.json()
    # Everything except the extraction itself: the vendor's own usage block, which is where
    # the cost is, kept as `raw` so a parsing mistake here is re-read rather than re-paid for.
    annotation = payload.get("document_annotation")
    if annotation is None:
        raise VendorError("200 with no document_annotation", status=200, body=r.text[:300])
    try:
        parsed = json.loads(annotation) if isinstance(annotation, str) else annotation
    except json.JSONDecodeError as exc:
        raise VendorError(f"document_annotation is not JSON: {exc}",
                          status=200, body=str(annotation)[:300]) from None
    if not isinstance(parsed, dict) or not parsed:
        raise VendorError("document_annotation was empty", status=200, body=r.text[:300])

    usage = payload.get("usage_info") or {}
    return Extraction(
        result=parsed,
        raw={k: v for k, v in payload.items() if k != "document_annotation"},
        cost=Cost.reported(usage.get("cost"), "usage_info.cost"))


def main() -> None:
    run_cli(extract, "mistral")


if __name__ == "__main__":
    main()
