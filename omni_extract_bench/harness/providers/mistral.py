"""Mistral OCR adapter: one `/v1/ocr` call with a `document_annotation_format` schema.

The PDF goes inline as a base64 data URL and the extraction comes back as
`document_annotation`, a JSON string. Single call, no polling.

Auth: MISTRAL_API_KEY.

    oeb run --system mistral
"""
from __future__ import annotations

import base64
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, ClassVar

import httpx
from omni_bench_core import adapter, errors

from .. import contract

MODEL = "mistral-ocr-latest"
URL = "https://api.mistral.ai/v1/ocr"


@dataclasses.dataclass(frozen=True)
class Mistral(contract.Extractor):
    """Mistral OCR. Nothing to steer: the document and the schema go in one call and that is all."""

    name: ClassVar[str] = "mistral"
    keys: ClassVar[tuple[str, ...]] = ("MISTRAL_API_KEY",)
    workers: ClassVar[int] = 5

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """One POST, one response.

        No polling, so the document's budget IS this request's timeout. httpx applies it per socket
        operation rather than to total elapsed time, so a server trickling bytes could outlast it; for a
        single call that is the closest an HTTP client gets to a wall-clock cap.
        """
        key = os.environ["MISTRAL_API_KEY"]

        b64 = base64.b64encode(pdf.read_bytes()).decode()
        body = {
            "model": MODEL,
            "document": {"type": "document_url", "document_url": f"data:application/pdf;base64,{b64}"},
            "document_annotation_format": {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": schema, "strict": False}},
        }
        r = httpx.post(URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                       json=body, timeout=timeout)
        if r.status_code >= 400:
            raise errors.VendorError(f"HTTP {r.status_code}: {r.text[:300]}", status=r.status_code, body=r.text)
        with adapter.reading("the response and its cost"):
            payload = r.json()
            usage = payload.get("usage_info") or {}
            cost = adapter.Cost.reported(usage.get("cost"), "usage_info.cost")
        return adapter.Call(raw=payload, cost=cost)

    def read(self, raw: Any, q: contract.Question) -> dict:
        """The extraction in `document_annotation`, a JSON string."""
        annotation = raw.get("document_annotation") if isinstance(raw, dict) else None
        if annotation is None:
            raise ValueError("the response has no document_annotation")
        try:
            parsed = json.loads(annotation) if isinstance(annotation, str) else annotation
        except json.JSONDecodeError as exc:
            raise ValueError(f"document_annotation is not JSON: {exc}") from None
        return contract.extraction(parsed)
