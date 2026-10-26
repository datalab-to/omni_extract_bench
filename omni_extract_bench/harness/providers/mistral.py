"""Mistral OCR: one `/v1/ocr` call, with the PDF inline as a base64 data URL and the schema as
`document_annotation_format`; the extraction comes back as `document_annotation`, a JSON string. Auth:
MISTRAL_API_KEY.

    oeb init mistral > config.yaml
    oeb run --config config.yaml"""
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
    """Mistral OCR: the document and the schema in one call, with nothing to configure."""

    name: ClassVar[str] = "mistral"
    keys: ClassVar[tuple[str, ...]] = ("MISTRAL_API_KEY",)
    workers: ClassVar[int] = 5

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """One POST, with the document's budget as its timeout. httpx applies it per socket operation, not
        to the whole request, which is as close to a wall-clock cap as one HTTP call gets."""
        key = os.environ["MISTRAL_API_KEY"]

        b64 = base64.b64encode(pdf.read_bytes()).decode()
        body = {
            "model": MODEL,
            "document": {"type": "document_url", "document_url": f"data:application/pdf;base64,{b64}"},
            "document_annotation_format": {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": schema, "strict": False}},
        }
        try:
            r = httpx.post(URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                           json=body, timeout=timeout)
        except httpx.TransportError as exc:
            raise contract.unreachable("submitting the document", exc) from exc
        if r.status_code >= 400:
            raise errors.SystemFailed(f"HTTP {r.status_code}: {r.text}", status=r.status_code, body=r.text)
        with adapter.reading("the response and its cost"):
            payload = r.json()
            usage = payload.get("usage_info") or {}
            cost = adapter.Cost.reported(usage.get("cost"), "usage_info.cost")
        return adapter.Call(raw=payload, cost=cost)

    def read(self, raw: Any, q: contract.Question) -> dict:
        """The extraction in `document_annotation`, a JSON string."""
        annotation = raw.get("document_annotation") if isinstance(raw, dict) else None
        if annotation is None:
            raise errors.SystemFailed("the response has no document_annotation")
        with adapter.reading("document_annotation"):
            parsed = json.loads(annotation) if isinstance(annotation, str) else annotation
        return contract.extraction(parsed)
