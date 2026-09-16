"""Datalab API adapter: POST /api/v1/extract, then poll for the result.

`mode` selects the extraction tier (`fast`, `balanced`, `accurate`). The published runs used
`balanced` (the default product tier) and `accurate` (the most capable one) as two separate
legs, because publishing only whichever scored higher would misrepresent the product.

Cost comes back as `cost_breakdown.final_cost_cents` — CENTS, converted once here. Reporting
cents as dollars overstates cost by 100x, which is the kind of error a cost table never
recovers from.

Auth: DATALAB_API_KEY.

    python -m omni_extract_bench.harness.providers.datalab --pdf doc.pdf --schema s.json --out out.json \\
        --mode balanced
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

from .envelope import write_output

DEFAULT_BASE_URL = "https://www.datalab.to"


def normalize_schema(schema: dict) -> dict:
    """Collapse nullable unions and de-require the fields that were nullable.

    The API expresses optionality by omission from `required`, not by a `["string", "null"]`
    union, so a schema written the JSON-Schema way has to be restated. Field names, types and
    descriptions are untouched: this is a dialect translation, not a change to the task.
    """
    schema = dict(schema)
    nullable: set[str] = set()
    props = schema.get("properties")
    if isinstance(props, dict):
        for k, v in props.items():
            if isinstance(v, dict) and isinstance(v.get("type"), list) and "null" in v["type"]:
                nullable.add(k)
        schema["properties"] = {k: normalize_schema(v) if isinstance(v, dict) else v
                                for k, v in props.items()}
    if "required" in schema and nullable:
        schema["required"] = [r for r in schema["required"] if r not in nullable]
        if not schema["required"]:
            del schema["required"]
    if isinstance(schema.get("type"), list):
        non_null = [t for t in schema["type"] if t != "null"]
        schema["type"] = non_null[0] if non_null else "string"
    if isinstance(schema.get("items"), dict):
        schema["items"] = normalize_schema(schema["items"])
    return schema


class DatalabAPI:
    def __init__(self, *, api_key: str, base_url: str = DEFAULT_BASE_URL, mode: str = "balanced",
                 poll_interval: float = 5, timeout: float = 1800):
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["X-Api-Key"] = api_key
        self.cost_usd: float | None = None

    def __call__(self, pdf: Path, schema: dict):
        url = f"{self.base_url}/api/v1/extract"
        with pdf.open("rb") as fh:
            resp = self.session.post(
                url,
                files={"file": (pdf.name, fh, "application/pdf")},
                data={"page_schema": json.dumps(normalize_schema(schema)),
                      "extraction_mode": self.mode, "output_format": "json"},
                timeout=120)
        if resp.status_code != 200:
            return {"__error__": f"HTTP {resp.status_code}: {resp.text[:300]}"}
        request_id = resp.json().get("request_id")
        if not request_id:
            return {"__error__": f"no request_id: {resp.text[:200]}"}
        return self._poll(request_id)

    def _poll(self, request_id: str):
        url = f"{self.base_url}/api/v1/extract/{request_id}"
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            r = self.session.get(url, timeout=60)
            if r.status_code != 200:
                return {"__error__": f"HTTP {r.status_code}: {r.text[:300]}"}
            body = r.json()
            cents = (body.get("cost_breakdown") or {}).get("final_cost_cents", body.get("total_cost"))
            if isinstance(cents, (int, float)):
                self.cost_usd = round(float(cents) / 100.0, 6)     # CENTS -> USD, once
            status = body.get("status")
            if status == "complete":
                if self.cost_usd is None:
                    # Billing is populated on a re-fetch AFTER completion: the poll that first
                    # reports `complete` carries `total_cost: null`. Without this second GET the
                    # cost column reads null and the vendor looks like it bills out of band.
                    self._refetch_cost(url)
                extraction = body.get("extraction_schema_json")
                if extraction is None:
                    return {"__error__": "status complete but no extraction_schema_json"}
                return json.loads(extraction) if isinstance(extraction, str) else extraction
            if status == "error":
                return {"__error__": f"vendor reported failure: {str(body.get('error'))[:300]}"}
            time.sleep(self.poll_interval)
        return {"__error__": f"timeout after {self.timeout}s; request {request_id} still running",
                "__timeout__": True}

    def _refetch_cost(self, url: str) -> None:
        try:
            body = self.session.get(url, timeout=60).json()
            cents = (body.get("cost_breakdown") or {}).get("final_cost_cents", body.get("total_cost"))
            if isinstance(cents, (int, float)):
                self.cost_usd = round(float(cents) / 100.0, 6)     # CENTS -> USD, once
        except Exception:  # noqa: BLE001 -- cost capture must never break extraction
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", default=os.environ.get("DATALAB_MODE", "balanced"),
                    choices=["fast", "balanced", "accurate"])
    ap.add_argument("--base-url", default=os.environ.get("DATALAB_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("OEB_TIMEOUT", 1800)))
    a = ap.parse_args()
    key = os.environ.get("DATALAB_API_KEY")
    if not key:
        raise SystemExit("DATALAB_API_KEY must be set")
    p = DatalabAPI(api_key=key, base_url=a.base_url, mode=a.mode, timeout=a.timeout)
    t0 = time.time()
    result = p(a.pdf, json.loads(a.schema.read_text()))
    write_output(a.out, provider=f"datalab-{a.mode}", result=result, latency_s=time.time() - t0,
                 usage={"mode": a.mode, "cost_usd": p.cost_usd})
    if isinstance(result, dict) and "__error__" in result:
        print(result["__error__"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
