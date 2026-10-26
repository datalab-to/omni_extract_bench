"""Datalab: POST /api/v1/extract, then poll for the result. `mode` is the tier (`fast`, `balanced`,
`accurate`); balanced and accurate are published as separate legs, never only the better one. Cost comes
back in cents (`cost_breakdown.final_cost_cents`) and is converted to dollars once, in `cost_of`. Auth:
DATALAB_API_KEY.

    oeb init datalab > config.yaml
    oeb run --config config.yaml --set system.datalab.mode=balanced"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, ClassVar

import httpx
from omni_bench_core import adapter, errors

from .. import contract

DEFAULT_BASE_URL = "https://www.datalab.to"


@dataclasses.dataclass(frozen=True)
class Datalab(contract.Extractor):
    """Datalab's extraction API, at one tier (`mode`)."""

    name: ClassVar[str] = "datalab"
    keys: ClassVar[tuple[str, ...]] = ("DATALAB_API_KEY",)
    workers: ClassVar[int] = 10
    mode: str = dataclasses.field(
        default="balanced",
        metadata={"choices": ["fast", "balanced", "accurate"],
                  "help": "extraction tier; the published runs use balanced and accurate"})
    base_url: str = DEFAULT_BASE_URL
    poll_interval: float = 5.0

    def prepare_schema(self, schema: dict) -> dict:
        """The schema in Datalab's dialect (`dialect`)."""
        return dialect(schema)

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """POSTs the document and polls until it is done; returns the answer and its cost. A failure is
        raised here, the one place that knows whether the vendor said no or the budget ran out while it
        worked."""
        key = os.environ["DATALAB_API_KEY"]

        deadline = adapter.Budget(timeout)
        base_url = self.base_url.rstrip("/")
        cost = adapter.Cost()
        polls = 0
        with httpx.Client(headers={"X-Api-Key": key}, timeout=120) as client:
            with pdf.open("rb") as fh:
                try:
                    resp = client.post(
                        f"{base_url}/api/v1/extract",
                        files={"file": (pdf.name, fh, "application/pdf")},
                        data={"page_schema": json.dumps(schema),
                              "extraction_mode": self.mode, "output_format": "json"})
                except httpx.TransportError as exc:
                    raise contract.unreachable("the upload didn't reach Datalab", exc) from exc
            if resp.status_code != 200:
                raise errors.SystemFailed(f"HTTP {resp.status_code}: {resp.text}",
                                          status=resp.status_code, body=resp.text)
            with adapter.reading("the request id"):
                request_id = resp.json().get("request_id")
            if not request_id:
                raise errors.SystemFailed(f"no request_id in the response: {resp.text}",
                                          status=resp.status_code, body=resp.text)

            url = f"{base_url}/api/v1/extract/{request_id}"
            retry = adapter.PollRetry(deadline)
            while True:
                deadline.check(f"request {request_id} was still running after {polls} polls")
                try:
                    r = client.get(url)
                except httpx.TransportError as exc:
                    if retry.again():
                        continue
                    raise contract.unreachable("polling failed", exc) from exc
                if r.status_code != 200:
                    if retry.again(r.status_code):
                        continue
                    raise errors.SystemFailed(f"HTTP {r.status_code}: {r.text}",
                                              status=r.status_code, body=r.text)
                retry.reset()
                polls += 1
                with adapter.reading("the request's status and cost"):
                    body = r.json()
                    if (found := cost_of(body)) is not None:
                        cost = found
                    status = body.get("status")
                if status == "error":
                    raise errors.SystemFailed(f"vendor reported failure: {str(body.get('error'))}",
                                              status=200, body=r.text, cost=cost)
                if status == "complete":
                    if cost.usd is None:
                        try:
                            with adapter.reading("the request's cost"):
                                if (found := cost_of(client.get(url).json())) is not None:
                                    cost = found
                        except Exception:  # noqa: BLE001 -- cost must never lose the extraction
                            pass
                    return adapter.Call(raw=body, cost=cost, job_id=request_id)
                time.sleep(self.poll_interval)

    def read(self, raw: Any, q: contract.Question) -> dict:
        """The extraction in `extraction_schema_json`, sent as a JSON string or an object."""
        x = raw.get("extraction_schema_json") if isinstance(raw, dict) else None
        if x is None:
            raise errors.SystemFailed("status complete but no extraction_schema_json")
        with adapter.reading("extraction_schema_json"):
            x = json.loads(x) if isinstance(x, str) else x
        return contract.extraction(x)


def cost_of(body: dict) -> adapter.Cost | None:
    """The cost this response states, with the field it came from as `Cost.source`, or None."""
    for field, value in (("cost_breakdown.final_cost_cents", (body.get("cost_breakdown") or {}).get("final_cost_cents")),
                         ("total_cost", body.get("total_cost"))):
        found = adapter.Cost.reported(value, field, cents=True)
        if found.usd is not None:
            return found
    return None


def dialect(schema: dict) -> dict:
    """The schema with nullable unions collapsed and nullable fields taken out of `required`, since the API
    expresses optionality by omission. Names, types and descriptions are unchanged."""
    schema = dict(schema)
    nullable: set[str] = set()
    props = schema.get("properties")
    if isinstance(props, dict):
        for k, v in props.items():
            if isinstance(v, dict) and isinstance(v.get("type"), list) and "null" in v["type"]:
                nullable.add(k)
        schema["properties"] = {k: dialect(v) if isinstance(v, dict) else v for k, v in props.items()}
    if "required" in schema and nullable:
        schema["required"] = [r for r in schema["required"] if r not in nullable]
        if not schema["required"]:
            del schema["required"]
    if isinstance(schema.get("type"), list):
        non_null = [t for t in schema["type"] if t != "null"]
        schema["type"] = non_null[0] if non_null else "string"
    if isinstance(schema.get("items"), dict):
        schema["items"] = dialect(schema["items"])
    return schema
