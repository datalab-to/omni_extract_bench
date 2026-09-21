"""Datalab API adapter: POST /api/v1/extract, then poll for the result.

`mode` selects the extraction tier (`fast`, `balanced`, `accurate`). The published runs used
`balanced` (the default product tier) and `accurate` (the most capable one) as two separate
legs, because publishing only whichever scored higher would misrepresent the product.

Cost comes back as `cost_breakdown.final_cost_cents` — CENTS, converted once here. Reporting
cents as dollars overstates cost by 100x, which is the kind of error a cost table never
recovers from.

Auth: DATALAB_API_KEY.

    oeb predict --provider datalab --doc doc.pdf --schema schema.json \
        --options '{"mode": "balanced"}'
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

import httpx

from ..extraction import (Budget, Cost, Extraction, MissingCredential, PollRetry,
                          VendorError)

DEFAULT_BASE_URL = "https://www.datalab.to"


@dataclasses.dataclass(frozen=True)
class Config:
    """What this vendor can be asked, and what it is asked at its maximum tier."""

    mode: str = dataclasses.field(
        default="balanced",
        metadata={"choices": ["fast", "balanced", "accurate"],
                  "help": "extraction tier; the published runs use balanced and accurate"})
    base_url: str = DEFAULT_BASE_URL
    poll_interval: float = 5.0


def prepare_schema(schema: dict) -> dict:
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
        schema["properties"] = {k: prepare_schema(v) if isinstance(v, dict) else v
                                for k, v in props.items()}
    if "required" in schema and nullable:
        schema["required"] = [r for r in schema["required"] if r not in nullable]
        if not schema["required"]:
            del schema["required"]
    if isinstance(schema.get("type"), list):
        non_null = [t for t in schema["type"] if t != "null"]
        schema["type"] = non_null[0] if non_null else "string"
    if isinstance(schema.get("items"), dict):
        schema["items"] = prepare_schema(schema["items"])
    return schema


def extract(pdf: Path, schema: dict, *, timeout: float = 1800.0,
            config: Config = Config()) -> Extraction:
    """POST the document, poll until it is done, return the extraction and what it cost.

    Raises rather than returning a failure: this function ran the poll loop, so it is the only
    place that knows the difference between "the vendor said no" and "we ran out of budget
    while it was still working" -- a distinction the harness used to reconstruct from an HTTP
    log afterwards.
    """
    key = os.environ.get("DATALAB_API_KEY")
    if not key:
        raise MissingCredential("DATALAB_API_KEY is not set")

    budget = Budget(timeout)
    base_url = config.base_url.rstrip("/")
    cost: Cost = Cost()
    polls = 0

    def read_cost(body: dict) -> Cost | None:
        """The cost the vendor stated, or None if this response does not say.

        Returns the SOURCE as well as the figure, because the two fields below are different
        claims and a cost that cannot be traced back to the vendor's own response cannot be
        checked. `bool` is excluded explicitly: `isinstance(True, int)` is True in Python, so
        a JSON `true` in a cost field would otherwise be read as one cent.
        """
        for field, value in (("cost_breakdown.final_cost_cents",
                              (body.get("cost_breakdown") or {}).get("final_cost_cents")),
                             ("total_cost", body.get("total_cost"))):
            found = Cost.reported(value, field, cents=True)
            if found.usd is not None:
                return found
        return None

    with httpx.Client(headers={"X-Api-Key": key}, timeout=120) as client:
        with pdf.open("rb") as fh:
            resp = client.post(
                f"{base_url}/api/v1/extract",
                files={"file": (pdf.name, fh, "application/pdf")},
                data={"page_schema": json.dumps(schema),
                      "extraction_mode": config.mode, "output_format": "json"})
        if resp.status_code != 200:
            raise VendorError(f"HTTP {resp.status_code}: {resp.text[:300]}",
                              status=resp.status_code, body=resp.text)
        request_id = resp.json().get("request_id")
        if not request_id:
            raise VendorError(f"no request_id in the response: {resp.text[:200]}",
                              status=resp.status_code, body=resp.text)

        url = f"{base_url}/api/v1/extract/{request_id}"
        retry = PollRetry(budget)
        while True:
            budget.check(f"request {request_id} was still running after {polls} polls")
            try:
                r = client.get(url)
            except httpx.TransportError as exc:
                if retry.again():
                    continue
                raise VendorError(f"polling failed: {exc}"[:300], status=None) from None
            if r.status_code != 200:
                if retry.again(r.status_code):
                    continue
                raise VendorError(f"HTTP {r.status_code}: {r.text[:300]}",
                                  status=r.status_code, body=r.text)
            retry.ok()
            polls += 1
            body = r.json()
            found = read_cost(body)
            if found is not None:
                cost = found
            status = body.get("status")
            if status == "error":
                raise VendorError(f"vendor reported failure: {str(body.get('error'))[:300]}",
                                  status=200, body=r.text)
            if status == "complete":
                if cost.usd is None:
                    try:
                        found = read_cost(client.get(url).json())
                        if found is not None:
                            cost = found
                    except Exception:  # noqa: BLE001 -- cost must never lose the extraction
                        pass
                extraction = body.get("extraction_schema_json")
                if extraction is None:
                    raise VendorError("status complete but no extraction_schema_json",
                                      status=200, body=r.text)
                return Extraction(
                    result=json.loads(extraction) if isinstance(extraction, str) else extraction,
                    raw=body,
                    cost=cost,
                    job_id=request_id)
            time.sleep(config.poll_interval)

