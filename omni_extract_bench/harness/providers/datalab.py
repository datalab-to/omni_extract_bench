"""Datalab API adapter: POST /api/v1/extract, then poll for the result.

`mode` selects the extraction tier (`fast`, `balanced`, `accurate`). The published runs used
`balanced` (the default product tier) and `accurate` (the most capable one) as two separate
legs, because publishing only whichever scored higher would misrepresent the product.

Cost comes back as `cost_breakdown.final_cost_cents`, in cents, and is converted to dollars once, here.
Read as dollars it would overstate the cost 100 times.

Auth: DATALAB_API_KEY.

    oeb run --providers datalab --options '{"datalab": {"mode": "balanced"}}'
"""
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
    workers: ClassVar[int] = 10
    mode: str = dataclasses.field(
        default="balanced",
        metadata={"choices": ["fast", "balanced", "accurate"],
                  "help": "extraction tier; the published runs use balanced and accurate"})
    base_url: str = DEFAULT_BASE_URL
    poll_interval: float = 5.0

    def prepare_schema(self, schema: dict) -> dict:
        """Collapse nullable unions and de-require the fields that were nullable (`dialect`)."""
        return dialect(schema)

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """POST the document, poll until it is done, return the answer and what it cost.

        Raises rather than returning a failure: this method ran the poll loop, so it is the only
        place that knows the difference between "the vendor said no" and "we ran out of budget
        while it was still working".
        """
        key = os.environ.get("DATALAB_API_KEY")
        if not key:
            raise errors.MissingCredential("DATALAB_API_KEY is not set")

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
                    # Note: a network fault on the upload is this document's failure, not a bug: no job was made.
                    raise errors.VendorUnreachable(f"the upload didn't reach Datalab: {type(exc).__name__}: {exc}"
                                                   ) from None
            if resp.status_code != 200:
                raise errors.VendorError(f"HTTP {resp.status_code}: {resp.text[:300]}",
                                         status=resp.status_code, body=resp.text)
            with adapter.reading("the request id"):
                request_id = resp.json().get("request_id")
            if not request_id:
                raise errors.VendorError(f"no request_id in the response: {resp.text[:200]}",
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
                    raise errors.VendorUnreachable(f"polling failed: {exc}"[:300]) from None
                if r.status_code != 200:
                    if retry.again(r.status_code):
                        continue
                    raise errors.VendorError(f"HTTP {r.status_code}: {r.text[:300]}",
                                             status=r.status_code, body=r.text)
                retry.reset()
                polls += 1
                with adapter.reading("the request's status and cost"):
                    body = r.json()
                    if (found := cost_of(body)) is not None:
                        cost = found
                    status = body.get("status")
                if status == "error":
                    raise errors.VendorError(f"vendor reported failure: {str(body.get('error'))[:300]}",
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
        """The extraction in `extraction_schema_json`, which the API sends as a JSON string or an object."""
        x = raw.get("extraction_schema_json") if isinstance(raw, dict) else None
        if x is None:
            raise ValueError("status complete but no extraction_schema_json")
        return contract.extraction(json.loads(x) if isinstance(x, str) else x)


def cost_of(body: dict) -> adapter.Cost | None:
    """The cost the vendor stated in this response, or None if it does not say.

    Two fields are tried, and the one the figure came from is named in `Cost.source`, so the figure
    can be traced back to the vendor's own response."""
    for field, value in (("cost_breakdown.final_cost_cents", (body.get("cost_breakdown") or {}).get("final_cost_cents")),
                         ("total_cost", body.get("total_cost"))):
        found = adapter.Cost.reported(value, field, cents=True)
        if found.usd is not None:
            return found
    return None


def dialect(schema: dict) -> dict:
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
