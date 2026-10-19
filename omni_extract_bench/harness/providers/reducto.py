#!/usr/bin/env python3
"""Run Reducto extraction against a PDF + JSON Schema.

It uploads the PDF, submits an /extract_async job and polls it. Every setting is set here: deep extraction
on, deep extraction model v2, citations off, and an empty system prompt by default (the schema's field
descriptions guide it).

Auth: REDUCTO_API_KEY.

Usage:
    oeb run --providers reducto

Adapted from longextract_bench (MIT, (c) 2026 Micro1) -- see providers/LICENSE-micro1.
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

from .. import contract, responses

BASE_URL = "https://platform.reducto.ai"
DEFAULT_DEEP_EXTRACT_MODEL = "v2"
AGENTIC_TABLE_MODE = "max"
_TERMINAL = {"Completed", "Failed", "Error", "Cancelled"}


_JOBS_PAGE = 200


def _server_duration(client: httpx.Client, api_key: str, job_id: str) -> float | None:
    """The request's server-side processing seconds, as Reducto records it on the
    job (the /jobs listing exposes `duration`; the /job result does not). This is
    the API request's own latency: it excludes our upload, the waits between polls, and
    the queue before processing starts.

    It is best effort, and it never raises. The extraction is already finished and paid for by the
    time this is called, so a cosmetic field must not be able to discard it -- `raise_for_status`
    here used to throw `httpx.HTTPStatusError` straight past `predict`, which catches
    `VendorError` only. A job that has scrolled past `_JOBS_PAGE` simply has no duration.
    """
    try:
        r = client.get(
            f"{BASE_URL}/jobs",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"limit": _JOBS_PAGE},
            timeout=60,
        )
        r.raise_for_status()
        with adapter.reading("the job's duration"):
            for j in r.json().get("jobs", []):
                if j.get("job_id") == job_id:
                    return j.get("duration")
    except Exception:            # noqa: BLE001 -- a recorded nicety, never the extraction
        return None
    return None


def _upload(client: httpx.Client, api_key: str, pdf: Path) -> str:
    r = client.post(
        f"{BASE_URL}/upload",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
        timeout=300,
    )
    r.raise_for_status()
    with adapter.reading("the uploaded file's id"):
        file_id = r.json().get("file_id")
    if not file_id:
        raise RuntimeError(f"upload {pdf.name}: no file_id: {r.text[:300]}")
    return file_id


def _build_payload(
    input_ref: str, schema: dict, system_prompt: str, deep_extract_model: str,
    agentic_table_mode: str = AGENTIC_TABLE_MODE,
) -> dict:
    instructions: dict[str, object] = {"schema": schema}
    if system_prompt:
        instructions["system_prompt"] = system_prompt
    return {
        "async": {"priority": False},
        "input": input_ref,
        "parsing": {"enhance": {"agentic": [{"scope": "table", "mode": agentic_table_mode}]}},
        "instructions": instructions,
        "settings": {
            "deep_extract": True,
            "citations": {"enabled": False},
            "alpha": {"deep_extract_model": deep_extract_model},
        },
    }


def _submit(client: httpx.Client, api_key: str, payload: dict) -> str:
    r = client.post(
        f"{BASE_URL}/extract_async",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"extract_async HTTP {r.status_code}: {r.text[:400]}")
    with adapter.reading("the job's id"):
        job_id = r.json().get("job_id")
    if not job_id:
        raise RuntimeError(f"extract_async: no job_id: {r.text[:300]}")
    return job_id


def _poll(client: httpx.Client, api_key: str, job_id: str, interval: int,
          deadline) -> dict:
    polls = 0
    retry = adapter.PollRetry(deadline)
    while True:
        deadline.check(f"job {job_id} was still running server-side after {polls} polls")
        try:
            r = client.get(
                f"{BASE_URL}/job/{job_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if retry.again(exc.response.status_code):
                continue
            raise errors.VendorError(f"poll HTTP {exc.response.status_code}: "
                                     f"{exc.response.text[:250]}",
                                     status=exc.response.status_code,
                                     body=exc.response.text) from None
        except httpx.TransportError as exc:
            if retry.again():
                continue
            raise errors.VendorUnreachable(f"polling failed: {exc}"[:300]) from None
        retry.reset()
        polls += 1
        with adapter.reading("the job's status"):
            body = r.json()
            status = body.get("status") or body.get("state") or ""
            done = status in _TERMINAL or body.get("result") is not None
        if done:
            return body
        time.sleep(interval)


@dataclasses.dataclass(frozen=True)
class Reducto(contract.Extractor):
    """Reducto's async extraction, at its maximum tier: deep extract on, `max` agentic tables."""

    name: ClassVar[str] = "reducto"
    workers: ClassVar[int] = 3
    deep_extract_model: str = DEFAULT_DEEP_EXTRACT_MODEL
    system_prompt: str = dataclasses.field(
        default="", metadata={"help": "extra system prompt; empty so the schema drives it"})
    agentic_table_mode: str = dataclasses.field(
        default=AGENTIC_TABLE_MODE,
        metadata={"choices": ["default", "max"],
                  "help": "agentic table enrichment; `max` enriches every table"})
    poll_interval: int = 5

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """Upload, submit an async extract, poll to completion.

        A job that outlives its budget is named in the record by `job_id`, so it can be chased by
        hand rather than being anonymous.
        """
        key = os.environ.get("REDUCTO_API_KEY")
        if not key:
            raise errors.MissingCredential("REDUCTO_API_KEY is not set")

        deadline = adapter.Budget(timeout)
        with httpx.Client() as client:
            try:
                file_id = _upload(client, key, pdf)
                job_id = _submit(client, key, _build_payload(
                    file_id, schema, self.system_prompt, self.deep_extract_model, self.agentic_table_mode))
            except httpx.HTTPStatusError as exc:
                raise errors.VendorError(f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
                                         status=exc.response.status_code, body=exc.response.text) from None
            except RuntimeError as exc:
                raise errors.VendorError(str(exc)[:300], status=200) from None
            body = _poll(client, key, job_id, self.poll_interval, deadline)
            latency_s = _server_duration(client, key, job_id)

        with adapter.reading("the job's status and cost"):
            status = body.get("status") or "?"
            usage = dict((body.get("result") or {}).get("usage") or {})
            credits = usage.get("credits")
        if status != "Completed":
            raise errors.VendorError(f"reducto {status}: {json.dumps(body)[:300]}", status=200)
        return adapter.Call(raw={**body, "latency_s": latency_s},
                            cost=adapter.Cost(credits=credits if isinstance(credits, (int, float))
                                              and not isinstance(credits, bool) else None,
                                              source="usage.credits" if credits is not None else None),
                            job_id=job_id)

    def read(self, raw: Any, q: contract.Question) -> dict:
        """The extraction in `result.result`. Reducto may answer with a list, one entry per extraction
        pass, which `responses.as_object` merges."""
        resp = (raw.get("result") or {}) if isinstance(raw, dict) else {}
        found = responses.as_object(resp.get("result", resp))
        if found is None:
            raise ValueError(f"completed with no extraction: {json.dumps(raw)[:300]}")
        return found
