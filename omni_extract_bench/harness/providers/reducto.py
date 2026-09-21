#!/usr/bin/env python3
"""Run Reducto extraction against a PDF + JSON Schema.

Submits a /extract_async job directly (upload -> extract_async -> poll). Every
setting is controlled here, so this script is the single source of truth for how
the benchmark calls Reducto:
  - deep_extract:        on
  - deep_extract_model:  v2
  - citations:           off
  - system_prompt:       empty by default (the schema field descriptions drive it)

Auth: REDUCTO_API_KEY.

Usage:
    oeb predict --provider reducto --doc doc.pdf --schema schema.json

Adapted from longextract_bench (MIT, (c) 2026 Micro1) -- see providers/LICENSE-micro1.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

import httpx

from ..extraction import (Budget, Cost, Extraction, MissingCredential, PollRetry,
                          VendorError, as_object)

BASE_URL = "https://platform.reducto.ai"
DEFAULT_DEEP_EXTRACT_MODEL = "v2"
AGENTIC_TABLE_MODE = "max"
_TERMINAL = {"Completed", "Failed", "Error", "Cancelled"}


@dataclasses.dataclass(frozen=True)
class Config:
    """What reducto can be asked, and what it is asked at its maximum tier."""

    deep_extract_model: str = DEFAULT_DEEP_EXTRACT_MODEL
    system_prompt: str = dataclasses.field(
        default="", metadata={"help": "extra system prompt; empty so the schema drives it"})
    agentic_table_mode: str = dataclasses.field(
        default=AGENTIC_TABLE_MODE,
        metadata={"choices": ["default", "max"],
                  "help": "agentic table enrichment; `max` enriches every table"})
    poll_interval: int = 5


_JOBS_PAGE = 200


def _server_duration(client: httpx.Client, api_key: str, job_id: str) -> float | None:
    """The request's server-side processing seconds, as Reducto records it on the
    job (the /jobs listing exposes `duration`; the /job result does not). This is
    the API request latency ONLY — it excludes our upload, poll sleeps, and the
    queue wait before processing starts.

    BEST EFFORT, and it never raises. The extraction is already finished and paid for by the
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
    job_id = r.json().get("job_id")
    if not job_id:
        raise RuntimeError(f"extract_async: no job_id: {r.text[:300]}")
    return job_id


def _poll(client: httpx.Client, api_key: str, job_id: str, interval: int,
          budget) -> dict:
    polls = 0
    retry = PollRetry(budget)
    while True:
        budget.check(f"job {job_id} was still running server-side after {polls} polls")
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
            raise VendorError(f"poll HTTP {exc.response.status_code}: "
                              f"{exc.response.text[:250]}",
                              status=exc.response.status_code,
                              body=exc.response.text) from None
        except httpx.TransportError as exc:
            if retry.again():
                continue
            raise VendorError(f"polling failed: {exc}"[:300], status=None) from None
        retry.ok()
        polls += 1
        body = r.json()
        status = body.get("status") or body.get("state") or ""
        if status in _TERMINAL or body.get("result") is not None:
            return body
        time.sleep(interval)


def prepare_schema(schema: dict) -> dict:
    """This vendor takes a JSON Schema as written; nothing to reshape."""
    return schema


def extract(pdf: Path, schema: dict, *, timeout: float = 1800.0,
            config: Config = Config()) -> Extraction:
    """Upload, submit an async extract, poll to completion.

    A job that outlives its budget is named in the record by `job_id`, so it can be chased by
    hand rather than being anonymous.
    """
    key = os.environ.get("REDUCTO_API_KEY")
    if not key:
        raise MissingCredential("REDUCTO_API_KEY is not set")

    budget = Budget(timeout)
    with httpx.Client() as client:
        try:
            file_id = _upload(client, key, pdf)
            job_id = _submit(client, key, _build_payload(
                file_id, schema, config.system_prompt, config.deep_extract_model,
                config.agentic_table_mode))
        except httpx.HTTPStatusError as exc:
            raise VendorError(f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
                              status=exc.response.status_code,
                              body=exc.response.text) from None
        except RuntimeError as exc:
            raise VendorError(str(exc)[:300], status=200) from None

        body = _poll(client, key, job_id, config.poll_interval, budget)
        try:
            latency_s = _server_duration(client, key, job_id)
        except Exception:  # noqa: BLE001
            latency_s = None

    status = body.get("status") or "?"
    if status != "Completed":
        raise VendorError(f"reducto {status}: {json.dumps(body)[:300]}", status=200)
    extract_resp = body.get("result") or {}
    extraction = as_object(extract_resp.get("result", extract_resp))
    if extraction is None:
        raise VendorError(f"completed with no extraction: {json.dumps(body)[:300]}", status=200)

    usage = dict(extract_resp.get("usage") or {})
    credits = usage.get("credits")
    return Extraction(result=extraction,
                      raw={**{k: v for k, v in body.items() if k != "result"},
                           "result": {k: v for k, v in extract_resp.items() if k != "result"},
                           "latency_s": latency_s},
                      cost=Cost(credits=credits if isinstance(credits, (int, float))
                                        and not isinstance(credits, bool) else None,
                                source="usage.credits" if credits is not None else None),
                      job_id=job_id)


