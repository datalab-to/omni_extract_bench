#!/usr/bin/env python3
"""Reducto: uploads the PDF, submits an /extract_async job and polls it, with every setting fixed here
(deep extraction on, its v2 model, citations off, and an empty system prompt by default, so the schema's
descriptions guide it). Auth: REDUCTO_API_KEY.

    oeb init reducto > config.yaml
    oeb run --config config.yaml

Adapted from longextract_bench (MIT, (c) 2026 Micro1); see providers/LICENSE-micro1."""

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
    """The job's server-side processing seconds, from the /jobs listing, or None. It never raises: the
    extraction is already paid for, so a cosmetic field must not discard it."""
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
        raise errors.SystemFailed(f"upload {pdf.name}: no file_id: {r.text}", status=r.status_code, body=r.text)
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
        raise errors.SystemFailed(f"extract_async HTTP {r.status_code}: {r.text}", status=r.status_code,
                                  body=r.text)
    with adapter.reading("the job's id"):
        job_id = r.json().get("job_id")
    if not job_id:
        raise errors.SystemFailed(f"extract_async: no job_id: {r.text}", status=r.status_code, body=r.text)
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
            raise errors.SystemFailed(f"poll HTTP {exc.response.status_code}: "
                                      f"{exc.response.text}",
                                      status=exc.response.status_code,
                                      body=exc.response.text) from exc
        except httpx.TransportError as exc:
            if retry.again():
                continue
            raise contract.unreachable("polling failed", exc) from exc
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
    keys: ClassVar[tuple[str, ...]] = ("REDUCTO_API_KEY",)
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
        """Uploads, submits an async extract, and polls it. A job that outlives its budget is named in the
        record by `job_id`."""
        key = os.environ["REDUCTO_API_KEY"]

        deadline = adapter.Budget(timeout)
        with httpx.Client() as client:
            try:
                file_id = _upload(client, key, pdf)
                job_id = _submit(client, key, _build_payload(
                    file_id, schema, self.system_prompt, self.deep_extract_model, self.agentic_table_mode))
            except httpx.HTTPStatusError as exc:
                raise errors.SystemFailed(f"HTTP {exc.response.status_code}: {exc.response.text}",
                                          status=exc.response.status_code, body=exc.response.text) from exc
            except httpx.TransportError as exc:
                raise contract.unreachable("submitting the document", exc) from exc
            body = _poll(client, key, job_id, self.poll_interval, deadline)
            latency_s = _server_duration(client, key, job_id)

        with adapter.reading("the job's status and cost"):
            status = body.get("status") or "?"
            usage = dict((body.get("result") or {}).get("usage") or {})
            credits = usage.get("credits")
        if status != "Completed":
            raise errors.SystemFailed(f"reducto {status}: {json.dumps(body)}", status=200)
        return adapter.Call(raw={**body, "latency_s": latency_s},
                            cost=adapter.Cost(credits=credits if isinstance(credits, (int, float))
                                              and not isinstance(credits, bool) else None,
                                              source="usage.credits" if credits is not None else None),
                            job_id=job_id)

    def read(self, raw: Any, q: contract.Question) -> dict:
        """The extraction in `result.result`; a list, one entry per pass, is merged by
        `responses.as_object`."""
        with adapter.reading("the job's result"):
            resp = (raw.get("result") or {}) if isinstance(raw, dict) else {}
            found = responses.as_object(resp.get("result", resp))
        if found is None:
            raise errors.SystemFailed(f"completed with no extraction: {json.dumps(raw)}")
        return found
