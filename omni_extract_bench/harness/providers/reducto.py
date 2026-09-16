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
    python -m longextract_bench.providers.reducto \
        --pdf path/to/document.pdf --schema path/to/schema.json --out /tmp/reducto_out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

from .envelope import write_output

BASE_URL = "https://platform.reducto.ai"
DEFAULT_DEEP_EXTRACT_MODEL = "v2"
# Agentic table enrichment mode. `default` enriches only tables a heuristic expects to benefit;
# `max` enriches every table and is the higher setting. Overridable, defaulting to max so the
# benchmark runs Reducto at its strongest -- Reducto is ahead on this benchmark, and running a
# competitor below their maximum would flatter our own result.
_AGENTIC_TABLE_MODE = os.environ.get("REDUCTO_AGENTIC_TABLE_MODE", "max")
_TERMINAL = {"Completed", "Failed", "Error", "Cancelled"}


def _server_duration(client: httpx.Client, api_key: str, job_id: str) -> float | None:
    """The request's server-side processing seconds, as Reducto records it on the
    job (the /jobs listing exposes `duration`; the /job result does not). This is
    the API request latency ONLY — it excludes our upload, poll sleeps, and the
    queue wait before processing starts."""
    r = client.get(
        f"{BASE_URL}/jobs",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"limit": 25},
        timeout=60,
    )
    r.raise_for_status()
    for j in r.json().get("jobs", []):
        if j.get("job_id") == job_id:
            return j.get("duration")
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
    input_ref: str, schema: dict, system_prompt: str, deep_extract_model: str
) -> dict:
    instructions: dict[str, object] = {"schema": schema}
    # empty system prompt -> omit entirely so the agent gets no extra instruction
    if system_prompt:
        instructions["system_prompt"] = system_prompt
    return {
        "async": {"priority": False},
        "input": input_ref,
        # parse: defaults only, except agentic table enrichment at its highest mode
        "parsing": {"enhance": {"agentic": [{"scope": "table", "mode": _AGENTIC_TABLE_MODE}]}},
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


def _poll(client: httpx.Client, api_key: str, job_id: str, interval: int) -> dict:
    # Resilient poll: the job keeps running server-side, so transient connection /
    # 5xx errors must NOT abandon it (that's how billed jobs got lost). Retry the GET.
    transient = 0
    while True:
        try:
            r = client.get(
                f"{BASE_URL}/job/{job_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60,
            )
            r.raise_for_status()
        except (httpx.TransportError, httpx.HTTPStatusError):
            transient += 1
            if transient > 60:
                raise
            time.sleep(min(30, 2 * transient))
            continue
        transient = 0
        body = r.json()
        status = body.get("status") or body.get("state") or ""
        print(f"  {status}", end="\r", flush=True)
        if status in _TERMINAL or body.get("result") is not None:
            print()
            return body
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reducto deep extract (v2, citations off)")
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output JSON file")
    ap.add_argument("--deep-extract-model", default=DEFAULT_DEEP_EXTRACT_MODEL)
    ap.add_argument(
        "--system-prompt", default="", help="extra system prompt; empty by default"
    )
    ap.add_argument("--poll-interval", type=int, default=5)
    args = ap.parse_args()

    api_key = os.environ.get("REDUCTO_API_KEY", "")
    if not api_key:
        sys.exit("REDUCTO_API_KEY not set")

    schema = json.loads(args.schema.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    side = args.out.with_suffix(".runid.json")
    resume_id = None
    if side.exists():
        try:
            resume_id = json.loads(side.read_text()).get("job_id")
        except (OSError, json.JSONDecodeError):
            resume_id = None

    with httpx.Client() as client:
        if resume_id:
            # A prior attempt already submitted this job — recover it instead of
            # re-submitting (no double-billing).
            print(f"Resuming existing job {resume_id} (sidecar)…")
            job_id = resume_id
        else:
            print(f"Uploading {args.pdf.name} ({args.pdf.stat().st_size // 1024} KB)…")
            file_id = _upload(client, api_key, args.pdf)
            print(f"  → {file_id}")
            payload = _build_payload(
                file_id, schema, args.system_prompt, args.deep_extract_model
            )
            print(
                f"Submitting extract (deep_extract_model={args.deep_extract_model}, "
                f"citations=off)…"
            )
            job_id = _submit(client, api_key, payload)
            # Persist the job id immediately so a killed/timed-out job stays pollable.
            try:
                side.write_text(
                    json.dumps(
                        {"provider": "reducto", "job_id": job_id, "file_id": file_id}
                    )
                )
            except OSError as e:
                print(f"  (warn: could not write runid sidecar: {e})")
        print(f"  → job_id: {job_id}\nPolling…")
        body = _poll(client, api_key, job_id, args.poll_interval)
        latency_s = _server_duration(client, api_key, job_id)

    # /job shape: {status, result: {usage: {num_pages, ...}, result: <data>}}
    extract_resp = body.get("result") or {}
    extraction = extract_resp.get("result", extract_resp)
    usage = dict(extract_resp.get("usage") or {})
    usage["job_id"] = job_id  # recorded so latency is auditable

    write_output(
        args.out,
        provider="reducto",
        result=extraction,
        latency_s=latency_s or 0.0,
        usage=usage,
    )
    status = body.get("status") or "?"
    print(f"Status: {status}  duration={latency_s}s\nSaved → {args.out}")


if __name__ == "__main__":
    main()
