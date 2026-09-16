#!/usr/bin/env python3
"""Extend extraction via the CURRENT processor API — one generic processor, any schema.

Background: `POST /extract_runs` 404s on the OLD api version (2025-04-21) -- it was read
as retired, but it is the current endpoint under 2026-02-09. Version, not deprecation.
Extraction now runs through processors. The obvious port — create a processor per schema —
would litter the customer's workspace with hundreds of objects (ExtractBench alone has a
distinct schema per document).

Probing the API showed a better route: `POST /processor_runs` accepts a `config` object and
VALIDATES the schema inside it, i.e. the schema can be supplied per run and overrides the
one baked into the processor version. Verified by sending a real schema with a deliberately
bogus fileId: the request passed schema validation and failed only on the file.

So this provider:
  * reuses an EXISTING processor as a generic shell (creates nothing), and
  * passes the document's own schema in `config` on every run.

Flow: POST /files/upload -> POST /processor_runs (config.schema = this doc's schema)
      -> poll GET /processor_runs/{id} -> result.

Auth note: organisation-level API keys REQUIRE `x-extend-workspace-id`; without it every
upload 400s. That was a real bug in the vendored provider.

Schemas must be reshaped before sending -- Extend validates strictly and rejects the JSON
Schema dialect most benchmarks emit. Use `omni_extract_bench.harness.dialects`:

    from omni_extract_bench.harness.dialects import (
        strip_benchmark_keys, resolve_refs, to_strict_dialect)
    payload_schema = to_strict_dialect(resolve_refs(strip_benchmark_keys(schema)))

Skipping that step produces a stream of 400s -- $ref, then `evaluation_config`, then `title`,
then non-nullable primitives -- one per request, which reads like a broken vendor and is not.

Usage: python3 -m omni_extract_bench.harness.providers.extend --pdf X.pdf --schema S.json --out O.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import httpx

BASE = os.environ.get("EXTEND_BASE_URL", "https://api.extend.ai")
# 2026-02-09 is the current version: a resource-based API with a dedicated /extract_runs
# endpoint that takes the schema inline, so no processor shell is needed. We were on
# 2025-04-21 -- stable, but two versions behind and missing the array options below.
API_VERSION = os.environ.get("EXTEND_API_VERSION", "2026-02-09")

# Extend's MAX array-extraction mode. The benchmark is array-heavy (3,000-row 13Fs,
# 2,200-row clinical tables), and this is the setting built for exactly that; the vendor
# flagged that we were benchmarking without it. Trades latency and credits for accuracy,
# which is the right side of that trade under a max-tier parity rule.
ARRAY_STRATEGY = os.environ.get("EXTEND_ARRAY_STRATEGY", "large_array_max_context")
POLL_S = 5
TIMEOUT_S = int(os.environ.get("EXTEND_TIMEOUT", "1800"))
TERMINAL = {"PROCESSED", "COMPLETED", "FAILED", "CANCELLED", "ERROR"}


def headers(key: str) -> dict:
    h = {"Authorization": f"Bearer {key}", "x-extend-api-version": API_VERSION}
    ws = os.environ.get("EXTEND_WORKSPACE_ID")
    if ws:
        h["x-extend-workspace-id"] = ws      # required for org-level keys
    return h


def generic_processor(c: httpx.Client) -> str:
    """Any existing EXTRACT processor works as a shell — the schema is overridden per run.

    Prefers one named for benchmarking if present. Creating a processor is avoided entirely
    so the customer's workspace is not mutated by running the benchmark.
    """
    env_pid = os.environ.get("EXTEND_PROCESSOR_ID")
    if env_pid:
        return env_pid
    r = c.get(f"{BASE}/processors")
    r.raise_for_status()
    d = r.json()
    procs = d.get("processors") or d.get("data") or []
    ex = [p for p in procs if p.get("type") == "EXTRACT"]
    if not ex:
        raise RuntimeError("no EXTRACT processor available to use as a generic shell")
    for p in ex:
        if "bench" in (p.get("name") or "").lower():
            return p["id"]
    return ex[0]["id"]


def upload(c: httpx.Client, pdf: Path) -> str:
    r = c.post(f"{BASE}/files/upload",
               files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")}, timeout=300)
    if r.status_code >= 400:
        raise RuntimeError(f"upload HTTP {r.status_code}: {r.text[:300]}")
    b = r.json()
    fid = (b.get("file") or {}).get("id") or b.get("id")
    if not fid:
        raise RuntimeError(f"upload: no file id in {r.text[:200]}")
    return fid


def submit(c: httpx.Client, processor_id: str, file_id: str, schema: dict) -> str:
    """Submit an extraction run against the current API.

    The schema goes inline in `config`, so the processor-shell workaround the old version
    needed is gone -- `processor_id` is accepted for signature compatibility and unused.
    `advancedOptions.arrayStrategy` selects MAX array extraction.
    """
    # 2026-02-09 renamed the file reference: {"fileId": X} -> {"id": X}
    body = {"file": {"id": file_id},
            "config": {"schema": schema,
                       "advancedOptions": {"arrayStrategy": {"type": ARRAY_STRATEGY}}}}
    r = c.post(f"{BASE}/extract_runs", json=body, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"extract_runs HTTP {r.status_code}: {r.text[:400]}")
    b = r.json()
    run = b.get("extractRun") or b.get("processorRun") or b.get("run") or b
    rid = run.get("id")
    if not rid:
        raise RuntimeError(f"extract_runs: no run id in {r.text[:250]}")
    return rid


def poll(c: httpx.Client, run_id: str) -> dict:
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        r = c.get(f"{BASE}/extract_runs/{run_id}", timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"poll HTTP {r.status_code}: {r.text[:250]}")
        b = r.json()
        run = b.get("extractRun") or b.get("processorRun") or b.get("run") or b
        st = str(run.get("status", "")).upper()
        if st in TERMINAL:
            return run
        time.sleep(POLL_S)
    raise RuntimeError(f"run {run_id} did not finish within {TIMEOUT_S}s")


def extraction_of(run: dict):
    """Pull the schema-shaped payload out of whatever envelope the run uses."""
    # Extend wraps the payload: run.output = {"value": <extraction>, "metadata": {...}}.
    # Unwrap `value` when present, else fall back to the container itself.
    for path in (("output", "value"), ("result", "value"), ("output",), ("result",),
                 ("edited", "output")):
        cur = run
        for k in path:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        if isinstance(cur, dict) and cur:
            if set(cur.keys()) == {"value", "metadata"}:
                cur = cur["value"]
            return cur if isinstance(cur, dict) else {"value": cur}
        if isinstance(cur, list) and cur and isinstance(cur[0], dict):
            return cur[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    key = os.environ.get("EXTEND_API_KEY")
    if not key:
        raise SystemExit("EXTEND_API_KEY not set")
    schema = json.loads(a.schema.read_text())
    with httpx.Client(headers=headers(key), timeout=120) as c:
        pid = generic_processor(c)
        print(f"processor (generic shell): {pid}", file=sys.stderr)
        fid = upload(c, a.pdf)
        print(f"uploaded: {fid}", file=sys.stderr)
        rid = submit(c, pid, fid, schema)
        print(f"run: {rid}", file=sys.stderr)
        run = poll(c, rid)
        print(f"status: {run.get('status')}", file=sys.stderr)
    res = extraction_of(run)
    a.out.write_text(json.dumps({"result": res, "_meta": {"run_id": rid, "processor": pid,
                                                          "status": run.get("status")}},
                                default=str))
    print(f"saved -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()


# Extend reserves the property name `id` ("Field key \"id\" is reserved for internal use", HTTP 400).
# Send it under an alias and restore the schema's own name on the way back -- dialect, not content.
RESERVED = {"id": "id__"}


def rename_reserved(node):
    if isinstance(node, list):
        return [rename_reserved(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "properties" and isinstance(v, dict):
            out[k] = {RESERVED.get(pk, pk): rename_reserved(pv) for pk, pv in v.items()}
        elif k == "required" and isinstance(v, list):
            out[k] = [RESERVED.get(x, x) for x in v]
        else:
            out[k] = rename_reserved(v)
    return out


def restore_reserved(obj):
    back = {v: k for k, v in RESERVED.items()}
    if isinstance(obj, list):
        return [restore_reserved(x) for x in obj]
    if isinstance(obj, dict):
        return {back.get(k, k): restore_reserved(v) for k, v in obj.items()}
    return obj
