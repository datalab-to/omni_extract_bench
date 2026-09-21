from __future__ import annotations
import dataclasses, json, os, time
from pathlib import Path
import httpx

from ..dialects import resolve_refs, to_strict_dialect
from ._cli import run_cli
from ..extraction import (Budget, Cost, Extraction, MissingCredential, PollRetry,
                          VendorError)

BASE = "https://api.extend.ai"
API_VERSION = "2026-02-09"

ARRAY_STRATEGY = "large_array_max_context"
POLL_S = 5
DEFAULT_TIMEOUT_S = 1800.0
TERMINAL = {"PROCESSED", "COMPLETED", "FAILED", "CANCELLED", "ERROR"}


@dataclasses.dataclass(frozen=True)
class Config:
    """What extend can be asked, and what it is asked at its maximum tier."""

    array_strategy: str = dataclasses.field(
        default=ARRAY_STRATEGY,
        metadata={"help": "MAX array extraction; this corpus is array-heavy"})
    api_version: str = API_VERSION
    base_url: str = BASE


def headers(key: str, api_version: str = API_VERSION) -> dict:
    h = {"Authorization": f"Bearer {key}", "x-extend-api-version": api_version}
    ws = os.environ.get("EXTEND_WORKSPACE_ID")
    if ws:
        h["x-extend-workspace-id"] = ws
    return h


def upload(c: httpx.Client, pdf: Path, base_url: str = BASE) -> str:
    r = c.post(f"{base_url}/files/upload",
               files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")}, timeout=300)
    if r.status_code >= 400:
        raise RuntimeError(f"upload HTTP {r.status_code}: {r.text[:300]}")
    b = r.json()
    fid = (b.get("file") or {}).get("id") or b.get("id")
    if not fid:
        raise RuntimeError(f"upload: no file id in {r.text[:200]}")
    return fid


def submit(c: httpx.Client, file_id: str, schema: dict, base_url: str = BASE,
           array_strategy: str = ARRAY_STRATEGY) -> str:
    """Submit an extraction run against the current API.

    The schema goes inline in `config`, which is what the current API version is for. The old
    one needed a saved processor to hang a schema on, so this adapter fetched any EXTRACT
    processor from the workspace and passed its id -- a wasted round trip per document, and a
    hard failure on an account that happens to have no processor saved. `submit` had already
    stopped using it.

    `advancedOptions.arrayStrategy` selects MAX array extraction.
    """
    body = {"file": {"id": file_id},
            "config": {"schema": schema,
                       "advancedOptions": {"arrayStrategy": {"type": array_strategy}}}}
    r = c.post(f"{base_url}/extract_runs", json=body, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"extract_runs HTTP {r.status_code}: {r.text[:400]}")
    b = r.json()
    run = b.get("extractRun") or b.get("processorRun") or b.get("run") or b
    rid = run.get("id")
    if not rid:
        raise RuntimeError(f"extract_runs: no run id in {r.text[:250]}")
    return rid


def poll(c: httpx.Client, run_id: str, budget, base_url: str = BASE) -> dict:
    polls = 0
    retry = PollRetry(budget)
    while True:
        budget.check(f"run {run_id} was still processing after {polls} polls")
        try:
            r = c.get(f"{base_url}/extract_runs/{run_id}", timeout=60)
        except httpx.TransportError as exc:
            if retry.again():
                continue
            raise VendorError(f"polling failed: {exc}"[:300], status=None) from None
        if r.status_code >= 400:
            if retry.again(r.status_code):
                continue
            raise VendorError(f"poll HTTP {r.status_code}: {r.text[:250]}",
                              status=r.status_code, body=r.text)
        retry.ok()
        polls += 1
        b = r.json()
        run = b.get("extractRun") or b.get("processorRun") or b.get("run") or b
        st = str(run.get("status", "")).upper()
        if st in TERMINAL:
            return run
        time.sleep(POLL_S)


def extraction_of(run: dict):
    """The extracted data, and the path it came from.

    `output.value` is the documented shape -- "extraction output splits into `output.value`
    (your data, shaped like the schema) and `output.metadata`" -- and a live run confirms it.
    This used to try five shapes in order and take the first that looked plausible, which is
    the wrong instinct in an extractor: if two could match, the wrong one is chosen silently
    and the wrong thing is scored. Better to read the documented field and fail loudly.
    """
    output = run.get("output")
    if isinstance(output, dict):
        value = output.get("value", output)
        if isinstance(value, dict) and value:
            return value, "output.value"
    return None, None


def extract(pdf: Path, schema: dict, *, timeout: float = DEFAULT_TIMEOUT_S,
            config: Config = Config()) -> Extraction:
    """Get a generic processor, upload, submit a run, poll it to a terminal state.

    Extend does not report a per-document cost, so `cost.usd` is None and the record says
    `billed_out_of_band` rather than inventing a figure from a price list.
    """
    key = os.environ.get("EXTEND_API_KEY")
    if not key:
        raise MissingCredential("EXTEND_API_KEY is not set")

    budget = Budget(timeout)
    with httpx.Client(headers=headers(key, config.api_version), timeout=120) as c:
        sent = to_strict_dialect(resolve_refs(rename_reserved(schema)))
        try:
            file_id = upload(c, pdf, config.base_url)
            run_id = submit(c, file_id, sent, config.base_url, config.array_strategy)
        except httpx.HTTPStatusError as exc:
            raise VendorError(f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
                              status=exc.response.status_code,
                              body=exc.response.text) from None
        except RuntimeError as exc:
            raise VendorError(str(exc)[:300], status=200) from None
        run = poll(c, run_id, budget, config.base_url)

    status = str(run.get("status", "")).upper()
    payload, envelope_path = extraction_of(run)
    result = restore_reserved(payload)
    if not isinstance(result, dict) or not result:
        raise VendorError(f"run finished {status} with no extraction: {json.dumps(run)[:300]}",
                          status=200)
    usage = run.get("usage") or {}
    credits = usage.get("totalCredits", usage.get("credits"))
    return Extraction(result=result,
                      raw={**{k: v for k, v in run.items()
                              if k not in ("output", "result", "edited")},
                           "envelope_path": envelope_path},
                      cost=Cost(credits=credits if isinstance(credits, (int, float))
                                        and not isinstance(credits, bool) else None,
                                source="usage.totalCredits" if credits is not None else None),
                      job_id=run_id)


def main() -> None:
    run_cli(extract, Config, "extend")


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


if __name__ == "__main__":
    main()
