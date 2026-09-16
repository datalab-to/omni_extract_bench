"""Run one provider over a document set, capturing everything needed to audit the result.

    python -m harness.run_provider <provider> --data-root data/ --out-root predictions/
    python -m harness.run_provider datalab --mode accurate --max-docs 5

This is the half of the benchmark the scorer cannot check: the scorer can only be fair about
what it is given, so the rules that decide what each vendor is given live here, in one file,
applied to every provider by the same code path.

WHAT IS UNIFORM, AND WHY EACH RULE EXISTS
-----------------------------------------
* ONE TIMEOUT for every provider (`--timeout`, default 1800s). The first run of this benchmark
  gave one provider 1800s and the raw-model legs 300s; one vendor lost 48 documents to
  "analysis timed out" under a cap another vendor never reached. A harness parameter must
  never be the thing that decides a vendor's coverage.
* MAXIMUM TIER for every provider. Parity is "as much as the vendor will give", not one number
  for everyone, so `PROVIDER_TIER` names the top tier per vendor and `MODEL_MAX_OUTPUT` the
  published output ceiling per model rather than a shared floor that silently truncates the
  models with bigger ones.
* THE SAME SCHEMA, with benchmark-only annotations removed (`evaluation_config`, `default`).
  Those tell a grader how to compare a value and tell a model nothing; one vendor validates
  strictly and rejected 8 of 40 documents over them while everyone else silently ignored them.
* CONVENTIONS ARE STATED, NOT ASSUMED (`schema_overlay`). Where the gold follows a convention
  the schema never states, the convention is written into the field description for every
  vendor alike -- rather than loosening the comparator, which would credit a vendor that
  dumps a paragraph.
* DIALECT TRANSLATION IS ALLOWED, CONTENT CHANGES ARE NOT. Inlining `$ref`, collapsing a
  nullable union, renaming a property a vendor reserves: same fields, same types, same
  descriptions. Excluding a vendor over how a schema is encoded is the harness deciding a
  score again.
* TRANSIENT FAILURES RETRY, REAL ANSWERS DO NOT. A 429 or a 5xx says nothing about whether a
  vendor can extract a document; a 400 is a real result (usually our schema, sometimes a
  genuine vendor limit) and retrying it would only hide the evidence.
* RAW FIRST, ALWAYS. The complete HTTP exchange, job ids, the exact schema sent and the
  vendor's own usage block are written before anything is parsed. An adapter that mis-parses
  a response with only the parsed value stored destroys already-paid calls; this cost 166 of
  them once. Raw-first means a parser bug is fixed by re-parsing on disk, never by re-paying.
* ACCOUNT-LEVEL FAILURES STOP THE RUN. A 402 is not about the document: one vendor marked 174
  documents "failed" in 60 seconds after hitting a credit ceiling, converting a recoverable
  pause into 174 stored zeros. Those documents are left untouched and resumable.

Set credentials in the environment (see `harness/README.md`). Nothing here reads a
machine-specific path: `--data-root` is the HF dataset layout,
`<root>/<subset>/<doc>/{document.pdf,schema.json}`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS.parent))

from harness import schema_overlay as SO                     # noqa: E402
from omni_extract_bench.prediction_io import usable          # noqa: E402

# ── parity constants ─────────────────────────────────────────────────────────────
DEFAULT_TIMEOUT = 1800
DEFAULT_MAX_OUTPUT_TOKENS = 64000        # floor for models with no published ceiling
MODEL_MAX_OUTPUT = {
    "anthropic/claude-opus-5": 128000,
    "openai/gpt-5.6-sol": 128000,
    "openai/gpt-5.6-sol-pro": 128000,
    "google/gemini-3.7-flash": 65536,
}
LLM_MODELS = {
    "claude": "anthropic/claude-opus-5",
    "gpt": "openai/gpt-5.6-sol",
    "gpt-pro": "openai/gpt-5.6-sol-pro",
    "gemini": "google/gemini-3.7-flash",
}
LLAMAEXTRACT_TIER = "agentic_plus"       # top of the v2 enum; verified live, not from a comment
PROVIDER_TIER = {
    "reducto": "super_agent",
    "llamaextract": LLAMAEXTRACT_TIER,
    "extend": "default processor",
    "mistral": "mistral-ocr-latest",
    "azure-cu": "gpt-4.1-mini (see --completion-model)",
    "datalab": "balanced",
    "datalab-accurate": "accurate",
    **{k: v for k, v in LLM_MODELS.items()},
}
PROVIDERS = sorted(set(PROVIDER_TIER) | {"datalab-accurate"})
WORKERS = {"reducto": 3, "llamaextract": 3, "azure-cu": 3, "datalab": 10, "datalab-accurate": 10}

BENCH_ONLY_KEYS = ("evaluation_config", "default")
TRANSIENT_ATTEMPTS = 4
TRANSIENT_BACKOFF = (20, 60, 120)
#: Ties the empty-200 message to the transient predicate. When these were two separate string
#: literals they drifted apart and the failure silently became non-retryable.
EMPTY_200_MARKER = "empty 200"


# ── schema preparation (dialect only) ────────────────────────────────────────────
def strip_bench_keys(node):
    if isinstance(node, list):
        return [strip_bench_keys(x) for x in node]
    if not isinstance(node, dict):
        return node
    return {k: strip_bench_keys(v) for k, v in node.items() if k not in BENCH_ONLY_KEYS}


def deref(schema, root=None, depth=0):
    """Inline local `$ref` so a schema is self-describing.

    Some vendors resolve `$defs` themselves; at least one rejects a bare `$ref` outright
    ("Array items must have a type property") because a `$ref` declares no type. Inlining
    changes no content.
    """
    if root is None:
        root = schema
    if depth > 12 or not isinstance(schema, (dict, list)):
        return schema
    if isinstance(schema, list):
        return [deref(x, root, depth + 1) for x in schema]
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        cur = root
        for part in ref[2:].split("/"):
            cur = (cur or {}).get(part) if isinstance(cur, dict) else None
        if isinstance(cur, dict):
            merged = {k: v for k, v in schema.items() if k != "$ref"}
            merged.update({k: v for k, v in cur.items() if k not in merged})
            return deref(merged, root, depth + 1)
    return {k: (deref(v, root, depth + 1) if k != "$defs" else v)
            for k, v in schema.items() if k != "$defs"}


# ── per-document capture state (thread-local: documents run in a pool) ───────────
_LOCAL = threading.local()


def _state():
    if not hasattr(_LOCAL, "d"):
        _LOCAL.d = {"http": [], "job_ids": [], "llm_calls": [], "envelope": None, "cost_usd": None}
    return _LOCAL.d


def _reset_state():
    _LOCAL.d = {"http": [], "job_ids": [], "llm_calls": [], "envelope": None, "cost_usd": None}


def _install_taps():
    """Install the transport taps from the SAME implementation the subprocess tap uses.

    One definition, deliberately: this tap was once maintained as three near-copies, and every
    gap found (async clients, then urllib, then an httpx fork) had to be fixed in each copy
    separately, which is how a fourth one hides.
    """
    sys.path.insert(0, str(HARNESS.parent / "omni_extract_bench" / "_tap"))
    from oeb_capture import Capture
    cap = Capture(sink=lambda: _state()["http"], id_sink=lambda: _state()["job_ids"])
    cap.install()
    return cap


CAPTURE = _install_taps()

_COST_PATHS = (
    (("cost_breakdown", "final_cost_cents"), 0.01),   # cents
    (("total_cost",), 0.01),                          # cents
    (("usage", "cost"), 1.0),                         # USD
    (("usage_info", "cost"), 1.0),
    (("cost",), 1.0),
)


def extract_cost(body):
    """(usd, raw_field) for the first recognised cost field, else (None, None).

    Vendors report cost under different names and units. Looking for only one shape is how six
    providers were once labelled "billed out of band" on the strength of our own adapters
    discarding the field. Credits are deliberately NOT converted: the rate is contract-specific,
    so a credits figure is recorded as itself rather than invented into dollars.
    """
    if not isinstance(body, dict):
        return None, None
    for path, mult in _COST_PATHS:
        cur = body
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, (int, float)):
            return round(float(cur) * mult, 6), f"{'.'.join(path)}={cur}"
    return None, None


def note_envelope(body):
    """Keep the non-extraction part of a vendor response, so a cost claim stays checkable."""
    if not isinstance(body, dict):
        return
    skip = {"extraction_schema_json", "json", "markdown", "html", "chunks", "images",
            "document_annotation", "output", "result", "data"}
    try:
        trimmed = {k: v for k, v in body.items() if k not in skip}
        _state()["envelope"] = json.loads(json.dumps(trimmed, default=str)[:4000])
    except Exception:  # noqa: BLE001
        try:
            _state()["envelope"] = {"_keys": sorted(body.keys())}
        except Exception:  # noqa: BLE001
            pass


# ── failure classification ───────────────────────────────────────────────────────
def is_transient(text) -> bool:
    if not text:
        return False
    blob = str(text)
    if "429" in blob or "RateLimit" in blob or "rate limit" in blob.lower():
        return True
    return any(m in blob for m in ("HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504",
                                   "unreachable_backend", "Service unavailable",
                                   "internal_server_error", "RemoteDisconnected",
                                   "Connection aborted", EMPTY_200_MARKER))


ACCOUNT_MARKERS = ("exceeded the maximum number of credits", "subscription has expired",
                   "insufficient_quota")


def is_account_failure(text) -> bool:
    blob = str(text or "")
    return "402" in blob or any(m in blob.lower() for m in ACCOUNT_MARKERS)


def _empty_completion(body: str) -> bool:
    """A 200 that carried no completion at all (keep-alive padding, or an empty content field).

    Narrow on purpose -- an EMPTY body, not "a body without choices" -- because providers
    legitimately return job handles and status envelopes that must not be mistaken for this.
    """
    if not body or not body.strip():
        return True
    stripped = body.strip()
    if '"choices"' not in stripped:
        return False
    try:
        parsed = json.loads(stripped)
    except Exception:  # noqa: BLE001 -- truncated capture: cannot judge, leave it alone
        return False
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        return True
    content = ((choices[0] or {}).get("message") or {}).get("content")
    return not (content and str(content).strip())


def _vendor_error_text(body: str):
    """The vendor's OWN error string, checked at the top level and one level down."""
    try:
        parsed = json.loads(body)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(parsed, dict):
        return None
    for key in ("error", "detail", "message", "error_message", "failure_reason"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:400]
        if isinstance(value, dict):
            for inner in ("message", "detail", "reason"):
                got = value.get(inner)
                if isinstance(got, str) and got.strip():
                    return got.strip()[:400]
    return None


def classify_no_output(wall_s, timeout):
    """Name WHY a provider produced nothing, from the transport records.

    All of these score zero, correctly. But "timed out at the shared limit while the vendor was
    still working" and "returned something unparseable" are different facts about a provider,
    and collapsing them throws the distinction away at the one point where the evidence exists.
    """
    log = _state()["http"]
    if not log:
        return {"__error__": "no output, and no HTTP was captured"}
    last = log[-1]
    body = last.get("body") or ""
    still_running = bool(re.search(
        r'"status"\s*:\s*"(running|processing|pending|queued|in_progress)"', body, re.I))
    if still_running and wall_s and wall_s >= timeout * 0.95:
        polls = sum(r.get("repeats", 1) for r in log if r.get("method") == "GET")
        return {"__error__": (f"timeout at the uniform {timeout}s limit; the vendor was still "
                              f"processing after {polls} polls over {wall_s:.0f}s"),
                "__timeout__": True}
    if last.get("status", 0) >= 400:
        return {"__error__": f"HTTP {last['status']}: {body[:300]}"}
    if last.get("status") == 200 and _empty_completion(body):
        took = f" after {wall_s:.0f}s" if isinstance(wall_s, (int, float)) else ""
        return {"__error__": (f"{EMPTY_200_MARKER}{took}: response body was keep-alive padding "
                              f"with no completion")}
    vendor_error = _vendor_error_text(body)
    if vendor_error:
        return {"__error__": f"vendor reported failure: {vendor_error}"}
    return {"__error__": f"no usable output; last response {last.get('status')}: {body[:300]}"}


def with_transient_retry(fn, label=""):
    last = None
    for attempt in range(TRANSIENT_ATTEMPTS):
        result = last = fn()
        err = result.get("__error__") if isinstance(result, dict) else None
        if not err or not is_transient(err):
            return result
        if attempt == TRANSIENT_ATTEMPTS - 1:
            break
        delay = TRANSIENT_BACKOFF[min(attempt, len(TRANSIENT_BACKOFF) - 1)]
        print(f"  {label}: transient failure, retry {attempt + 1}/{TRANSIENT_ATTEMPTS - 1} "
              f"in {delay}s -- {str(err)[:90]}", flush=True)
        time.sleep(delay)
    return last


# ── adapters ─────────────────────────────────────────────────────────────────────
def as_extraction(r):
    """Normalise an adapter result to the schema-shaped dict.

    One vendor returns a LIST (one entry per extraction pass); a single-entry list is just the
    extraction, and multi-entry lists are merged shallowly, first writer wins, so array fields
    from separate passes are preserved rather than discarded.
    """
    if isinstance(r, dict):
        return r
    if isinstance(r, list):
        dicts = [x for x in r if isinstance(x, dict)]
        if len(dicts) == 1:
            return dicts[0]
        if len(dicts) > 1:
            merged = {}
            for d in dicts:
                for k, v in d.items():
                    if k not in merged or merged[k] in (None, [], {}):
                        merged[k] = v
            return merged
    return {"__error__": f"non-dict:{type(r).__name__}"}


def run_cli_adapter(module: str, pdf: Path, schema: dict, timeout: float, extra_args=()):
    """Run one adapter as a subprocess, capturing its HTTP from inside the child.

    `sitecustomize` installs the transport tap in the CHILD interpreter: the parent's
    monkey-patch cannot reach a subprocess, so without this the adapter's HTTP is invisible.
    """
    env = dict(os.environ)
    tap_dir = str(HARNESS.parent / "omni_extract_bench" / "_tap")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [tap_dir, str(HARNESS.parent), env.get("PYTHONPATH")]))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sf:
        json.dump(schema, sf); schema_path = sf.name
    out_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    http_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    env["OEB_HTTP_LOG"] = http_path
    cmd = [sys.executable, "-m", module, "--pdf", str(pdf), "--schema", schema_path,
           "--out", out_path, *extra_args]
    proc = subprocess.run(cmd, cwd=str(HARNESS.parent), env=env, capture_output=True,
                          text=True, timeout=timeout + 60)
    _merge_subprocess_http(http_path)
    try:
        envelope = json.loads(Path(out_path).read_text())
    except Exception:  # noqa: BLE001
        envelope = None
    finally:
        for p in (schema_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass
    if envelope is None:
        tail = (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1]
        return {"__error__": tail[:900]}
    note_envelope(envelope)
    meta = envelope.get("_meta") or {}
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    # An adapter that already resolved cost (datalab converts cents once, at the source) passes
    # it through `usage.cost_usd`; otherwise look for the known vendor shapes in the envelope.
    if isinstance(usage.get("cost_usd"), (int, float)):
        _state()["cost_usd"] = float(usage["cost_usd"])
    else:
        for blob in (usage, envelope.get("usage"), envelope):
            usd, _ = extract_cost(blob if isinstance(blob, dict) else {})
            if usd is not None:
                _state()["cost_usd"] = usd
                break
    # LLM legs record every completion pre-parse, so a fenced or unparseable answer can be
    # salvaged from the record rather than re-paid for, and billing is per attempt.
    calls = usage.get("calls")
    if isinstance(calls, list):
        _state()["llm_calls"].extend(calls)
    result = envelope.get("result", envelope)
    if proc.returncode != 0 and isinstance(result, dict) and "__error__" in result:
        return result
    return as_extraction(result)


def _merge_subprocess_http(path: str):
    """Pull the child's transport records into this thread's log, then remove the file.

    Runs for failed calls too: an adapter that exits non-zero is exactly the case where the
    vendor's response body is the only thing that explains why.
    """
    try:
        records = json.loads(Path(path).read_text())
        for rec in records if isinstance(records, list) else []:
            _state()["http"].append(rec)
            try:
                CAPTURE.note_job_id(json.loads(rec.get("body") or ""),
                                    f"{rec.get('method')} {str(rec.get('url', '')).rsplit('/', 1)[-1]}")
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def extract_one(provider: str, pdf: Path, schema: dict, args):
    """Dispatch one document to one provider, at that provider's maximum tier."""
    timeout = args.timeout
    if provider == "reducto":
        return with_transient_retry(
            lambda: run_cli_adapter("harness.providers.reducto", pdf, schema, timeout), provider)
    if provider == "llamaextract":
        os.environ["LLAMAEXTRACT_TIER"] = LLAMAEXTRACT_TIER
        return with_transient_retry(
            lambda: run_cli_adapter("harness.providers.llamaextract", pdf, schema, timeout), provider)
    if provider == "extend":
        return with_transient_retry(
            lambda: run_cli_adapter("harness.providers.extend", pdf, deref(schema), timeout), provider)
    if provider == "mistral":
        return with_transient_retry(
            lambda: run_cli_adapter("harness.providers.mistral", pdf, schema, timeout), provider)
    if provider == "azure-cu":
        return with_transient_retry(
            lambda: run_cli_adapter("harness.providers.azure_cu", pdf, schema, timeout,
                                    ("--completion-model", args.completion_model)), provider)
    if provider in LLM_MODELS:
        model = LLM_MODELS[provider]
        return with_transient_retry(
            lambda: run_cli_adapter("harness.providers.llm_single_shot", pdf, schema, timeout,
                                    ("--model", model, "--max-output-tokens",
                                     str(MODEL_MAX_OUTPUT.get(model, DEFAULT_MAX_OUTPUT_TOKENS)))),
            provider)
    if provider.startswith("datalab"):
        mode = args.mode or ("accurate" if provider.endswith("accurate") else "balanced")
        return with_transient_retry(
            lambda: run_cli_adapter("harness.providers.datalab", pdf, schema, timeout,
                                    ("--mode", mode)), provider)
    raise SystemExit(f"unknown provider {provider}")


# ── document set ─────────────────────────────────────────────────────────────────
def load_documents(data_root: Path, subsets=None):
    """Every `<data-root>/<subset>/<doc>/` holding a document.pdf and a schema.json."""
    docs = []
    for sub in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if subsets and sub.name not in subsets:
            continue
        for d in sorted(p for p in sub.iterdir() if p.is_dir()):
            if (d / "document.pdf").exists() and (d / "schema.json").exists():
                docs.append((sub.name, d.name))
    return docs


def needs_run(out_path: Path) -> bool:
    """True when a document has no prediction, or holds a TRANSIENT failure worth re-attempting.

    Skipping anything with a file on disk turns a network blip into a permanent zero on an
    unattended run. Genuine vendor results are never re-attempted: a context-length rejection
    is an answer, and `is_transient` is the same predicate the in-run retry uses, so
    "transient" has one definition rather than two.
    """
    if not out_path.exists():
        return True
    try:
        result = json.loads(out_path.read_text()).get("result")
    except Exception:  # noqa: BLE001
        return True
    if usable(result):
        return False
    err = (result or {}).get("__error__") if isinstance(result, dict) else None
    return bool(err) and is_transient(err)


def cost_record(wall_s: float):
    """Per-document cost: real billing where the vendor returns it, null where it does not.

    A guessed cost sitting next to a measured one, in the same column, is how a cost comparison
    becomes fiction. Wall-clock and attempt count are recorded instead so the gap is visible.
    """
    st = _state()
    calls = st["llm_calls"]
    usd = st["cost_usd"]
    if usd is None and calls:
        costs = [c.get("cost_usd") for c in calls if c.get("cost_usd") is not None]
        usd = round(sum(costs), 6) if costs else None
    return {
        "usd": usd,
        "source": ("openrouter usage.cost" if calls and usd is not None
                   else ("vendor response" if usd is not None else None)),
        "tokens_in": sum((c.get("usage") or {}).get("prompt_tokens") or 0 for c in calls) or None,
        "tokens_out": sum((c.get("usage") or {}).get("completion_tokens") or 0 for c in calls) or None,
        "attempts": len(calls) or None,
        "billed_out_of_band": usd is None,
        "wall_s": round(wall_s, 1),
        "vendor_usage": CAPTURE.usage() or None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("provider", choices=PROVIDERS)
    ap.add_argument("--data-root", required=True, type=Path,
                    help="HF dataset layout: <root>/<subset>/<doc>/{document.pdf,schema.json}")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--subsets", nargs="*", help="limit to these subset directories")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("OEB_TIMEOUT", DEFAULT_TIMEOUT)),
                    help="uniform per-document budget, seconds (the same for every provider)")
    ap.add_argument("--mode", help="datalab extraction mode (default: balanced, or accurate for datalab-accurate)")
    ap.add_argument("--completion-model", default=os.environ.get("AZURE_CU_COMPLETION_MODEL", "gpt-4.1-mini"),
                    help="azure-cu completion deployment; publish this beside the score")
    ap.add_argument("--no-overlay", action="store_true",
                    help="do not write benchmark conventions into field descriptions")
    a = ap.parse_args(argv)

    provider = a.provider
    os.environ["OEB_TIMEOUT"] = str(a.timeout)
    os.environ.setdefault("OEB_DATA_ROOT", str(a.data_root))
    out_dir = a.out_root / provider
    docs = load_documents(a.data_root, a.subsets)
    todo = [(s, d) for s, d in docs if needs_run(out_dir / s / f"{d}.json")]
    if a.max_docs:
        todo = todo[:a.max_docs]
    print(f"{provider}: {len(docs)} documents, {len(todo)} to run, timeout {a.timeout:.0f}s, "
          f"tier {PROVIDER_TIER.get(provider)}", flush=True)
    if not todo:
        return 0

    stop = threading.Event()
    reason: list[str] = []
    counts = {"ok": 0, "err": 0, "skip": 0}
    lock = threading.Lock()

    def run_one(job):
        subset, doc = job
        if stop.is_set():
            return "skip"                      # leave no file: the document stays resumable
        src = a.data_root / subset / doc
        out_path = out_dir / subset / f"{doc}.json"
        raw_path = out_dir / "_raw" / subset / f"{doc}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        _reset_state()
        t0 = time.time()
        schema = json.loads((src / "schema.json").read_text())
        sent = strip_bench_keys(schema if a.no_overlay else SO.apply_overlay(schema))
        try:
            result = extract_one(provider, src / "document.pdf", sent, a)
        except Exception as exc:  # noqa: BLE001 -- the message is the result
            result = {"__error__": f"{type(exc).__name__}: {exc}"[:600]}
        if isinstance(result, dict) and not result:
            result = classify_no_output(time.time() - t0, a.timeout)
        st = _state()
        try:
            raw_path.write_text(json.dumps({
                "raw": result, "provider": provider, "subset": subset, "doc": doc,
                "llm_calls": st["llm_calls"] or None,
                "cost": cost_record(time.time() - t0),
                "vendor_envelope": st["envelope"],
                "http": st["http"],
                "job_ids": st["job_ids"],
                "schema_sent": sent,
                "run_manifest": {
                    "timeout_s": a.timeout,
                    "tier": PROVIDER_TIER.get(provider),
                    "model": LLM_MODELS.get(provider),
                    "max_output_tokens": MODEL_MAX_OUTPUT.get(LLM_MODELS.get(provider),
                                                              DEFAULT_MAX_OUTPUT_TOKENS),
                    "conventions_applied": not a.no_overlay and json.dumps(sent) != json.dumps(strip_bench_keys(schema)),
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }}, default=str))
        except Exception:  # noqa: BLE001 -- capture must never lose the prediction
            pass
        if isinstance(result, dict) and is_account_failure(result.get("__error__")):
            if not stop.is_set():
                reason.append(str(result.get("__error__"))[:200])
                stop.set()
            return "skip"
        out_path.write_text(json.dumps({"result": result, "_secs": round(time.time() - t0, 1)},
                                       default=str))
        return "err" if (isinstance(result, dict) and "__error__" in result) else "ok"

    workers = a.workers or WORKERS.get(provider, 5)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, status in enumerate(ex.map(run_one, todo), 1):
            with lock:
                counts[status] += 1
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] ok={counts['ok']} err={counts['err']}"
                      + (f" skipped={counts['skip']}" if counts["skip"] else ""), flush=True)
    if stop.is_set():
        print(f"{provider.upper()} STOPPED: account-level failure, {counts['skip']} documents left "
              f"untouched and resumable -- {(reason or [''])[0]}", flush=True)
    print(f"{provider.upper()} DONE ok={counts['ok']} err={counts['err']}"
          + (f" skipped={counts['skip']}" if counts["skip"] else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
