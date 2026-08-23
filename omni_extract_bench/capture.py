"""Capture vendor responses at the transport layer, so a run is paid for once.

Benchmarking extraction vendors means spending real money on calls that are slow, sometimes
nondeterministic, and often async. What you keep from each call decides whether a failure
costs you an explanation or another invoice.

The mistake this module exists to prevent -- made here, twice, in the same benchmark -- is
capturing at the WRONG LAYER. Wrapping the adapter records its return value: the parsed
extraction. By then the interesting parts are gone.

  * the model's text, when ``json.loads`` failed  -> a provider looks unreliable, and you
    cannot tell whether it returned prose, a fenced block, or nothing
  * the HTTP error body, reduced to a status code -> failures are unattributable, so a
    harness bug is indistinguishable from a vendor limitation
  * the vendor's own billing fields                -> the absence gets reported as "this
    vendor does not report cost", which is a claim about your harness

All three are one placement error, and all three were separately patched before the cause was
seen. Capturing at the transport means the response body is recorded before anything gets a
chance to interpret it, and every one of those questions stays answerable for free.

Two things to install, because most harnesses have two execution shapes:

    install_taps()                      # in-process SDK calls
    env = subprocess_env(os.environ)    # adapters launched as their own interpreter

The second is not optional-in-practice. Monkey-patching cannot reach a subprocess, so a
harness that taps only the parent captures nothing for exactly the providers whose adapters it
shells out to -- while still writing a capture file that looks complete. Verify contents, not
the presence of a key: see ``tests/test_capture.py``.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

_LOCAL = threading.local()

#: Response keys that identify a job to the vendor. Async extraction APIs hand back an id and
#: keep the job; that id is what lets you recover usage LATER from job history instead of
#: re-running the document. Recovering a number a vendor will still give you, by paying to
#: generate it again, is the worst available trade -- and for a nondeterministic provider it
#: does not even reproduce the answer that was scored.
ID_KEYS = ("request_id", "job_id", "jobId", "run_id", "runId", "id",
           "operation_id", "operationId", "task_id")

#: Usage fields worth keeping, in the vendor's own units. Several report `credits`, which is
#: NOT dollars: the credit rate is contract-specific. Recording credits in a column named for
#: dollars invents a number, so units stay attached to the value and conversion is the
#: caller's explicit decision.
USAGE_FIELDS = ("credits", "num_pages", "pages", "num_fields", "cost", "extract_mode", "tier")

MAX_BODY = 20000

#: Long base64 runs are elided from captured REQUESTS. A vendor 400 reading "Request contains
#: an invalid argument" names no argument, so without the request the failure is not
#: diagnosable and the only way to learn more is to pay for the call again. What makes keeping
#: requests affordable is that their bulk is an encoded document, while the part that causes
#: argument errors -- schema, model, generation parameters -- is small.
_B64_RUN = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")


def scrub_payload(raw):
    """Return a request body as text, with long base64 runs replaced by a length marker."""
    if raw is None:
        return None
    try:
        text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    except Exception:  # noqa: BLE001
        return None
    return _B64_RUN.sub(lambda m: f"<base64 {len(m.group(0))} chars elided>", text)[:MAX_BODY]


def records():
    """Transport records captured on this thread.

    Thread-local, not a module list: documents are typically graded by a pool, and a shared
    list interleaves them -- storing one document's response inside another document's
    capture, which is worse than storing nothing because it looks right.
    """
    if not hasattr(_LOCAL, "http"):
        _LOCAL.http = []
    return _LOCAL.http


def job_ids():
    """Vendor job identifiers seen on this thread, as ``{"key", "id", "source"}`` dicts."""
    if not hasattr(_LOCAL, "ids"):
        _LOCAL.ids = []
    return _LOCAL.ids


def reset():
    """Clear this thread's capture. Call between documents."""
    _LOCAL.http = []
    _LOCAL.ids = []


def note_job_id(body, source=""):
    """Record the first identifier found in a response body, including one level of nesting."""
    if not isinstance(body, dict):
        return
    for key in ID_KEYS:
        value = body.get(key)
        if isinstance(value, str) and 6 <= len(value) <= 80:
            entry = {"key": key, "id": value, "source": source}
            if entry not in job_ids():
                job_ids().append(entry)
            return
    for nested in ("extractRun", "processorRun", "run", "job", "data", "result"):
        if isinstance(body.get(nested), dict):
            note_job_id(body[nested], source or nested)
            return


def record(method, url, status, body, elapsed=None, request=None):
    """Append one transport record. Never raises: capture must not break the call it observes."""
    try:
        text = body if isinstance(body, str) else str(body)
        try:
            note_job_id(json.loads(text), f"{method} {str(url).split('?')[0].rsplit('/', 1)[-1]}")
        except Exception:  # noqa: BLE001 -- most bodies are not JSON, and that is fine
            pass
        records().append({
            "method": method,
            "url": str(url).split("?")[0],       # query strings carry keys
            "status": status,
            "elapsed_s": round(elapsed, 2) if elapsed is not None else None,
            "body": text[:MAX_BODY],
            "truncated": len(text) > MAX_BODY,
            "request": request,
        })
    except Exception:  # noqa: BLE001
        pass


def install_taps():
    """Tap ``httpx.Client.send`` and ``requests.Session.send`` for in-process calls.

    Both, because vendor SDKs disagree about which to use and you should not have to know.
    Idempotent, and a missing library is not an error.
    """
    for module_name, class_name in (("httpx", "Client"), ("requests", "Session")):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        cls = getattr(module, class_name)
        original = cls.send
        if getattr(original, "_oeb_tapped", False):
            continue

        def send(self, request, _original=original, **kwargs):
            started = time.time()
            response = _original(self, request, **kwargs)
            try:
                record(request.method, request.url, response.status_code,
                       response.text, time.time() - started,
                       scrub_payload(getattr(request, "content", None)
                                     or getattr(request, "body", None)))
            except Exception:  # noqa: BLE001
                pass
            return response

        send._oeb_tapped = True
        cls.send = send


def tap_dir():
    """Directory holding the ``sitecustomize`` that taps a subprocess."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tap")


def subprocess_env(env, log_path):
    """Environment that makes a child interpreter capture its own HTTP to ``log_path``.

    Python imports ``sitecustomize`` at start-up, before the adapter or its SDK loads, so the
    tap is in place in time and no adapter needs to change.
    """
    child = dict(env)
    existing = child.get("PYTHONPATH", "")
    child["PYTHONPATH"] = f"{tap_dir()}{os.pathsep}{existing}" if existing else tap_dir()
    child["OEB_HTTP_LOG"] = log_path
    return child


def merge_subprocess_log(log_path, remove=True):
    """Fold a child's records into this thread's capture.

    Call it for failed adapters too: a non-zero exit is precisely when the vendor's response
    body is the only thing that explains what happened.
    """
    try:
        with open(log_path) as handle:
            child_records = json.load(handle)
        for rec in child_records if isinstance(child_records, list) else []:
            records().append(rec)
            try:
                note_job_id(json.loads(rec.get("body") or ""),
                            f"{rec.get('method')} {str(rec.get('url', '')).rsplit('/', 1)[-1]}")
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        if remove:
            try:
                os.unlink(log_path)
            except OSError:
                pass


def usage_from_records(recs=None):
    """Harvest vendor usage out of captured bodies, keeping the vendor's units.

    Returns a flat dict of whatever was found -- ``credits``, ``num_pages``, ``extract_mode``
    and friends. ``extract_mode`` is worth keeping beside the billing: it is the vendor's own
    statement of which tier served the request, which is better evidence than your config
    claiming it asked for one.
    """
    found = {}
    for rec in (records() if recs is None else recs):
        body = rec.get("body") or ""
        if '"usage"' not in body:
            continue
        try:
            parsed = json.loads(body)
        except Exception:  # noqa: BLE001
            continue
        usage = None
        for candidate in ((parsed.get("result") or {}), parsed):
            if isinstance(candidate, dict) and isinstance(candidate.get("usage"), dict):
                usage = candidate["usage"]
                break
        if not usage:
            continue
        for key in USAGE_FIELDS:
            if usage.get(key) is not None:
                found.setdefault(key, usage[key])
        found.setdefault("_endpoint", rec.get("url"))
    return found
