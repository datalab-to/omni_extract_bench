"""Capture vendor responses at the transport layer, so a run is paid for once.

Benchmarking extraction vendors means spending real money on calls that are slow, sometimes
nondeterministic, and often async. What you keep from each call decides whether a failure costs
you an explanation or another invoice.

The mistake this module prevents is capturing at the WRONG LAYER. Wrapping the adapter records
its return value: the parsed extraction. By then the interesting parts are gone.

  * the model's text, when ``json.loads`` failed  -> a provider looks unreliable, and you cannot
    tell whether it returned prose, a fenced block, or nothing
  * the HTTP error body, reduced to a status code -> failures are unattributable, so a harness
    bug is indistinguishable from a vendor limitation
  * the vendor's own billing fields                -> the absence gets reported as "this vendor
    does not report cost", which is a claim about your harness

All three are one placement error, and all three were separately patched here before the cause
was seen.

This module is a thin, module-level API over ``_tap.oeb_capture.Capture``, which holds the only
implementation. The taps live there rather than here because a subprocess ``sitecustomize`` must
import them without this package being installed in the child environment -- and because the
same tap maintained as several near-copies is how three of the four transport gaps below
survived a round of "capture is fixed now":

    httpx.Client · httpx.AsyncClient · requests.Session · urllib.request

Cover every transport the code *could* use, not the one you believe it uses. A capture gap is
invisible from the outside: the file exists, the key is present, the list is empty.

Usage:

    from omni_extract_bench import capture

    capture.install_taps()
    capture.reset()                        # per document; records are thread-local
    result = run_one_document(pdf, schema)
    record = {"result": result, "http": capture.records(),
              "job_ids": capture.job_ids(), "usage": capture.usage_from_records()}

For adapters launched as their own process, patching the parent reaches nothing:

    env = capture.subprocess_env(os.environ, log_path)
    subprocess.run(cmd, env=env, ...)
    capture.merge_subprocess_log(log_path)     # on failure too -- especially then
"""
from __future__ import annotations

import json
import os
import sys
import threading

_TAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tap")
if _TAP_DIR not in sys.path:
    sys.path.insert(0, _TAP_DIR)

from oeb_capture import ID_KEYS, MAX_BODY, USAGE_FIELDS, Capture, scrub_payload  # noqa: E402,F401

_LOCAL = threading.local()


def _http():
    """Thread-local record list.

    Thread-local, not a module list: documents are typically graded by a pool, and a shared list
    interleaves them -- storing one document's response inside another document's capture, which
    is worse than storing nothing because it looks right.
    """
    if not hasattr(_LOCAL, "http"):
        _LOCAL.http = []
    return _LOCAL.http


def _ids():
    if not hasattr(_LOCAL, "ids"):
        _LOCAL.ids = []
    return _LOCAL.ids


CAPTURE = Capture(sink=_http, id_sink=_ids)


def install_taps():
    """Install every transport tap in this process. Idempotent."""
    CAPTURE.install()


def records():
    """Transport records captured on this thread."""
    return CAPTURE.records()


def job_ids():
    """Vendor job identifiers seen on this thread, as ``{"key", "id", "source"}`` dicts."""
    return CAPTURE.job_ids()


def reset():
    """Clear this thread's capture. Call between documents."""
    CAPTURE.reset()


def record(method, url, status, body, elapsed=None, request=None, headers=None):
    """Append one record by hand (the taps call this for you)."""
    CAPTURE.record(method, url, status, body, elapsed, request, headers=headers)


def note_job_id(body, source=""):
    """Record the first job identifier found in a response body."""
    CAPTURE.note_job_id(body, source)


def usage_from_records():
    """Vendor usage from the captured bodies, in the vendor's own units."""
    return CAPTURE.usage()


def tap_dir():
    """Directory holding the ``sitecustomize`` that taps a subprocess."""
    return _TAP_DIR


def subprocess_env(env, log_path):
    """Environment that makes a child interpreter capture its own HTTP to ``log_path``.

    Python imports ``sitecustomize`` at start-up, before the adapter or its SDK loads, so the
    taps are in place in time and no adapter needs to change.
    """
    child = dict(env)
    existing = child.get("PYTHONPATH", "")
    child["PYTHONPATH"] = f"{_TAP_DIR}{os.pathsep}{existing}" if existing else _TAP_DIR
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
