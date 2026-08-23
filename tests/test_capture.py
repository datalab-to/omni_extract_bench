#!/usr/bin/env python3
"""Tests for transport capture — including the subprocess path, which is where it broke.

The bug these guard against was not "capture is missing". A capture file existed for every
document, with a populated key named `raw`. What it held was the adapter's parsed return
value, so an audit that counted files and checked for the key reported full coverage while
the response bodies, error bodies and billing fields were all being discarded.

So these tests assert CONTENTS. The last one runs a real subprocess, because the in-process
tap passing tells you nothing about the child interpreter a shelled-out adapter runs in --
that gap is the whole reason capture silently covered only some providers.

Run: python3 tests/test_capture.py
"""
import json
import os
import subprocess
import sys
import tempfile


# run from anywhere: `python tests/x.py` puts tests/ on the path, not the repo root
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench import capture  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


print("\n[1] a record keeps the evidence, not a summary of it")
capture.reset()
capture.record("POST", "https://api.example.com/extract?api_key=SECRET", 200,
               '{"job_id": "abc123def", "usage": {"credits": 14.5, "num_pages": 2}}', 0.42)
rec = capture.records()[0]
check("status kept", rec["status"] == 200)
check("body kept", '"credits": 14.5' in rec["body"])
check("query string stripped from url", "SECRET" not in rec["url"], rec["url"])
check("timing kept", rec["elapsed_s"] == 0.42)

print("\n[2] job ids are harvested — they are what makes cost recoverable without re-running")
check("id found", capture.job_ids()[0]["id"] == "abc123def", str(capture.job_ids()))
capture.reset()
capture.record("POST", "https://api.example.com/run", 200,
               '{"extractRun": {"id": "nested-run-42"}}')
check("nested id found", capture.job_ids()[0]["id"] == "nested-run-42", str(capture.job_ids()))

print("\n[3] usage is harvested in the VENDOR'S units")
capture.reset()
capture.record("GET", "https://api.example.com/job/1", 200,
               '{"result": {"usage": {"credits": 17.47, "num_pages": 2, '
               '"extract_mode": "super_agent"}}}')
usage = capture.usage_from_records()
check("credits found", usage["credits"] == 17.47, str(usage))
check("pages found", usage["num_pages"] == 2)
check("vendor's own tier statement kept", usage["extract_mode"] == "super_agent")
check("no invented dollar figure", "usd" not in usage and "cost_usd" not in usage, str(usage))

print("\n[4] capture never breaks the call it observes")
capture.reset()
capture.record("GET", None, None, object())          # nonsense arguments
check("bad input does not raise", True)
capture.record("GET", "https://x/y", 500, "not json at all")
check("non-json body still recorded", capture.records()[-1]["body"] == "not json at all")

print("\n[5] a large body is truncated but FLAGGED, never silently cut")
capture.reset()
capture.record("GET", "https://x/y", 200, "z" * (capture.MAX_BODY + 500))
check("truncation flagged", capture.records()[0]["truncated"] is True)


print("\n[6] the request is kept too, minus the payload")
# Without the request, a 400 that says "Request contains an invalid argument" is not
# diagnosable, and the only way to learn which argument is to pay for the call again.
capture.reset()
payload = json.dumps({"model": "some-model", "max_tokens": 64000,
                      "image": "A" * 5000, "schema": {"type": "object"}})
capture.record("POST", "https://api.example.com/v1/chat", 400,
               '{"error": {"message": "Request contains an invalid argument."}}',
               0.3, request=capture.scrub_payload(payload))
req = capture.records()[0]["request"]
check("request kept", req is not None)
check("the argument-bearing fields survive", '"max_tokens": 64000' in req and
      '"model": "some-model"' in req, (req or "")[:120])
check("base64 payload elided", "AAAA" not in req, (req or "")[:160])
check("elision is marked, not silent", "elided" in req, (req or "")[:160])
check("request is far smaller than the payload", len(req) < len(payload) / 4,
      f"{len(req)} vs {len(payload)}")


print("\n[7] ASYNC clients are tapped too")
# A sync-only tap captures nothing for an async SDK while still writing a capture file that
# looks populated. One provider in this benchmark did exactly that, after the subprocess gap
# had already been found and fixed.
capture.reset()
capture.install_taps()
try:
    import asyncio
    import httpx

    async def _go():
        async with httpx.AsyncClient(timeout=10) as client:
            await client.get("https://example.com")

    asyncio.run(_go())
except Exception as exc:  # noqa: BLE001
    print(f"  SKIP  async client unavailable or offline ({type(exc).__name__})")
else:
    got = capture.records()
    check("async request captured", len(got) >= 1, f"{len(got)} records")
    if got:
        check("async record carries a status", got[0]["status"] is not None)
        check("async record carries a body", bool(got[0]["body"]))


print("\n[8] a poll storm collapses, but its duration survives")
capture.reset()
for _ in range(50):
    capture.record("GET", "https://api.example.com/job/1", 200, '{"status":"Running"}', 0.2)
capture.record("GET", "https://api.example.com/job/1", 200, '{"status":"Succeeded"}', 0.2)
recs = capture.records()
check("identical polls collapsed", len(recs) == 2, f"{len(recs)} records")
check("repeat count retained", recs[0].get("repeats") == 50, str(recs[0].get("repeats")))
check("the DIFFERING response is kept", "Succeeded" in recs[1]["body"])


print("\n[9] urllib is tapped — an adapter may use no SDK at all")
# The provider that exposed this called the REST API with the standard library, so three
# rounds of "the tap is fixed" (sync, then async, then subprocess) all still captured nothing
# for it. A capture layer must also not BREAK its caller: reading a urllib response consumes
# it, so the caller is handed back an equivalent response over the bytes already read.
capture.reset()
capture.install_taps()
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
check("urlopen is tapped", getattr(urllib.request.urlopen, "_oeb_tapped", False))
try:
    resp = urllib.request.urlopen("https://example.com", timeout=15)
    body = resp.read()
except Exception as exc:  # noqa: BLE001
    print(f"  SKIP  network unavailable ({type(exc).__name__})")
else:
    check("caller still receives the body", len(body) > 0, f"{len(body)} bytes")
    check("caller still receives the status", resp.status == 200, str(resp.status))
    check("caller still receives getcode()", resp.getcode() == 200)
    check("the call was captured", any(r["url"] == "https://example.com"
                                       for r in capture.records()))

print("\n[10] THE SUBPROCESS PATH — a child interpreter captures its own HTTP")
# The in-process tap cannot reach a child, and this is the case that silently failed: the
# harness recorded an empty `http` list for every provider whose adapter it shelled out to.
capture.reset()
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
    log_path = handle.name
env = capture.subprocess_env(os.environ, log_path)
check("sitecustomize is on the child PYTHONPATH", capture.tap_dir() in env["PYTHONPATH"])

child = subprocess.run(
    [sys.executable, "-c",
     "import httpx\n"
     "try:\n"
     "    httpx.Client(timeout=5).get('https://example.com')\n"
     "except Exception:\n"
     "    pass\n"],
    env=env, capture_output=True, text=True, timeout=90)

capture.merge_subprocess_log(log_path)
got = capture.records()
if child.returncode != 0 and not got:
    print("  SKIP  child could not run httpx — no network or httpx missing")
else:
    check("child's request reached the parent's capture", len(got) >= 1,
          f"{len(got)} records, child rc={child.returncode}")
    if got:
        check("child record carries a status", got[0]["status"] is not None, str(got[0])[:120])
        check("child record carries a body", got[0]["body"] is not None)
check("child log file cleaned up", not os.path.exists(log_path))

print(f"\n{'ALL CAPTURE TESTS PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
