"""Transport capture for provider adapters that run in a SUBPROCESS.

The parent harness taps httpx/requests so every response body is recorded before anything
parses it. That tap covers the LLM providers, which run in-process -- and covered NONE of the
vendor providers, which are launched as `uv run python -m longextract_bench.providers.<x>` in
a separate interpreter with its own site-packages. Monkey-patching in the parent cannot reach
them. The captures for those providers held the adapter's parsed return value and an empty
`http` list, which is the exact failure the transport tap was written to end: a capture that
looks complete because the key is present.

Python imports `sitecustomize` automatically at interpreter start-up, before the adapter or
its SDK is imported, so the tap is installed in time and no adapter needs to change. Records
are flushed at exit to the path in OEB_HTTP_LOG for the parent to merge.
"""
import atexit
import json
import os
import re
import time

# The expensive part of an extraction request is a base64 PDF; the part that causes argument
# errors is everything else. Eliding the former makes capturing the latter affordable.
_B64_RUN = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")


def _scrub(raw):
    try:
        text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    except Exception:  # noqa: BLE001
        return None
    return _B64_RUN.sub(lambda m: f"<base64 {len(m.group(0))} chars elided>", text)[:20000]

_LOG = []
_OUT = os.environ.get("OEB_HTTP_LOG")


def _record(method, url, status, body, elapsed, request=None):
    try:
        text = body if isinstance(body, str) else str(body)
        entry = {
            "method": method, "url": str(url).split("?")[0], "status": status,
            "elapsed_s": round(elapsed, 2), "body": text[:20000],
            "truncated": len(text) > 20000, "request": request,
        }
        # collapse a poll storm; the repeat count is kept so duration stays recoverable
        if _LOG:
            prev = _LOG[-1]
            if (prev["method"], prev["url"], prev["status"], prev["body"]) == (
                    entry["method"], entry["url"], entry["status"], entry["body"]):
                prev["repeats"] = prev.get("repeats", 1) + 1
                prev["elapsed_s"] = entry["elapsed_s"]
                return
        _LOG.append(entry)
    except Exception:  # noqa: BLE001 -- capture must never break a call
        pass


def _wrap(cls, attr):
    original = getattr(cls, attr)
    if getattr(original, "_oeb_tapped", False):
        return

    def send(self, request, **kw):
        t0 = time.time()
        resp = original(self, request, **kw)
        try:
            _record(request.method, request.url, resp.status_code, resp.text,
                    time.time() - t0,
                    _scrub(getattr(request, "content", None) or getattr(request, "body", None)))
        except Exception:  # noqa: BLE001
            pass
        return resp

    send._oeb_tapped = True
    setattr(cls, attr, send)


def _wrap_async(cls, attr):
    """Tap an ASYNC client. Not optional: several vendor SDKs are async underneath, and a
    sync-only tap captures nothing for them while still writing a capture file that looks
    populated -- the same silent gap as tapping only the parent process."""
    original = getattr(cls, attr)
    if getattr(original, "_oeb_tapped", False):
        return

    async def send(self, request, **kw):
        t0 = time.time()
        resp = await original(self, request, **kw)
        try:
            try:
                text = resp.text
            except Exception:  # noqa: BLE001 -- streaming response not yet read
                await resp.aread()
                text = resp.text
            _record(request.method, request.url, resp.status_code, text, time.time() - t0,
                    _scrub(getattr(request, "content", None)))
        except Exception:  # noqa: BLE001
            pass
        return resp

    send._oeb_tapped = True
    setattr(cls, attr, send)


if _OUT:
    for _mod, _cls in (("httpx", "Client"), ("requests", "Session")):
        try:
            mod = __import__(_mod)
            _wrap(getattr(mod, _cls), "send")
        except Exception:  # noqa: BLE001 -- an SDK that uses the other library still works
            pass
    try:
        import httpx as _httpx
        _wrap_async(_httpx.AsyncClient, "send")
    except Exception:  # noqa: BLE001
        pass

    @atexit.register
    def _flush():
        try:
            with open(_OUT, "w") as fh:
                json.dump(_LOG, fh)
        except Exception:  # noqa: BLE001
            pass
