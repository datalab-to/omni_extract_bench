"""The single transport-capture implementation. Imported by both the parent harness and the
subprocess `sitecustomize`, so there is exactly one definition of "record a call".

Why one definition: this tap was written three times -- once in the harness, once for
subprocesses, once for the public package -- and every gap found so far had to be fixed in each
copy. Two implementations of the same idea that must agree eventually will not.

Four transports are covered, and each was discovered only after a provider silently captured
nothing while still writing a capture file that looked populated:

  httpx.Client        the obvious one
  httpx.AsyncClient   several vendor SDKs are async underneath
  requests.Session    other SDKs use requests
  urllib.request      an adapter that calls the REST API directly, with no SDK at all

The lesson is in the shape, not the list: a capture gap is invisible from the outside, so cover
every transport the code could use rather than the one you believe it uses. Each of these was
"surely the last one".
"""
from __future__ import annotations

import io
import json
import re
import time

MAX_BODY = 20000

ID_KEYS = ("request_id", "job_id", "jobId", "run_id", "runId", "id",
           "operation_id", "operationId", "task_id")

# Usage field names seen so far, kept as DOCUMENTATION rather than as a filter. Several vendors
# report `credits`, which is NOT dollars: the rate is contract-specific, so units stay attached
# to the value and conversion is the caller's explicit decision.
#
# Nothing is filtered against this list. An earlier version used it to SELECT keys, and so
# discarded a vendor's `num_pages_billed` purely because the name was not on it -- repeating,
# one level down, the hard-coded-path mistake that had already made a provider look like it
# reported no cost at all. Whatever the vendor puts in its usage block is what gets kept.
USAGE_FIELDS = ("credits", "num_pages", "num_pages_billed", "num_pages_extracted", "pages",
                "num_fields", "cost", "extract_mode", "tier")

# The bulk of an extraction request is an encoded document; the part that causes argument
# errors is everything else. Eliding the former makes keeping the latter affordable.
_B64_RUN = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")


def scrub_payload(raw):
    """Request body as text, with long base64 runs replaced by a length marker."""
    if raw is None:
        return None
    try:
        text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    except Exception:  # noqa: BLE001
        return None
    return _B64_RUN.sub(lambda m: f"<base64 {len(m.group(0))} chars elided>", text)[:MAX_BODY]


def _as_text(body):
    if isinstance(body, (bytes, bytearray)):
        return body.decode("utf-8", "replace")
    return body if isinstance(body, str) else str(body)


def _find_usage(parsed, depth=0):
    """Find a usage block anywhere in a response, not just at a fixed path.

    Vendors nest it differently -- top level, under `result`, under `data` -- and hard-coding
    one path is how a provider gets recorded as "does not report cost" when it does.
    """
    if depth > 4 or not isinstance(parsed, dict):
        return None
    usage = parsed.get("usage")
    if isinstance(usage, dict) and usage:
        return {k: v for k, v in usage.items() if not isinstance(v, (dict, list))}
    for value in parsed.values():
        if isinstance(value, dict):
            found = _find_usage(value, depth + 1)
            if found:
                return found
    return None


class Capture:
    """Collects transport records and the job ids seen in them.

    ``sink`` supplies the list to append to. The harness passes a thread-local list, because
    documents are graded by a pool and a shared list interleaves them -- storing one document's
    response inside another document's capture, which is worse than storing nothing.
    """

    def __init__(self, sink=None, id_sink=None):
        self._records = []
        self._ids = []
        self._sink = sink or (lambda: self._records)
        self._id_sink = id_sink or (lambda: self._ids)

    def records(self):
        return self._sink()

    def job_ids(self):
        return self._id_sink()

    def reset(self):
        del self.records()[:]
        del self.job_ids()[:]

    def note_job_id(self, body, source=""):
        """Record the first identifier in a response body, including one level of nesting.

        Async APIs keep the job, so this id is what lets usage be recovered from job history
        later instead of paying to run the document again.
        """
        if not isinstance(body, dict):
            return
        for key in ID_KEYS:
            value = body.get(key)
            if isinstance(value, str) and 6 <= len(value) <= 80:
                entry = {"key": key, "id": value, "source": source}
                if entry not in self.job_ids():
                    self.job_ids().append(entry)
                return
        for nested in ("extractRun", "processorRun", "run", "job", "data", "result"):
            if isinstance(body.get(nested), dict):
                self.note_job_id(body[nested], source or nested)
                return

    def record(self, method, url, status, body, elapsed=None, request=None):
        """Append one record. Never raises: capture must not break the call it observes."""
        try:
            text = _as_text(body)
            usage = None
            try:
                parsed = json.loads(text)
                self.note_job_id(parsed,
                                 f"{method} {str(url).split('?')[0].rsplit('/', 1)[-1]}")
                usage = _find_usage(parsed)
            except Exception:  # noqa: BLE001 -- most bodies are not JSON
                pass
            entry = {
                "method": method,
                "url": str(url).split("?")[0],        # query strings carry keys
                "status": status,
                "elapsed_s": round(elapsed, 2) if elapsed is not None else None,
                "body": text[:MAX_BODY],
                "truncated": len(text) > MAX_BODY,
                "request": request,
            }
            # Billing lives at the END of a JSON response as often as the start, so a
            # head-only truncation quietly discards the one field the capture exists to keep.
            # Usage is parsed from the FULL text before truncating, and a tail slice is kept so
            # a truncated body is still diagnosable rather than merely present.
            if usage:
                entry["usage"] = usage
            if len(text) > MAX_BODY:
                entry["body_tail"] = text[-4000:]
            log = self.records()
            # Collapse a poll storm. One provider polled 582 times on a single document with
            # byte-identical responses; keeping each buys no evidence and buries the records
            # that differ. The count is kept, so duration stays recoverable.
            if log:
                prev = log[-1]
                if (prev.get("method"), prev.get("url"), prev.get("status"),
                        prev.get("body")) == (entry["method"], entry["url"],
                                              entry["status"], entry["body"]):
                    prev["repeats"] = prev.get("repeats", 1) + 1
                    prev["elapsed_s"] = entry["elapsed_s"]
                    return
            log.append(entry)
        except Exception:  # noqa: BLE001
            pass

    def usage(self):
        """Vendor usage from the captured records, in the VENDOR'S units.

        Keys are whatever the vendor used. Nothing is filtered against a known-names list,
        because doing that silently discarded a vendor's page counts once already.
        """
        found = {}
        for rec in self.records():
            usage = rec.get("usage")          # parsed at record time, before truncation
            if not usage:
                body = rec.get("body") or ""
                if '"usage"' not in body:
                    continue
                try:
                    usage = _find_usage(json.loads(body))
                except Exception:  # noqa: BLE001
                    continue
            if not usage:
                continue
            for key, value in usage.items():
                if value is not None:
                    found.setdefault(key, value)
            found.setdefault("_endpoint", rec.get("url"))
        return found

    # ── transports ───────────────────────────────────────────────────────────────

    def install(self):
        """Install every tap. Idempotent; a missing library is not an error."""
        self._tap_httpx_sync()
        self._tap_httpx_async()
        self._tap_requests()
        self._tap_urllib()

    def _tap_httpx_sync(self):
        try:
            import httpx
        except ImportError:
            return
        original = httpx.Client.send
        if getattr(original, "_oeb_tapped", False):
            return

        def send(client, request, **kwargs):
            started = time.time()
            response = original(client, request, **kwargs)
            try:
                self.record(request.method, request.url, response.status_code,
                            response.text, time.time() - started,
                            scrub_payload(getattr(request, "content", None)))
            except Exception:  # noqa: BLE001
                pass
            return response

        send._oeb_tapped = True
        httpx.Client.send = send

    def _tap_httpx_async(self):
        try:
            import httpx
        except ImportError:
            return
        original = httpx.AsyncClient.send
        if getattr(original, "_oeb_tapped", False):
            return

        async def send(client, request, **kwargs):
            started = time.time()
            response = await original(client, request, **kwargs)
            try:
                try:
                    text = response.text
                except Exception:  # noqa: BLE001 -- streaming response not read yet
                    await response.aread()
                    text = response.text
                self.record(request.method, request.url, response.status_code, text,
                            time.time() - started,
                            scrub_payload(getattr(request, "content", None)))
            except Exception:  # noqa: BLE001
                pass
            return response

        send._oeb_tapped = True
        httpx.AsyncClient.send = send

    def _tap_requests(self):
        try:
            import requests
        except ImportError:
            return
        original = requests.Session.send
        if getattr(original, "_oeb_tapped", False):
            return

        def send(session, request, **kwargs):
            started = time.time()
            response = original(session, request, **kwargs)
            try:
                self.record(request.method, request.url, response.status_code,
                            response.text, time.time() - started,
                            scrub_payload(getattr(request, "body", None)))
            except Exception:  # noqa: BLE001
                pass
            return response

        send._oeb_tapped = True
        requests.Session.send = send

    def _tap_urllib(self):
        """Tap ``urllib.request.urlopen``.

        Needed because an adapter can skip the vendor SDK entirely and call the REST API with
        the standard library -- which is exactly what one provider here did, capturing nothing
        through three rounds of "the tap is fixed now".

        The response body has to be read to be recorded, and reading consumes it, so the caller
        is handed back an equivalent response over the bytes already read. HTTPError is a
        response too, and is the case that matters most: it carries the vendor's explanation.
        """
        import urllib.error
        import urllib.request
        import urllib.response

        original = urllib.request.urlopen
        if getattr(original, "_oeb_tapped", False):
            return

        def _replay(raw, url, code, headers):
            # addinfourl derives .status from .code, and .status has no setter -- assigning it
            # raises and breaks the caller, which is a capture layer doing the one thing it
            # must never do.
            return urllib.response.addinfourl(io.BytesIO(raw), headers, url, code)

        def urlopen(url, *args, **kwargs):
            method = url.get_method() if hasattr(url, "get_method") else "GET"
            full = getattr(url, "full_url", url)
            body = getattr(url, "data", None)
            started = time.time()
            try:
                response = original(url, *args, **kwargs)
            except urllib.error.HTTPError as err:
                raw = err.read()
                self.record(method, full, err.code, raw, time.time() - started,
                            scrub_payload(body))
                raise urllib.error.HTTPError(
                    err.url, err.code, err.reason, err.headers, io.BytesIO(raw)) from None
            try:
                raw = response.read()
            except Exception:  # noqa: BLE001 -- unreadable body: leave the caller's object alone
                return response
            code = getattr(response, "status", None) or response.getcode()
            self.record(method, full, code, raw, time.time() - started, scrub_payload(body))
            return _replay(raw, full, code, response.headers)

        urlopen._oeb_tapped = True
        urllib.request.urlopen = urlopen
