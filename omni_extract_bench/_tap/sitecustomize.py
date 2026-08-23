"""Transport capture for provider adapters that run in a SUBPROCESS.

Python imports `sitecustomize` at interpreter start-up, before the adapter or any SDK loads, so
the taps are in place in time and no adapter needs to change. Monkey-patching in the parent
cannot reach a `uv run python -m ...` child, so without this the vendor providers capture
nothing while still writing a capture file that looks complete.

The taps themselves live in oeb_capture, next to this file, so the parent harness and this
child run exactly the same code. Records are flushed at exit to OEB_HTTP_LOG for the parent.
"""
import atexit
import json
import os

_OUT = os.environ.get("OEB_HTTP_LOG")

if _OUT:
    try:
        from oeb_capture import Capture

        CAPTURE = Capture()
        CAPTURE.install()

        @atexit.register
        def _flush():
            try:
                with open(_OUT, "w") as handle:
                    json.dump(CAPTURE.records(), handle)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 -- capture must never prevent the adapter from running
        pass
