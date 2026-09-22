#!/usr/bin/env python3
"""`predict` -- the whole per-document procedure, checked without paying a vendor.

Every parity rule has to hold inside this one function, and they are invisible when they break:
an unannotated schema still returns an extraction, a retried 400 still returns an extraction, a
recorded credit failure still returns an extraction. All of it looks like a working run and
scores like a different benchmark.

So the ADAPTER is stubbed -- the vendor and its transport are not what is under test -- and
what it receives, and what `predict` does with what it returns, is.

Run: python3 tests/test_predict.py
"""
import dataclasses
import json
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

# TWO handles, because they are two modules and the tests PATCH both: `document` holds
# `predict` and the retry policy, `registry` holds the lookup and the `Config`. Aliasing them
# to one name silently sends a patch to the wrong module -- `TRANSIENT_BACKOFF = (0, 0, 0)`
# set on the registry leaves `document` sleeping the real 20/60/120.
from omni_extract_bench.harness import document, registry  # noqa: E402
from omni_extract_bench.harness.contract import Cost, Extraction
from omni_extract_bench.harness.errors import AccountFailure, MissingCredential, MissingDependency, VendorError, VendorTimeout

FAILS = []
document.TRANSIENT_BACKOFF = (0, 0, 0)
_REAL_ADAPTER = registry.adapter
_REAL_DOC_ADAPTER = document.adapter


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


SCHEMA = {
    "type": "object",
    "properties": {
        "capex": {"type": "number", "description": "Capital expenditures.",
                  "evaluation_config": {"tolerance": 0.01}},
        "vendor": {"type": "string", "default": "unknown"},
    },
}
PDF = Path(tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name)
PDF.write_bytes(b"%PDF-1.4\n")

seen = {}


def stub(fn):
    """Install `fn` as every provider's `extract`, keeping the real adapter otherwise.

    The stub's own signature does not matter: the options come from the REAL adapter's
    `Config`, which is a declaration rather than something inferred from whatever is standing
    in for `extract`. A stub used to have to replicate the signature to be believed.

    `Config` and `prepare_schema` come from the real adapter for the same reason -- a stand-in
    that answered those itself would be testing the stand-in.
    """
    def counted(pdf, schema, *, timeout=None, config=None):
        opts = dataclasses.asdict(config) if config is not None else {}
        seen.clear()
        seen.update(pdf=pdf, schema=schema, opts=opts, timeout=timeout)
        counted.calls += 1
        return fn(pdf, schema, timeout=timeout, **opts)
    counted.calls = 0

    def lookup(provider):
        real = _REAL_ADAPTER(provider)
        # __name__ too: an adapter is a MODULE, and `resolve` reads it off whatever the
        # lookup returns.
        return types.SimpleNamespace(__name__=real.__name__, Config=real.Config,
                                     prepare_schema=real.prepare_schema,
                                     extract=counted)
    # Substituted on `document`, which is where `predict` reads the adapter. `registry` keeps
    # the real one, so `config_for` still builds the real Config -- this stub fakes the CALL,
    # not the declaration.
    document.adapter = lookup
    return counted


def _refused(name):
    try:
        registry.resolve(name)
        return False
    except ValueError as exc:
        return "datalab" in str(exc)


OK = Extraction(result={"capex": 98.2}, raw={"body": "..."},
                cost=Cost(usd=0.031, source="usage.cost"))

print("WHAT THE VENDOR IS HANDED")
stub(lambda pdf, schema, **o: OK)
rec = document.predict("mistral", PDF, SCHEMA)
sent = seen["schema"]
report("benchmark-only annotations never reach the vendor",
       "evaluation_config" not in json.dumps(sent) and '"default"' not in json.dumps(sent),
       json.dumps(sent))
report("conventions are stated in the description the vendor reads",
       len(sent["properties"]["capex"]["description"])
       > len(SCHEMA["properties"]["capex"]["description"]))
report("...and the record says so", rec["run_manifest"]["conventions_applied"] is True)
report("overlay=False leaves the description alone",
       document.predict("mistral", PDF, SCHEMA, overlay=False)["run_manifest"]
       ["conventions_applied"] is False)
report("the caller's schema is never mutated",
       "evaluation_config" in SCHEMA["properties"]["capex"])
report("the provider's maximum-tier options are passed to the adapter",
       document.predict("datalab", PDF, SCHEMA) is not None
       and seen["opts"]["mode"] == "balanced", str(seen["opts"]))
report("the uniform budget is passed to the adapter",
       document.predict("mistral", PDF, SCHEMA, timeout=900) is not None
       and 899 < seen["timeout"] <= 900, str(seen["timeout"]))

_given, _backoff = [], document.TRANSIENT_BACKOFF
document.TRANSIENT_BACKOFF = (0, 0, 0)
stub(lambda pdf, schema, **o: (_given.append(o["timeout"]),
                               (_ for _ in ()).throw(VendorError("HTTP 503", status=503)))[1])
_rec = document.predict("mistral", PDF, SCHEMA, timeout=5)
document.TRANSIENT_BACKOFF = _backoff
report("a retried document does not get a fresh budget each time",
       all(a > b for a, b in zip(_given, _given[1:])), str([round(t, 3) for t in _given]))
report("...and the whole run stays inside one budget",
       _given and _given[0] <= 5 and _given[-1] > 0, str([round(t, 3) for t in _given]))
stub(lambda pdf, schema, **o: OK)

print("\nTHE ANSWER TRAVELS WITH ITS EVIDENCE")
rec = document.predict("mistral", PDF, SCHEMA)
for key in ("result", "raw", "cost", "schema_sent", "run_manifest", "error", "job_id"):
    report(f"the record carries `{key}`", key in rec)
report("the cost the vendor reported is kept", rec["cost"]["usd"] == 0.031)
report("...with the field it was read from", rec["cost"]["source"] == "usage.cost")
report("a vendor that reports no cost is billed out of band, not guessed",
       (stub(lambda p, s, **o: Extraction(result={"a": 1})) is not None
        and document.predict("extend", PDF, SCHEMA)["cost"]["billed_out_of_band"] is True))
_sent = document.predict("datalab", PDF, SCHEMA)["run_manifest"]["settings"]
report("what the provider was actually sent is recorded, not a name for it",
       _sent == seen["opts"], f'{_sent} vs {seen["opts"]}')
report("...and the contract's own arguments are not settings", "timeout" not in _sent)

print("\nA RUN SAYS WHAT IT ASKED FOR")
stub(lambda pdf, schema, **o: OK)
import re as _re
report("a run is named for the vendor and a digest of what it was sent",
       _re.fullmatch(r"datalab-[0-9a-f]{8}", registry.out_name("datalab")) is not None,
       registry.out_name("datalab"))
_rec = document.predict("datalab", PDF, SCHEMA, mode="accurate")
report("an overridden option reaches the adapter", seen["opts"]["mode"] == "accurate")
report("...and is recorded in the run manifest",
       _rec["run_manifest"]["settings"]["mode"] == "accurate", str(_rec["run_manifest"]))
report("an option equal to the stock value names the same run",
       registry.out_name("datalab", {"mode": "balanced"}) == registry.out_name("datalab"),
       registry.out_name("datalab", {"mode": "balanced"}))

def _refused_with(provider, options, wanted):
    try:
        registry.config_for(provider, options)
        return False
    except ValueError as exc:
        return wanted in str(exc)


print("\nA MODEL IS NAMED IN FULL, NOT ALIASED")
report("a model id routes to the single-shot adapter",
       registry.resolve("openai/gpt-5.6-sol") == "llm_single_shot"
       and registry.settings_for("openai/gpt-5.6-sol")["model"] == "openai/gpt-5.6-sol")
report("a vendor name still routes to its own adapter",
       registry.resolve("datalab") == "datalab")
report("an OpenRouter suffix is part of the id, not a separator",
       registry.settings_for("mistralai/mistral-medium-3-5:batch")["model"]
       == "mistralai/mistral-medium-3-5:batch")
report("...and cannot be overridden into naming a different one than it ran",
       _refused_with("openai/gpt-5.6-sol", {"model": "anthropic/claude-opus-5"},
                     "is the provider name"))
report("a bare alias is refused, naming the vendors",
       (lambda: [False for _ in ()] or _refused("gpt"))())
report("a model id gets one directory, not a nested pair",
       registry.out_name("openai/gpt-5.6-sol").startswith("openai__gpt-5.6-sol-"),
       registry.out_name("openai/gpt-5.6-sol"))

print("\nRETRY ONLY WHAT A RETRY CAN FIX")
for status, label, want in ((503, "a 5xx", document.TRANSIENT_ATTEMPTS),
                            (429, "a rate limit", document.TRANSIENT_ATTEMPTS),
                            (400, "a real answer", 1)):
    called = stub(lambda p, s, st=status, **o: (_ for _ in ()).throw(
        VendorError(f"HTTP {st}", status=st)))
    rec = document.predict("mistral", PDF, SCHEMA)
    report(f"{label} ({status}) is attempted {want}x", called.calls == want,
           f"called {called.calls}x")
    report(f"...and the record says transient={status != 400}",
           rec["error"]["transient"] is (status != 400), str(rec["error"]))

called = stub(lambda p, s, **o: (_ for _ in ()).throw(VendorTimeout("still running after 412 polls")))
rec = document.predict("mistral", PDF, SCHEMA)
report("a timeout is never retried -- it would spend another full budget", called.calls == 1)
report("...and is marked in the manifest", rec["run_manifest"]["timed_out"] is True)
report("...and the reason survives, counted by the adapter that polled",
       "412 polls" in rec["error"]["message"], rec["error"]["message"])

print("\nNOT FACTS ABOUT THE DOCUMENT -- RAISED, NOT RECORDED")
for exc, kind in ((VendorError("HTTP 402: no credits", status=402), AccountFailure),
                  (VendorError("insufficient_quota", status=403), AccountFailure),
                  (MissingCredential("MISTRAL_API_KEY is not set"), MissingCredential)):
    stub(lambda p, s, e=exc, **o: (_ for _ in ()).throw(e))
    try:
        document.predict("mistral", PDF, SCHEMA)
        report(f"{type(exc).__name__} raises {kind.__name__}", False, "it returned a record")
    except kind:
        report(f"{type(exc).__name__} raises {kind.__name__}", True)
    except Exception as other:                                       # noqa: BLE001
        report(f"{type(exc).__name__} raises {kind.__name__}", False, f"got {other!r}")

document.adapter = _REAL_DOC_ADAPTER

print("\nA MISSING SDK IS FOUND AT THE IMPORT, NOT IN A MESSAGE")
# Importing the harness imports every adapter, so a missing SDK is one message at the import
# rather than a per-adapter surprise. Checked in a subprocess because this one already has it.
_blocked = subprocess.run(
    [sys.executable, "-c", textwrap.dedent("""
        import builtins
        _real_import = builtins.__import__
        def _no_httpx(name, *a, **k):
            if name == "httpx" or name.startswith("httpx."):
                # what Python itself raises for a module that is not installed -- an
                # `ImportError` here would be a different case, and the guard tells them apart
                raise ModuleNotFoundError("No module named 'httpx'", name="httpx")
            return _real_import(name, *a, **k)
        builtins.__import__ = _no_httpx
        try:
            import omni_extract_bench.harness
        except BaseException as exc:
            print(type(exc).__name__)
            print(exc)
    """)],
    capture_output=True, text=True, cwd=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
report("a missing SDK raises MissingDependency",
       "MissingDependency" in _blocked.stdout, _blocked.stdout + _blocked.stderr[-300:])
report("...naming the extra to install",
       "omni-extract-bench[harness]" in _blocked.stdout, _blocked.stdout)

print("\nAND AN UNKNOWN PROVIDER IS REFUSED BY NAME")
try:
    registry.adapter("nonesuch")
    report("an unknown provider raises", False, "it was accepted")
except ValueError as exc:
    report("an unknown provider raises, naming the ones that exist", "datalab" in str(exc))

PDF.unlink(missing_ok=True)
print(f"\n{'predict HOLDS THE PARITY RULES' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
sys.exit(1 if FAILS else 0)
