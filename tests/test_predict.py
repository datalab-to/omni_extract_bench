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
import json
import sys
import tempfile
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from omni_extract_bench.harness import vendor                                  # noqa: E402
from omni_extract_bench.harness.extraction import (                            # noqa: E402
    AccountFailure, Cost, Extraction, MissingCredential, MissingDependency,
    VendorError, VendorTimeout)

FAILS = []
vendor.TRANSIENT_BACKOFF = (0, 0, 0)          # no sleeping in a test
_REAL_ADAPTER = vendor.adapter


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


# A schema carrying both things the harness must change before a vendor sees it: a
# benchmark-only annotation, and a field name a convention matches on.
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
    """Install `fn` as every provider's adapter."""
    def counted(pdf, schema, **opts):
        seen.clear()
        seen.update(pdf=pdf, schema=schema, opts=opts)
        counted.calls += 1
        return fn(pdf, schema, **opts)
    counted.calls = 0
    vendor.adapter = lambda provider: counted
    return counted


def _refused(name):
    try:
        vendor.resolve(name)
        return False
    except ValueError as exc:
        return "datalab" in str(exc)


OK = Extraction(result={"capex": 98.2}, raw={"body": "..."},
                cost=Cost(usd=0.031, source="usage.cost"))

print("WHAT THE VENDOR IS HANDED")
stub(lambda pdf, schema, **o: OK)
rec = vendor.predict("mistral", PDF, SCHEMA)
sent = seen["schema"]
report("benchmark-only annotations never reach the vendor",
       "evaluation_config" not in json.dumps(sent) and '"default"' not in json.dumps(sent),
       json.dumps(sent))
report("conventions are stated in the description the vendor reads",
       len(sent["properties"]["capex"]["description"])
       > len(SCHEMA["properties"]["capex"]["description"]))
report("...and the record says so", rec["run_manifest"]["conventions_applied"] is True)
report("overlay=False leaves the description alone",
       vendor.predict("mistral", PDF, SCHEMA, overlay=False)["run_manifest"]
       ["conventions_applied"] is False)
report("the caller's schema is never mutated",
       "evaluation_config" in SCHEMA["properties"]["capex"])
report("the provider's maximum-tier options are passed to the adapter",
       vendor.predict("datalab-accurate", PDF, SCHEMA) is not None
       and seen["opts"]["mode"] == "accurate", str(seen["opts"]))
report("the uniform budget is passed to the adapter",
       vendor.predict("mistral", PDF, SCHEMA, timeout=900) is not None
       and 899 < seen["opts"]["timeout"] <= 900, str(seen["opts"]))

# One budget for the document, not one per attempt. Four attempts at the full timeout plus
# backoff could spend 7,400s on a document documented as taking 1,800s end to end -- the same
# mistake `Budget` fixed inside the adapters, one level up.
_given, _backoff = [], vendor.TRANSIENT_BACKOFF
vendor.TRANSIENT_BACKOFF = (0, 0, 0)
stub(lambda pdf, schema, **o: (_given.append(o["timeout"]),
                               (_ for _ in ()).throw(VendorError("HTTP 503", status=503)))[1])
_rec = vendor.predict("mistral", PDF, SCHEMA, timeout=5)
vendor.TRANSIENT_BACKOFF = _backoff
report("a retried document does not get a fresh budget each time",
       all(a > b for a, b in zip(_given, _given[1:])), str([round(t, 3) for t in _given]))
report("...and the whole run stays inside one budget",
       _given and _given[0] <= 5 and _given[-1] > 0, str([round(t, 3) for t in _given]))
stub(lambda pdf, schema, **o: OK)          # the sections below expect a successful adapter

print("\nTHE ANSWER TRAVELS WITH ITS EVIDENCE")
rec = vendor.predict("mistral", PDF, SCHEMA)
for key in ("result", "raw", "cost", "schema_sent", "run_manifest", "error", "job_id"):
    report(f"the record carries `{key}`", key in rec)
report("the cost the vendor reported is kept", rec["cost"]["usd"] == 0.031)
report("...with the field it was read from", rec["cost"]["source"] == "usage.cost")
report("a vendor that reports no cost is billed out of band, not guessed",
       (stub(lambda p, s, **o: Extraction(result={"a": 1})) is not None
        and vendor.predict("extend", PDF, SCHEMA)["cost"]["billed_out_of_band"] is True))
report("the tier the provider ran at is recorded",
       rec["run_manifest"]["tier"] == vendor.PROVIDER_TIER["mistral"])

print("\nA RUN THAT IS NOT STOCK SAYS SO")
# The benchmark's claim is that every vendor ran at its maximum. An option passed by the caller
# can turn that down, so each one is recorded on every document -- a figure produced with a
# vendor dialled back must not be able to look stock afterwards.
stub(lambda pdf, schema, **o: OK)
report("a stock run records no overrides",
       vendor.predict("datalab", PDF, SCHEMA)["run_manifest"]["overrides"] is None)
_rec = vendor.predict("datalab", PDF, SCHEMA, mode="accurate")
report("an overridden option reaches the adapter", seen["opts"]["mode"] == "accurate")
report("...and is recorded in the run manifest",
       _rec["run_manifest"]["overrides"] == {"mode": "accurate"}, str(_rec["run_manifest"]))
report("an option equal to the default is not an override",
       vendor.predict("datalab", PDF, SCHEMA,
                      mode="balanced")["run_manifest"]["overrides"] is None)

print("\nA MODEL IS NAMED IN FULL, NOT ALIASED")
# `gpt` said nothing about which model produced a row, and changed meaning whenever the alias
# was repointed. A slash is the discriminator: no vendor name has one, every model id does.
report("a model id routes to the single-shot adapter",
       vendor.resolve("openai/gpt-5.6-sol") == ("llm_single_shot",
                                                {"model": "openai/gpt-5.6-sol"}))
report("a vendor name still routes to its own adapter",
       vendor.resolve("datalab")[0] == "datalab")
report("an OpenRouter suffix is part of the id, not a separator",
       vendor.resolve("mistralai/mistral-medium-3-5:batch")[1]["model"]
       == "mistralai/mistral-medium-3-5:batch")
report("a bare alias is refused, naming the vendors",
       (lambda: [False for _ in ()] or _refused("gpt"))())
report("a model id gets one directory, not a nested pair",
       vendor.out_name("openai/gpt-5.6-sol") == "openai__gpt-5.6-sol")

print("\nRETRY ONLY WHAT A RETRY CAN FIX")
for status, label, want in ((503, "a 5xx", vendor.TRANSIENT_ATTEMPTS),
                            (429, "a rate limit", vendor.TRANSIENT_ATTEMPTS),
                            (400, "a real answer", 1)):
    called = stub(lambda p, s, st=status, **o: (_ for _ in ()).throw(
        VendorError(f"HTTP {st}", status=st)))
    rec = vendor.predict("mistral", PDF, SCHEMA)
    report(f"{label} ({status}) is attempted {want}x", called.calls == want,
           f"called {called.calls}x")
    report(f"...and the record says transient={status != 400}",
           rec["error"]["transient"] is (status != 400), str(rec["error"]))

called = stub(lambda p, s, **o: (_ for _ in ()).throw(VendorTimeout("still running after 412 polls")))
rec = vendor.predict("mistral", PDF, SCHEMA)
report("a timeout is never retried -- it would spend another full budget", called.calls == 1)
report("...and is marked in the manifest", rec["run_manifest"]["timed_out"] is True)
report("...and the reason survives, counted by the adapter that polled",
       "412 polls" in rec["error"]["message"], rec["error"]["message"])

print("\nNOT FACTS ABOUT THE DOCUMENT -- RAISED, NOT RECORDED")
# Each is identical for all 620 documents and fixable in one step. Recorded, it becomes a
# settled failure that no resume re-attempts, and the provider reads as 0% coverage.
for exc, kind in ((VendorError("HTTP 402: no credits", status=402), AccountFailure),
                  (VendorError("insufficient_quota", status=403), AccountFailure),
                  (MissingCredential("MISTRAL_API_KEY is not set"), MissingCredential)):
    stub(lambda p, s, e=exc, **o: (_ for _ in ()).throw(e))
    try:
        vendor.predict("mistral", PDF, SCHEMA)
        report(f"{type(exc).__name__} raises {kind.__name__}", False, "it returned a record")
    except kind:
        report(f"{type(exc).__name__} raises {kind.__name__}", True)
    except Exception as other:                                       # noqa: BLE001
        report(f"{type(exc).__name__} raises {kind.__name__}", False, f"got {other!r}")

vendor.adapter = _REAL_ADAPTER                 # stop stubbing: the next two test dispatch itself

print("\nA MISSING SDK IS FOUND AT THE IMPORT, NOT IN A MESSAGE")
# The dispatch imports the adapter module on demand, so this is structural: no grepping
# "ModuleNotFoundError" out of a stderr tail, which is what the subprocess design forced.
import omni_extract_bench.harness.vendor as _v                                 # noqa: E402
_real = _v.importlib.import_module
_v.importlib.import_module = lambda *a, **k: (_ for _ in ()).throw(ImportError("No module named 'httpx'"))
try:
    _v.adapter("mistral")
    report("a missing SDK raises MissingDependency", False, "it returned an adapter")
except MissingDependency as exc:
    report("a missing SDK raises MissingDependency", True)
    report("...naming the extra to install", "omni-extract-bench[harness]" in str(exc))
finally:
    _v.importlib.import_module = _real

print("\nAND AN UNKNOWN PROVIDER IS REFUSED BY NAME")
try:
    vendor.adapter("nonesuch")
    report("an unknown provider raises", False, "it was accepted")
except ValueError as exc:
    report("an unknown provider raises, naming the ones that exist", "datalab" in str(exc))

PDF.unlink(missing_ok=True)
print(f"\n{'predict HOLDS THE PARITY RULES' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
sys.exit(1 if FAILS else 0)
