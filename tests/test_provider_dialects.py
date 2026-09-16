#!/usr/bin/env python3
"""Tests for the vendor adapters' schema handling, without calling a vendor.

`run_provider`'s contract is that a schema may be RE-ENCODED for a vendor but never CHANGED:
"same fields, same types, same descriptions". That is the part of the harness a score depends
on -- a transform that drops a field asks a vendor a smaller question and then grades it as if
it had been asked the whole one -- and it is pure, so it needs no API key to check.

What is NOT covered here: the HTTP itself. Those need credentials and cost money per call, and
a mocked transport would only assert that the mock was written to match the adapter.

Run: python3 tests/test_provider_dialects.py
"""
import json
import sys

# run from anywhere: `python tests/x.py` puts tests/ on the path, not the repo root
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from omni_extract_bench.harness import schema_overlay  # noqa: E402
from omni_extract_bench.harness.providers import datalab  # noqa: E402
from omni_extract_bench.harness.run_provider import (  # noqa: E402
    deref, is_account_failure, is_transient, strip_bench_keys,
)

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


def leaves(node, path="", out=None):
    """Every leaf address a schema declares, so two encodings can be compared as tasks."""
    out = {} if out is None else out
    if not isinstance(node, dict):
        return out
    props = node.get("properties")
    if isinstance(props, dict):
        for k, v in props.items():
            leaves(v, f"{path}.{k}" if path else k, out)
    it = node.get("items")
    if isinstance(it, dict):
        leaves(it, f"{path}[]", out)
    if not props and not isinstance(it, dict) and path:
        t = node.get("type")
        t = tuple(sorted(t)) if isinstance(t, list) else t
        out[path] = t
    return out


SAMPLE = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "record id"},
        "issued": {"type": ["string", "null"]},
        "rows": {"type": "array", "items": {"type": "object", "properties": {
            "amount": {"type": ["number", "null"], "description": "Dollar amount as printed"},
            "page": {"type": ["integer", "null"], "evaluation_config": "integer_exact"},
        }}},
    },
    "required": ["id", "issued"],
    "evaluation_config": "array_llm",
}

print("[1] extend reserves `id` -- the rename must be reversible")
from omni_extract_bench.harness.providers import extend  # noqa: E402

renamed = extend.rename_reserved(SAMPLE)
check("`id` is aliased on the way out", "id__" in renamed["properties"])
check("the original name is gone", "id" not in renamed["properties"])
check("`required` follows the rename", "id__" in renamed["required"])
check("a nested non-reserved name is untouched",
      "amount" in renamed["properties"]["rows"]["items"]["properties"])
check("restoring a vendor RESPONSE renames the payload key back",
      extend.restore_reserved({"id__": "A-1", "rows": [{"id__": "r1"}]})
      == {"id": "A-1", "rows": [{"id": "r1"}]})
# The pair is deliberately not symmetric: `rename_reserved` walks a SCHEMA (so it knows to
# follow `properties` keys and the `required` list), while `restore_reserved` walks a
# RESPONSE, where every dict key is a field name and there is no `required`. Round-tripping a
# schema therefore leaves `required` holding the alias. That is harmless as used -- the schema
# goes out, the response comes back -- but it is a real edge, so it is pinned here rather than
# left for someone to discover by calling restore on a schema.
back = extend.restore_reserved(json.loads(json.dumps(renamed)))
check("restore fixes property keys", "id" in back["properties"])
check("restore does NOT fix `required` (it walks responses, not schemas)",
      back["required"] == ["id__", "issued"], f"{back['required']}")

print("\n[2] datalab normalize_schema -- dialect only")
norm = datalab.normalize_schema(SAMPLE)
check("same leaf addresses", set(leaves(norm)) == set(leaves(SAMPLE)),
      f"{set(leaves(SAMPLE)) ^ set(leaves(norm))}")
check("nullable union collapsed to one type", norm["properties"]["issued"]["type"] == "string")
check("a nullable field is no longer required", "issued" not in norm.get("required", []))
check("a non-nullable field stays required", "id" in norm.get("required", []))
check("descriptions survive",
      norm["properties"]["rows"]["items"]["properties"]["amount"]["description"]
      == "Dollar amount as printed")
check("input not mutated in place", SAMPLE["properties"]["issued"]["type"] == ["string", "null"])

print("\n[3] parity rules: benchmark-only keys out, $ref inlined, task unchanged")
stripped = strip_bench_keys(SAMPLE)
check("evaluation_config removed at every depth",
      "evaluation_config" not in json.dumps(stripped))
check("stripping changes no leaf", set(leaves(stripped)) == set(leaves(SAMPLE)))
reffed = {"$defs": {"Row": {"type": "object", "properties": {"n": {"type": "string"}}}},
          "type": "object",
          "properties": {"rows": {"type": "array", "items": {"$ref": "#/$defs/Row"}}}}
d = deref(reffed)
check("$ref inlined", d["properties"]["rows"]["items"]["properties"]["n"]["type"] == "string")
check("$defs dropped once inlined", "$defs" not in d)
check("a schema with no $ref is unchanged", deref(SAMPLE) == SAMPLE)

print("\n[4] retry classification -- a 400 is an answer, a 429 is not")
check("429 retries", is_transient("HTTP 429 Too Many Requests"))
check("rate limit wording retries", is_transient("rate limit exceeded"))
check("5xx retries", is_transient("HTTP 503 Service unavailable"))
check("empty 200 retries", is_transient("empty 200"))
check("400 does NOT retry", not is_transient("HTTP 400: schema invalid"))
check("404 does NOT retry", not is_transient("HTTP 404 no such processor"))
check("no error is not transient", not is_transient(""))
check("402 is an account failure", is_account_failure("HTTP 402 payment required"))
# One condition, one classification. `outcome_error` has two places that report "a 200 that
# yielded no usable output", and they once disagreed: 91 predictions took the fallthrough and
# were never retried while 18 identical cases matched the marker and were. Both messages are
# asserted here so a future edit to either has to come past this test.
from omni_extract_bench.harness.run_provider import EMPTY_200_MARKER  # noqa: E402
for msg in (f"{EMPTY_200_MARKER}: response body was keep-alive padding with no completion",
            f"{EMPTY_200_MARKER}: no usable output; last response 200: \n\n\n"):
    check(f"every empty-200 message is transient ({msg[:22]}...)", is_transient(msg))
check("a non-200 with no usable output is NOT transient",
      not is_transient("no usable output; last response 204: "))
check("credit ceiling is an account failure",
      is_account_failure("you have exceeded the maximum number of credits"))
check("a 400 is not an account failure", not is_account_failure("HTTP 400"))

print("\n[5] schema_overlay states a convention without changing the task")
ov = schema_overlay.apply_overlay(SAMPLE)
check("leaf addresses unchanged", set(leaves(ov)) == set(leaves(SAMPLE)))
check("types unchanged", leaves(ov) == leaves(SAMPLE))
check("input not mutated", "evaluation_config" in SAMPLE)
check("applying twice is idempotent", schema_overlay.apply_overlay(ov) == ov)

print("\n[6] envelope -- a failure must stay distinguishable from a result")
import tempfile, pathlib  # noqa: E402
from omni_extract_bench.harness.providers.envelope import write_output  # noqa: E402
from omni_extract_bench.harness.prediction_io import usable  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    good = pathlib.Path(td) / "good.json"
    write_output(good, provider="x", result={"a": 1}, latency_s=1.5, usage={"cost_usd": 0.01})
    body = json.loads(good.read_text())
    check("result preserved", body["result"] == {"a": 1})
    # `result` stays at the top: everything else a run records is under `_meta`, so a scorer
    # reading a prediction never has to know which keys were bookkeeping.
    check("latency recorded under _meta", body["_meta"]["latency_s"] == 1.5)
    check("usage carried through", body["_meta"]["usage"] == {"cost_usd": 0.01})
    check("provider recorded", body["_meta"]["provider"] == "x")
    check("a real result is usable", usable(body["result"]))
    check("an error envelope is NOT usable", not usable({"__error__": "HTTP 500"}))

print("\n[7] every real corpus schema survives every vendor's transform")
CORPUS = _os.environ.get("OEB_CORPUS")
if not CORPUS or not _os.path.isdir(CORPUS):
    print("  SKIP  set OEB_CORPUS=<corpus dir> to check these against the real schemas")
else:
    import glob
    paths = sorted(glob.glob(_os.path.join(CORPUS, "*", "schema.json")))
    drops = {"datalab": [], "overlay": [], "strip": []}
    for p in paths:
        s = json.loads(open(p).read())
        base = set(leaves(s))
        name = _os.path.basename(_os.path.dirname(p))
        if set(leaves(datalab.normalize_schema(s))) != base:
            drops["datalab"].append(name)
        if set(leaves(schema_overlay.apply_overlay(s))) != base:
            drops["overlay"].append(name)
        if set(leaves(strip_bench_keys(s))) != base:
            drops["strip"].append(name)
    print(f"  checked {len(paths)} real schemas")
    for k, v in drops.items():
        check(f"{k} preserves every leaf of every schema", not v, f"{len(v)}: {v[:3]}")

print(f"\n{'ALL PROVIDER DIALECT TESTS PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
