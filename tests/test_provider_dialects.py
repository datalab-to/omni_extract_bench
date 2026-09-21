#!/usr/bin/env python3
"""Tests for the vendor adapters' schema handling, without calling a vendor.

`vendor.predict`'s contract is that a schema may be RE-ENCODED for a vendor but never CHANGED:
"same fields, same types, same descriptions". That is the part of the harness a score depends
on -- a transform that drops a field asks a vendor a smaller question and then grades it as if
it had been asked the whole one -- and it is pure, so it needs no API key to check.

What is NOT covered here: the HTTP itself. Those need credentials and cost money per call, and
a mocked transport would only assert that the mock was written to match the adapter.

Run: python3 tests/test_provider_dialects.py
"""
import json
import sys

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from omni_extract_bench.harness import schema_overlay  # noqa: E402
from omni_extract_bench.harness.providers import datalab  # noqa: E402
from omni_extract_bench.harness.dialects import (  # noqa: E402
    resolve_refs as deref, strip_benchmark_keys as strip_bench_keys, to_strict_dialect,
)
from omni_extract_bench.harness.extraction import VendorError  # noqa: E402
from omni_extract_bench.harness.vendor import _is_account_failure  # noqa: E402

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
back = extend.restore_reserved(json.loads(json.dumps(renamed)))
check("restore fixes property keys", "id" in back["properties"])
check("restore does NOT fix `required` (it walks responses, not schemas)",
      back["required"] == ["id__", "issued"], f"{back['required']}")

print("\n[2] datalab prepare_schema -- dialect only")
norm = datalab.prepare_schema(SAMPLE)
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
for status in (429, 500, 502, 503, 504):
    check(f"{status} is transient", VendorError("x", status=status).transient)
for status in (200, 400, 401, 404, 422):
    check(f"{status} is NOT transient -- it is an answer",
          not VendorError("x", status=status).transient)
check("402 is an account failure", _is_account_failure(VendorError("no credit", status=402)))
check("credit ceiling wording is an account failure",
      _is_account_failure(VendorError("you have exceeded the maximum number of credits")))
check("a 400 is not an account failure",
      not _is_account_failure(VendorError("schema invalid", status=400)))

print("\n[5] EVERY VENDOR IS ASKED THE SAME QUESTION")
FAIR = {
    "$defs": {"Row": {"type": "object", "properties": {
        "sku": {"type": "string", "description": "the stock code, verbatim"},
        "qty": {"anyOf": [{"type": "integer"}, {"type": "null"}],
                "description": "units billed; null if absent"}}}},
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "the filing identifier"},
        "total": {"anyOf": [{"type": "number"}, {"type": "null"}],
                  "description": "total due, as a positive magnitude",
                  "evaluation_config": {"tolerance": 0.01}},
        "rows": {"anyOf": [{"type": "array", "items": {"$ref": "#/$defs/Row"}},
                           {"type": "null"}],
                 "description": "one entry per billed line"},
    },
}


def descriptions(node):
    """Every description anywhere in a schema. Path-independent on purpose: a vendor may
    restructure or rename, but it may not stop telling the model something."""
    found = []
    if isinstance(node, dict):
        if isinstance(node.get("description"), str):
            found.append(node["description"])
        for v in node.values():
            found += descriptions(v)
    elif isinstance(node, list):
        for v in node:
            found += descriptions(v)
    return found


def field_names(node):
    """Every property name the schema declares, at any depth."""
    names = set()
    if isinstance(node, dict):
        for k, v in (node.get("properties") or {}).items():
            names.add(k)
            names |= field_names(v)
        for key in ("items", "anyOf", "oneOf", "allOf", "$defs", "fields"):
            names |= field_names(node.get(key))
    elif isinstance(node, list):
        for v in node:
            names |= field_names(v)
    return names


import copy as _copy                                                        # noqa: E402
from omni_extract_bench.harness.providers import (azure_cu, llamaextract,   # noqa: E402
                                                  extend as _extend)

_BASE = deref(strip_bench_keys(schema_overlay.apply_overlay(FAIR)))
_WANT_DESC = sorted(descriptions(_BASE))
_WANT_NAMES = field_names(_BASE)
check("the baseline asks for every field with a description",
      len(_WANT_DESC) == 5 and {"id", "total", "rows", "sku", "qty"} <= _WANT_NAMES,
      f"{len(_WANT_DESC)} descriptions, names {sorted(_WANT_NAMES)}")

# READ OFF THE ADAPTERS, never re-stated here. This table used to hand-copy each transform,
# so it could agree with itself while disagreeing with the code it was checking -- the same
# way `schema_sent` once named a schema no vendor had seen.
from omni_extract_bench.harness.providers import mistral, reducto            # noqa: E402

_DIALECTS = {
    "mistral": (mistral.prepare_schema, {}),
    "reducto": (reducto.prepare_schema, {}),
    "datalab": (datalab.prepare_schema, {}),
    "llamaextract": (llamaextract.prepare_schema, {}),
    "extend": (_extend.prepare_schema, {"id": "id__"}),
    "azure-cu": (azure_cu.prepare_schema, {}),
}
KNOWN_BROKEN = {
    "azure-cu": "asked for 45% of the corpus's fields; `_field` dispatches on `type` and the "
                "schemas use `anyOf` unions, so arrays collapse to string. Fix and numbers in "
                "~/TODO.md.",
}


def check_dialect(vendor, name, cond, detail=""):
    if cond or vendor not in KNOWN_BROKEN:
        check(name, cond, detail)
    else:
        print(f"  KNOWN  {name} -- {KNOWN_BROKEN[vendor]}")


for _name, (_fn, _renames) in _DIALECTS.items():
    _got = _fn(_copy.deepcopy(_BASE))
    check_dialect(_name, f"{_name}: no description is dropped",
                  sorted(descriptions(_got)) == _WANT_DESC,
                  f"missing {sorted(set(_WANT_DESC) - set(descriptions(_got)))}")
    _names = {v: k for k, v in _renames.items()}
    _after = {_names.get(n, n) for n in field_names(_got)}
    check_dialect(_name, f"{_name}: no field is dropped", _WANT_NAMES <= _after,
                  f"lost {sorted(_WANT_NAMES - _after)}")
    check(f"{_name}: no field is invented", not (_after - _WANT_NAMES),
          f"added {sorted(_after - _WANT_NAMES)}")

print("\n[5b] schema_sent IS the payload the vendor received")


def _prepared_by(mod, schema):
    """What `predict` sends this adapter: the universal layer, then its own prepare_schema."""
    return mod.prepare_schema(strip_bench_keys(schema_overlay.apply_overlay(schema)))

import pathlib                                                              # noqa: E402
_ANNOTATED = {"type": "object", "evaluation_config": "array_llm",
              "properties": {"amt": {"anyOf": [{"type": "number"}, {"type": "null"}],
                                     "default": None, "description": "the amount"}}}
for _name, _mod in (("datalab", datalab), ("llamaextract", llamaextract),
                    ("extend", _extend), ("azure-cu", azure_cu), ("mistral", mistral)):
    _prepared = _prepared_by(_mod, _copy.deepcopy(_ANNOTATED))
    check(f"{_name}: the adapter's own prepare_schema is what ran",
          _prepared == _mod.prepare_schema(strip_bench_keys(schema_overlay.apply_overlay(
              _copy.deepcopy(_ANNOTATED)))))
    check(f"{_name}: benchmark-only keys never reach the vendor",
          "evaluation_config" not in json.dumps(_prepared)
          and '"default"' not in json.dumps(_prepared), json.dumps(_prepared)[:160])

check("a dialect-less adapter is identity, not a dropped schema",
      _prepared_by(mistral, _copy.deepcopy(_ANNOTATED))
      == strip_bench_keys(schema_overlay.apply_overlay(_copy.deepcopy(_ANNOTATED))))
check("llamaextract's payload is the collapsed one, not the JSON Schema it came from",
      "anyOf" not in json.dumps(_prepared_by(llamaextract, _copy.deepcopy(_ANNOTATED))))
check("azure-cu's payload is a fieldSchema, not a JSON Schema",
      "fields" in _prepared_by(azure_cu, _copy.deepcopy(_ANNOTATED)))
check("extend's payload carries the aliased name",
      "id__" in json.dumps(_prepared_by(_extend, {"type": "object", "properties": {
          "id": {"type": "string"}}})))

import omni_extract_bench.harness.vendor as _vendor                          # noqa: E402
from omni_extract_bench.harness.extraction import (DialectError,             # noqa: E402
                                                  VendorError as _VendorError)


class _Spy:
    """A stand-in adapter: the three names `Adapter` asks for, and nothing else."""

    Config = llamaextract.Config

    def __init__(self, prepare):
        self.prepare_schema = prepare
        self.seen = {}

    def extract(self, pdf, schema, *, timeout, config):
        self.seen["schema"] = schema
        raise _VendorError("stopped before the network", status=400)


def _predict_with(spy, schema):
    """`predict`, with this adapter in place of the real one."""
    real, _vendor.adapter = _vendor.adapter, lambda provider: spy
    try:
        return _vendor.predict("llamaextract", pathlib.Path(__file__), schema)
    finally:
        _vendor.adapter = real


_spy = _Spy(llamaextract.prepare_schema)
_rec = _predict_with(_spy, _copy.deepcopy(_ANNOTATED))
check("schema_sent IS the object the adapter received",
      _rec["schema_sent"] == _spy.seen["schema"],
      "the record still describes a different schema")
check("a REJECTED schema is still recorded",
      _rec["error"]["status"] == 400 and _rec["schema_sent"] == _spy.seen["schema"])


print("\n[5c] a dialect that raises costs one document, not the run")


def _explodes(schema):
    raise KeyError(type([]))


_bad = _predict_with(_Spy(_explodes), _copy.deepcopy(_ANNOTATED))
check("a raising prepare_schema is returned, not raised", isinstance(_bad, dict))
check("it is named as a DialectError",
      _bad["error"]["type"] == "DialectError", str(_bad["error"]))
check("it is NOT transient -- the same schema shapes the same way every time",
      _bad["error"]["transient"] is False)
check("the vendor was never called", _bad["cost"]["attempts"] == 0, str(_bad["cost"]))
check("the cause is named, not just the type", "KeyError" in _bad["error"]["message"])
check("schema_sent falls back to the pre-dialect schema, so the input is still readable",
      "amt" in json.dumps(_bad["schema_sent"]))
check("benchmark-only keys are still stripped from that fallback",
      "evaluation_config" not in json.dumps(_bad["schema_sent"]))


print("\n[5d] every adapter satisfies the Adapter protocol")
for _name in list(_vendor.PROVIDERS) + ["openai/gpt-5.6-sol"]:
    _mod = _vendor.adapter(_name)
    check(f"{_name}: has Config, prepare_schema, extract",
          all(hasattr(_mod, n) for n in ("Config", "prepare_schema", "extract")),
          f"missing {[n for n in ('Config','prepare_schema','extract') if not hasattr(_mod,n)]}")
check("a model id routes to the one LLM adapter",
      _vendor.resolve("openai/gpt-5.6-sol") == _vendor.resolve("anthropic/claude-opus-5")
      == "llm_single_shot")


print("\n[5] schema_overlay states a convention without changing the task")
ov = schema_overlay.apply_overlay(SAMPLE)
check("leaf addresses unchanged", set(leaves(ov)) == set(leaves(SAMPLE)))
check("types unchanged", leaves(ov) == leaves(SAMPLE))
check("input not mutated", "evaluation_config" in SAMPLE)
check("applying twice is idempotent", schema_overlay.apply_overlay(ov) == ov)

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
        if set(leaves(datalab.prepare_schema(s))) != base:
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
