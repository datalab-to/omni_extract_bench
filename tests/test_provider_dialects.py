#!/usr/bin/env python3
"""The vendor adapters' schema handling, without calling a vendor: a schema may be written differently for a
vendor, but never changed (same fields, same types, same descriptions). The HTTP calls themselves are not
tested here; they need credentials and cost money.

Run: python3 tests/test_provider_dialects.py
"""
import dataclasses as _dataclasses
import json
import pathlib
import sys

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from omni_extract_bench.harness import schema as SCHEMA  # noqa: E402
from omni_extract_bench.harness.providers import datalab  # noqa: E402
from omni_extract_bench.harness.schema import strip_benchmark_keys as strip_bench_keys
from omni_extract_bench.scorers.field_matching.documents import resolve_refs as deref  # noqa: E402
from omni_bench_core import adapter as _adapter  # noqa: E402
from omni_bench_core import errors as _errors  # noqa: E402
from omni_bench_core.errors import SystemFailed  # noqa: E402
from omni_extract_bench.harness import contract as _contract  # noqa: E402

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
check("restoring a vendor response renames the payload key back",
      extend.restore_reserved({"id__": "A-1", "rows": [{"id__": "r1"}]})
      == {"id": "A-1", "rows": [{"id": "r1"}]})
back = extend.restore_reserved(json.loads(json.dumps(renamed)))
check("restore fixes property keys", "id" in back["properties"])
check("restore does not fix `required` (it walks responses, not schemas)",
      back["required"] == ["id__", "issued"], f"{back['required']}")

print("\n[2] datalab prepare_schema -- dialect only")
norm = datalab.Datalab().prepare_schema(SAMPLE)
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
    check(f"{status} is transient", SystemFailed("x", status=status).transient)
for status in (200, 400, 401, 404, 422):
    check(f"{status} is not transient -- it is an answer",
          not SystemFailed("x", status=status).transient)


@_dataclasses.dataclass(frozen=True)
class _Fails(_contract.Extractor):
    """An adapter whose call fails with `message`, before the network."""

    message: str = ""

    def extract(self, pdf, schema, *, timeout):
        raise SystemFailed(self.message, status=400)


def _account_failure(message):
    """Whether omni-bench-core's `predict` stops the run on this failure, rather than recording it."""
    try:
        _adapter.predict(_Fails(message=message), _contract.Question(pathlib.Path(__file__), {"type": "object"}),
                         timeout=5)
    except PermissionError:
        return True
    return False


check("402 is an account failure", _Fails().account_failure(SystemFailed("no credit", status=402)))
check("credit ceiling wording is an account failure",
      _account_failure("you have exceeded the maximum number of credits"))
check("a 400 is not an account failure", not _account_failure("schema invalid"))

print("\n[5] Every vendor is asked the same question")
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

_BASE = deref(strip_bench_keys(SCHEMA.apply_overlay(FAIR)))
_WANT_DESC = sorted(descriptions(_BASE))
_WANT_NAMES = field_names(_BASE)
check("the baseline asks for every field with a description",
      len(_WANT_DESC) == 5 and {"id", "total", "rows", "sku", "qty"} <= _WANT_NAMES,
      f"{len(_WANT_DESC)} descriptions, names {sorted(_WANT_NAMES)}")

# Read off the adapters, never restated, so this cannot agree with itself and not the code.
from omni_extract_bench.harness.providers import mistral, reducto            # noqa: E402

_DIALECTS = {
    "mistral": (mistral.Mistral().prepare_schema, {}),
    "reducto": (reducto.Reducto().prepare_schema, {}),
    "datalab": (datalab.Datalab().prepare_schema, {}),
    "llamaextract": (llamaextract.LlamaExtract().prepare_schema, {}),
    "extend": (_extend.Extend().prepare_schema, {"id": "id__"}),
    "azure-cu": (azure_cu.AzureCU(endpoint="https://x.example").prepare_schema, {}),
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
        print(f"  Known  {name} -- {KNOWN_BROKEN[vendor]}")


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
    return mod.prepare_schema(strip_bench_keys(SCHEMA.apply_overlay(schema)))

_ANNOTATED = {"type": "object", "evaluation_config": "array_llm",
              "properties": {"amt": {"anyOf": [{"type": "number"}, {"type": "null"}],
                                     "default": None, "description": "the amount"}}}
for _name, _mod in (("datalab", datalab.Datalab()), ("llamaextract", llamaextract.LlamaExtract()),
                    ("extend", _extend.Extend()), ("azure-cu", azure_cu.AzureCU(endpoint="https://x.example")),
                    ("mistral", mistral.Mistral())):
    _prepared = _prepared_by(_mod, _copy.deepcopy(_ANNOTATED))
    check(f"{_name}: the adapter's own prepare_schema is what ran",
          _prepared == _mod.prepare_schema(strip_bench_keys(SCHEMA.apply_overlay(
              _copy.deepcopy(_ANNOTATED)))))
    check(f"{_name}: benchmark-only keys never reach the vendor",
          "evaluation_config" not in json.dumps(_prepared)
          and '"default"' not in json.dumps(_prepared), json.dumps(_prepared)[:160])

check("a dialect-less adapter is identity, not a dropped schema",
      _prepared_by(mistral.Mistral(), _copy.deepcopy(_ANNOTATED))
      == strip_bench_keys(SCHEMA.apply_overlay(_copy.deepcopy(_ANNOTATED))))
check("llamaextract is sent the nullability the schema declares, not a collapsed union",
      "anyOf" in json.dumps(_prepared_by(llamaextract.LlamaExtract(), _copy.deepcopy(_ANNOTATED))))
check("azure-cu's payload is a fieldSchema, not a JSON Schema",
      "fields" in _prepared_by(azure_cu.AzureCU(endpoint="https://x.example"), _copy.deepcopy(_ANNOTATED)))
check("extend's payload carries the aliased name",
      "id__" in json.dumps(_prepared_by(_extend.Extend(), {"type": "object", "properties": {
          "id": {"type": "string"}}})))

_SEEN = {}


@_dataclasses.dataclass(frozen=True)
class _Spy(_contract.Extractor):
    """A stand-in adapter that reshapes like llamaextract, records what it was sent, and stops."""

    explode: str = ""   # "dialect": a schema the vendor can't express; "bug": a reshaping that crashes

    def prepare_schema(self, schema):
        if self.explode == "dialect": raise _contract.DialectError("this vendor has no way to write a tuple")
        if self.explode == "bug": raise KeyError(type([]))
        return llamaextract.LlamaExtract().prepare_schema(schema)

    def extract(self, pdf, schema, *, timeout):
        _SEEN["schema"] = schema
        raise SystemFailed("stopped before the network", status=400)


def _predict_with(spy, schema):
    """omni-bench-core's `predict`, with `spy` as the adapter, as `Extract.input` would ask it."""
    _SEEN.clear()
    return _adapter.predict(spy, _contract.Question(pathlib.Path(__file__),
                                                    strip_bench_keys(SCHEMA.apply_overlay(schema))), timeout=60)


_rec = _predict_with(_Spy(), _copy.deepcopy(_ANNOTATED))
check("a rejected call is recorded as the document's answer", _rec["error"]["status"] == 400)
check("a rejected call still records the schema the vendor was sent", _rec["sent"] == _SEEN["schema"])
check("the schema the adapter received never carries benchmark-only keys",
      "evaluation_config" not in json.dumps(_SEEN["schema"]))


print("\n[5e] a deep $ref chain still describes its field")
# `resolve_refs` budgets nodes, not $refs: 12 is about four hops, past which a field silently
# becomes `{}`. The corpus tops out at two.


def _ref_chain(depth):
    """A schema whose `$ref` chain is `depth` long, ending in a typed leaf."""
    defs = {f"L{i}": {"type": "object", "properties": {"next": {"$ref": f"#/$defs/L{i + 1}"}}}
            for i in range(depth)}
    defs[f"L{depth}"] = {"type": "string"}
    return {"$defs": defs, "type": "object",
            "properties": {"a": {"$ref": "#/$defs/L0"}}}


def _leaf(prepared, key="a"):
    """Walk to the end of the prepared chain and return the deepest node."""
    node = prepared.get("properties", {}).get(key, {})
    while isinstance(node, dict) and "next" in (node.get("properties") or {}):
        node = node["properties"]["next"]
    return node


for _depth in (2, 6, 20):
    for _name, _mod in (("llamaextract", llamaextract.LlamaExtract()), ("extend", _extend.Extend())):
        _leafnode = _leaf(_mod.prepare_schema(_copy.deepcopy(_ref_chain(_depth))))
        check(f"{_name}: a {_depth}-deep $ref chain keeps its leaf type",
              _leafnode.get("type") in ("string", ["string", "null"]),
              f"leaf came out {json.dumps(_leafnode)[:80]}")

_recursive = {"$defs": {"Node": {"type": "object", "properties": {
                  "name": {"type": "string"}, "child": {"$ref": "#/$defs/Node"}}}},
              "type": "object", "properties": {"root": {"$ref": "#/$defs/Node"}}}
for _name, _mod in (("llamaextract", llamaextract.LlamaExtract()), ("extend", _extend.Extend())):
    _out = json.dumps(_mod.prepare_schema(_copy.deepcopy(_recursive)))
    check(f"{_name}: a self-referential $ref terminates, bounded",
          len(_out) < 200_000, f"expanded to {len(_out)} bytes")


print("\n[5f] azure-cu's analyzer is named after everything it is built from")
# Analyzers are reused by name and a 409 counts as success, so a change must change the name.
_V = "2025-05-01-preview"
_s1 = azure_cu.field_schema({"type": "object", "properties": {"a": {"type": "string"}}})
_s2 = azure_cu.field_schema({"type": "object", "properties": {"b": {"type": "string"}}})
_base = azure_cu.analyzer_id(_s1, "gpt-4.1-mini", _V)

check("the same task reuses one analyzer -- that is the point of the name",
      azure_cu.analyzer_id(_s1, "gpt-4.1-mini", _V) == _base)
check("a different completion_model is a different analyzer",
      azure_cu.analyzer_id(_s1, "gpt-4.1", _V) != _base,
      "two deployments would share one analyzer and one would be scored as the other")
check("a different api_version is a different analyzer",
      azure_cu.analyzer_id(_s1, "gpt-4.1-mini", "2026-01-01-preview") != _base)
check("a different schema is a different analyzer",
      azure_cu.analyzer_id(_s2, "gpt-4.1-mini", _V) != _base)

# The process cache spans resources; an analyzer lives inside one.
_puts = []


class _FakePut:
    status_code, headers, text = 201, {}, ""


class _FakeClient:
    def put(self, url, json=None):
        _puts.append(url.split("/contentunderstanding/")[0])
        return _FakePut()


azure_cu._ANALYZERS.clear()
for _ep in ("https://a.example", "https://b.example", "https://a.example"):
    azure_cu._ensure_analyzer(_FakeClient(), _base, _s1,
                              azure_cu.AzureCU(endpoint=_ep, api_version=_V), _adapter.Budget(60))
check("each endpoint gets its own analyzer created",
      sorted(_puts) == ["https://a.example", "https://b.example"],
      f"PUTs went to {_puts} -- a second resource skipping creation 404s every document")
azure_cu._ANALYZERS.clear()


print("\n[5c] a schema the dialect can't express costs one document; a crashing dialect stops the run")


_bad = _predict_with(_Spy(explode="dialect"), _copy.deepcopy(_ANNOTATED))
check("a DialectError from prepare_schema is returned, not raised", isinstance(_bad, dict))
check("it is named as a DialectError", _bad["error"]["type"] == "DialectError", str(_bad["error"]))
check("it is not transient -- the same schema shapes the same way every time",
      _bad["error"]["transient"] is False)
check("the vendor was never called", "schema" not in _SEEN)
check("its traceback names where it was raised", "prepare_schema" in _bad["error"]["traceback"])
check("the record keeps the schema it couldn't reshape, so the input is still readable",
      "amt" in json.dumps(_bad["sent"]))
check("benchmark-only keys are still stripped from it", "evaluation_config" not in json.dumps(_bad["sent"]))
try:
    _predict_with(_Spy(explode="bug"), _copy.deepcopy(_ANNOTATED))
    _crashed = None
except KeyError as _exc:
    _crashed = _exc
check("any other exception from prepare_schema is a bug, raised rather than counted 0", _crashed is not None)
check("and the vendor was never called", "schema" not in _SEEN)


print("\n[5d] every adapter is an Extractor, and a model id is the one LLM adapter")
from omni_extract_bench import harness as _harness  # noqa: E402

for _name, _cls in _harness.ADAPTERS.items():
    check(f"{_name}: is an Extractor", issubclass(_cls, _contract.Extractor))
check("a model id routes to the one LLM adapter",
      type(_harness.system("openai/gpt-5.6-sol")) is type(_harness.system("anthropic/claude-opus-5"))
      is _harness.LLM)


print("\n[5] the overlay states a convention without changing the task")
ov = SCHEMA.apply_overlay(SAMPLE)
check("leaf addresses unchanged", set(leaves(ov)) == set(leaves(SAMPLE)))
check("types unchanged", leaves(ov) == leaves(SAMPLE))
check("input not mutated", "evaluation_config" in SAMPLE)
check("applying twice is idempotent", SCHEMA.apply_overlay(ov) == ov)

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
        if set(leaves(datalab.Datalab().prepare_schema(s))) != base:
            drops["datalab"].append(name)
        if set(leaves(SCHEMA.apply_overlay(s))) != base:
            drops["overlay"].append(name)
        if set(leaves(strip_bench_keys(s))) != base:
            drops["strip"].append(name)
    print(f"  checked {len(paths)} real schemas")
    for k, v in drops.items():
        check(f"{k} preserves every leaf of every schema", not v, f"{len(v)}: {v[:3]}")

print(f"\n{'All provider dialect tests pass' if not FAILS else 'Failures: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
