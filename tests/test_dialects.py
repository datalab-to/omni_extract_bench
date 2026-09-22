#!/usr/bin/env python3
"""Tests for the per-vendor schema dialects.

Every case here is a real vendor rejection observed while running this benchmark, not a
hypothetical. The comments name what failed, because the value of these transforms is only
obvious once you have seen a provider score zero on documents it could handle.

Run: python3 tests/test_dialects.py
"""
import json
import sys


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench.harness.schema import resolve_refs
from omni_extract_bench.harness.providers.llamaextract import to_typed_enum_dialect
from omni_extract_bench.harness.providers.extend import to_strict_dialect
from omni_extract_bench.harness.responses import cost_from_response, parse_model_json
from omni_extract_bench.harness.schema import strip_benchmark_keys

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


print("\n[1] universal: harness metadata is removed, fields are not")
schema = {
    "type": "object",
    "properties": {
        "amount": {"type": "number", "evaluation_config": "number_tolerance", "default": 0},
        "rows": {"type": "array", "items": {"type": "string", "evaluation_config": "x"}},
    },
}
clean = strip_benchmark_keys(schema)
blob = json.dumps(clean)
check("evaluation_config removed", "evaluation_config" not in blob)
check("default removed", '"default"' not in blob)
check("fields preserved", set(clean["properties"]) == {"amount", "rows"})
check("types preserved", clean["properties"]["amount"]["type"] == "number")

print("\n[2] $ref resolution — a bare $ref declares no type, which strict vendors reject")
ref_schema = {
    "type": "object",
    "$defs": {"row": {"type": "object", "properties": {"a": {"type": "string"}}}},
    "properties": {"rows": {"type": "array", "items": {"$ref": "#/$defs/row"}}},
}
resolved = resolve_refs(ref_schema)
items = resolved["properties"]["rows"]["items"]
check("refs inlined", "$ref" not in json.dumps(resolved))
check("$defs removed", "$defs" not in resolved)
check("items now declare a type", items.get("type") == "object")
check("inlined content intact", "a" in (items.get("properties") or {}))

print("\n[3] strict dialect — the rules differ BY POSITION")
strict = to_strict_dialect(resolve_refs(strip_benchmark_keys({
    "type": "object",
    "properties": {
        "name": {"type": "string", "title": "Name"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "level": {"enum": ["LOW", "HIGH"]},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
})))
props = strict["properties"]
check("property primitive is nullable", props["name"]["type"] == ["string", "null"],
      str(props["name"]))
check("ARRAY ITEM stays a bare type", props["tags"]["items"]["type"] == "string",
      str(props["tags"]["items"]))
described = to_strict_dialect(resolve_refs(strip_benchmark_keys({
    "type": "object",
    "properties": {
        "rows": {"type": "array", "items": {"type": "object", "properties": {
            "cases": {"type": "array", "description": "list",
                      "items": {"type": "string", "description": "one case"}}}}},
        "name": {"type": "string", "description": "kept on a property"},
    },
})))
leaf = described["properties"]["rows"]["items"]["properties"]["cases"]["items"]
check("scalar array item carries ONLY type", set(leaf) == {"type"}, str(leaf))
check("object array item keeps properties",
      "properties" in described["properties"]["rows"]["items"])
check("a property keeps its description",
      described["properties"]["name"].get("description") == "kept on a property")
check("enum gains null", None in props["level"]["enum"], str(props["level"]))
check("nullable union collapsed then re-nulled",
      props["note"]["type"] == ["string", "null"], str(props["note"]))
check("disallowed key dropped", "title" not in props["name"])

print("\n[4] typed-enum dialect — enum needs a type, and then the null must go")
typed = to_typed_enum_dialect({
    "type": "object",
    "properties": {
        "severity": {"enum": ["MILD", "MODERATE", "SEVERE", None]},
        "skills": {"anyOf": [
            {"type": "array", "items": {"type": "string"}},
            {"type": "object", "additionalProperties": {"type": "array"}},
            {"type": "null"},
        ]},
    },
})
sev = typed["properties"]["severity"]
check("type inferred from enum values", sev.get("type") == "string", str(sev))
check("null removed once typed", None not in sev["enum"], str(sev))
check("nested union collapsed", "anyOf" not in json.dumps(typed["properties"]["skills"]))
ap = to_typed_enum_dialect({"type": "object", "additionalProperties": {"type": "array"}})
check("additionalProperties reduced to boolean", ap["additionalProperties"] is True)

print("\n[5] fenced JSON is still JSON")
check("bare json", parse_model_json('{"a": 1}') == {"a": 1})
check("```json fence", parse_model_json('```json\n{"a": 1}\n```') == {"a": 1})
check("bare fence", parse_model_json('```\n{"a": 1}\n```') == {"a": 1})
check("fence containing backticks",
      parse_model_json('```json\n{"x": "has ``` inside"}\n```') == {"x": "has ``` inside"})
check("non-json returns None", parse_model_json("I could not find that") is None)
check("empty returns None", parse_model_json("") is None)

print("\n[6] cost — units are the trap")
usd, raw = cost_from_response({"cost_breakdown": {"final_cost_cents": 6.0}})
check("cents converted to USD", usd == 0.06, f"{usd} from {raw}")
check("openrouter USD passthrough",
      cost_from_response({"usage": {"cost": 0.0431}})[0] == 0.0431)
check("absent cost is None, not zero", cost_from_response({"pages": 3})[0] is None)
check("booleans are not costs", cost_from_response({"cost": True})[0] is None)

print(f"\n{'ALL DIALECT TESTS PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
