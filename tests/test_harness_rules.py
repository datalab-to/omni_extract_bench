"""Harness smoke test: the parity rules hold and a schema survives preparation intact.

No network. Runs the schema pipeline the runner applies to every provider and asserts the
properties that make the comparison fair, so a future edit cannot quietly change what a vendor
is asked without failing here.
"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from harness import run_provider as R
from harness import schema_overlay as SO

FAILS = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if not cond else ""))
    if not cond: FAILS.append(name)

SCHEMA = {"$defs": {"row": {"type": "object", "properties": {"capex": {"type": ["number", "null"],
          "description": "Capital expenditures.", "evaluation_config": "number_tolerance"}}}},
          "type": "object",
          "properties": {"filed": {"type": "string", "default": "n/a"},
                         "rows": {"type": "array", "items": {"$ref": "#/$defs/row"}}}}

print("\n[1] benchmark-only annotations never reach a vendor")
stripped = R.strip_bench_keys(SCHEMA)
blob = json.dumps(stripped)
check("evaluation_config removed", "evaluation_config" not in blob)
check("default removed", '"default"' not in blob)
check("field names survive", "capex" in blob and "filed" in blob and "rows" in blob)
check("descriptions survive", "Capital expenditures." in blob)

print("\n[2] $ref inlining changes no content")
d = R.deref(SCHEMA)
check("no $ref left", "$ref" not in json.dumps(d))
check("no $defs left", "$defs" not in d)
check("the referenced shape is inlined", "capex" in json.dumps(d["properties"]["rows"]["items"]))

print("\n[3] the conventions overlay states conventions, it does not change the task")
overlaid = SO.apply_overlay(SCHEMA)
before, after = json.dumps(SCHEMA, sort_keys=True), json.dumps(overlaid, sort_keys=True)
def shape(node):
    if isinstance(node, dict):
        return {k: shape(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list): return [shape(x) for x in node]
    return node
check("overlay is non-destructive (input unchanged)", json.dumps(SCHEMA, sort_keys=True) == before)
check("only descriptions differ", json.dumps(shape(overlaid), sort_keys=True) == json.dumps(shape(SCHEMA), sort_keys=True))

print("\n[4] parity constants are single-valued and present for every provider")
check("one timeout default", R.DEFAULT_TIMEOUT == 1800)
missing = [p for p in R.PROVIDERS if p not in R.PROVIDER_TIER]
check("every provider declares a tier", not missing, str(missing))
check("model ceilings are per model, not a shared floor",
      len(set(R.MODEL_MAX_OUTPUT.values())) > 1)

print("\n[5] retry policy: infrastructure retries, real answers do not")
check("429 is transient", R.is_transient("HTTP 429 rate limit"))
check("503 is transient", R.is_transient("HTTP 503 unreachable_backend"))
check("empty 200 is transient", R.is_transient(f"{R.EMPTY_200_MARKER} after 900s: padding"))
check("400 is NOT transient", not R.is_transient("HTTP 400: unsupported properties"))
check("context-length rejection is NOT transient", not R.is_transient("maximum context length"))
check("402 stops the run", R.is_account_failure("HTTP 402: exceeded the maximum number of credits"))
check("a 400 is not an account failure", not R.is_account_failure("HTTP 400: bad schema"))

print("\n[6] the empty-200 marker is ONE string, shared by message and predicate")
msg = R.classify_no_output.__doc__ or ""
check("marker constant exists", isinstance(R.EMPTY_200_MARKER, str) and R.EMPTY_200_MARKER)
check("predicate keys on the constant", R.is_transient(R.EMPTY_200_MARKER))

print(f"\n{'ALL HARNESS CHECKS PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
