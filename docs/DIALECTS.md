# Providers and schema dialects

Every vendor accepts a different subset of JSON Schema, and the strict ones reject what the
permissive ones ignore. Send one shape to everyone and the strict vendors score zero on
documents they could have handled — a fact about your harness, reported as a fact about them.

`omni_extract_bench.harness.dialects` holds the transforms, split by intent:

```python
from omni_extract_bench.harness.dialects import (
    strip_benchmark_keys,    # remove YOUR grader metadata — every vendor
    resolve_refs,            # inline $ref so the schema is self-describing
    to_strict_dialect,       # allowlisted keys; nullable properties, bare array items
    to_typed_enum_dialect,   # enums need a matching type, then the null must go
    parse_model_json,        # fenced JSON is still JSON
    cost_from_response,      # vendor cost, converted to USD from whatever unit
)
```

Each exists because of a measured failure, not a hypothetical:

| transform | what it prevents |
| --- | --- |
| `strip_benchmark_keys` | shipping scoring annotations to vendors as part of the task |
| `resolve_refs` | a bare `$ref` declaring no type |
| `to_strict_dialect` | nullable rules that **differ by position** — properties need `["string","null"]`, array items need a bare `string` |
| `to_typed_enum_dialect` | `{"enum": [...]}` with no type, then a `null` member that violates the inferred type |
| `parse_model_json` | discarding a correct answer wrapped in a markdown fence |
| `cost_from_response` | reporting a cents-denominated field as dollars (100× overstatement) |

In one run these took a provider from 8th place with 19 of 40 documents failed to 4th with
none — the extraction quality never changed. Treat a vendor's low coverage as a harness bug
until proven otherwise.

`omni_extract_bench/harness/providers/extend.py` is a reference adapter showing the shape an integration takes.
Provider APIs change; treat it as an example rather than a maintained client.

