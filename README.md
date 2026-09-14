# Omni Extract Bench

One scorer for document-extraction benchmarks.

Extraction benchmarks tend to ship their own grader, so scores are not comparable across them
and format differences get scored as errors. This is the scoring half: a single metric, applied
identically to every subset and every provider, with the properties it claims written down and
tested rather than asserted.

It grades a predicted JSON object against a ground-truth object and a JSON Schema. It does not
run extractors, and it ships no benchmark data.

## Data

The frozen 169-document evaluation set, with verified ground-truth corrections already applied,
is published separately:
[`datalab-to/omni_extract_bench`](https://huggingface.co/datasets/datalab-to/omni_extract_bench).

This repository holds the scorer only; it ships no benchmark data and no benchmark results.

## Install

```bash
pip install -e .
```

Python 3.11+. One dependency: `scipy`, for the assignment solver
(`linear_sum_assignment`).

## Use

```python
from omni_extract_bench import grade

result = grade(prediction, ground_truth, schema)
result["leaf_accuracy"]   # 0-100, the headline number
result["recall"]          # gold rows matched / gold rows
result["precision"]       # gold rows matched / predicted rows
result["matching_exact"]  # False if an array was too large to solve exactly
```

From the command line:

```bash
omni-extract-bench score      --pred p.json --gt g.json --schema s.json
omni-extract-bench score-dir  --pred-dir preds/ --gt-dir gt/ --schema-dir schemas/
omni-extract-bench leaderboard --pred-root baselines/ --gt-dir gt/ --schema-dir schemas/
```

`leaderboard` scores every provider directory under `--pred-root` over the same document list.

## What the metric does

**Leaf value accuracy.** Every scalar in the ground truth is one point. The score is the
fraction matched, counting spurious predicted leaves against you as well as missing ones.

**Arrays are matched optimally.** Rows are paired by maximum-weight bipartite matching
(Hungarian / Jonker-Volgenant, via `scipy`), not by index or by a guessed key, so a provider is
never punished for row order. Where a document is too large to solve exactly, the fallback is
approximate and **says so** in `matching_exact` — an approximate score is never reported as
though it were exact.

**Format is free; content is not.** `10/31/2024` equals `2024-10-31`; `5`, `5.0` and `"5.00"`
agree; `(98.2)` equals `-98.2`. But `-98.2` never equals `98.2`, and ID-like integers stay
exact so `8303911426` never equals `8303511426`.

**Omission is charged.** Returning 44 of 349 rows scores about 12, not 100. This is the single
most important property: a metric that lets an extractor skip rows for free will rank a
truncating system above a complete one.

**No output scores zero.** A document a provider failed to return is not dropped from its mean,
or a system that fails on hard documents outranks one that attempts them. Coverage is reported
alongside, and a "score on returned documents only" view separates *processes documents badly*
from *silently fails on hard documents*.

**One definition of equality.** `canon_key(value)` is the only comparison rule, used by leaf
scoring, row-pair weighting, and blocking alike. There is no per-field or per-vendor mode.

Full specification, including all sixteen properties: [`docs/METRIC_SPEC.md`](docs/METRIC_SPEC.md).

## Why the properties are tested

Most of them are regressions. Each of these reached a leaderboard before it was caught:

| property | the bug it prevents |
| --- | --- |
| P11 nesting invariance | omission was free for top-level arrays but charged when nested — 44 of 349 rows scored **100.0** |
| P13 scalar arrays scored | a top-level array of strings had denominator 0 |
| P14 pairing/scoring agree | pairing demanded literal equality while scoring accepted date formats, so a correct extraction scored **50.0** |
| P16 blocking equality | blocking used a third notion of equality, so rows differing only in **capitalisation** scored **0.0** |

Four of these were one root cause: several implementations of "are these two values equal?"
that had to agree, and did not. They are now one function, which makes a disagreement
unrepresentable rather than something tests have to catch.

```bash
python tests/test_metric_properties.py        # P1-P17, generative
python tests/test_metric_structural_audit.py  # wrapping invariance over 300 generated documents
python tests/test_grader_invariants.py        # identity, determinism, traps
python tests/test_dialects.py                 # per-vendor schema dialects
python tests/test_capture.py                  # transport capture, including a real subprocess
```

The structural audit is the strongest of these: if wrapping a document in an extra level cannot
change its score, no depth-dependent scoring path can exist.

## Capturing runs so you pay for them once

Benchmarking extraction vendors costs real money on calls that are slow, often async, and
sometimes nondeterministic. What you keep from each call decides whether a failure costs you an
explanation or another invoice — so `omni_extract_bench.harness.capture` records responses at
the **transport layer**, before anything parses them.

```python
from omni_extract_bench.harness import capture

capture.install_taps()                 # in-process SDK calls (httpx + requests)
capture.reset()                        # per document; records are thread-local
result = run_one_document(pdf, schema)
record = {
    "result": result,
    "http": capture.records(),         # status, body, timing, and the request minus its payload
    "job_ids": capture.job_ids(),      # what makes cost recoverable WITHOUT re-running
    "usage": capture.usage_from_records(),
}
```

For adapters you launch as their own process, patching the parent reaches nothing:

```python
env = capture.subprocess_env(os.environ, log_path)   # sitecustomize taps the child
subprocess.run(cmd, env=env, ...)
capture.merge_subprocess_log(log_path)               # merge on failure too — especially then
```

Three things are worth stating plainly, because each was learned the expensive way:

**Capture below the parser, or you capture the parser's opinion.** Wrapping the adapter records
its return value, so a model's text is gone whenever `json.loads` failed, an HTTP error body
becomes a status code, and the vendor's billing fields vanish with the envelope. Those look like
three bugs and are one placement error.

**Test the contents, not the key.** An audit here counted capture files, found a populated field
named `raw` on every one, and reported full coverage — while the field held parsed output. The
same audit passed while *subprocess* providers captured no HTTP whatsoever.

**Tap every transport the code *could* use, not the one you believe it uses.** Four are covered
— `httpx.Client`, `httpx.AsyncClient`, `requests.Session`, `urllib.request` — plus a
`sitecustomize` for subprocesses. Each gap was found separately, *after* the previous one had
supposedly fixed capture, because they all look identical from the outside: the file exists, the
key is present, the list is empty. The last one was a provider adapter that used no SDK at all
and called the REST API with the standard library.

They live in one module (`harness/_tap/oeb_capture.py`) for the same reason: maintained as
near-copies,
each gap has to be found and fixed once per copy, which is how the fourth one survived the
first three fixes.

**Keep job ids.** Async APIs hand back an id and keep the job, so usage can usually be recovered
from job history later for free. Re-running a document to recover a number the vendor will still
hand you is the worst available trade, and for a nondeterministic provider it does not even
reproduce the answer you scored.

**A parsing mistake should cost a re-parse, not another invoice.** Because the bodies are on
disk, a bug in how usage is *read* is fixed by re-reading them. That happened here: the usage
parser filtered keys against a list of known names and so discarded a vendor's
`num_pages_billed` for not being on the list — page counts recovered afterwards from stored
responses, with nothing re-run. Keep whatever the vendor puts in its usage block; a known-names
filter is the hard-coded-path mistake one level down.

Units deserve the same care as values: several vendors report `credits`, which is not dollars —
the rate is contract-specific. `usage_from_records` keeps the vendor's own field names so no
conversion happens by accident, and `dialects.cost_from_response` converts only where the unit
is known (a field named in cents reported as dollars overstates by 100x).

## Providers and schema dialects

Every vendor accepts a different subset of JSON Schema, and the strict ones reject what the
permissive ones ignore. Send one shape to everyone and the strict vendors score zero on
documents they could have handled — a fact about your harness, reported as a fact about them.

`omni_extract_bench.dialects` holds the transforms, split by intent:

```python
from omni_extract_bench.dialects import (
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

`harness/providers/extend_provider.py` is a reference adapter showing the shape an integration
takes.
Provider APIs change; treat it as an example rather than a maintained client.

## Licence and attribution

Apache 2.0 — see [`LICENSE`](LICENSE).

Value canonicalisation builds on the `longextract_bench` grader (MIT, © Micro1), vendored under
`omni_extract_bench/vendor/` with its licence intact. See [`NOTICE`](NOTICE).
