# Omni Extract Bench

One scorer for document-extraction benchmarks.

Extraction benchmarks tend to ship their own grader, so scores are not comparable across them
and format differences get scored as errors. This is the scoring half: a single metric, applied
identically to every subset and every provider, with the properties it claims written down and
tested rather than asserted.

It grades a predicted JSON object against a ground-truth object and a JSON Schema. It does not
run extractors, and it ships no benchmark data.

## Install

```bash
pip install omni-extract-bench            # the scorer; one dependency, scipy
pip install 'omni-extract-bench[benchmark]'   # plus running a whole benchmark
```

Python 3.11+.

## Quick start

Score a directory of predictions against the published benchmark:

```bash
oeb score --predictions preds/ --out run/
```

`preds/` holds one `<doc_id>.json` per document, containing the extraction itself. The corpus
is downloaded for you.

To score one pair with no benchmark involved:

```python
from omni_extract_bench import grade

r = grade(prediction, ground_truth, schema)
r["accuracy"]         # 0-100, the headline number
r["recall"]           # gold rows matched / gold rows
r["precision"]        # gold rows matched / predicted rows
r["matching_exact"]   # False if an array was too large to solve exactly
```

## Bring your own ground truth

A corpus is a directory of document directories, plus an atlas that says which of them are in
the benchmark:

```
<corpus>/corpus.parquet            the atlas -- the source of truth
<corpus>/<doc_id>/ground_truth.json
<corpus>/<doc_id>/schema.json
```

```bash
oeb build-corpus --corpus my-benchmark/                    # declare what it contains
oeb score --corpus my-benchmark/ --predictions preds/ --out run/
```

**The atlas is what you explore and what you run.** It is a plain parquet — `doc_id`, where the
two files are, and whatever higher-level metadata you have alongside. Scoring follows its rows,
so filtering the table is how you choose a subset:

```python
import pyarrow.parquet as pq, pyarrow as pa
t = pq.read_table("my-benchmark/corpus.parquet")
pq.write_table(pa.Table.from_pylist([r for r in t.to_pylist() if r["suite"] == "invoices"]),
               "my-benchmark/corpus.parquet")
```

Predictions for documents the atlas no longer lists are skipped and counted. A row naming a
file that is not there stops the run — the atlas is what the run follows, so it has to resolve.

`--corpus` also takes a **specific atlas file**, so a subset can live beside the full list
rather than replacing it:

```bash
oeb score --corpus my-benchmark/invoices-only.parquet --predictions preds/
```

Rows name their files relative to the atlas, so a filtered copy written next to
`corpus.parquet` describes the same documents and needs no rewriting.

```bash
oeb verify --corpus my-benchmark/     # what has changed since the atlas was written
```

The published benchmark is one instance of that contract, not a special case -- it carries
PDFs, provenance and a parquet atlas alongside, and none of that is required of yours.

A schema is required and is never inferred. Without one, an `additionalProperties` subtree
would be graded silently, and a value the model invented could not be told from one the schema
offered it a slot for.

## Seeing why a score is what it is

A run records every address, not just the total:

```
run/
  summary.parquet              one row per (doc_id, prediction_id)
  verdicts/<doc_id>.parquet    every address, with gold and pred
```

They are ordinary parquet files, so exploring them needs no library:

```sql
select doc_id, address, gold, pred
from 'run/verdicts/*.parquet'
where verdict = 'misread';
```

Or for one document, without SQL:

```bash
oeb explain --run run/ --doc <doc_id>
```

This is worth doing before trusting an accuracy number. On a sample of real vendor output,
every wrong value was a boundary disagreement rather than a misreading -- `"Glenmere Robotics"`
against `"Glenmere Robotics Inc."`, `"14 March 2026"` against `"Updated 14 March 2026"`. That is
invisible in a score and obvious per address.

Verdicts come back from the pass that computes the score, so they cost time you have already
spent: 43.8s against 41.4s on an 89,000-leaf document, where asking for both separately took
90.9s. They do cost memory -- one record per address, and the largest document in the corpus
has 410,012 -- so `--no-verdicts` turns them off.

### Or beside the page it was read from

```bash
oeb ui --run run/ --run run-other/ --corpus my-benchmark/ --out site/
python -m http.server -d site/
```

Every address of a document, next to the PDF, with the browser's own find working on the page
-- which is how you tell "the vendor misread this" from "the value is not in the document".
Repeat `--run` to put vendors side by side; each run is one prediction set, and the column is
labelled with its `--source`.

The site is an index plus one file per document, because the whole run does not fit in a page.
The corpus at nine vendors is ~1.1 GB of addresses; what a browser loads is a 420 KB index over
all 660 documents, then ~1.7 MB for the one you opened. The PDFs are symlinked, so the corpus's
774 MB of them cost nothing.

Full guide: [`docs/USING.md`](docs/USING.md).

## Three outcomes, kept apart

Every row records a `kind`, and the distinctions are the point:

| kind | meaning |
| --- | --- |
| `graded` | scored on its merits |
| `unusable` | the provider returned nothing scoreable: an error payload, an empty object, bytes that are not JSON |
| `failed` | **this harness** could not score it |

Metrics on the last two are null, never zero. Zero is an interpretation, and `mean(accuracy)`
without filtering on `kind` adopts it silently -- which makes a rate-limited run look like a bad
model. Keeping `failed` apart matters for the same reason in reverse: a broken schema in your
corpus must not read as a vendor scoring badly.

An earlier version of this scorer had two different answers to the `graded`/`unusable`
question, and a provider read as 100% coverage while 37 of its 45 outputs were empty.

## The benchmark data

The frozen evaluation set, with verified ground-truth corrections applied, is published
separately: [`datalab-to/omni_extract_bench`](https://huggingface.co/datasets/datalab-to/omni_extract_bench).

This repository holds the scorer. It ships no benchmark data and no benchmark results.

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
python tests/test_bench.py                    # the corpus contract, and graded/unusable/failed
python tests/test_cli.py                      # the command line, end to end
python tests/test_scores.py                   # the scores table and corpus versioning
```

The structural audit is the strongest of these: if wrapping a document in an extra level cannot
change its score, no depth-dependent scoring path can exist.

## Capturing runs so you pay for them once

Vendor calls are slow, async, and billed. `omni_extract_bench.harness.capture` records responses at the
**transport layer**, before anything parses them, so a parsing bug costs a re-parse rather than
another invoice — and job ids are kept, because usage can usually be recovered from job history
for free.

Four transports are tapped (`httpx` sync and async, `requests`, `urllib`) plus a
`sitecustomize` for adapters launched as their own process. Each gap was found separately,
after the previous one had supposedly fixed capture, because they all look identical from the
outside: the file exists, the key is present, the list is empty.

Details, and the three mistakes that produced them: [`docs/CAPTURE.md`](docs/CAPTURE.md).

## Providers and schema dialects

Every vendor accepts a different subset of JSON Schema, and the strict ones reject what the
permissive ones ignore. Send one shape to everyone and the strict vendors score zero on
documents they could have handled — a fact about your harness, reported as a fact about them.

`omni_extract_bench.harness.dialects` holds the per-vendor transforms, each one written for a measured
failure. In one run they took a provider from 8th place with 19 of 40 documents failed to 4th
with none, and the extraction quality never changed. **Treat a vendor's low coverage as a
harness bug until proven otherwise.**

The transforms and what each prevents: [`docs/DIALECTS.md`](docs/DIALECTS.md).

## Licence and attribution

Apache 2.0 — see [`LICENSE`](LICENSE).

All code here is Datalab's own.
