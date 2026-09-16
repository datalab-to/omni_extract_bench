# Omni Extract Bench

One scorer for document-extraction benchmarks.

Extraction benchmarks tend to ship their own grader, so scores are not comparable across them
and format differences get scored as errors. This is the scoring half: a single metric, applied
identically to every document and every provider, with the properties it claims written down
and tested rather than asserted.

```bash
pip install omni-extract-bench                 # the scorer; one dependency, scipy
pip install 'omni-extract-bench[benchmark]'    # running a table of them
```

Python 3.11+.

## Scoring

**A run is one table.** Each row names a document and its three inputs — there is no corpus
directory, no naming convention, no registry:

| column | |
|---|---|
| `doc_id` | your name for it; unique within the table |
| `gt_path` | the ground truth, JSON |
| `pred_path` | the prediction, JSON |
| `schema` | the JSON Schema itself, inline |

The two documents are paths because a ground truth runs to megabytes; the schema is inline
because it is small and because it is the *question* the document was asked. Paths go through
fsspec, so `s3://`, `gs://` and a local path are the same thing.

```bash
oeb score --manifest jobs.parquet --out run/
```

```
run/scores/part-00000.parquet      one row per manifest row
run/verdicts/part-00000-0.parquet  one row per address
```

Two tables because they are read on opposite schedules: a leaderboard reads every score and no
verdicts, an audit reads one document's verdicts and no scores, and verdicts are two orders of
magnitude larger.

**Every other column you put in the manifest rides through to `scores` untouched** — vendor,
suite, capture date, cost. The scorer never learns what they are, and without that anyone
comparing two vendors rebuilds the registry as a join on the side.

**Every row comes back, including the ones that failed.** A row that could not be graded has
`status="error"`, null metrics, and an `error` saying why: a traceback where something raised,
the system's own words where the prediction recorded its own failure. Null rather than zero,
because a zero claims the model tried and missed every field — so when you average, filter on
`status == "scored"` and say how many documents that was.

Walk through it on data in this repo, including what each verdict means:
[`tutorials/quickstart_scoring.md`](tutorials/quickstart_scoring.md).

One pair, no table:

```bash
oeb score-one --gt gold.json --schema schema.json --pred pred.json
```

## Predicting

Same shape, one column fewer, and the provider is a run rather than a column — so **one
manifest serves every vendor**:

| column | |
|---|---|
| `doc_id` | unique |
| `doc_path` | the PDF |
| `schema` | inline |

```bash
oeb predict --manifest docs.parquet --out preds/ --provider datalab
```

```
preds/predictions/<doc_id>.json     the bare extraction
preds/manifest/part-00000.parquet   a row per document, with pred_path filled in
```

**That output table is a score manifest.** If the input carried `gt_path`, score it with
nothing joined and nothing assembled:

```bash
oeb score --manifest preds/manifest --out run/
```

Per-prediction accounting (`provider`, `pred_status`, `pred_latency_s`, `pred_cost_usd`) rides
through scoring to sit beside the accuracy. A document the vendor failed is written as
`{"__error__": ...}` and stays a row with a reason.

Vendor adapters live behind an extra, so a machine that only scores installs no vendor SDK:
`pip install 'omni-extract-bench[harness]'`.

## On Modal

Same scorer, one container per batch, for when one machine is the bottleneck:

```bash
modal run --detach -m omni_extract_bench.run_score_modal \
    --manifest s3://bucket/jobs.parquet --out s3://bucket/run --rows 16
```

Your machine needs no bucket credentials — it passes paths as strings, and only the containers
read and write. `--detach` is what lets the laptop close mid-run.

## What the metric does

**Leaf value accuracy.** Every scalar either document asserts is one address. The score is the
share of addresses both used *and* agreed on, so spurious predicted leaves count against you as
well as missing ones.

**Arrays are matched optimally.** Rows are paired by maximum-weight bipartite matching
(Hungarian / Jonker-Volgenant, via `scipy`), not by index or a guessed key, so a provider is
never punished for row order. Where a document is too large to solve exactly the fallback is
approximate and **says so** in `matching_exact`.

**Format is free; content is not.** `10/31/2024` equals `2024-10-31`; `5`, `5.0` and `"5.00"`
agree; `(98.2)` equals `-98.2`. But `-98.2` never equals `98.2`, and ID-like integers stay exact
so `8303911426` never equals `8303511426`.

**Omission is charged.** Returning 44 of 349 rows scores about 12, not 100. A metric that lets
an extractor skip rows for free ranks a truncating system above a complete one.

**One definition of equality.** `canon_key(value)` is the only comparison rule, used by leaf
scoring, row-pair weighting and blocking alike. No per-field or per-vendor mode.

Every address carries exactly one verdict — `matched`, `misread`, `unfound`, `fabricated`,
`invented_item`, `invented_field` — so the six partition the score and sum to it.

Full specification: [`docs/METRIC_SPEC.md`](docs/METRIC_SPEC.md).

## Why the properties are tested

Most are regressions. Each of these reached a leaderboard before it was caught:

| property | the bug it prevents |
|---|---|
| P11 nesting invariance | omission was free for top-level arrays but charged when nested — 44 of 349 rows scored **100.0** |
| P13 scalar arrays scored | a top-level array of strings had denominator 0 |
| P14 pairing/scoring agree | pairing demanded literal equality while scoring accepted date formats, so a correct extraction scored **50.0** |
| P16 blocking equality | blocking used a third notion of equality, so rows differing only in **capitalisation** scored **0.0** |

Four were one root cause: several implementations of "are these two values equal?" that had to
agree and did not. They are now one function, which makes the disagreement unrepresentable.

```bash
python tests/test_metric_properties.py   # P1-P17, generative
python tests/test_run_score.py           # the manifest contract
python tests/test_score_properties.py    # what must be true of the scorer
python tests/test_dialects.py            # per-vendor schema dialects
```

## Elsewhere

- Schema dialects, and why low coverage is a harness bug until proven otherwise:
  [`docs/DIALECTS.md`](docs/DIALECTS.md)
- Capturing vendor responses at the transport layer, so you pay for a call once:
  [`docs/CAPTURE.md`](docs/CAPTURE.md)
- Measurements taken and deliberately not acted on:
  [`docs/FOLLOW_UPS.md`](docs/FOLLOW_UPS.md)

The evaluation set is published separately; this repository holds the scorer and ships no
benchmark data.

## Licence

Apache 2.0 — see [`LICENSE`](LICENSE). All code here is Datalab's own.
