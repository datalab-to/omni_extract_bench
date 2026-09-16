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

To score requires a table containing or pointing to the necessary inputs. The required columsn are:

| column | |
|---|---|
| `doc_id` | your name for it; unique within the table |
| `gt_path` | the ground truth, JSON |
| `pred_path` | the prediction, JSON |
| `schema` | the JSON Schema itself, inline |

The two documents are paths because ground-truth and predictions are unbounded; the schema is inline. 
Paths go through fsspec, so `s3://`, `gs://` and a local path are the same thing.

```bash
oeb score --manifest jobs.parquet --out run/
```

```
run/scores/part-00000.parquet      one row per manifest row
run/verdicts/part-00000-0.parquet  one row per verdict
```

Two tables are outputted because they are read differently: a leaderboard reads every score and no
verdicts, an audit reads one document's verdicts and no scores, and verdicts are two orders of
magnitude larger. Every other column you put in the manifest rides through to `scores` untouched.

Every row comes back, including the ones that failed. A row that could not be graded has
`status="error"`, null metrics, and an `error` saying why: a traceback where something raised.
Null rather than zero, because a zero claims the model tried and missed every field — so when you average, filter on
`status == "scored"` and say how many documents that was.

Walk through it on data in this repo, including what each verdict means:
[`tutorials/quickstart_scoring.md`](tutorials/quickstart_scoring.md).

One pair, no table:

```bash
oeb score-one --gt gold.json --schema schema.json --pred pred.json
```

### On Modal

Same scorer, one container per batch, for when one machine is the bottleneck:

```bash
modal run --detach -m omni_extract_bench.run_score_modal \
    --manifest s3://bucket/jobs.parquet --out s3://bucket/run --rows 16
```

Your machine needs no bucket credentials but modal does: see [`tutorials/quickstart_scoring.md`](tutorials/quickstart_scoring.md) for details.

## Predicting

Required columns:

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


Vendor adapters live behind an extra:
`pip install 'omni-extract-bench[harness]'`.


## What the metric does

![gif](./docs/animation/scoring.gif)

- Normalize document;
- Flatten prediction and gold JSON dictionary to addresses mapped to their scalar values;
- Normalize scalar values of the flattened addresses; and
- For each array that appears, Hungarian match (recursively for nested arrays) based on array element content to align ambiguous predicted and gold addresses (there may unmatched predicted addresses — false positives, and unmatched gold addresses — false negatives).

For each document, this process produces one `Verdict` per unique scalar address, aligned via Hungarian matching when needed. The options are:

- `matched`: was matched and the values match;
- `misread`: was matched and the values don’t match;
- `unfound`: ground-truth has the address, prediction doesn’t;
- `fabricated`: schema offered the address, ground-truth is silent but prediction exists;
- `invented_item`: an array element’s scalar prediction that paired with nothing; or
- `invented_field`: an address the schema never declared.

These are mutually exclusive in our code and also semantically. The one interesting judgement call we made here is that an address falls under `invented_item` it falls within an unpaired item (i.e. row), even if the address was an invented field *within* that array element’s schema. We think this is the right call: it signals that this was counted against the model for inventing an item. Addresses outside of arrays that the schema never declared is `invented_field`.

Full specification: [`docs/METRIC_SPEC.md`](docs/METRIC_SPEC.md).

## Licence

Apache 2.0 — see [`LICENSE`](LICENSE). All code here is Datalab's own.
