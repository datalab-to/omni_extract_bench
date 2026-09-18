# API

This is the public facing API for the toolkit. There are three main entry points:

- `score`: score a document's prediction against the ground truth.
- `predict`: predict extractions for a document with one of our providers.
- `benchmark`: run our benchmark.

The first two are the primitives the last one is built around. Each one works in your code and
from the cli, and they do the same thing either way — the cli is a thin wrapper.

1. **[score](#score)** — grade one prediction against one ground truth.
2. **[predict](#predict)** — one document, one vendor, under the parity rules.
3. **[benchmark](#benchmark)** — the whole corpus, every vendor, resumable.
4. **[providers](#providers)** — what you can run, and what each one takes.
5. **[What a benchmark run writes](#what-a-benchmark-run-writes)** — the files, and what's in them.


## score

Needs the base install. No network, no API keys, no vendor packages.

```python
from omni_extract_bench import score

score(pred, gt, schema, order_matters=(), verdicts=False) -> dict
```

| argument | what it does |
| --- | --- |
| `pred` | the prediction, as a dict |
| `gt` | the ground truth, as a dict |
| `schema` | the JSON Schema the prediction was generated against |
| `order_matters` | arrays whose order is part of the answer, e.g. `["steps", "books[*].chapters"]`. Default: none — the order rows appear in a document is usually an accident of layout |
| `verdicts` | also return one verdict per address: what happened there, and what each side was compared as |

```python
result = score(prediction, ground_truth, schema)
result["accuracy"]    # matched addresses / addresses either document used
result["precision"]
result["recall"]
```

or from the command line.

```bash
oeb score --pred pred.json --gt gold.json --schema schema.json
```

It writes the result to stdout as JSON, so `oeb score ... | jq .accuracy` is one pipe. Add
`--verdicts` for the per-address breakdown and `--order-matters steps 'books[*].chapters'` for
arrays where position is the answer.

The README has a [worked example](../README.md#score) and
[`docs/METRIC_SPEC.md`](./METRIC_SPEC.md) is the full specification.


## predict

Needs the harness extra: `uv pip install 'omni-extract-bench[harness]'`.

```python
from omni_extract_bench.harness import predict

predict(provider, pdf, schema, *, timeout=1800.0, overlay=True, **options) -> dict
```

| argument | what it does |
| --- | --- |
| `provider` | a vendor name (`datalab`) or an OpenRouter model id (`openai/gpt-5.6-sol`) |
| `pdf` | path to the document |
| `schema` | the JSON Schema to extract against, as a dict |
| `timeout` | seconds this document may take, end to end — upload, submit and poll together |
| `overlay` | write the gold's conventions into the field descriptions. On for the benchmark, so a vendor is told a convention rather than expected to guess it |
| `**options` | anything the provider takes, e.g. `mode="accurate"`. `oeb providers <name>` lists them |

Credentials come from the environment and nowhere else. `DATALAB_API_KEY`, `REDUCTO_API_KEY`,
`OPENROUTER_API_KEY` and so on — an option you pass is recorded in the run, and a key should
never be.

```python
record = predict("datalab", "invoice.pdf", schema, mode="accurate")
record["result"]         # the extraction, shaped like your schema
record["raw"]            # the vendor's response, as received
record["cost"]           # what the vendor said this cost, plus wall_s and attempts
record["error"]          # None, or what went wrong
record["run_manifest"]   # what the vendor was actually sent
```

You can score it directly.

```python
from omni_extract_bench import score
score(record["result"], gold, schema)
```

From the cli:

```bash
oeb predict --provider datalab --doc invoice.pdf --schema schema.json \
    --options '{"mode": "accurate"}'
```

It prints the whole record, not just the answer — the cost and the response are how you check
a number later. Exit code is 1 when the record carries an error.

Three things are raised rather than returned, because none of them is a fact about the
document: `MissingCredential` (an unset API key), `MissingDependency` (an adapter whose SDK
isn't installed), and `AccountFailure` (the account can't pay). Everything else comes back in
`record["error"]` with the prediction beside it.


## benchmark

Needs the benchmark extra: `uv pip install 'omni-extract-bench[benchmark,harness]'`.

```python
from pathlib import Path
from omni_extract_bench.benchmark import run

run(providers, *, out=Path("runs"), data_root=Path("benchmark"), suites=None,
    limit=0, timeout=1800.0, predict_workers=None, score_workers=0,
    verdicts=False, rescore=False, score_only=False, options=None) -> dict
```

| argument | cli flag | what it does |
| --- | --- | --- |
| `providers` | `--providers` | vendors and/or model ids |
| `out` | `--out` | where the runs go. Default: `runs/` |
| `data_root` | `--data-root` | where the corpus is downloaded to. Default: `benchmark/` |
| `suites` | `--suites` | limit to these suites |
| `limit` | `--limit` | first N documents; for a smoke test |
| `timeout` | `--timeout` | seconds one document may take, the same for every vendor |
| `predict_workers` | `--predict-workers` | documents in flight at one vendor. `{"reducto": 25}` or one number for all. Default: the harness's per-vendor limit |
| `score_workers` | `--score-workers` | processes used to grade. Default: one per core, capped at 8 |
| `verdicts` | `--verdicts` | also write one verdict per address, per document |
| `rescore` | `--rescore` | grade every document again, ignoring the scores on disk |
| `score_only` | `--score-only` | score the predictions already on disk; call no vendor |
| `options` | `--options` | per-provider settings; a list runs that provider once per entry |

It returns the summary, keyed by run.

```python
summary = run(["datalab", "reducto"], limit=5)
summary["datalab-f46415c9"]["unified"]
```

Keyword arguments and no argparse, so this stays callable from a notebook, and it raises
rather than exits for the same reason.

The cli is the same thing:

```bash
oeb benchmark --out runs/ --limit 1 --providers datalab reducto
```

### A run is a provider plus its options

`--options` may give one provider a **list**, and each entry is its own run — its own
directory, its own summary row, its own line in the progress display. This is how you compare
a vendor against itself:

```bash
oeb benchmark \
  --providers datalab reducto \
  --options '{"datalab":  [{"mode": "balanced"}, {"mode": "accurate"}],
              "reducto": [{"agentic_table_mode": "max"},
                          {"agentic_table_mode": "default"}]}' \
  --out runs/
```

That's four runs. An option name the provider doesn't have is refused before anything is
downloaded, and so is an `--options` key that isn't in `--providers` — a typo there used to be
dropped silently and the run would report as stock.

### Resuming

It's resumable, and safe to run twice.

- A document is **predicted** again only when it has no record. The record is written last, so
  its presence means the prediction beside it is complete.
- A document is **graded** again only when it has no row in `scores.jsonl`.

So a Ctrl-C, a crash or a credit ceiling costs you the documents that were in flight, and
nothing else. Reinvoke the same command to carry on.

Nothing checks that the metric hasn't changed under a resume. If you've edited the scorer,
`--rescore` is how you say so:

```bash
oeb benchmark --providers datalab --score-only --rescore
```

`--rescore` regrades from the predictions on disk; `--score-only` makes sure no vendor is
called for whatever is still missing. Note that this regrades — it doesn't re-parse. The
predictions on disk are already parsed, so a fix to an *adapter's* parsing needs new
predictions, not a rescore.


## providers

```bash
oeb providers
```

lists the vendors, plus a few model ids to show the shape. Any OpenRouter `org/model` works,
so the list of models is an example and not the set.

```bash
oeb providers datalab
```

shows what `--options` takes for that provider and what each one is by default. The defaults
are the vendor's maximum tier — parity here is "as much as the vendor will give", not one
number for everyone.

In your code:

```python
from omni_extract_bench.harness import PROVIDERS, settings_for

PROVIDERS                       # the named vendors
settings_for("datalab")         # {'mode': 'balanced', 'base_url': ..., 'poll_interval': 5.0}
settings_for("datalab", {"mode": "accurate"})
```


## What a benchmark run writes

```
runs/
├── datalab-f46415c9/
│   ├── settings.json
│   ├── summary.json
│   ├── predictions/<doc_id>.json
│   ├── records/<doc_id>.json
│   ├── scores.jsonl
│   └── verdicts/<doc_id>.jsonl      # only with --verdicts
├── datalab-01a72762/
└── reducto-e54d3a1d/
```

One directory per run, named for the provider and an eight-character digest of everything it
was asked. The digest is what keeps two configurations of one vendor apart, so
`datalab-f46415c9` is the balanced run and `datalab-01a72762` is the accurate one and neither
can be handed the other's answers. It isn't meant to be read — `settings.json` is where you
read what a run was.

There's no summary across runs. `runs/*/summary.json` is one, aggregated however you like, and
a file sitting above them would only be one opinion about that, stale the moment another run
lands beside it.

### settings.json

What this run asked for, written **before** the first document — so an interrupted run still
says what it is.

```json
{
  "provider": "datalab",
  "settings": {
    "mode": "balanced",
    "base_url": "https://www.datalab.to",
    "poll_interval": 5.0
  },
  "timeout_s": 1800.0
}
```

`settings` is the whole resolved configuration, not just what you passed. The benchmark's
claim is that every vendor ran at its maximum, so a run has to state what that came to on the
day rather than leave it to the defaults in whatever version of the code you read next.

### summary.json

What the run came to, written as soon as **that** run is graded rather than when the whole
invocation finishes.

```json
{
  "run": "datalab-f46415c9",
  "provider": "datalab",
  "settings": {"mode": "balanced", "base_url": "https://www.datalab.to", "poll_interval": 5.0},
  "documents": 1,
  "scored": 1,
  "coverage": 1.0,
  "unified": 0.9555555555555556,
  "flat_mean": 0.9555555555555556,
  "mean_over_scored": 0.9555555555555556,
  "per_suite": {
    "extractbench": {"documents": 1, "accuracy": 0.9555555555555556}
  }
}
```

Three numbers because they have three different denominators, and seeing them differ is the
point:

| field | the mean over | a document that failed |
| --- | --- | --- |
| `unified` | the **suite** means — one vote per suite | counts as 0 |
| `flat_mean` | every **document** | counts as 0 |
| `mean_over_scored` | the documents that **scored** | excluded |

`unified` is the published number: equal weight per suite, so a large suite can't decide the
benchmark ([`METRIC_SPEC.md`](./METRIC_SPEC.md) §7). `flat_mean` sits next to it so you can see
the skew instead of taking it on trust, and `mean_over_scored` is what the vendor's number
would look like if you only counted the documents it managed — reported, never used, because
averaging only successes pays a vendor for failing on the hard ones.

They're all equal when there's one suite and nothing failed.

### predictions/&lt;doc_id&gt;.json

The bare extraction, shaped like the schema. This is what scoring reads, and it's the same
object `predict` returns as `record["result"]`.

### records/&lt;doc_id&gt;.json

Everything else about that document — kept separate because the two are read at different
times and are very different sizes. Scoring wants the answer; an audit wants all of it, and
only one of them is worth loading 620 of.

```json
{
  "result":  "...",
  "raw":     "...",
  "error":   null,
  "provider": "datalab",
  "cost": {
    "usd": 1.4,
    "source": "cost_breakdown.final_cost_cents",
    "tokens_in": null,
    "tokens_out": null,
    "credits": null,
    "wall_s": 172.4,
    "attempts": 1,
    "billed_out_of_band": false
  },
  "job_id": "32RBrjAbYxN0yDlOF__lDQ",
  "schema_sent": {"...": "the schema this vendor actually received"},
  "run_manifest": {
    "timeout_s": 1800,
    "settings": {"mode": "balanced", "base_url": "https://www.datalab.to", "poll_interval": 5.0},
    "model": null,
    "timed_out": false,
    "conventions_applied": false,
    "captured_at": "2026-09-18T15:06:13"
  }
}
```

A few of these are worth knowing about:

- `raw` is the vendor's response as received, before our parsing. It's kept because the
  expensive mistake is a parser bug: with only the parsed value stored, fixing the parser means
  paying for every call again.
- `cost.source` names the field the figure was read from, so you can check it against the
  vendor's own response rather than trust it. `credits` is there for vendors that bill in their
  own unit and never converted to dollars, because the rate is contract-specific.
- `cost.billed_out_of_band` is `true` when the vendor reported no per-document cost. That's
  different from free, and different from a figure we made up from a price list.
- `schema_sent` is the schema `predict` handed the adapter: the benchmark-only annotations
  stripped out and the conventions overlay applied. An adapter that needs a dialect reshape
  does that afterwards, on its way out — vendors don't all accept the same JSON Schema.
- `run_manifest.timed_out` distinguishes "the budget ran out while the vendor was still
  working" from "the vendor answered and we couldn't use it".

### scores.jsonl

One line per document, in document order whatever order the workers finished in, so a run stays
comparable with the one before it.

```json
{
  "doc_id": "long__dd1155_schedule_continuation_0011",
  "suite": "extractbench",
  "provider": "datalab",
  "status": "scored",
  "accuracy": 0.9555555555555556,
  "precision": 0.9555555555555556,
  "recall": 1.0,
  "f1": 0.9772727272727273,
  "total": 90,
  "matched": 86,
  "misread": 0,
  "unfound": 0,
  "fabricated": 4,
  "invented_item": 0,
  "invented_field": 0,
  "asserted": 90,
  "addresses_found": 0.9555555555555556,
  "addresses_read_right": 1.0,
  "gt_rows": 20,
  "pred_rows": 20,
  "matched_rows": 20,
  "matching_exact": true,
  "approximated": [],
  "skipped_open_maps": []
}
```

Every document gets a line, including the ones with no usable prediction — coverage is only
visible if a failure occupies a row. Those rows carry no metrics at all, just what happened:

```json
{"doc_id": "long__dd1155_schedule_continuation_0011", "suite": "extractbench",
 "provider": "datalab", "status": "error", "error": "DATALAB_API_KEY must be set"}
```

Not zeros, which would claim the vendor tried and missed every field. `summarise` counts them
as zero when it averages, and `coverage` is what says how often the vendor answered at all.

### verdicts/&lt;doc_id&gt;.jsonl

Only with `--verdicts`. One line per address: what happened there, and what each side was
compared as. This is the same shape `score(..., verdicts=True)` returns, and it's how you get
from "0.95" to the four fields that went wrong.

One line looks like this:

```json
{"address": [["k", "line_items"], ["i", 0], ["k", "unit_price"]], "gold_raw": 12.5, "pred_raw": 12.05, "gold_canon": "12.5", "pred_canon": "12.05", "verdict": "misread"}
```

`address` is the path to the scalar, one step at a time: `["k", name]` walks into a key,
`["i", n]` into an array element. So that one reads `line_items[0].unit_price`.

`gold_raw` and `pred_raw` are what each document said; `gold_canon` and `pred_canon` are what
they were actually compared as, after normalising. Both are there because the comparison is
where a score is won or lost, and you want to see it rather than trust it — `"03/31/2024"` and
`"2024-03-31"` both canonicalise to `#d2024-03-31`, which is a `matched`, and if you disagree
with that you can see exactly what happened:

```json
{"address": [["k", "invoice_date"]], "gold_raw": "2024-03-31", "pred_raw": "03/31/2024", "gold_canon": "#d2024-03-31", "pred_canon": "#d2024-03-31", "verdict": "matched"}
```

The rest of the vocabulary, one real line each (these are the [README's example](../README.md#score)):

```json
{"address": [["k", "line_items"], ["i", 0], ["k", "description"]], "gold_raw": "Hex bolt, M8", "pred_raw": null, "gold_canon": "hexboltm8", "pred_canon": null, "verdict": "unfound"}
{"address": [["k", "purchase_order"]], "gold_raw": null, "pred_raw": "PO-88231", "gold_canon": null, "pred_canon": "po88231", "verdict": "fabricated"}
{"address": [["k", "currency"]], "gold_raw": null, "pred_raw": "USD", "gold_canon": null, "pred_canon": "usd", "verdict": "invented_field"}
{"address": [["k", "line_items"], ["i", "p2"], ["k", "description"]], "gold_raw": null, "pred_raw": "Freight surcharge", "gold_canon": null, "pred_canon": "freightsurcharge", "verdict": "invented_item"}
```

| verdict | what it means |
| --- | --- |
| `matched` | both documents addressed it and agree, after normalising |
| `misread` | both addressed it and they disagree |
| `unfound` | the gold has it, the prediction doesn't |
| `fabricated` | the schema offered the slot, the document is silent, the model asserted a value anyway |
| `invented_field` | a name the schema never declared |
| `invented_item` | a value under an array row that paired with nothing |

Two of those are worth a second look. `fabricated` keys off the **schema**, not gold's `null`s
— the schema offered `purchase_order`, the document doesn't cite one, and the model produced
`PO-88231` from nowhere.

And in the `invented_item` line the index is `"p2"`, not a number. A predicted row that paired
with a gold row takes its partner's index; one that paired with nothing gets a made-up label
instead. If it borrowed a spare gold index it would read as a real row that happened to be
wrong, and if it were dropped, inventing rows would be free — the label is how it gets counted,
and counted against.

[`METRIC_SPEC.md`](./METRIC_SPEC.md) §4 has the full vocabulary and how the counts add up.
