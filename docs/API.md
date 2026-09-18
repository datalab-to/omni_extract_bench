# API

This is the public facing API for the toolkit. There are three main entry points:

- `score`: score a document's prediction against the ground truth.
- `predict`: predict extractions for a document with one of our providers.
- `benchmark`: run our benchmark.

The first two are the primitives the last one is built around. Each works in your code and from
the cli, and does the same thing either way -- the cli is a thin wrapper.

1. **[score](#score)** — grade one prediction against one ground truth.
2. **[predict](#predict)** — one document, one vendor.
3. **[benchmark](#benchmark)** — the whole corpus, every vendor, resumable.
4. **[providers](#providers)** — what you can run, and what each one takes.
5. **[What benchmark writes](#what-benchmark-writes)** — the files, and what's in them.


## score

Just the base install. No network, no API keys, no vendor packages.

Use in your own code.

```python
from omni_extract_bench import score

score(
  pred,               # the prediction, a dict
  gt,                 # the ground truth, a dict
  schema,             # the JSON Schema the prediction was generated against
  order_matters=(),   # arrays where position is part of the answer, e.g. ["steps"]
  verdicts=False,     # also return one verdict per address
)
```

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

Every argument is a flag of the same name, underscores as dashes, so `order_matters` is
`--order-matters`. Results go to stdout as JSON, so `oeb score ... | jq .accuracy` is one pipe.

The README has a [worked example](../README.md#score) and
[`docs/METRIC_SPEC.md`](./METRIC_SPEC.md) is the full specification.


## predict

Needs the harness extra.

```bash
uv pip install 'omni-extract-bench[harness]'
```

Use in your own code.

```python
from omni_extract_bench.harness import predict

predict(
  provider,           # a vendor name, or an OpenRouter model id
  pdf,                # path to the document
  schema,             # the JSON Schema to extract against
  timeout=1800.0,     # seconds this document may take, end to end
  overlay=True,       # state the gold's conventions in the field descriptions
  **options,          # anything the provider takes, e.g. mode="accurate"
)
```

Keys come from the environment and nowhere else -- `DATALAB_API_KEY`, `REDUCTO_API_KEY`,
`OPENROUTER_API_KEY` and so on. Options are recorded in the run, and a key should never be.

You get back the answer and the evidence for it.

```python
record = predict("datalab", "invoice.pdf", schema, mode="accurate")
record["result"]         # the extraction, shaped like your schema
record["raw"]            # the vendor's response, as received
record["cost"]           # what the vendor said this cost, plus wall_s and attempts
record["error"]          # None, or what went wrong
record["run_manifest"]   # what the vendor was actually sent
```

Score directly from the prediction.

```python
from omni_extract_bench import score
score(record["result"], gold, schema)
```

You can also use the cli:

```bash
oeb predict --provider datalab --doc invoice.pdf --schema schema.json \
    --options '{"mode": "accurate"}'
```

Same flag names, except `pdf` is `--doc`. It prints the whole record, not just the answer --
the cost and the response are how you check a number later.

Three things raise instead of returning, because none of them is a fact about the document:

```python
MissingCredential    # an unset API key
MissingDependency    # an adapter whose SDK isn't installed
AccountFailure       # the account can't pay
```

Everything else comes back in `record["error"]`, with the prediction beside it.


## benchmark

Needs the benchmark extra.

```bash
uv pip install 'omni-extract-bench[benchmark,harness]'
```

Use in your own code.

```python
from omni_extract_bench.benchmark import run

run(
  providers,              # vendors and/or model ids
  out="runs",             # where the runs go
  data_root="benchmark",  # where the corpus is downloaded to
  suites=None,            # limit to these suites
  limit=0,                # first N documents; for a smoke test
  timeout=1800.0,         # seconds one document may take, the same for every vendor
  predict_workers=None,   # documents in flight at one vendor: {"reducto": 25}, or one number
  score_workers=0,        # processes used to grade. Default: one per core, capped at 8
  verdicts=False,         # also write one verdict per address, per document
  rescore=False,          # grade every document again, ignoring the scores on disk
  score_only=False,       # score the predictions already on disk; call no vendor
  options=None,           # per-provider settings: {"datalab": {"mode": "accurate"}}
)
```

`out` and `data_root` are paths, and everything but `providers` is keyword-only. It returns the
summary, keyed by run.

```python
summary = run(["datalab", "reducto"], limit=5)
summary["datalab-f46415c9"]["unified"]
```

Keyword arguments and no argparse, so this stays callable from a notebook, and it raises rather
than exits for the same reason.

The cli is the same thing, one flag per argument:

```bash
oeb benchmark --out runs/ --limit 1 --providers datalab reducto
```

### A run is a provider plus its options

`--options` can give one provider a **list**, and each entry is its own run -- its own
directory, its own summary, its own line in the progress display. This is how you compare a
vendor against itself. For example:

```bash
oeb benchmark \
  --providers datalab reducto \
  --options '{"datalab":  [{"mode": "balanced"}, {"mode": "accurate"}],
              "reducto": [{"agentic_table_mode": "max"},
                          {"agentic_table_mode": "default"}]}' \
  --out runs/
```

That's 4 runs. An option name the provider doesn't have is refused before anything downloads,
and so is an `--options` key that isn't in `--providers`:

```
--options names 'datalb', which is not in --providers (datalab).
Its options would be silently ignored.
```

### Resuming

It's **resumable**, and safe to run twice.

- A document is predicted again only when it has no record. The record is written last, so
  its presence means the prediction beside it is complete.
- A document is graded again only when it has no row in `scores.jsonl`.

So a Ctrl-C, a crash or a credit ceiling costs you the documents in flight, and nothing else.
Reinvoke the same command to carry on.

Nothing notices if you edit the metric, though. `--rescore` is how you say you did:

```bash
oeb benchmark --providers datalab --score-only --rescore
```

`--rescore` grades everything again from the predictions on disk, and `--score-only` makes sure
no vendor is called for whatever is still missing.

**!!NOTE!!**: that regrades, it doesn't re-parse. The predictions on disk are already parsed,
so fixing an *adapter's* parsing needs new predictions, not a rescore.


## providers

See providers:

```bash
oeb providers
```

Any OpenRouter `org/model` works, so the model ids it lists are examples and not the set.

See what settings each takes and its default values. For example, datalab:

```bash
oeb providers datalab
```
```
datalab

  mode           balanced
  base_url       https://www.datalab.to
  poll_interval  5.0

  oeb benchmark --providers datalab --options '{"datalab": {"mode": ...}}'
```

Those defaults are the vendor's maximum tier. Parity here is "as much as the vendor will give",
not one number for everyone.

In your code:

```python
from omni_extract_bench.harness import PROVIDERS, settings_for

PROVIDERS
# ['azure-cu', 'datalab', 'extend', 'llamaextract', 'mistral', 'reducto']

settings_for("datalab")
# {'mode': 'balanced', 'base_url': 'https://www.datalab.to', 'poll_interval': 5.0}

settings_for("datalab", {"mode": "accurate"})
# {'mode': 'accurate', 'base_url': 'https://www.datalab.to', 'poll_interval': 5.0}
```


## What benchmark writes

One directory per run.

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

Each is named for the provider and an eight-character digest of everything it was asked. That's
what keeps two configurations of one vendor apart -- `datalab-f46415c9` is the balanced run and
`datalab-01a72762` is the accurate one, and neither can be handed the other's answers. The
digest isn't meant to be read. `settings.json` is where you read what a run was.

There's no summary across runs. `runs/*/summary.json` is one, aggregated however you like.

### settings.json

What the run asked for, written **before** the first document -- so an interrupted run still
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

`settings` is the whole resolved configuration, not just what you passed.

### summary.json

What it came to, written as soon as that run is graded rather than when the whole invocation
finishes.

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

Three numbers because they have three different denominators:

```python
summary["unified"]           # mean of the SUITE means -- one vote per suite, failures count 0
summary["flat_mean"]         # mean over every DOCUMENT -- failures count 0
summary["mean_over_scored"]  # mean over the documents that SCORED -- failures excluded
```

`unified` is the published number: equal weight per suite, so a large suite can't decide the
benchmark ([`METRIC_SPEC.md`](./METRIC_SPEC.md) §7). `flat_mean` sits next to it so you can see
the skew instead of taking it on trust, and `mean_over_scored` is what a vendor's number would
look like if you only counted the documents it managed. All three are equal when there's one
suite and nothing failed.

### predictions/&lt;doc_id&gt;.json

The bare extraction, shaped like the schema. This is what scoring reads, and it's the same
thing `predict` returns as `record["result"]`.

### records/&lt;doc_id&gt;.json

Everything else about that document. Two files because they're read at different times and are
very different sizes -- scoring wants the answer, an audit wants all of it, and only one of
them is worth loading 620 of.

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
  "schema_sent": {"...": "the schema this vendor received"},
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

A few worth knowing about:

```python
record["raw"]                         # the response before our parsing, so fixing a parser
                                      # means re-reading a file, not buying 620 calls again
record["cost"]["source"]              # the field the figure was read from, so you can check it
record["cost"]["credits"]             # vendors that bill in their own unit. Never converted:
                                      # the rate is contract-specific
record["cost"]["billed_out_of_band"]  # true when the vendor reported no per-document cost.
                                      # Different from free, and from a made-up figure
record["schema_sent"]                 # what predict handed the adapter. An adapter that needs
                                      # a dialect reshape does that afterwards
record["run_manifest"]["timed_out"]   # the budget ran out while the vendor was still working,
                                      # as opposed to the vendor answering unusably
```

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

Every document gets a line, including the ones with no usable prediction -- coverage is only
visible if a failure occupies a row. Those carry no metrics at all, just what happened:

```json
{"doc_id": "long__dd1155_schedule_continuation_0011", "suite": "extractbench",
 "provider": "datalab", "status": "error", "error": "DATALAB_API_KEY must be set"}
```

Not zeros, which would claim the vendor tried and missed every field. They count as zero when
`unified` and `flat_mean` average, and `coverage` is what says how often the vendor answered.

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
they were actually compared as, after normalising.

| verdict | what it means |
| --- | --- |
| `matched` | both documents addressed it and agree, after normalising |
| `misread` | both addressed it and they disagree |
| `unfound` | the gold has it, the prediction doesn't |
| `fabricated` | the schema offered the slot, the document is silent, the model asserted a value anyway |
| `invented_field` | a name the schema never declared |
| `invented_item` | a value under an array row that paired with nothing |

Two of those are worth a second look. `fabricated` keys off the **schema**, not gold's `null`s
-- the schema offered `purchase_order`, the document doesn't cite one, and the model produced
`PO-88231` from nowhere.

And in the `invented_item` line the index is `"p2"`, not a number. A predicted row that paired
with a gold row takes gold's index; one that paired with nothing gets a made-up label
instead.

[`METRIC_SPEC.md`](./METRIC_SPEC.md) §4 has the full vocabulary and how the counts add up.
