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
```json
{
  "accuracy": 0.5294117647058824,
  "precision": 0.5625,
  "recall": 0.8181818181818182,
  "f1": 0.6666666666666666,
  ...truncated for display
```

Every argument is a flag of the same name, underscores as dashes, so `order_matters` is
`--order-matters`. It goes to stdout, so `oeb score ... | jq .accuracy` is one pipe.

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
```json
{
  "result": {"invoice_id": "INV-4417", "total_due": 1240.0, ...},
  "error": null,
  "provider": "datalab",
  "cost": {"usd": 1.4, "wall_s": 172.4, "attempts": 1, ...},
  ...truncated for display
```

Same flag names, except `pdf` is `--doc`. It prints the whole record, not just the answer. The
cost and the response are how you check a number later.

Three things raise instead of returning, because none of them is about the document:

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
from omni_extract_bench.benchmark import BenchmarkRun

BenchmarkRun(
  providers,              # vendors and/or model ids
  out="runs",             # where the runs go
  data_root=None,         # where the corpus goes, and what a relative manifest path is from
  manifest=None,          # a parquet manifest of your own instead of ours
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

Everything but `providers` is keyword-only. `go()` runs it and returns the summary, keyed by
run.

```python
summary = BenchmarkRun(["datalab", "reducto"], limit=5).go()
summary["datalab-f46415c9"]["accuracy"]
```

`run(providers, **kwargs)` is the same thing in one call, if you don't want the object.

Keyword arguments and no argparse, so this stays callable from a notebook, and it raises rather
than exits for the same reason. Nothing here reads stdin either.

### Three levels, three classes

```python
BenchmarkRun   # the invocation      every Run, grouped by adapter, predicted then graded
ProviderRun    # one adapter         every Run on it, through one pool sized to that service
Run            # one configuration   a provider plus its options
```

A **Run** is a provider plus its options -- `datalab` at `mode=accurate` -- and it is what
everything is keyed by: it names a directory, a summary row and a progress line.

A **ProviderRun** is every Run on one adapter, sharing one pool. The cap belongs to the vendor,
so two datalab tiers split its ten rather than taking ten each. And every `org/model` id is one
`llm_single_shot`, one OpenRouter endpoint and one key, so three models named separately still
share one budget.

### See the plan before you spend

Construct one and ask. No download, no vendor call:

```python
print(BenchmarkRun(["datalab"], options={"datalab": [{"mode": "balanced"},
                                                     {"mode": "accurate"}]}).describe())
```
```
benchmark: 2 runs over 1 adapter, 1800s per document, our corpus -> runs
╭─────────┬─────────┬──────────────────┬─────────────────────────────────┬─────────┬───────╮
│ adapter │ at once │ run              │ settings                        │ predict │ grade │
├─────────┼─────────┼──────────────────┼─────────────────────────────────┼─────────┼───────┤
│ datalab │      10 │ datalab-f46415c9 │ base_url=https://www.datalab.to │         │       │
│         │         │                  │ ─────────────────────────────── │         │       │
│         │         │                  │ mode=balanced                   │         │       │
│         │         │                  │ ─────────────────────────────── │         │       │
│         │         │                  │ poll_interval=5.0               │         │       │
│         │         │ datalab-01a72762 │ ...truncated for display         │         │       │
╰─────────┴─────────┴──────────────────┴─────────────────────────────────┴─────────┴───────╯
```

`predict` and `grade` are empty because there is no corpus yet. Give it one and it says what a
resume would cost, per run:

```python
b = BenchmarkRun(["datalab", "reducto"], limit=1)
b.describe(b.corpus())
```
```
benchmark: 3 runs over 2 adapters, 1800s per document, our corpus, 1 document selected -> runs
╭─────────┬─────────┬──────────────────┬─────────────────────────────────┬─────────┬───────╮
│ adapter │ at once │ run              │ settings                        │ predict │ grade │
├─────────┼─────────┼──────────────────┼─────────────────────────────────┼─────────┼───────┤
│ datalab │      10 │ datalab-f46415c9 │ base_url=https://www.datalab.to │       1 │     1 │
│         │         │                  │ ─────────────────────────────── │         │       │
│         │         │                  │ mode=balanced                   │         │       │
│         │         │                  │ ─────────────────────────────── │         │       │
│         │         │                  │ poll_interval=5.0               │         │       │
│         │         │ datalab-01a72762 │ base_url=https://www.datalab.to │       1 │     1 │
│         │         │                  │ ...                             │         │       │
├─────────┼─────────┼──────────────────┼─────────────────────────────────┼─────────┼───────┤
│ reducto │       3 │ reducto-e54d3a1d │ agentic_table_mode=max          │       1 │     1 │
│         │         │                  │ ...truncated for display        │         │       │
╰─────────┴─────────┴──────────────────┴─────────────────────────────────┴─────────┴───────╯
```

They're counted separately because they're independent: every prediction can be on disk with
every grade still owed, which is what `--rescore` means.

### From the cli

The cli is the same thing, one flag per argument. It prints that plan and waits.

**!!NOTE!!**: this will cost money and you will need your API keys set.

```bash
oeb benchmark --out runs/ --limit 1 --providers datalab reducto
```
```
benchmark: 2 runs over 2 adapters, 1800s per document, our corpus, 1 document selected -> runs/
...the table above
  proceed? [y/N]
```

`-y` skips the question. A run with no terminal to ask -- a pipe, a cron job -- never waits
anyway, so a script is not the thing that hangs.

Then it runs:

```
 run                                    done   ok   err   in flight       cost     avg
 ────────────────────────────────────────────────────────────────────────────────────────────────────
 datalab-f46415c9   ━━━━━━━━━━━━━━━━   20/20   20     0                  $6.20   1m58s   done in 3.0s
 datalab-01a72762   ━━━━━━━━━━━━━━━━   20/20   20     0                  $6.20   2m02s   done in 3.0s
 reducto-e54d3a1d   ━━━━━━━━━━━━━━━━   20/20   19     1               5,820 cr   2m19s   done in 3.0s
 mistral-44136fa3   ━━━━━━━━━━━━━━━━   20/20   20     0                          2m13s   done in 3.0s
80/80 documents  1 failed  $12.40 + 5,820 cr  3.0s elapsed
```

Piped to a file or running in CI there is nothing to redraw, so the same counters come out as a
log line per provider instead. You don't choose: it reads the stream and decides.

### Settings per provider

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

That's 4 runs.

### A manifest of your own

Instead of our corpus, which then isn't downloaded at all:

```bash
oeb benchmark --providers datalab --manifest my/corpus/manifest.parquet
```

Same parquet either way. Columns: `doc_id`, `suite`, `doc_path`, `gt_path`, and a `schema`
column holding JSON.

A `doc_id` is a filename -- it names `predictions/<doc_id>.json`, its record and its row in
`scores.jsonl` -- so one with a `/` in it is refused, naming the row.

An absolute `doc_path` is used as it is. A relative one resolves against `--data-root`, or the
manifest's own directory when you don't give one, so a manifest sitting next to its PDFs needs
no flags.

**!!NOTE!!**: an absolute path won't survive a container or somebody else's machine.

### Resuming

It's **resumable**, and safe to run twice.

- A document is predicted again only when it has no record. The record is written last, so
  its presence means the prediction beside it is complete.
- A document is graded again only when it has no row in `scores.jsonl`.

So a Ctrl-C, a crash or a credit ceiling costs you the documents in flight, and nothing else.
Reinvoke the same command to carry on.


## providers

See providers:

```bash
oeb providers
```
```
azure-cu
datalab
extend
llamaextract
mistral
reducto
openai/gpt-5.6-sol
anthropic/claude-opus-5
google/gemini-3.7-flash
```

Any OpenRouter `org/model` works, so the model ids are examples and not the set.

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
what keeps two configurations of one vendor apart. `datalab-f46415c9` is the balanced run,
`datalab-01a72762` is the accurate one, and neither can be handed the other's answers.

The digest isn't meant to be read. `settings.json` is where you read what a run was.

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
  "documents": 6,
  "scored": 6,
  "coverage": 1.0,
  "accuracy": 0.8076406248606786,
  "precision": 0.8101146573593331,
  "recall": 0.9539514921288181,
  "misread_rate": 0.03748563782596681,
  "unfound_rate": 0.002711187188462559,
  "fabricated_rate": 0.14784472888803452,
  "invented_item_rate": 0.0043178212368574645,
  "invented_field_rate": 0.0,
  "per_suite": {
    "extractbench": {"documents": 6, "scored": 6, "coverage": 1.0, "accuracy": 0.8076406248606786,
                     "...": "the same block, per suite"}
  }
}
```

The run and each suite are the same shape, so whatever you read at the top you can read per
suite as well.


### predictions/&lt;doc_id&gt;.json

The bare extraction, shaped like the schema. This is what scoring reads, and it's the same
thing `predict` returns as `record["result"]`.

### records/&lt;doc_id&gt;.json

Everything else about that document. Two files because scoring wants the answer and an audit
wants all of it.

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


### scores.jsonl

One line per document, in document order whatever order the workers finished in.

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


### verdicts/&lt;doc_id&gt;.jsonl

Only written when `--verdicts` passed. One line per address: what happened there, and what
each side was compared as. This is the same shape `score(..., verdicts=True)` returns, and
it's a low-level address breakdown of how scoring happened.

One line looks like this:

```json
{"address": [["k", "line_items"], ["i", 0], ["k", "unit_price"]], "gold_raw": 12.5, "pred_raw": 12.05, "gold_canon": "12.5", "pred_canon": "12.05", "verdict": "misread"}
```

`address` is the path to the scalar, one step at a time: `["k", name]` walks into a key,
`["i", n]` into an array element. So that one reads `line_items[0].unit_price`.

`gold_raw` and `pred_raw` are what each document said; `gold_canon` and `pred_canon` are what
they were actually compared as, after normalising.

There are six verdicts, and every address gets exactly one:

```
matched          both documents addressed it and agree
misread          both addressed it and the values disagree
unfound          the gold has it, the prediction doesn't
fabricated       the schema offered the slot, the document is silent
invented_field   a name the schema never declared
invented_item    a value under an array row that paired with nothing
```

`fabricated` keys off the **schema**, not gold's `null`s. The schema offered `purchase_order`,
the document doesn't cite one, and the model produced `PO-88231` from nowhere.

And in the `invented_item` line the index is `"p2"`, not a number. A predicted row that paired
with a gold row takes gold's index; one that paired with nothing gets a made-up label
instead.

[`METRIC_SPEC.md`](./METRIC_SPEC.md) §4 has the full vocabulary and how the counts add up.
