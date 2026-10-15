# API

This is the public facing API for the toolkit. There are three main entry points:

- `score`: score a document's prediction against the ground truth.
- `predict`: predict extractions for a document with one of our providers.
- `benchmark`: run our benchmark.

The first two are the primitives the last one is built around. Each works in your code and
from the cli. The cli is a thin wrapper, except that `benchmark` shows you the plan and waits
-- nothing in the library ever reads stdin.

1. **[score](#score)** — grade one prediction against one ground truth.
2. **[predict](#predict)** — one document, one vendor.
3. **[benchmark](#benchmark)** — the whole corpus, every vendor, resumable.
4. **[providers](#providers)** — what you can run, and what each one takes.
5. **[What a benchmark writes](#what-a-benchmark-writes)** — the files, and what's in them.


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
record["result"]         # the extraction, shaped like your schema; None when the call failed
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

Three types of errors raise:

```python
MissingCredential    # an unset API key
MissingDependency    # an adapter whose SDK isn't installed
AccountFailure       # the account can't pay
```

Everything else comes back in `record["error"]` (type, HTTP status, whether it's transient, and
the message), with `record["result"]` set to `None`.


## benchmark

Needs the benchmark extra.

```bash
uv pip install 'omni-extract-bench[benchmark]'
```

**!!NOTE!!**: running this will cost money and you will need your API keys set.

Use in your own code.

```python
from omni_extract_bench import benchmark

p = benchmark.plan(
  providers,              # e.g. ["datalab"]
  out="runs",             # where runs go
  dataset=DEFAULT_DATASET,  # a folder, or hf://datasets/<org>/<name>[@rev]. Default: ours
  manifest="manifest.parquet",  # which documents: a parquet inside the dataset
  suites=None,            # limit to these suites
  limit=0,                # first N documents
  timeout=1800.0,
  predict_workers=None,   # documents in flight per vendor
  score_workers=0,        # processes used to score
  verdicts=True,          # write verdicts; False to skip them
  rescore=False,          # score every document again
  score_only=False,       # score the predictions; call no vendor
  options=None,           # per-provider settings: {"datalab": {"mode": "accurate"}}
)
print(benchmark.describe(p))   # what it would do, with no vendor call
summary = benchmark.execute(p)
summary["datalab-f46415c9"]["headline"] # the mean accuracy, e.g. 0.9145
```

`benchmark.run(providers, **kwargs)` does the three in one call.

### The functions, by what they work on

| | reading data (`data`) | predicting (`harness`, `benchmark`) | scoring (`scorer`, `benchmark`) |
| --- | --- | --- | --- |
| one document | `doc.doc_path`, `doc.gt_path` | `harness.predict` | `scorer.gold(doc)`, `score` |
| a dataset | `download_dataset`, `read_manifest`, `select`, `dataset_id` | | `scorer.summarize(lines)` |
| one run | | `prepare`, `needs`, `record`, `recorded` | `grade(run, docs)` |
| a plan | | `plan`, `predict_all` | `grade_all` |
| both | | `execute`; `describe` | |

The same names, in the same modules, as omni-parse-bench's (whose rows are pages, not documents).
`data.read_manifest(dataset, manifest)` gives the dataset's folder and every document, each with
its file and gold as full local paths; `data.select(docs, suites, limit)` picks the ones to run.

### Two levels of data

```python
Plan   # the invocation: every Run, and the documents they cover
Run    # one configuration: a provider plus its options, and the directory it writes to
```

- A `Run` is a provider plus its options. For example, `datalab` at `mode=accurate`. It has its own self-contained directory of results.
- Concurrency is per adapter, not per run: every run of one adapter (two datalab modes, every OpenRouter model) shares one pool. The cap belongs to the provider.
- `execute(plan)` is `predict_all(plan)`, then `grade_all(plan)`.



### From the cli


```bash
oeb benchmark --out runs/ --limit 1 --providers datalab reducto
```
```
benchmark
out         runs
runs        2 over 2 adapters
dataset     hf://datasets/datalab-to/omni_extract_bench
manifest    manifest.parquet
documents   1 document selected
timeout     1800s per document
score only  false
rescoring   false

╭─────────┬─────────┬──────────────────┬─────────────────────────────────┬─────────┬───────╮
│ adapter │ at once │ run              │ settings                        │ predict │ grade │
├─────────┼─────────┼──────────────────┼─────────────────────────────────┼─────────┼───────┤
│ datalab │      10 │ datalab-f46415c9 │ base_url=https://www.datalab.to │       1 │     1 │
│         │         │                  │ ─────────────────────────────── │         │       │
│         │         │                  │ mode=balanced                   │         │       │
│         │         │                  │ ─────────────────────────────── │         │       │
│         │         │                  │ poll_interval=5.0               │         │       │
├─────────┼─────────┼──────────────────┼─────────────────────────────────┼─────────┼───────┤
│ reducto │       3 │ reducto-e54d3a1d │ agentic_table_mode=max          │       1 │     1 │
│         │         │                  │ ─────────────────────────────── │         │       │
│         │         │                  │ deep_extract_model=v2           │         │       │
│         │         │                  │ ─────────────────────────────── │         │       │
│         │         │                  │ poll_interval=5                 │         │       │
│         │         │                  │ ─────────────────────────────── │         │       │
│         │         │                  │ system_prompt=''                │         │       │
╰─────────┴─────────┴──────────────────┴─────────────────────────────────┴─────────┴───────╯
  proceed? [y/N]
```

You can pass `-y` to skip the interactive confirmation. The execution looks like:

```
 run                                    done   ok   err   in flight       cost     avg    dur
 ────────────────────────────────────────────────────────────────────────────────────────────
 datalab-f46415c9   ━━━━━━━━━━━━━━━━   20/20   20     0                  $6.20   1m58s   3.0s
 datalab-01a72762   ━━━━━━━━━━━━━━━━   20/20   20     0                  $6.20   2m02s   3.0s
 reducto-e54d3a1d   ━━━━━━━━━━━━━━━━   20/20   19     1               5,820 cr   2m19s   3.0s
 mistral-44136fa3   ━━━━━━━━━━━━━━━━   20/20   20     0                          2m13s   3.0s
80/80 documents  1 failed  $12.40 + 5,820 cr  3.0s elapsed
```


### Settings per provider

`--options` can give one provider a **list**, and each entry is its own `Run`. For example:

```bash
oeb benchmark \
  --providers datalab reducto \
  --options '{"datalab":  [{"mode": "balanced"}, {"mode": "accurate"}],
              "reducto": [{"agentic_table_mode": "max"},
                          {"agentic_table_mode": "default"}]}' \
  --out runs/
```

That's 4 runs.

### Choosing the documents: `--dataset` and `--manifest`

A dataset is a folder holding one or more manifests and the files they point at. A manifest is
a parquet file with one row per document, with columns:

- `doc_id`
- `suite`
- `doc_path`
- `gt_path`
- `schema`

`doc_path` and `gt_path` are relative to the dataset's folder, wherever the manifest is in it.
It is the same shape as [ours](https://huggingface.co/datasets/datalab-to/omni_extract_bench).

With no `--dataset`, a run uses ours, `hf://datasets/datalab-to/omni_extract_bench`, and with no
`--manifest`, the dataset's `manifest.parquet`. Otherwise you can pass:

*A subset of ours.* Our dataset holds more than one manifest; this one is 40 documents. For example,

```bash
oeb benchmark --providers datalab \
  --manifest manifests/sample40-predictive-across-vendors-2026-09-22.parquet
```

*Any HuggingFace dataset*, pinned to a revision with `@`. It is downloaded whole, at that
revision, into the HuggingFace cache (ours is 0.9 GB), so its documents and gold are always one
commit's, and a second run downloads nothing. For example,

```bash
oeb benchmark --providers datalab --dataset hf://datasets/someone/their-corpus@v2
```

*A folder on disk.* Nothing is downloaded. For example:

```bash
oeb benchmark --providers datalab --dataset my/corpus
```

A run folder holds one dataset: pointing it at another is refused before anything is read.
Two manifests of the same dataset may share a run folder, since they are the same documents
and gold.

### Resuming

It's **resumable**, and safe to run twice.

- A document is predicted again only when it has no record. The record is written last, so
  its presence means the prediction beside it is complete.
- A document is graded again only when its line in `scores.jsonl` is missing or stale. Each line
  says which prediction (`prediction`, a digest of its file), which gold (`gold`, a digest of the
  document's id, suite, schema and gold file) and which scoring code (`scorer`, the release and a
  digest of the scoring files) it graded. A line whose prediction, gold or scorer has changed is
  graded again, which calls no vendor, so a finished grading's lines are one scorer's against one
  gold. `--rescore` grades the selected documents again anyway, and drops every other line.

So a Ctrl-C, a crash or a credit ceiling costs you the documents in flight, and nothing else.
Reinvoke the same command to carry on.


### A loop of your own

`oeb benchmark` is one loop over documents. To predict elsewhere (on another machine, on a
scheduler, through a queue) and still get the same run folder, keep the run with the functions
`oeb benchmark` uses itself. A `benchmark.Run` is an address, as a `Path` is: what the run holds
is whatever its folder holds when it is read.

```python
from omni_extract_bench import benchmark
from omni_extract_bench.harness import predict

p = benchmark.plan(["datalab"], out="runs")
(run,) = p.runs
benchmark.prepare(run, p.dataset, p.timeout, manifest=p.manifest)
for doc in p.docs:
    if benchmark.needs(run, doc):
        result = predict(run.provider, doc.doc_path, doc.schema, timeout=p.timeout, **run.options)  # wherever
        benchmark.record(run, doc.doc_id, result)
benchmark.grade(run, list(p.docs))
```

- `prepare(run, dataset, timeout, commit=None, manifest="manifest.parquet")` makes the folder
  and writes `settings.json`;
- `needs(run, doc)` is True when the run has no readable record of the document. A loop that
  predicts everything every time can skip it; it is what makes a loop resumable;
- `record(run, doc_id, result)` writes a result from `predict`: its prediction, if the call
  succeeded, then the record;
- `recorded(run, doc_id)` reads the document's record, or None; and
- `grade(run, docs)` grades the documents whose line is missing or stale, writes `scores.jsonl`
  and `summary.json`, and returns the summary of `docs`' lines (`read_scores(run.out)` gives the
  lines). A document with no prediction is graded as failed.

The folder is then exactly one `oeb benchmark` writes, so either can resume or grade it.

## providers

See providers:

```bash
oeb providers
```
```
provider
───────────────────────
azure-cu
datalab
extend
llamaextract
mistral
reducto

openai/gpt-5.6-sol
anthropic/claude-opus-5
google/gemini-3.7-flash

the three model ids are examples: any OpenRouter org/model id works.
```

See what settings each takes and its default values. For example, datalab:

```bash
oeb providers datalab
```
```
datalab
option          default
──────────────────────────────────────
mode            balanced
base_url        https://www.datalab.to
poll_interval   5.0

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

### Your own adapter

An adapter is a module with three names:

- `Config`: a frozen dataclass. Its fields are the options `--options` sets, with their defaults;
- `prepare_schema(schema)`: the JSON Schema translated to whatever your system takes; and
- `extract(pdf, schema, *, timeout, config)`: runs one document and returns an `Extraction` result.

`omni_extract_bench/harness/contract.py` states the full contract.

To run it from the cli, declare it in your package's `pyproject.toml` under the
`omni_extract_bench.adapters` entry-point group:

```toml
[project.entry-points."omni_extract_bench.adapters"]
my-agent = "my_package.oeb_adapter"
```

```bash
oeb providers
oeb providers my-agent
oeb benchmark --providers my-agent --out runs/ --limit 1
```

Or in your code:

```python
from omni_extract_bench.harness import add_adapter, predict
import my_adapter

add_adapter("my-agent", my_adapter)
predict("my-agent", "invoice.pdf", schema)
```


## What a benchmark writes

One directory per run.

```
runs/
├── datalab-f46415c9/
│   ├── settings.json
│   ├── summary.json
│   ├── predictions/<doc_id>.json
│   ├── records/<doc_id>.json
│   ├── scores.jsonl
│   └── verdicts/<doc_id>.jsonl      # unless --no-verdicts
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
  "run": "datalab-f46415c9",
  "provider": "datalab",
  "settings": {
    "mode": "balanced",
    "base_url": "https://www.datalab.to",
    "poll_interval": 5.0
  },
  "dataset": "hf://datasets/datalab-to/omni_extract_bench",
  "commit": "8d3f0c1a...",
  "manifest": "manifest.parquet",
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
  "headline": 0.8076406248606786,
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

`headline` is the mean accuracy, the one number to compare runs by. The run and each suite are the
same shape, so whatever you read at the top you can read per suite as well.


### predictions/&lt;doc_id&gt;.json

The bare extraction, shaped like the schema. This is what scoring reads, and it's the same
thing `predict` returns as `record["result"]`. Only answers are written here: a failed call has
no prediction file, and its error is in its record. A document with no prediction is graded as
failed (`"status": "error"`), whether the call failed or the file is missing from a folder you
brought, so it lowers coverage instead of being left out of it.

### records/&lt;doc_id&gt;.json

Everything else about that document. Two files because scoring wants the answer and an audit
wants all of it.

**`schema_sent` is not the schema you passed in.** Vendors accept different subsets of JSON
Schema, so the harness re-encodes yours to fit each one before sending it.

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
  "schema_sent": {"...": "the schema this vendor was actually given -- see below"},
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
  "scorer": "0.1.8+3f9a0c2e71b4",
  "gold": "a41c09e2d5f7b803",
  "prediction": "5d0e72c9a1b34f60",
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

One line per address: what happened there, and what each side was compared as. This is the
same shape `score(..., verdicts=True)` returns, and it's a low-level address breakdown of how
scoring happened.

Written by default, because it's what lets a number be argued with rather than only quoted.
It's also the bulk of what a run writes -- around five times the gold it grades, so a
four-provider run over our corpus is a couple of gigabytes. Pass `--no-verdicts` and you get
`scores.jsonl` and `summary.json` and nothing else.

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
