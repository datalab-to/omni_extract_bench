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
  gold,               # the ground truth, a dict
  schema,             # the JSON Schema the prediction was generated against
  order_matters=(),   # arrays where position is part of the answer, e.g. ["steps"]; True for all
  verdicts=False,     # also return one verdict per address
  absent=states_nothing,  # which values get no address; default: null, "", [], {}
)
```

`absent=lambda v: False` makes `null`, `""`, `[]` and `{}` values the prediction must
reproduce: gold `null` vs predicted `""` is `misread`, a missing key is `unfound`. Scores under
different `absent`s are not comparable, since the number of addresses changes.

```python
result = score(prediction, ground_truth, schema)
result["accuracy"]    # matched addresses / addresses either document used
result["precision"]
result["recall"]
```

or from the command line.

```bash
oeb score --pred pred.json --gold gold.json --schema schema.json
```
```json
{
  "accuracy": 0.5294117647058824,
  "precision": 0.5625,
  "recall": 0.8181818181818182,
  "f1": 0.6666666666666666,
  ...truncated for display
```

Every argument except `absent` is a flag of the same name, underscores as dashes, so
`order_matters` is `--order-matters`. It goes to stdout, so `oeb score ... | jq .accuracy` is one pipe.

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

These raise, since they are the same for every document:

```python
MissingCredential    # an unset API key                     (a Stop)
MissingDependency    # an adapter whose SDK isn't installed (a Stop)
AccountFailure       # the account can't pay                (a Stop)
UsageError           # a PDF that doesn't exist, an unknown provider or option
```

A schema that isn't a JSON object raises `TypeError`. `UsageError` (a `ValueError`) and `Stop` are in
`omni_extract_bench.errors`: the two kinds of error a user meets; anything else raised is a bug of ours.

A failure the vendor reports comes back in `record["error"]` (type, HTTP status, whether `predict`
retried it within the call, and the message), with `record["result"]` set to `None`. A 429 or 5xx is
retried within the call first, after 20, 60 and 120 seconds, inside the document's `timeout`; a
timeout or a failed poll is not, since the first would spend a second budget and the second may buy
the extraction twice. Anything else an adapter raises is a bug, not an answer: `predict_all` stops
that adapter's runs, naming the document, and keeps what was recorded (an adapter whose `extract`
returns something other than an `Extraction` raises `TypeError` saying so).


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
  sample=0,               # N documents spread over every suite
  only=None,              # these doc ids only (a frozenset); each must be in the dataset
  timeout=1800.0,
  workers=0,              # documents in flight per adapter: N, or {adapter: N}
  score_workers=0,        # processes used to score
  verdicts=True,          # write verdicts; False to skip them
  score_only=False,       # score the predictions; call no vendor
  retry_failed=False,     # predict again documents whose failure may not recur
  options=None,           # per-provider settings: {"datalab": {"mode": "accurate"}}
)
print(benchmark.describe(p))   # what it would do, with no vendor call
summary = benchmark.execute(p)
summary["datalab-f46415c9"]["headline"] # the mean accuracy, a failure counting 0, e.g. 0.9145;
                                        # null while any document is pending
```

### The functions, by what they work on

| | reading data (`data`) | predicting (`harness`, `benchmark`) | scoring (`scorer`, `benchmark`) |
| --- | --- | --- | --- |
| one document | `doc.doc_path`, `doc.gt_path` | `harness.predict` | `scorer.gold(doc)`, `scorer.score` |
| a dataset | `download_dataset`, `read_manifest`, `select`, `dataset_id` | | `scorer.check(docs)`, `scorer.summarize(lines)` |
| one run | | `prepare`, `needs`, `record`, `recorded`, `predicted` | `grade(run, docs)`, `read_scores(run)` |
| a folder | `settled`, `declare` (its settings.json), `gradable` | | `scorer_errors(summary)` |
| a plan | | `plan`, `predict_all`, `owed(plan, run)` | `grade_all` |
| both | | `execute`; `describe` | |

`scorer` is the package's one scorer, `omni_extract_bench.scorers.field_matching` (each scorer
is a package owning its gold shape); `data`, `scorers`, `gold`, `score`, `summarize`, `check` and
`Verdict` are exported from the package top.

The same names, in the same modules, as omni-parse-bench's (whose rows are pages, not documents).
`data.read_manifest(dataset, manifest)` gives the dataset's folder and every document, each with
its file and gold as full local paths (a manifest that breaks the dataset's rules raises
`data.ManifestError`, naming what's wrong); `data.select(docs, suites, limit, sample, only)` picks the ones to run: by suite (an unknown suite
is refused), by doc id, `sample` documents spread over every suite (the same ones each time), then
the first `limit`. On the command line, `--sample N` and `--docs FILE` (doc ids, one per line; blank
lines and `#` comments are skipped). `--sample` larger than the selection takes all of it, and with
`--docs` it samples among those documents. A null or empty `suite` in a manifest is read as
`"other"` (`data.NO_SUITE`), so it merges with a suite really named "other". Rows may share an input
file (one PDF asked with several schemas); each gold is named for its row, `<doc_id>.json`.

`owed(plan, run)` is two counts, `(to predict, to grade)`: how many of the plan's documents the run has
no record of (with `retry_failed`, also the retryable failures), and how many of the manifest's
documents will have a line once it is graded (the rest are pending). `describe` shows both. `scorer.check(docs)` raises naming
the first document whose gold can't be read or isn't a JSON object, as `oeb check` does.

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

With `-y`, or `--score-only`, which spends nothing, the plan is printed and the run goes on without
asking; a declined plan prints `oeb benchmark: nothing run`. The execution looks like:

```
 run                                    done   ok   err   in flight       cost     avg    dur
 ────────────────────────────────────────────────────────────────────────────────────────────
 datalab-f46415c9   ━━━━━━━━━━━━━━━━   20/20   20     0                  $6.20   1m58s   3.0s
 datalab-01a72762   ━━━━━━━━━━━━━━━━   20/20   20     0                  $6.20   2m02s   3.0s
 reducto-e54d3a1d   ━━━━━━━━━━━━━━━━   20/20   19     1               5,820 cr   2m19s   3.0s
 mistral-44136fa3   ━━━━━━━━━━━━━━━━   20/20   20     0                          2m13s   3.0s
80/80 documents  1 failed  $12.40 + 5,820 cr  3.0s elapsed
```

Then each run's summary is printed to stdout, by run name, as its summary.json holds it, and each
run's closing line to stderr; when documents failed it says that `--retry-failed` may predict some
again, it says how many. A `UsageError` (what was asked is wrong) or a `Stop` (an account, a key or a
dependency) is one line, `oeb benchmark: ...`, and exits 1; after a `Stop`, the summaries grading wrote
are printed to stdout first. Ctrl-C exits 130 and grades nothing. Either way, what was predicted is
kept, and the same command resumes. A document whose grading raises in our scorer is a bug of ours:
grading stops with the traceback, naming the run and the document, and that run writes no scores or summary (a lost prediction may
already have been rebuilt from its record), since a
summary with it left out would be wrong. Runs are graded one after another, so those graded before it
keep their new files, each a whole grading. Fix the bug, then grade again with `--score-only`. A gold that
can't be read stops it too, as a `GoldError`.


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

`doc_path` and `gt_path` are relative to the dataset's folder, wherever the manifest is in it, and
a gold is named for its document, `<doc_id>.json`, in any folder. An input may have any name, and
documents may share one, such as one PDF asked with several schemas. A doc id appears once.
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
commit's, and a second run downloads nothing. A run's `settings.json` records the commit; when
`--out` holds a run made at another commit, `plan` warns and carries on, so its old predictions
are graded against the new gold. Pass another `--out` to keep them apart. For example,

```bash
oeb benchmark --providers datalab --dataset hf://datasets/someone/their-corpus@v2
```

*A folder on disk.* Nothing is downloaded. For example:

```bash
oeb benchmark --providers datalab --dataset my/corpus
```

A run folder holds one dataset and one manifest: pointing it at another of either is refused before
anything is read, since a grading of the other would replace its summary. To predict a sample, then
the whole, use another `--out` for the sample.

### Resuming

It's **resumable**, and safe to run twice.

- A document is predicted again only when it has no record. The prediction is written before the
  record, so a record never stands beside a half-written prediction, and the record is the source of
  truth for an answer: a prediction lost or broken beside a record of an answer is rebuilt from it, with no call,
  whenever the run is graded (`--score-only`, `oeb grade` and the grading after predicting). A
  prediction of `{}` is an answer like any other: scored, with every field the gold states unfound.
- `--retry-failed` predicts again documents whose recorded failure may not recur: a timeout, an
  outage (`VendorUnreachable`: the vendor couldn't be reached), a 429 or a 5xx (500, 502-504,
  520-524). A 400, a refusal or an unusable answer is an answer, and is not retried. Within one
  call, a 429 or 5xx is tried again after 20, 60 and 120 seconds before it is recorded. It decides from the error's type and status;
  `record["error"]["transient"]` says something else: whether `predict` retried within the call,
  which it never does for a timeout or a failed poll.
- Grading is fresh every time: every document of the manifest is graded again, which calls no
  vendor, and `verdicts/`, `scores.jsonl` and then `summary.json` are written when it finishes. So a
  run's scores are always one grading's, a gold correction or a new scorer reaches a run on its next
  grading, and `--suites`, `--limit`, `--sample` and `--docs` narrow what is predicted, never what is
  scored. An interrupted grading writes no scores or summary; the last finished grading's files stand. Each line
  says which scoring code (`scorer`, the release and a digest of the scoring files) graded it.
- A document is pending, not failed, when it was never attempted: no line, counted in the
  summary's `pending` (and each suite's). While any is pending the `headline` is null, since a
  headline is a finished run's; the closing line says `<run>: headline — (3 pending: a headline needs
  every document); 5 of 8 documents graded, 1 failed`. A recorded failure is failed. Only a run
  (its settings.json names a provider, `is_run`) has pending documents: in a folder of predictions
  you bring, a document missing from it is failed, since it is your submission.

So a Ctrl-C, a crash or a credit ceiling costs you the documents in flight, and nothing else.
Reinvoke the same command to carry on: it predicts what is left, then grades everything.


### A loop of your own

`oeb benchmark` is one loop over documents. To predict elsewhere (on another machine, on a
scheduler, through a queue) and still get the same run folder, keep the run with the functions
`oeb benchmark` uses itself. A `benchmark.Run` is an address, as a `Path` is: what the run holds
is whatever its folder holds when it is read.

```python
from omni_extract_bench import benchmark, data
from omni_extract_bench.harness import predict

p = benchmark.plan(["datalab"], out="runs")
(run,) = p.runs
benchmark.prepare(run, p.dataset, p.timeout, p.manifest)
for doc in p.docs:
    if benchmark.needs(run, doc):
        result = predict(run.provider, doc.doc_path, doc.schema, timeout=p.timeout, **run.options)  # wherever
        benchmark.record(run, doc.doc_id, result)
_, every = data.read_manifest(p.dataset, p.manifest)
benchmark.grade(run, list(every))
```

- `prepare(run, dataset, timeout, manifest="manifest.parquet")` makes the folder and writes
  `settings.json`, with the commit of a HuggingFace dataset;
- `needs(run, doc, retry_failed=False)` is True when the run has no readable record of the
  document, or, with `retry_failed`, when its recorded failure may not recur. A loop that
  predicts everything every time can skip it; it is what makes a loop resumable;
- `record(run, doc_id, result)` writes a result from `predict`: its prediction, if the call
  succeeded, then the record, and returns the record;
- `recorded(run, doc_id)` reads the document's record, or None;
- `predicted(run, doc_id)` reads the document's prediction, or None: what grading reads; and
- `grade(run, docs, say=print, *, settings=None, verdicts=True, score_workers=0, progress=NULL)`
  grades the folder against `docs` (pass the whole manifest, so a smaller loop never shrinks a run's
  scores), fresh, writes `verdicts/`, `scores.jsonl` and then `summary.json`, says its closing line,
  and returns the summary (`read_scores(run)` gives the lines). A document a run never attempted is
  pending; a recorded failure, or a missing prediction in a folder of your own, is failed. A folder
  of your own none of whose files names a document is refused, writing nothing. A folder that doesn't exist or has no `predictions/` is refused (`gradable`).
  `score_workers` is the number of scoring processes (0: one per core, capped at 8), the same name
  `plan` uses; `verdicts=False` skips the per-address verdict files; `settings` is the folder's
  settings.json as it will be, when not yet written;
- `settled(out, dataset=None, manifest=None)` is the folder's settings.json (`{}` when it has none),
  refusing another dataset or manifest than it names; `declare(out, dataset, manifest, fields)` writes
  it, with the commit; and
- `restore(run, doc_id)` rebuilds a document's lost or unreadable prediction from its record of an
  answer, with no call, and says whether it did; `is_run(settings)` says whether a folder's settings
  are a run's; and
- `pools(plan)` is each adapter's documents at once, and `write_json(path, obj)` writes a file whole.

The folder is then exactly one `oeb benchmark` writes, so either can resume or grade it.

### Grading your own predictions

Write each document's extraction to `predictions/<doc_id>.json`, and grade the folder:

```bash
oeb grade mine/
oeb grade mine/ --dataset my/corpus --manifest manifests/sub.parquet
```

The first grades against the dataset and manifest the folder's `settings.json` names, else ours;
the second against a dataset of your own, and one manifest in it. A folder's first grading that
succeeds writes its dataset and manifest into its `settings.json` (a run's already has them), so a
later `oeb grade mine/` grades against the same ones, and another dataset or manifest is refused
(delete `mine/settings.json` to start the folder over); a grading that fails writes no settings, scores or summary. A grading
none of whose files names a document of the manifest is refused, so a forgotten `--dataset` can't
grade your documents against ours. Grading covers the whole manifest; `per_suite` in the summary
says how each suite went. Files in `predictions/` named for no document of the manifest, and
predictions that aren't a JSON object (graded as failed; each line says why: empty, not UTF-8, not
JSON, or a list), are counted in one warning each. A file saved with a UTF-8 byte-order mark reads as
its JSON. A prediction is read only from a file named exactly
`<doc_id>.json`, case included, so a folder grades the same on macOS and Linux.

Before grading a dataset of your own, `oeb check --dataset my/corpus` says whether oeb can run it:
its manifest, that every file it names exists, and that every gold is a JSON object. The first
problem is printed, naming its row. A dataset that passes is printed as JSON: `documents` (rows),
`suites`, and `inputs`, the distinct input files (fewer than `documents` when rows share one).

The folder is your submission, so a document with no prediction fails: a document left out is one
your system didn't answer. The folder gets `scores.jsonl` and `summary.json`, as a run's does; the
summary is printed to stdout and the closing line to stderr. A `UsageError` or a `Stop` is one line,
`oeb grade: ...`, and exits 1; a scorer that raises is a traceback, with no scores or summary written.

From Python, a folder of your own is a `Run` whose `out` is the folder itself (`provider` names it in
the score lines; its options are `{}`), graded against the whole manifest:

```python
from pathlib import Path
from omni_extract_bench import benchmark, data

_, docs = data.read_manifest("my/corpus")
run = benchmark.Run("mine", {}, Path("mine"))
summary = benchmark.grade(run, docs, score_workers=1)
lines = benchmark.read_scores(run)        # scores.jsonl, back
```

`benchmark.grade` doesn't write `settings.json`; `oeb grade` does, once grading succeeds.

## providers

See providers (a table in a terminal; piped, one name per line):

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

Those defaults are each adapter's `Config`, and not always the vendor's highest tier (datalab's
is `balanced`): `--options` asks for another. An option is given as its field's type (a whole number
for a float field is the same setting), and a value of another type is refused.

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
  "scorer": "0.1.8+3f9a0c2e71b4",
  "headline": 0.8076406248606786,
  "documents": 6,
  "scored": 6,
  "failed": 0,
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
    "extractbench": {"headline": 0.8076406248606786, "documents": 6, "scored": 6, "failed": 0,
                     "pending": 0, "...": "the same block, per suite"}
  },
  "pending": 0,
  "usd": 1.86,
  "credits": null
}
```

`headline` is the mean accuracy over every document, a failed one counting 0: the one number to
compare runs by, which a system can't raise by failing on its hard documents. It exists only for a
finished run: while any document is pending it is null, overall and in each suite with pending
documents, and every per-suite block carries its own `pending`. It is `accuracy *
coverage`; `accuracy`, `precision` and `recall` are over the scored documents only. `documents` is the
documents graded, `scored` those with a usable prediction, `failed` the rest. With none scored,
`accuracy`, `precision`, `recall` and the rates are null, since there is nothing to average. `usd` and
`credits` are what the records say the calls cost; `credits` is null when none reported any. Each suite
is the same block as the run's numbers, with its own `pending`; `run`, `provider`, `settings`,
`scorer`, `usd` and `credits` are the run's alone. A sample's headline (`--manifest manifests/...`) is
over that sample, so it isn't comparable with the whole manifest's.


### predictions/&lt;doc_id&gt;.json

The bare extraction, shaped like the schema. This is what scoring reads, and it's the same
thing `predict` returns as `record["result"]`. Only answers are written here: a failed call has
no prediction file, and its error is in its record. A document with no prediction is graded as
failed (`"status": "failed"`), whether the call failed or the file is missing from a folder you
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

One line per document, in document order whatever order the workers finished in. Its `status` is
`scored` or `failed` (no usable prediction: the line says why).

```json
{
  "run": "datalab-f46415c9",
  "doc_id": "long__dd1155_schedule_continuation_0011",
  "suite": "extractbench",
  "provider": "datalab",
  "scorer": "0.1.8+3f9a0c2e71b4",
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

One line per address: what happened there, and what each side was compared as. The folder is one
grading's, swapped in when it finishes; `--no-verdicts` removes it. This is the
same shape `score(..., verdicts=True)` returns, and it's a low-level address breakdown of how
scoring happened.

Written by default, because it's what lets a number be argued with rather than only quoted.
It's also the bulk of what a run writes -- around five times the gold it grades, so a
four-provider run over our corpus is a couple of gigabytes. Pass `--no-verdicts` and you get
`scores.jsonl` and `summary.json` and nothing else.

One line looks like this:

```json
{"address": [["k", "line_items"], ["i", 0], ["k", "unit_price"]], "gold_raw": 12.5, "pred_raw": 12.05, "gold_canon": "12.5", "pred_canon": "12.05", "result": "misread"}
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
