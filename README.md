# Omni Extract Bench

Omni Extract Bench measures how well a system extracts a JSON object from a PDF, given a JSON Schema. A system is
sent a document and a schema, and returns a JSON object. The object is compared with the ground truth, field by
field, by our scorer, field matching.

This repository holds:

- the scorer, which works on its own: one prediction, its ground truth and its schema in, a score out;
- the benchmark: our dataset on [Hugging Face](https://huggingface.co/datasets/datalab-to/omni_extract_bench), or
  your own, run through a system and graded; and
- the adapters: one per vendor, and one for any model on OpenRouter, to get predictions in your own code.

Contents:

1. [How scoring works](#how-scoring-works): what field matching compares, and how;
2. [Install](#install): the scorer alone, or with the adapters and the dataset;
3. [Run the benchmark](#run-the-benchmark): one system on the dataset, from a config;
4. [Grade](#grade): a run again, or predictions of your own;
5. [Score one prediction](#score-one-prediction): the scorer on its own, in code or from the command line;
6. [Predict one document](#predict-one-document): one document and its schema through one system;
7. [Write your own scorer or adapter](#write-your-own-scorer-or-adapter);
8. [Tests](#tests); and
9. [License](#license).

[`docs/API.md`](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md) documents every public
function and command in more depth.

![oeb-usage](https://github.com/datalab-to/omni_extract_bench/blob/main/assets/oeb_usage.gif?raw=true)

## How scoring works

![scoring.gif](https://github.com/datalab-to/omni_extract_bench/blob/main/assets/scoring.gif?raw=true)

Field matching is deterministic. It works in four stages:

- Normalize the document;
- Flatten the prediction and the ground truth to addresses (the path to a value, such as
  `line_items[0].qty`), each mapped to its scalar value;
- Normalize each scalar value, so `"1,240.00"` and `1240.0` are the same number; and
- Pair the addresses of the two documents:
  - If unambiguous, pair exactly; and
  - If ambiguous (the items of an array), Hungarian match on the items' content, recursively for nested arrays.

A predicted address with no partner is a false positive; a ground-truth address with no partner is a false
negative. Accuracy is the matched addresses over the addresses either document used. The full specification is
[`docs/METRIC_SPEC.md`](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/METRIC_SPEC.md).

## Install

```bash
uv pip install omni-extract-bench                 # the scorer, and grading a dataset folder on disk
uv pip install 'omni-extract-bench[harness]'      # + the adapters, to get predictions
uv pip install 'omni-extract-bench[benchmark]'    # + huggingface_hub, to download our dataset
```

The base install brings scipy, pyarrow, pydantic, pyyaml and rich; scipy brings numpy. It scores and grades
(`oeb score`, `oeb grade`, `oeb check`) with no vendor package. Reading a dataset from the Hugging Face Hub also
needs `huggingface_hub`, which `[benchmark]` installs.

## Run the benchmark

A run is one system on one dataset. It works in three stages:

1. Plan: check the config, the system's keys and the dataset's files, before any call;
2. Predict: send each document to the system and save each answer in the run's folder; and
3. Grade: score the saved answers. Grading calls no system, so it can be done again for free.

### A config

A run is one config: a YAML file with every key written out. `oeb init` writes one for a system, every option at
its default, on our dataset:

```bash
oeb init datalab > datalab.yaml
```

```yaml
dataset: hf://datasets/datalab-to/omni_extract_bench   # a folder, or a Hugging Face dataset
manifest: manifest.parquet                             # which of its documents, with which files
scorers:
  field_matching: {}
rows:                    # sample N evenly spaced, or limit to the first N (0 for every document), or only these ids
  sample: 0
  limit: 0
  only: null
timeout: 1800.0          # seconds each document may take
workers: 0               # documents in flight at once; 0 for the system's own
system:
  datalab:
    mode: balanced
    ...
```

`--set path=value` changes one value the file has, read as YAML, and refuses a path it hasn't. This runs Datalab
in accurate mode on one document:

```bash
export DATALAB_API_KEY=...
oeb run --config datalab.yaml --set system.datalab.mode=accurate --set rows.limit=1 --out runs/
```

`oeb run` shows the plan and asks before calling any system. With no terminal to ask (a pipe, a notebook, CI), it
goes ahead without asking, so a script runs as written. It prints its grading as one JSON object, whose `"run"` is
the run folder's name.

### Keys

Calling a system costs money, and each system reads its key from the environment:

| system | environment variable |
| --- | --- |
| `datalab` | `DATALAB_API_KEY` |
| `reducto` | `REDUCTO_API_KEY` |
| `extend` | `EXTEND_API_KEY`, and `EXTEND_WORKSPACE_ID` when the key reaches several workspaces |
| `llamaextract` | `LLAMA_CLOUD_API_KEY` (or its older name, `LLAMAPARSE_API_KEY`) |
| `mistral` | `MISTRAL_API_KEY` |
| `azure-cu` | `AZURE_CU_KEY`, and the `endpoint` option |
| an OpenRouter model id, such as `openai/gpt-5.6-sol` | `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) |

A missing key stops the run before any call, and says what to set. `oeb systems <name>` prints a system's options
and the variables it needs.

### Choosing the documents

`dataset` is a folder on disk, or a Hugging Face dataset (ours, as `oeb init` writes it; `--set dataset=<folder>`
runs on another). Ours is about 856 MB. It is downloaded whole the first time, even for one document, because
grading needs all of it; later runs read the copy in the Hugging Face cache. Set `HF_TOKEN` (or run
`hf auth login`) to download faster and avoid the Hub's rate limits.

`manifest` picks which of the dataset's documents a run can ask about, such as a 40-document subset of ours:

```bash
oeb run --config datalab.yaml --out runs/ \
  --set manifest=manifests/sample40-predictive-across-vendors-2026-09-22.parquet
```

`rows` picks which of the manifest's documents this run asks about: `sample: N` for N evenly spaced, `limit: N`
for the first N, or `only: [ids]`.

### Comparing systems

A run is one system. To compare systems, or two settings of one system, write one config per system, with the
same `rows`, so they answer the same documents:

```bash
oeb init reducto > reducto.yaml
oeb run --config datalab.yaml --set rows.sample=20 --out runs/
oeb run --config reducto.yaml --set rows.sample=20 --out runs/
```

### The run folder

Each run is a folder under `--out`, named after its system and a run id: `datalab-20261021-160012-3f9a`. It holds:

- `config.yaml`: the config, every `--set` applied and every option written out, so
  `oeb run --config runs/<run>/config.yaml` asks the same question again;
- `settings.json`: the question the run answers (the system and its options, the dataset, the manifest and the
  documents chosen), and when and how it was launched;
- `records/` and `predictions/`: each document's answer as the system gave it, and as it is graded; and
- `field_matching/`: the grading.

A run folder answers one exact question. So:

- every `oeb run` is a new run in a new folder, so a system redeployed since an earlier run is never mixed into it;
  and
- `oeb resume <its folder>` continues that run ([Resuming a run](#resuming-a-run)).

[`docs/API.md`](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md#what-a-run-writes) says
what each file holds, and [how to run on your own dataset](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md#your-own-dataset).

### When something goes wrong

One rule decides what counts as 0:

- a document counts 0 only when code says, on purpose, that the system failed, by raising `errors.SystemFailed`
  (or one of its kinds, `SystemTimeout`, `SystemUnreachable` and `Unreadable`). Its record keeps the message and
  the traceback of where it was raised; and
- any other exception stops the run and is raised as it is, with its traceback. It is a bug in an adapter or a
  scorer, or a problem with the setup (a refused key, an account out of credits), not a failure of the system.

Every case (a refusal, a timeout, a wrong key, an interrupt, a bug) follows from that rule. After a run stops, fix
the cause and resume it: the answers already paid for are kept.

### Resuming a run

A run that stopped part way is continued with `oeb resume <its folder>`. A run stops when you press Ctrl-C, when
its key is refused or its account runs out, or when an adapter has a bug. Resuming asks only for the documents
with no record yet, so nothing already answered is paid for twice:

```bash
oeb resume runs/datalab-20261021-160012-3f9a
```

It reads the run's own `config.yaml`, so it asks the same question. `--set` may change what isn't part of the
question, `timeout` and `workers`. A change to anything else (the system, its options, the dataset, the manifest,
`rows`) is refused, naming the keys that differ: a different question is a new run.

A failed document has a record, so a resume doesn't ask for it again, unless you add `--retry-failed`. That asks
again for the documents whose failure may not happen a second time: a timeout, a system that couldn't be reached,
a rate limit (429) or a server error (5xx). A refusal (such as a 400) or an answer that couldn't be read stays
failed, since asking again would give the same answer. To give documents that timed out more time:

```bash
oeb resume runs/datalab-20261021-160012-3f9a --retry-failed --set timeout=3600
```

Each document's record keeps the timeout it was asked with.

An answer that `read` crashed on is kept, unread. Resuming the run after the fix reads it again, with no call.

### In Python

```python
from omni_extract_bench import benchmark
from omni_extract_bench.harness.providers.datalab import Datalab

if __name__ == "__main__":
    b = benchmark.default()
    g = b.run(Datalab(mode="accurate"), out="runs", sample=20)    # one system; g["run"] names its folder
```

To compare systems, call `run` once per system with the same `sample` (or `limit`, `only`).
`harness.system("datalab", {"mode": "accurate"})` makes the same adapter from a system name. To resume a run, give
its name as `resume` (`b.run(Datalab(mode="accurate"), out="runs", resume="datalab-20261021-160012-3f9a")`), and
`retry_failed=True` to ask again for its failures that may not recur.

Code that runs or grades a benchmark goes under `if __name__ == "__main__":`. Grading scores documents in a pool
of processes, and on macOS and Windows each process imports your script again.

## Grade

`oeb run` grades its run when it finishes. To grade a run again, or a folder of your own predictions
(`predictions/<doc_id>.json`):

```bash
oeb grade runs/datalab-20261021-160012-3f9a        # a run knows its own dataset
oeb grade mine/ --dataset my-dataset/              # your predictions on your own dataset
oeb grade mine/ --dataset my-dataset/ --partial    # count unfinished documents as 0; the summary counts them
```

Without `--dataset`, a folder of your own predictions is graded against our public dataset.

A run is graded only when it is finished: every document has an answer or a recorded failure. Otherwise
`oeb grade` refuses and names the documents; `--partial` counts them as 0, and the summary says so.

Grading is fast on most documents, but field matching on a very large one takes long. On our dataset,
`long__real_ofac_ssi_full` alone takes about 2 minutes, and all 620 documents take about 7 to 8 minutes of CPU,
spread over up to 8 processes. Each document's score is kept in the run folder, so grading again only scores the
documents whose prediction or gold changed.

## Score one prediction

The scorer works on its own, with no dataset and no network. This example's files are `pred.json`, `gold.json`
and `schema.json` in [`data/`](https://github.com/datalab-to/omni_extract_bench/blob/main/data) on GitHub. A pip
install doesn't include them, so download them from there, or clone the repository and run the commands below
from its folder:

```bash
git clone https://github.com/datalab-to/omni_extract_bench && cd omni_extract_bench
```

```python
from omni_extract_bench import score

schema = {
  "type": "object",
  "properties": {
    "invoice_id": {
      "type": "string",
      "description": "The invoice number as printed on the document."
    },
    "invoice_date": {
      "type": "string",
      "description": "Date of issue, ISO 8601 (YYYY-MM-DD)."
    },
    "total_due": {
      "type": "number",
      "description": "Total amount payable, in the invoice currency."
    },
    "purchase_order": {
      "type": "string",
      "description": "Buyer's purchase order number, where the invoice cites one."
    },
    "line_items": {
      "type": "array",
      "description": "One entry per billed line.",
      "items": {
        "type": "object",
        "properties": {
          "sku": {"type": "string", "description": "Stock code for the item."},
          "description": {"type": "string", "description": "Item description as printed."},
          "qty": {"type": "integer", "description": "Units billed."},
          "unit_price": {"type": "number", "description": "Price per unit."}
        }
      }
    }
  }
}

prediction = {
  "invoice_id": "INV-4417",
  "invoice_date": "03/31/2024",
  "total_due": "1,240.00",
  "purchase_order": "PO-88231",
  "currency": "USD",
  "line_items": [
    {"sku": "BX-2201", "description": "Washer, 8mm", "qty": 100, "unit_price": 9.4},
    {"sku": "AX-9910", "qty": 24, "unit_price": 12.05},
    {"sku": "ZZ-0000", "description": "Freight surcharge", "qty": 1, "unit_price": 45.0}
  ]
}

ground_truth = {
  "invoice_id": "INV-4417",
  "invoice_date": "2024-03-31",
  "total_due": 1240.0,
  "line_items": [
    {"sku": "AX-9910", "description": "Hex bolt, M8", "qty": 24, "unit_price": 12.5},
    {"sku": "BX-2201", "description": "Washer, 8mm", "qty": 100, "unit_price": 9.4}
  ]
}

result = score(prediction, ground_truth, schema)
result["accuracy"]    # 0.5294 -- matched addresses / addresses either document used
result["precision"]   # 0.5625
result["recall"]      # 0.8182
```

The same from the command line:

```bash
oeb score --pred data/pred.json --gold data/gold.json --schema data/schema.json
```

```json
{
  "accuracy": 0.5294117647058824,
  "precision": 0.5625,
  "recall": 0.8181818181818182,
  "f1": 0.6666666666666666,
  "total": 17,
  "matched": 9,
  "misread": 1,
  "unfound": 1,
  "fabricated": 1,
  "invented_item": 4,
  "invented_field": 1,
  "asserted": 16,
  "addresses_found": 0.5882352941176471,
  "addresses_read_right": 0.9,
  "gold_rows": 2,
  "pred_rows": 3,
  "matched_rows": 2,
  "matching_exact": true,
  "approximated": [],
  "skipped_open_maps": []
}
```

To see where a prediction went wrong, ask for the verdicts: one per address, with its raw and normalized values
on both sides and its result.

```python
result = score(prediction, ground_truth, schema, verdicts=True)
print(result["verdicts"][0])
# Verdict(address=(('k', 'currency'),), gold_raw=None, pred_raw='USD', gold_canon=None,
#         pred_canon='usd', result='invented_field')
```

From the command line, `--details` adds them:

```bash
oeb score --pred data/pred.json --gold data/gold.json --schema data/schema.json --details
```

```
{
  "accuracy": 0.5294117647058824,
  ...
  "verdicts": [
    {
      "address": [["k", "currency"]],
      "gold_raw": null,
      "pred_raw": "USD",
      "gold_canon": null,
      "pred_canon": "usd",
      "result": "invented_field"
    },
    ...
```

## Predict one document

With `[harness]` installed, each vendor is an adapter class. `harness.system` makes one from a system name and its
options, and `harness.predict` sends one document under the same rules a run uses (one time budget, retries,
failures recorded) and returns its record:

```python
from pathlib import Path

from omni_extract_bench import harness, score
from omni_extract_bench.harness import contract

datalab = harness.system("datalab", {"mode": "accurate"})
record = harness.predict(datalab, contract.Question(Path("invoice.pdf"), schema))
record["output"]    # the extracted object
record["raw"]       # the answer as the system gave it
record["cost"]      # what it cost, its wall time and attempts

score(record["output"], gold, schema)
```

This sends the schema as you give it. A run first applies the benchmark's conventions to the schema
(`task.Extract().input`), so use `oeb predict` to send exactly what a run sends. It prints the record as JSON, and
exits 1 when the call failed. `--options` takes the bare options, or the same keyed by the system's name:

```bash
oeb predict --system datalab --doc invoice.pdf --schema schema.json --options '{"mode": "accurate"}'
oeb predict --system datalab --doc invoice.pdf --schema schema.json --options '{"datalab": {"mode": "accurate"}}'
```

## Write your own scorer or adapter

A scorer is a class with `gold`, `score` and `summarize`; an adapter is a class with `call` (for extraction,
`extract`). [`docs/API.md`](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md#your-own-adapter)
shows an extraction adapter and how it signals each outcome, and
[how to grade with a scorer of your own](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md#your-own-scorer)
beside field matching.

## Tests

```bash
uv run --group test --extra benchmark pytest -n auto
```

## License

Apache 2.0, see [`LICENSE`](https://github.com/datalab-to/omni_extract_bench/blob/main/LICENSE). All code here is
Datalab's own.
