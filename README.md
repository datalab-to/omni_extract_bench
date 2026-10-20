# Omni Extract Bench

Omni Extract Bench measures how well a system extracts a JSON object from a PDF, given a JSON Schema. This repository holds:

- the scorer, field matching, which compares a prediction with its ground truth value by value;
- the benchmark: our dataset on [Hugging Face](https://huggingface.co/datasets/datalab-to/omni_extract_bench), or your own, run through vendor systems and graded; and
- the lower-level pieces (`score` and `predict`), to use in your own pipelines.

Contents:

1. [Install](#install): the scorer on its own, or with the vendor adapters;
2. [Run our benchmark](#run-our-benchmark): run vendors on the benchmark dataset, and grade them;
3. [Score](#score): score one prediction against its ground truth, in code or from the command line;
4. [Predict](#predict): one document and its schema through one system;
5. [Write your own scorer or adapter](#write-your-own-scorer-or-adapter);
6. [Tests](#tests); and
7. [Licence](#licence).

[`docs/API.md`](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md) documents the public functions and commands in more depth.

## Install

```bash
uv pip install omni-extract-bench                 # the scorer, and grading a dataset folder on disk
uv pip install 'omni-extract-bench[harness]'      # + the vendor adapters, to produce predictions
uv pip install 'omni-extract-bench[benchmark]'    # + huggingface_hub, to download our dataset
```

The base install brings scipy and [omni-bench-core](https://github.com/datalab-to/omni-bench-core), which brings pyarrow and rich; scipy brings numpy. It scores, and grades (`oeb grade`, `oeb check`) without any vendor package. Reading a dataset from the Hugging Face Hub also needs `huggingface_hub`, which the `[benchmark]` extra installs.

![oeb-usage](https://github.com/datalab-to/omni_extract_bench/blob/main/assets/oeb_usage.gif?raw=true)

## Run our benchmark

The benchmark runs on [omni-bench-core](https://github.com/datalab-to/omni-bench-core). A run is one system: it sends each document to the system, saves each answer, and grades the answers. This runs Datalab in accurate mode on one document:

```bash
oeb run --system datalab --options '{"mode": "accurate"}' --limit 1 --out runs/
```

`--options` is the system's own options, as `oeb predict` takes them. This costs money, and each system needs its API key in the environment:

| system | environment variable |
| --- | --- |
| `datalab` | `DATALAB_API_KEY` |
| `reducto` | `REDUCTO_API_KEY` |
| `extend` | `EXTEND_API_KEY`, and `EXTEND_WORKSPACE_ID` when the key reaches several workspaces |
| `llamaextract` | `LLAMA_CLOUD_API_KEY` (or its older name, `LLAMAPARSE_API_KEY`) |
| `mistral` | `MISTRAL_API_KEY` |
| `azure-cu` | `AZURE_CU_KEY`, and the `endpoint` option |
| an OpenRouter model id, such as `openai/gpt-5.6-sol` | `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) |

A missing key stops the run before any call, and says what to set, such as `export DATALAB_API_KEY=...`. `oeb systems <name>` prints a system's options and the variables it needs.

A run is one system. To compare systems, or one system's options, run one command per system, with the same choice of documents (`--sample`, `--limit`, `--only` or `--suites`), so they answer the same rows:

```bash
oeb run --system datalab --options '{"mode": "accurate"}' --sample 20 --out runs/
oeb run --system reducto --sample 20 --out runs/
```

Each prints its grading as one JSON object, whose `"run"` is the run folder's name. `--workers N` sets how many documents are in flight at once; by default, the system's own number.

`oeb run` shows the plan and asks before calling any system. With no terminal to ask (a pipe, a notebook, CI), it goes ahead without asking, so a script runs as written.

`--dataset` picks the dataset: a folder on disk, or a Hugging Face dataset (ours by default). Ours is about 856 MB, and it is downloaded whole the first time, even for one document, because grading needs all of it; later runs use the copy in the Hugging Face cache. Set `HF_TOKEN` (or run `hf auth login`) to download faster and avoid the Hub's rate limits. `--manifest` picks which of its documents to run, such as a 40-document subset of ours:

```bash
oeb run --system datalab --out runs/ \
  --manifest manifests/sample40-predictive-across-vendors-2026-09-22.parquet
```

Each run is a folder under `--out`, named after everything that changes its answers: the system, its options, the dataset, the manifest and which documents (`--suites`, `--sample`, `--limit`, `--only`). So:

- running the same command again resumes the same folder, and pays for nothing already answered; and
- changing any of those uses a new folder, and leaves the old one as it was.

The timeout isn't one of them. To give documents that timed out more time, run the same command again with a longer `--timeout` and `--retry-failed`: it uses the same folder, and each document's record keeps the timeout it was run with.

A run is graded only when it is finished: every document has an answer or a recorded failure. One rule decides what counts as 0 ([The rules](https://github.com/datalab-to/omni-bench-core/blob/main/OVERVIEW.md#the-rules) in omni-bench-core's overview):

- a document counts 0 only when code says, on purpose, that the system failed, by raising `errors.SystemFailed` (or one of its kinds, `SystemTimeout`, `SystemUnreachable` and `Unreadable`). Its record keeps the message and the traceback of where it was raised; and
- any other exception stops the run and prints its traceback. It is a bug in an adapter or a scorer, not a failure of the system.

When the run stops (its key is refused, its account is out of credits, an adapter or a scorer has a bug), the command exits 1 saying why, and running it again resumes. An answer that `read` crashed on is kept, unread, and running the same command again after the fix reads it with no call.

To grade a run again, or a folder of your own predictions (`predictions/<doc_id>.json`):

```bash
oeb grade runs/datalab-1a2b3c4d                    # a run knows its own dataset
oeb grade mine/ --dataset my-dataset/              # your predictions on your own dataset
oeb grade mine/ --dataset my-dataset/ --partial    # count unfinished documents as 0; the summary counts them
```

Without `--dataset`, a folder of your own predictions is graded against our public dataset (`hf://datasets/datalab-to/omni_extract_bench`).

Grading is fast on most documents, but field matching on a very large document takes long. On our dataset, `long__real_ofac_ssi_full` alone takes about 2 minutes, and all 620 documents take about 7 to 8 minutes of CPU, spread over up to 8 processes. Each document's score is kept in the run folder, so grading again only scores the documents whose prediction or gold changed.

What happens in every other case (a refusal, a timeout, a wrong key, an interrupt, a bug in an adapter) is in the table under [The rules](https://github.com/datalab-to/omni-bench-core/blob/main/OVERVIEW.md#the-rules) in omni-bench-core's overview. [`docs/API.md`](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md#what-a-run-writes) says what a run writes, and [how to run on your own dataset](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md#your-own-dataset).

In Python:

```python
from omni_extract_bench import benchmark
from omni_extract_bench.harness.providers.datalab import Datalab

if __name__ == "__main__":
    b = benchmark.default()
    g = b.run(Datalab(mode="accurate"), out="runs", sample=20)    # one system; g["run"] names its folder
```

To compare systems, call `run` once per system with the same `sample` (or `suites`, `limit`, `only`), so each is graded on the same documents. `harness.system("datalab", {"mode": "accurate"})` makes the same adapter from a system name.

Code that runs or grades a benchmark goes under `if __name__ == "__main__":`. Grading scores documents in a pool of processes, and on macOS and Windows each process imports your script again.

## Score

The scorer works on its own, with no dataset and no network. The files of this example are `pred.json`, `gold.json` and `schema.json` in [`data/`](https://github.com/datalab-to/omni_extract_bench/blob/main/data) on GitHub. A pip install doesn't include them, so download them from there, or clone the repository and run the commands below from its folder:

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

or from the command line.


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

Return with verdicts to dive deeper into exact places where the model failed.

```python

result = score(prediction, ground_truth, schema, verdicts=True)
print(result["verdicts"][0]) # print the first verdict
# Verdict(address=(('k', 'currency'),), gold_raw=None, pred_raw='USD', gold_canon=None,
#         pred_canon='usd', result='invented_field')
```

From the command line.

```bash
oeb score --pred data/pred.json --gold data/gold.json --schema data/schema.json --details
```

On the example above, printing one line each:

```
{
  "accuracy": 0.5294117647058824,
  "precision": 0.5625,
  "recall": 0.8181818181818182,
  "f1": 0.6666666666666666,
  ...
  "verdicts": [
    {
      "address": [
        [
          "k",
          "currency"
        ]
      ],
      "gold_raw": null,
      "pred_raw": "USD",
      "gold_canon": null,
      "pred_canon": "usd",
      "result": "invented_field"
    },
    ...truncated for display
```


### What the metric does

![scoring.gif](https://github.com/datalab-to/omni_extract_bench/blob/main/assets/scoring.gif?raw=true)

- Normalize document;
- Flatten prediction and gold JSON dictionary to addresses mapped to their scalar values;
- Normalize scalar values of the flattened addresses; and
- For each array that appears, Hungarian match (recursively for nested arrays) based on array element content to align ambiguous predicted and gold addresses (there may be unmatched predicted addresses, false positives, and unmatched gold addresses, false negatives).

Full specification: [`METRIC_SPEC.md`](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/METRIC_SPEC.md).

## Predict

Predict with our vendor adapters.

```bash
uv pip install 'omni-extract-bench[harness]'
```

Each vendor is an adapter class; `harness.system` makes one from a system name and its options. omni-bench-core's `predict` sends one document under the same rules a run uses (one time budget, retries, failures recorded) and returns its record.

```python
from pathlib import Path

from omni_bench_core import adapter
from omni_extract_bench import harness
from omni_extract_bench.harness import contract

datalab = harness.system("datalab", {"mode": "accurate"})
record = adapter.predict(datalab, contract.Question(Path("invoice.pdf"), schema))
record["output"]
record["raw"]
record["cost"]
```

This sends the schema as you give it. `oeb run` and `oeb predict` first apply the benchmark's conventions to the schema (`task.Extract().input`), so use `oeb predict` to send exactly what a run sends.

Score directly from the predictions.

```python
from omni_extract_bench import score
score(record["output"], gold, schema)
```

You can also use the command line, which prints the record as JSON and exits 1 when the call failed. `--options` takes the bare options, as `oeb run --system` does, or the same keyed by the system's name:

```bash
oeb predict --system datalab --doc invoice.pdf --schema schema.json --options '{"mode": "accurate"}'
oeb predict --system datalab --doc invoice.pdf --schema schema.json --options '{"datalab": {"mode": "accurate"}}'
```

## Write your own scorer or adapter

A scorer is a class with `gold` and `score`; an adapter is a class with `call` (for extraction, `extract`). omni-bench-core's [overview](https://github.com/datalab-to/omni-bench-core/blob/main/OVERVIEW.md) explains both with small examples. [`docs/API.md`](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md#your-own-adapter) shows an extraction adapter and how it signals each outcome, and [how to grade with a scorer of your own](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md#your-own-scorer) beside field matching.

## Tests

```bash
uv run --group test --extra benchmark pytest -n auto
```

## Licence

Apache 2.0, see [`LICENSE`](https://github.com/datalab-to/omni_extract_bench/blob/main/LICENSE). All code here is Datalab's own.
