# Omni Extract Bench

[![test](https://github.com/datalab-to/omni_extract_bench/actions/workflows/test.yml/badge.svg)](https://github.com/datalab-to/omni_extract_bench/actions/workflows/test.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://github.com/datalab-to/omni_extract_bench/blob/main/pyproject.toml)
[![Code: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-blue)](https://github.com/datalab-to/omni_extract_bench/blob/main/LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)](https://huggingface.co/datasets/datalab-to/omni_extract_bench)
[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-omni__extract__bench-yellow)](https://huggingface.co/datasets/datalab-to/omni_extract_bench)

This repository contains a way to benchmark extraction systems using our deterministic field-matching scorer and dataset as well as your own. Concretely, this repository holds:

- the scorer, which works on its own: a prediction its ground truth and its schema as input, the scoring as output;
- the benchmark: run over our dataset on [Hugging Face](https://huggingface.co/datasets/datalab-to/omni_extract_bench), or
  your own; and
- the systems: different registered systems that you can easily evaluate. 

Contents:

1. [How scoring works](#how-scoring-works): what field matching compares, and how;
2. [Install](#install): the scorer alone, or with the adapters and the dataset;
3. [Run the benchmark](#run-the-benchmark): one system on the dataset, from a config;
4. [Grade](#grade): a run again, or predictions of your own;
5. [Score on your own](#score-on-your-own): the scorer on its own, in code or from the command line;
6. [Predict one document](#predict-one-document): one document and its schema through one system;
7. [Tests](#tests); and
8. [License](#license).

[`docs/API.md`](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md) documents every public
function and command in more depth.


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

A predicted address with no matching ground-truth address is a false positive and a ground-truth address with no matching predicted address is a false
negative. Accuracy is the matched addresses over the addresses either document used. The full specification is
[`docs/METRIC_SPEC.md`](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/METRIC_SPEC.md).

## Install

```bash
uv pip install omni-extract-bench                 # the scorer, and grading a dataset folder on disk
uv pip install 'omni-extract-bench[harness]'      # + the adapters, to get predictions
uv pip install 'omni-extract-bench[benchmark]'    # + huggingface_hub, to download our dataset
```

## Run the benchmark

A run is one system on one dataset. It works in three stages:

1. Plan: check the config, the system's keys and the dataset's files before any call;
2. Predict: send each document to the system and save each answer in the run's folder; and
3. Grade: score the saved answers. Grading calls no system, so it can be done again for free.

### A config

A run is one config: a YAML file with every key written out. `oeb init` writes one for a registered system, every option at
its default, and on our dataset:

```bash
oeb init datalab > datalab.yaml
```

```yaml
dataset: hf://datasets/datalab-to/omni_extract_bench   # a folder, or a Hugging Face dataset
manifest: manifest.parquet                             # which of its documents, with which files
scorers:
  field_matching: {}
rows:                    
  sample: 0
  limit: 0
  only: null
timeout: 1800.0          # seconds
workers: 0               # documents in flight at once; 0 for the system's default
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

`oeb run` shows the plan and asks before calling any system.

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


### Run folder shape

```bash
runs/datalab-20261026-185908-1517/     
  config.yaml                         # resolved config; point to this to rerun
  settings.json                       # + launch information (run's record), resume uses this
  records/<id>.json                   # response from system in full (cost, raw response)
  predictions/<id>.json               # prediction portion of resopnse
  field_matching/summary.json         # summary of scores 
  field_matching/scores.jsonl         # scores (line-per-document)
  field_matching/details/<id>.jsonl   # detailed info (verdicts) per document
```


[`docs/API.md`](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md#what-a-run-writes) has more information, including [how to run on your own dataset](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/API.md#your-own-dataset).

### Resume

You can resume a run by giving a path to its folder.

```bash
oeb resume runs/<run>                                         # continue where it stopped
oeb resume runs/<run> --retry-failed                          # also ask again for timeouts, etc.
oeb resume runs/<run> --retry-failed --set timeout=3600       # increase timeout
```


### In Python

```python
from omni_extract_bench import benchmark
from omni_extract_bench.harness.providers.datalab import Datalab

if __name__ == "__main__":
    bench = benchmark.default()
    results = bench.run(Datalab(mode="accurate"), out="runs", sample=20)   
```

Code that runs or grades a benchmark goes under `if __name__ == "__main__":`. Grading scores documents in a pool of processes, and on macOS and Windows each process imports your script again.

## Grade

`oeb run` grades its run when it finishes. To grade a run again, or a folder of your own predictions:

```bash
oeb grade runs/datalab-20261021-160012-3f9a        # a run knows its own dataset
oeb grade mine/ --dataset my-dataset/              # your predictions on your own dataset
```

Grading is fast on most documents, but field matching on a very large one takes long. On our dataset, `long__real_ofac_ssi_full` alone takes about 2 minutes, and all 620 documents take about 7 to 8 minutes of CPU, spread over up to 8 processes. Each document's score is kept in the run folder, so grading again only scores the documents whose prediction or gold changed.

## Score on your own

The scorer works on its own, with no dataset and no network.

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

To see where a prediction went wrong, ask for the verdicts: one per address, with its raw and normalized values on both sides and its result.

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

With `[harness]` installed, each vendor is an adapter class. `harness.system` makes one from a system name and its options and `harness.predict` sends one document under the same rules a run uses (one time budget, retries, failures recorded) and returns its record:

```python
from pathlib import Path

from omni_extract_bench import harness, score
from omni_extract_bench.harness import contract

datalab = harness.system("datalab", {"mode": "accurate"})
record = harness.predict(datalab, contract.Question(Path("invoice.pdf"), schema))
record["output"]    # the extracted object
record["raw"]       # the answer as the system gave it
record["cost"]      # what it cost, its wall time and attempts

score(record["output"], ground_truth, schema)
```

This sends the schema as you give it. A run first applies the benchmark's conventions to the schema (`task.Extract().input`), so use `oeb predict` to send exactly what a run sends. 

```bash
oeb predict --system datalab --doc invoice.pdf --schema schema.json --options '{"mode": "accurate"}'
oeb predict --system datalab --doc invoice.pdf --schema schema.json --options '{"datalab": {"mode": "accurate"}}'
```

## Tests

```bash
uv run --group test --extra benchmark pytest -n auto
```

## License

Apache 2.0, see [`LICENSE`](https://github.com/datalab-to/omni_extract_bench/blob/main/LICENSE). All code here is
Datalab's own.
