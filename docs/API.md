# API

This is the public API. There are three entry points:

- `score`: score one document's prediction against its ground truth;
- `predict`: one document through one vendor; and
- `run`: the benchmark, every document through every vendor, resumable.

Predicting and running are built on [omni-bench-core](https://github.com/datalab-to/omni-bench), which holds
the loop, the run folders and the rules every call and every grading follows. This package adds what is
particular to extraction: the rows (`data.Doc`), the task (`task.Extract`), the vendor adapters
(`harness`) and the scorer (`scorers.field_matching.FieldMatching`, over the functions of `scorers.field_matching`).

1. **[score](#score)**: grade one prediction against one ground truth.
2. **[predict](#predict)**: one document, one vendor.
3. **[run](#run)**: the whole corpus, every vendor, resumable.
4. **[providers](#providers)**: what you can run, what each one takes, and adding your own.
5. **[What a run writes](#what-a-run-writes)**: the files, and what's in them.


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
  absent=values.states_nothing,  # which values get no address; default: null, "", [], {}
)                     # values: from omni_extract_bench.scorers.field_matching import values
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

A vendor is an adapter class; `harness.system` makes one from a provider name and its options, checked.
omni-bench-core's `adapter.predict` sends one document under the same rules a run uses: one time budget per
document, transient failures retried, other failures recorded as the document's answer.

```python
from pathlib import Path

from omni_bench_core import adapter
from omni_extract_bench import harness
from omni_extract_bench.harness import contract

datalab = harness.system("datalab", {"mode": "accurate"})
record = adapter.predict(datalab, contract.Question(Path("invoice.pdf"), schema), timeout=1800)
```

From the cli, the record is printed as JSON, and the command exits 1 when the call failed:

```bash
oeb predict --provider datalab --doc invoice.pdf --schema schema.json --options '{"mode": "accurate"}'
```

The record holds `output` (the extraction, None on a failure), `raw` (the vendor's answer as received),
`sent` (the schema as the vendor was sent it, kept on a failure too), `error`, `cost` (a failed call's too,
when the vendor billed it), `job_id`, `settings` and `timeout_s`. A
benchmark run sends the schema with the benchmark's conventions applied and its own annotations removed
(`task.Extract.input`); call that, or `harness.schema`, to send exactly what a run sends.


## run

```bash
oeb run --providers datalab reducto --out runs/ --sample 20
oeb run --providers datalab --options '{"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]}'
oeb grade runs/datalab-69059f7c                 # grade again, with no call
oeb grade mine/ --dataset <dataset>             # a folder of your own predictions/<doc_id>.json
oeb run --providers datalab --docs ids.txt      # only the documents listed, one id per line
oeb check --dataset <dataset>                   # is it a dataset oeb can run? no call is made
```

`oeb check` reads the manifest, checks every document's PDF exists and every gold reads, and prints
`{"dataset", "manifest", "rows", "suites"}`; it exits 1 naming the first problem (every missing PDF at once).

In code, `benchmark.default(dataset, manifest)` is Omni Extract Bench as an omni-bench-core `Benchmark`:

```python
from omni_extract_bench import benchmark, harness

oeb = benchmark.default()
oeb.run([harness.system("datalab", {"mode": "accurate"})], out="runs", sample=20)
oeb.grade("mine/")
```

`run` plans (checks every option, folder and gold before any call), predicts every document a run has no
record of, then grades each run. Running it again calls nothing already answered. A document never
attempted is pending, and the headline is null while any document is pending; a document with no usable
answer counts as 0. `--dataset` is a folder or `hf://datasets/<org>/<name>[@<revision>]` (ours by default),
and `--manifest` a manifest inside it, such as `manifests/sample40-predictive-across-vendors-2026-09-22.parquet`.
omni-bench-core's `Benchmark` docstring has the rest: `plan`, `predict`, `grade`, `reread`.


## providers

```bash
oeb providers                 # every name
oeb providers datalab         # what --options takes for it, and its defaults
```

A provider is a vendor (`datalab`, `reducto`, `extend`, `llamaextract`, `mistral`, `azure-cu`) or any
OpenRouter model id (`openai/gpt-5.6-sol`), which the single-shot LLM adapter runs with that model.

### Your own adapter

Subclass `harness.contract.Extractor`, a frozen dataclass whose fields are the vendor's options:

```python
import dataclasses
from typing import ClassVar

from omni_bench_core import adapter
from omni_extract_bench.harness import contract


@dataclasses.dataclass(frozen=True)
class Mine(contract.Extractor):
    name: ClassVar[str] = "mine"
    mode: str = "best"

    def extract(self, pdf, schema, *, timeout):          # the vendor's answer, as received
        return adapter.Call(raw={"extraction": call_my_vendor(pdf, schema, self.mode)})
```

`prepare_schema` reshapes the schema into the vendor's dialect, and `read` reads the extraction out of the
raw answer (by default `raw["extraction"]`). Pass an instance to `benchmark.default().run([...])`, or
declare the class under the entry-point group `omni_extract_bench.adapters` so `oeb run --providers mine`
finds it.


## What a run writes

omni-bench-core's run folder, one per vendor and options:

```
runs/datalab-69059f7c/
  settings.json                    what the run asked, its dataset, commit and manifest
  records/<doc_id>.json            each document's record, as `predict` returns it
  predictions/<doc_id>.json        each document's extraction alone; all grading reads
  field_matching/
    scores.jsonl                   one line per graded document: id, suite, status, and score's numbers
    summary.json                   the summary: the headline, counts, the metrics, and per suite
    details/<doc_id>.jsonl         one verdict per address (`--no-verdicts` skips them)
```

The folder is named for the vendor and a digest of every option, so two configurations never share one.
`oeb run` and `oeb grade` print each run's cost (from its records) beside its summary.
