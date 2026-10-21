# API

This is the public API. There are three entry points:

- `score`: score one document's prediction against its ground truth;
- `predict`: one document through one vendor; and
- `run`: the benchmark, every document through one system, resumable.

Predicting and running are built on [omni-bench-core](https://github.com/datalab-to/omni-bench-core), which holds
the loop, the run folders and the rules every call and every grading follows. This package adds what is
particular to extraction: the rows (`data.Doc`), the task (`task.Extract`), the vendor adapters
(`harness`) and the scorer (`scorers.field_matching.FieldMatching`, over the functions of `scorers.field_matching`).

The sections:

1. [score](#score): grade one prediction against one ground truth;
2. [predict](#predict): one document, one vendor;
3. [run](#run): the whole corpus through one system, resumable;
4. [providers](#providers): what you can run, what each one takes and needs, and adding your own;
5. [Your own dataset](#your-own-dataset): its folder, its manifest and its gold files;
6. [Your own scorer](#your-own-scorer): grading with a scorer of yours beside field matching; and
7. [What a run writes](#what-a-run-writes): the files, and what's in them.


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

On the command line, `pred`, `gold` and `schema` are `--pred`, `--gold` and `--schema` (each a JSON file),
`order_matters` is `--order-matters`, and `verdicts=True` is `--details`. `absent` has no flag. The result
goes to stdout, so `oeb score ... | jq .accuracy` is one pipe.

The README has a [worked example](../README.md#score) and
[`docs/METRIC_SPEC.md`](./METRIC_SPEC.md) is the full specification.


## predict

A vendor is an adapter class; `harness.system` makes one from a system name and its options, checked.
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

From the command line, the record is printed as JSON, and the command exits 1 when the call failed.
`--options` takes the bare options, as `oeb run` does, or the same keyed by the system's name
(`'{"datalab": {"mode": "accurate"}}'`):

```bash
oeb predict --system datalab --doc invoice.pdf --schema schema.json --options '{"mode": "accurate"}'
```

`oeb predict`, like `adapter.predict` from Python, refuses a missing API key before the call, saying what to set
(`export DATALAB_API_KEY=...`), as a run's plan does. As in
a run, a refused connection is raised with a note on how to start the server, an account that can't pay is a
`PermissionError`, and any other exception is raised as it is, with a note naming the adapter.

The record holds `system`, `options`, `package`, `output` (the extraction, null on a failure), `raw`
(the vendor's answer as received), `sent` (the schema as the vendor was sent it), `error`, `cost`,
`job_id`, `timeout_s` and `captured_at`; [What a run writes](#what-a-run-writes) says what each is. A
benchmark run sends the schema with the benchmark's conventions applied and its own annotations removed
(`task.Extract.input`); call that, or `harness.schema`, to send exactly what a run sends.


## run

```bash
oeb run --system datalab --options '{"mode": "accurate"}' --out runs/ --sample 20
oeb run --system datalab --workers 4             # 4 documents in flight at once; default: the system's own
oeb grade runs/datalab-69059f7c                 # grade again, with no call
oeb grade mine/ --dataset <dataset>             # a folder of your own predictions/<doc_id>.json
oeb grade mine/ --dataset <dataset> --partial   # grade an unfinished folder, counting what's missing as 0
oeb run --system datalab --only ids.txt         # only the documents listed, one id per line
oeb check --dataset <dataset>                   # is it a dataset oeb can run? no call is made
oeb --version
```

A run is one system. To compare systems, run one command per system, with the same choice of documents
(`--sample`, `--limit` or `--only`), so they answer the same rows:

```bash
oeb run --system datalab --options '{"mode": "accurate"}' --sample 20 --out runs/
oeb run --system reducto --sample 20 --out runs/
```

`oeb run` prints the run's grading as one JSON object, whose `"run"` is the run folder's name.

`--options` is the system's options as an object (`'{"mode": "accurate"}'`), or the same keyed by the system's
name (`'{"datalab": {"mode": "accurate"}}'`). Options keyed by another system are refused.

`oeb grade` grades a run against the dataset and manifest its settings.json names. A folder of your own
predictions has no settings.json, so without `--dataset` it is graded against the public dataset
(`hf://datasets/datalab-to/omni_extract_bench`) and its `manifest.parquet`.

`oeb check` reads the manifest, checks every document's PDF exists and every gold reads, and prints
`{"dataset", "manifest", "rows"}`; it raises a `ValueError` naming the first problem (every missing PDF at once).

In code, `benchmark.default(dataset, manifest)` is Omni Extract Bench as an omni-bench-core `Benchmark`:

```python
from omni_extract_bench import benchmark
from omni_extract_bench.harness.providers.datalab import Datalab
from omni_extract_bench.harness.providers.reducto import Reducto

if __name__ == "__main__":
    b = benchmark.default()
    for system in (Datalab(mode="accurate"), Reducto()):     # one run per system, on the same documents
        g = b.run(system, out="runs", sample=20)
        print(g["run"], g["scorers"]["field_matching"]["headline"])
    b.grade("mine/", partial=True)
```

`run` takes one adapter instance; `harness.system("datalab", {"mode": "accurate"})` makes the same one from a
system name. It returns the grading: `{"run", "cost", "scorers"}`.

Code that runs or grades a benchmark goes under `if __name__ == "__main__":`, because grading scores
documents in a pool of processes, and on macOS and Windows each process imports your script again.

`run` has three steps:

1. plan: checks the options, the API key, every document and every gold before any call;
2. predict: calls the system on every document the run has no record of; and
3. grade: grades the run, when it is finished.

A run folder answers one exact question: the system and its options, the dataset, the manifest and which
documents (`--sample`, `--limit`, `--only`). Every launch is a new run in a new folder, named after the
question and a run id. `--resume <its folder name>` (`resume=` in Python) continues a run and calls nothing
already answered, and refuses a run of another question. A `--limit` or `--sample` that covers every
document is the same question as neither.

The timeout isn't part of the question. Resuming a run with a longer `--timeout` and
`--retry-failed` gives the documents that timed out more time, in the same folder. Each record keeps the
`timeout_s` it was run with, and settings.json holds the latest.

A run is graded over its own documents, and only when it is finished: every document has an answer or a
recorded failure. A document with no usable answer counts as 0. A run that stops part way (a key the system
refuses, an account out of credits, a server of yours that isn't running, a bug in an adapter, an interrupt)
is not graded: `oeb run` raises the exception that stopped it, with its traceback, and resuming the run
(`--resume`) finishes it.
`oeb grade --partial` (`partial=True` in Python) grades an unfinished run anyway, counting its unfinished
documents as 0; its summary counts them under `partial`, and names a few.

Grading is fast on most documents, but field matching on a very large one takes long, because pairing the
rows of large arrays is slow. On the public dataset, `long__real_ofac_ssi_full` takes about 2 minutes on its
own, and all 620 documents take about 7 to 8 minutes of CPU, spread over up to 8 processes. Each document's
score is kept (`field_matching/cache/`), so grading again only scores the documents whose prediction or gold
changed (or all of them, after a change to the scorer's code).

What happens in each case is the table under [The rules](https://github.com/datalab-to/omni-bench-core/blob/main/OVERVIEW.md#the-rules) in
omni-bench-core's overview.

`--dataset` is a folder or `hf://datasets/<org>/<name>[@<revision>]` (ours by default, about 856 MB,
downloaded whole once; set `HF_TOKEN` to download faster), and `--manifest` a manifest inside it, such as
`manifests/sample40-predictive-across-vendors-2026-09-22.parquet`. omni-bench-core's `Benchmark` docstring
has the rest: `plan`, `predict`, `grade`, `reread`.


## providers

```bash
oeb systems                 # every name
oeb systems datalab         # what --options takes for it, its defaults, and the key it needs
```

A system is a vendor (`datalab`, `reducto`, `extend`, `llamaextract`, `mistral`, `azure-cu`) or any
OpenRouter model id (`openai/gpt-5.6-sol`), which the single-shot LLM adapter runs with that model. Each
needs its API key in the environment:

| system | environment variable |
| --- | --- |
| `datalab` | `DATALAB_API_KEY` |
| `reducto` | `REDUCTO_API_KEY` |
| `extend` | `EXTEND_API_KEY`, and `EXTEND_WORKSPACE_ID` when the key reaches several workspaces |
| `llamaextract` | `LLAMA_CLOUD_API_KEY` (or its older name, `LLAMAPARSE_API_KEY`) |
| `mistral` | `MISTRAL_API_KEY` |
| `azure-cu` | `AZURE_CU_KEY`, and the `endpoint` option: `--options '{"endpoint": "https://<resource>.cognitiveservices.azure.com"}'` |
| an OpenRouter model id | `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) |

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
    keys: ClassVar[tuple[str, ...]] = ("MINE_API_KEY",)   # checked before any call
    mode: str = "best"

    def extract(self, pdf, schema, *, timeout):          # the vendor's answer, as received
        return adapter.Call(raw={"extraction": call_my_vendor(pdf, schema, self.mode)})
```

`prepare_schema` reshapes the schema into the vendor's dialect, and `read` reads the extraction out of the
raw answer (by default `raw["extraction"]`). Pass an instance to `benchmark.default().run(Mine(), out="runs")`,
or declare the class under the entry-point group `omni_extract_bench.adapters` so `oeb run --system mine`
finds it.

`extract` runs in several threads at once, one per document in flight (the class's `workers`, 5 unless you
set it, or `--workers`), so anything it shares between calls, such as a client or a cache, must be safe to
use from several threads.

One rule decides what an exception does, in `extract`, `prepare_schema`, `read` and a scorer's
`prediction` alike: a document counts 0 only when code raises `errors.SystemFailed` on purpose. Any other
exception stops the run and is raised as it is, with a note naming the adapter and the document; resuming the
run after the fix costs nothing for the documents already answered.

In `extract`:

- raise `errors.SystemFailed(message, status=...)`, or one of its kinds, for a vendor failure. It is recorded
  with its traceback and counts 0; a 429 or 5xx is retried first, and a 401 or 402 stops the run as a
  `PermissionError`;
- read the fields you need while calling (a job id, a status, a cost) inside `adapter.reading("what it is")`,
  which turns a `ValueError`, `KeyError`, `IndexError`, `TypeError` or `AttributeError` into a
  `SystemFailed`;
- for a request that got no answer, `raise contract.unreachable("what was sent", exc) from exc`: a
  `SystemTimeout` or a `SystemUnreachable`, which `--retry-failed` tries again. A refused connection in its
  chain stops the run instead, since nothing listens at the address; and
- use a native exception, whose message says what to fix, for anything that isn't the vendor's failure (a
  `ValueError` for a wrong option).

`prepare_schema` raises `contract.DialectError`, a `SystemFailed`, for a schema the vendor's dialect can't
express: the document is recorded as failed, with the schema, and no call is made.

`read` takes the raw answer and returns the extraction. Raise `errors.SystemFailed` for an answer with no
usable extraction (`contract.extraction(x)` does, for anything but a non-empty JSON object); the document is
recorded as `Unreadable` and keeps its raw answer. Anything else is a bug: the answer is saved unread, the
run stops, and resuming the run after the fix reads it with no call.

For a vendor whose answer holds the extraction as a JSON string under `result`:

```python
import json

from omni_bench_core import adapter, errors


@dataclasses.dataclass(frozen=True)
class Stringly(Mine):
    def read(self, raw, q):
        if not isinstance(raw, dict) or "result" not in raw:
            raise errors.SystemFailed(f"the answer has no result: {json.dumps(raw)[:300]}")
        with adapter.reading("result, a JSON string"):
            x = json.loads(raw["result"])
        return contract.extraction(x)
```

omni-bench-core's [overview](https://github.com/datalab-to/omni-bench-core/blob/main/OVERVIEW.md#adapter)
has the full rules, and [The rules](https://github.com/datalab-to/omni-bench-core/blob/main/OVERVIEW.md#the-rules)
has the table of every case.


## Your own dataset

A dataset is a folder (or a Hugging Face dataset with the same files). `help(omni_extract_bench.data)` has
the same in short:

```
my-dataset/
  manifest.parquet          one row per document
  manifests/*.parquet       other manifests, if any, such as a subset of the same documents
  pdfs/invoice-1.pdf        the documents that are sent
  gold/invoice-1.json       each document's gold
```

The manifest is a parquet file with one row per document and these columns:

| column | holds |
| --- | --- |
| `doc_id` | the document's id, a string. It names the document's files (`predictions/<doc_id>.json`), so it must be a file name (no `/`), and no two ids may be the same ignoring case |
| `suite` | optional: the kind of document, a string such as `"invoices"`. Field matching's summary is broken down by suite. A missing column, or an empty or null value, is `"other"` |
| `doc_path` | the PDF's path, relative to the dataset folder, such as `pdfs/invoice-1.pdf`. Several rows may share one PDF |
| `gt_path` | the gold file's path, relative to the dataset folder, named after its row: `<doc_id>.json`, such as `gold/invoice-1.json` |
| `schema` | the JSON Schema to extract against, as a JSON string (`json.dumps(schema)`), not a nested object. It must be a non-empty JSON object |

Any other column is kept, unread, in each row's `extra` (`data.Doc.extra`), for scorers of your own. The
folder names `pdfs/` and `gold/` are only a convention: any paths inside the dataset work.

A gold file is one JSON object: the extraction a perfect system would return for that document and its
schema. For example, for a schema with `invoice_id` and `total_due`:

```json
{"invoice_id": "INV-4417", "total_due": 1240.0}
```

A run sends each schema with the benchmark's conventions written into its field descriptions
(`task.Extract`, `overlay=True`), and with every key named `evaluation_config` or `default` removed, at any
depth; these are the public dataset's notes for graders, and are never sent.

To write a manifest:

```python
import json

import pyarrow as pa
import pyarrow.parquet as pq

rows = [{"doc_id": "invoice-1", "suite": "invoices", "doc_path": "pdfs/invoice-1.pdf",
         "gt_path": "gold/invoice-1.json", "schema": json.dumps(schema)}]
pq.write_table(pa.Table.from_pylist(rows), "my-dataset/manifest.parquet")
```

Then check it, run on it, and grade it:

```bash
oeb check --dataset my-dataset/                                      # no call; names the first problem
oeb run --system datalab --dataset my-dataset/ --out runs/            # predict, then grade
oeb grade runs/datalab-1a2b3c4d                                      # grade again; the run knows its dataset
oeb grade mine/ --dataset my-dataset/                                # your own predictions/<doc_id>.json
```

`oeb check` prints `{"dataset", "manifest", "rows"}`. Another manifest inside the folder is
`--manifest manifests/<name>.parquet`. In Python, the same dataset is `benchmark.default("my-dataset/")`.


## Your own scorer

Field matching is the benchmark's scorer, but a benchmark can have several. To grade with one of your own
beside it, build the benchmark directly. A scorer is a frozen dataclass with `gold(row)`, which reads a
row's gold, `score(pred, gold)`, which returns a `scorer.Score`, and `summarize(lines)`, which turns the scored
documents' lines into a summary with a `headline` (0 to 1) and `headline_is`:

```python
import dataclasses
import json

from omni_bench_core import benchmark, scorer
from omni_extract_bench import data, task
from omni_extract_bench.harness.providers.datalab import Datalab
from omni_extract_bench.scorers import field_matching


@dataclasses.dataclass(frozen=True)
class KeysFound(scorer.Scorer):
    """The share of the gold's top-level keys that the prediction has."""

    def gold(self, row):
        return json.loads(row.gt_path.read_text())

    def score(self, pred, gold):
        return scorer.Score({"keys_found": sum(k in pred for k in gold) / len(gold) if gold else 1.0})

    def summarize(self, lines):
        found = sum(x["keys_found"] for x in lines) / len(lines)
        return {"headline": found, "headline_is": "keys_found"}


if __name__ == "__main__":
    b = benchmark.Benchmark(task.Extract(), data.DEFAULT_DATASET,
                            [field_matching.FieldMatching(), KeysFound()], data.MANIFEST)
    b.run(Datalab(mode="accurate"), out="runs", sample=20)
    b.grade("runs/datalab-1a2b3c4d")          # a run you already have: graded again, with no call
```

A scorer can also write `prediction(output)`, which reads what it needs of a document's prediction. Raise
`errors.SystemFailed` there for a prediction it can't use: that document counts 0, and its line in
scores.jsonl has the message and `traceback`. Field matching does this for a prediction that isn't a JSON
object. Anything else `prediction` raises is a bug, and stops grading.

The scorers are not part of a run's question, so the runs you already have keep their folders, and `grade`
adds a `keys_found/` folder beside `field_matching/`. The `oeb` command grades with field matching only.
omni-bench-core's [overview](https://github.com/datalab-to/omni-bench-core/blob/main/OVERVIEW.md#scorer)
explains scorers, and its [toy task](https://github.com/datalab-to/omni-bench-core/blob/main/tests/test_core.py)
is a whole benchmark with its own scorer.


## What a run writes

omni-bench-core's run folder, one per question (system, options, dataset, manifest, task and documents):

```
runs/datalab-69059f7c/
  settings.json                    the question, and the timeout
  records/<doc_id>.json            each document's record, as `predict` returns it
  predictions/<doc_id>.json        each document's extraction alone; all grading reads
  field_matching/
    scores.jsonl                   one line per document: id, status, row (its suite), and score's numbers
    summary.json                   the summary: the headline, counts, and under scored the metrics, overall
                                   and per suite
    details/<doc_id>.jsonl         one verdict per address (`--no-details` skips them)
    cache/<doc_id>.json            each document's kept score, reused while its prediction and gold don't change
```

The folder is named for the system's label and a digest of the question, so two configurations, or two
choices of documents, never share one.

settings.json is written before the first call, and holds:

- `run`: the folder's name;
- `system`: the system's name, such as `datalab`;
- `options`: every option of the system, defaults included;
- `dataset`: the dataset, as an `hf://datasets/<org>/<name>` address (without its revision) or a folder's
  full path;
- `manifest`: the manifest's path inside the dataset;
- `task`: the task's name and options, `{"name": "extract", "options": {"overlay": true}}`;
- `rows`: which documents, as chosen by `--only`, `--sample` and `--limit`, such as
  `{"sample": 20}`. `{}` means every document of the manifest;
- `commit`: the commit the dataset was downloaded at, for an `hf://` dataset only; and
- `timeout_s`: the seconds each document may take, as of the latest run of this folder. It isn't part of
  the question (see [run](#run)).

A record, `records/<doc_id>.json`, holds:

- `system` and `options`: the system and its options, as in settings.json;
- `package`: the adapter's package and its version, such as `"omni-extract-bench 1.0.0"`, so answers from
  two versions of an adapter can be told apart; null for an adapter of your own outside any package;
- `output`: the extraction, what predictions/ holds; null on a failure;
- `raw`: the vendor's answer as received, kept so that a fixed `read` can read it again with no call
  (`Benchmark.reread`);
- `sent`: the schema as the vendor was sent it, kept on a failure too;
- `error`: null, or the failure's `type` (`SystemFailed`, or a kind of it: `SystemTimeout`,
  `SystemUnreachable`, `Unreadable`, `DialectError`), `status`, `transient` (whether a retry could help),
  `message` and `traceback`. The traceback is of where the failure was raised, and of what it was raised from
  (an answer read inside `adapter.reading`), so it shows which line decided the vendor failed, and whether the
  vendor's answer changed or the reading is wrong;
- `cost`: `usd` and `credits` as the vendor reported them (a failed call's too, when it was billed),
  `source` (the response field the cost was read from), `tokens_in`, `tokens_out`, `wall_s` (the seconds
  the document took) and `attempts`;
- `job_id`: the vendor's id for the job, when it made one;
- `timeout_s`: the seconds this document was given;
- `captured_at`: when the answer was recorded; and
- `unread`: true when `read` raised a bug on this answer. Such a record has its `raw` answer, no `output` and no
  `error`; the document isn't finished, and resuming the run reads it with no call.

A summary holds what omni-bench-core gives every summary (`scorer`, `options`, `run`, `headline`,
`headline_is`, `rows`, `failed`, `coverage`, and `partial` when graded with `--partial`), and under `scored`
field matching's own numbers over the scored documents only: `headline` (their accuracy), `rows` (how many
were scored), `accuracy`, `precision`, `recall`, one `<error>_rate` per kind of error, and `per_suite`, the
same numbers for each suite. The top `headline` is `scored.headline` times coverage, so a failed document
counts as 0:

```json
{"headline": 0.5, "headline_is": "scored.headline × coverage", "rows": 2, "failed": 1, "coverage": 0.5,
 "scored": {"headline": 1.0, "headline_is": "accuracy", "rows": 1, "accuracy": 1.0, "precision": 1.0, ...,
            "per_suite": {"s": {"rows": 1, "accuracy": 1.0, ...}}}}
```

`oeb run` and `oeb grade` print each run's cost (from its records) beside its summary.
