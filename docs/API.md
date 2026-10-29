# API

The public functions and commands of omni-extract-bench. It is built on
[omni-bench-core](https://github.com/datalab-to/omni-bench-core), which runs the loop: planning, predicting,
resuming and grading. This package adds what is particular to extraction: the rows (`data.Doc`), the task
(`task.Extract`), the adapters (`harness`) and the scorer (`scorers.field_matching`).

Contents:

1. [Run](#run): one system on a dataset, from a config;
2. [Resume](#resume): a run that stopped part way;
3. [Grade](#grade): a run again, or predictions of your own;
4. [Score](#score): one prediction against its ground truth;
5. [Predict](#predict): one document through one system;
6. [Adapters](#adapters): the systems, and your own;
7. [Your own dataset](#your-own-dataset): its folder, its manifest and its gold files;
8. [Your own scorer](#your-own-scorer): grading with a scorer of yours beside field matching; and
9. [What a run writes](#what-a-run-writes): the files, and what's in them.

## Run

```bash
oeb init datalab > datalab.yaml                                   # a config at the system's defaults, on our dataset
oeb run --config datalab.yaml --set rows.sample=20 --out runs/    # predict, then grade: a new run
oeb run --config datalab.yaml --set 'rows.only=[invoice-1, invoice-2]'
oeb run --config datalab.yaml --set dataset=my-dataset/           # another dataset
oeb run --config runs/<run>/config.yaml                           # the same question again, as a new run
```

A config has every key written out, and nothing has a default:

- `dataset`: a folder, or `hf://datasets/<org>/<name>[@<revision>]`;
- `manifest`: a manifest inside the dataset;
- `scorers`: `field_matching: {}`;
- `rows`: `sample` N evenly spaced, `limit` the first N (0 for every document), or `only` these ids;
- `timeout`: seconds each document may take;
- `workers`: documents in flight at once, 0 for the system's default; and
- `system`: one system and every option it has.

`--set path=value` changes one value the file has, read as YAML. A path the file hasn't is refused, with the
nearest one it has. A missing key, an unknown one or a value of the wrong kind is refused at its key.

`oeb run` shows the plan and asks before calling any system; `-y`, or no terminal, skips the question. It prints
the grading as one JSON object, `{"run", "cost", "scorers"}`, whose `run` is the run folder's name.
`--no-details` skips the per-address breakdown.

A run works in three stages:

1. plan: checks the options, the keys, every document and every gold before any call;
2. predict: calls the system on every document the run has no record of; and
3. grade: grades the run, when it is finished.

In Python:

```python
from omni_extract_bench import benchmark
from omni_extract_bench.harness.providers.datalab import Datalab

if __name__ == "__main__":
    b = benchmark.default()                                  # or benchmark.default("my-dataset/", manifest)
    g = b.run(Datalab(mode="accurate"), out="runs", sample=20)
```

`run` takes one adapter. `harness.system("datalab", {"mode": "accurate"})` makes the same one from a system
name. It takes `sample`, `limit` and `only`, `timeout` and `workers`, as a config does.

Code that runs or grades a benchmark goes under `if __name__ == "__main__":`. Grading scores documents in a pool
of processes, and on macOS and Windows each process imports your script again.

## Resume

```bash
oeb resume runs/<run>                                       # continue where it stopped
oeb resume runs/<run> --retry-failed                        # also ask again for timeouts, outages, 429s and 5xx
oeb resume runs/<run> --retry-failed --set timeout=3600     # more time per document
```

A run folder answers one question: the system and its options, the dataset, the manifest, the task and `rows`.
A resume reads the run's own `config.yaml`, and asks only for the documents with no record. `--set` may change
`timeout` and `workers`, which aren't part of the question; a change to anything else is refused, naming the keys
that differ.

A failed document has a record, so it is asked again only with `--retry-failed`, and only when its failure may
not recur: a timeout, a system that couldn't be reached, a 429 or a 5xx. A refusal or an answer that couldn't
be read stays failed.

In Python, `b.run(system, out="runs", resume="<run>", retry_failed=True)`.

## Grade

```bash
oeb grade runs/<run>                                # the dataset and manifest its settings.json names
oeb grade mine/ --dataset my-dataset/               # your own predictions/<doc_id>.json
oeb grade runs/<run> --partial                      # an unfinished run: what's missing counts as failed
oeb check --dataset my-dataset/                     # is it a dataset oeb can run? no call
```

```python
if __name__ == "__main__":
    benchmark.default().grade("runs/<run>")
    benchmark.default("my-dataset/").grade("mine/", partial=True)
```

A run is graded only when it is finished: every document has an answer or a recorded failure. Otherwise grading
refuses and names the documents. `--partial` counts them as 0, and the summary counts them under `partial`.

A folder of your own predictions has no settings.json, so without `--dataset` it is graded against our dataset.
`--manifest` picks another manifest inside the dataset.

`oeb check` reads the manifest, checks that every PDF exists and every gold reads, and prints
`{"dataset", "manifest", "rows"}`. A problem is a `ValueError` naming it.

Grading is fast on most documents, but field matching on a very large one takes long. On our dataset, all 620
documents take about 7 to 8 minutes of CPU, spread over up to 8 processes. Each document's score is kept in
`field_matching/cache/`, so grading again only scores the documents whose prediction, gold or scorer code changed.

## Score

```python
from omni_extract_bench import score

score(
  pred,                          # the prediction, a dict
  gold,                          # the ground truth, a dict
  schema,                        # the JSON Schema the prediction was made against
  order_matters=(),              # arrays where position is part of the answer, e.g. ["steps"]; True for all
  verdicts=False,                # also return one verdict per address
  absent=values.states_nothing,  # values that get no address: null, "", [], {}
)
```

```bash
oeb score --pred pred.json --gold gold.json --schema schema.json [--details] [--order-matters [ARRAY ...]]
```

`--details` is `verdicts=True`; `--order-matters` given bare is every array. `absent` has no flag. The result goes
to stdout, so `oeb score ... | jq .accuracy` is one pipe.

`absent=lambda v: False` makes `null`, `""`, `[]` and `{}` values the prediction must reproduce. Scores under
different `absent`s aren't comparable, since the number of addresses changes. `values` is
`omni_extract_bench.scorers.field_matching.values`.

[`METRIC_SPEC.md`](./METRIC_SPEC.md) is the full specification.

## Predict

```python
from pathlib import Path

from omni_extract_bench import harness
from omni_extract_bench.harness import contract

datalab = harness.system("datalab", {"mode": "accurate"})
record = harness.predict(datalab, contract.Question(Path("invoice.pdf"), schema), timeout=1800)
```

```bash
oeb predict --system datalab --doc invoice.pdf --schema schema.json --options '{"mode": "accurate"}' [--timeout 1800]
```

`harness.predict` sends one document under the rules a run uses: one time budget, transient failures retried,
other failures recorded as the document's answer. It returns the record ([What a run writes](#what-a-run-writes)).
A missing key is refused before the call.

`harness.predict` sends the schema as you give it. A run first applies the benchmark's conventions to it
(`task.Extract().input`), and `oeb predict` does the same, so it sends exactly what a run sends. `oeb predict`
prints the record as JSON and exits 1 when the call failed. `--options` takes the bare options, or the same
keyed by the system's name.

## Adapters

```bash
oeb systems              # every system
oeb systems datalab      # one system's options, their defaults, and the key it needs
```

A system is a vendor (`datalab`, `reducto`, `extend`, `llamaextract`, `mistral`, `azure-cu`) or any OpenRouter
model id (`openai/gpt-5.6-sol`), which the single-shot LLM adapter runs. Each reads its key from the environment,
and a missing key is refused before any call.

### Your own adapter

Subclass `harness.contract.Extractor` as a frozen dataclass whose fields are its options:

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

    def extract(self, pdf, schema, *, timeout):          # the answer, as received
        return adapter.Call(raw={"extraction": call_my_vendor(pdf, schema, self.mode)})
```

It has three methods:

- `extract(pdf, schema, *, timeout)`: sends the document and returns the answer as received;
- `prepare_schema(schema)`: optional, the schema in the vendor's dialect; and
- `read(raw, q)`: optional, the extraction in the answer; by default `raw["extraction"]`.

Give an instance to `benchmark.default().run(Mine(), out="runs")`, or declare the class under the entry-point
group `omni_extract_bench.adapters` so a config's `system: {mine: {...}}` finds it (`oeb init mine` writes one).
`extract` runs on several threads at once (`workers`, 5 by default), so anything it shares between calls must be
thread-safe.

A document fails only when code raises `errors.SystemFailed` on purpose. Any other exception stops the run,
and resuming after the fix costs nothing for the documents already answered. So:

- in `extract`, raise `errors.SystemFailed(message, status=...)` for a vendor failure. A 429 or 5xx is retried
  first, and a 401 or 402 stops the run as a `PermissionError`;
- read the vendor's fields (a job id, a status) inside `adapter.reading("what it is")`, which turns a
  `ValueError`, `KeyError`, `IndexError`, `TypeError` or `AttributeError` into a `SystemFailed`;
- for a request that got no answer, `raise contract.unreachable("what was sent", exc) from exc`: a
  `SystemTimeout` or a `SystemUnreachable`, which `--retry-failed` asks again. A refused connection stops the
  run instead;
- in `prepare_schema`, raise `contract.DialectError` for a schema the vendor can't express: the document fails
  with no call;
- in `read`, raise `errors.SystemFailed` for an answer with no usable extraction (`contract.extraction(x)` does,
  for anything but a non-empty JSON object). The document is recorded as `Unreadable` and keeps its raw answer.
  Anything else is a bug: the answer is kept unread, and resuming after the fix reads it with no call; and
- use a native exception for anything that isn't the vendor's failure, such as a `ValueError` for a wrong option.

omni-bench-core's [overview](https://github.com/datalab-to/omni-bench-core/blob/main/OVERVIEW.md#adapter) has
the full rules.

## Your own dataset

```
my-dataset/
  manifest.parquet          one row per document
  manifests/*.parquet       other manifests, if any, such as a subset
  pdfs/invoice-1.pdf        the documents that are sent
  gold/invoice-1.json       each document's ground truth
```

| column | holds |
| --- | --- |
| `doc_id` | the document's id, a string. It names its files (`predictions/<doc_id>.json`), so it has no `/`, and no two ids are the same ignoring case |
| `suite` | optional: the kind of document, such as `"invoices"`. Field matching's summary is broken down by it. A missing column, or an empty or null value, is `"other"` |
| `doc_path` | the PDF's path, relative to the dataset folder. Several rows may share one PDF |
| `gt_path` | the gold file's path, relative to the dataset folder, named `<doc_id>.json` |
| `schema` | the JSON Schema, as a JSON string (`json.dumps(schema)`), a non-empty object |

Any other column is kept, unread, in each row's `extra`, for scorers of your own. A gold file is one JSON object,
the extraction a perfect system would return:

```json
{"invoice_id": "INV-4417", "total_due": 1240.0}
```

A run sends each schema with the benchmark's conventions written into its field descriptions, and with every key
named `evaluation_config` or `default` removed, at any depth: these are notes for graders, never sent.

To write a manifest, then check it and run on it:

```python
import json

import pyarrow as pa
import pyarrow.parquet as pq

rows = [{"doc_id": "invoice-1", "suite": "invoices", "doc_path": "pdfs/invoice-1.pdf",
         "gt_path": "gold/invoice-1.json", "schema": json.dumps(schema)}]
pq.write_table(pa.Table.from_pylist(rows), "my-dataset/manifest.parquet")
```

```bash
oeb check --dataset my-dataset/
oeb run --config datalab.yaml --set dataset=my-dataset/ --out runs/
```

## Your own scorer

A scorer is a frozen dataclass with four methods:

- `gold(row)`: reads the row's gold;
- `score(pred, gold)`: returns a `scorer.Score`;
- `score_failure(gold)`: a failed document's `scorer.Score`, the worst it can get; and
- `summarize(lines)`: every document's line into a summary with `headline` (0 to 1) and `headline_is`.

To grade with one beside field matching, build the benchmark directly:

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

    def score_failure(self, gold):
        return scorer.Score({"keys_found": 0.0})

    def summarize(self, lines):
        return {"headline": sum(x["keys_found"] for x in lines) / len(lines), "headline_is": "keys_found"}


if __name__ == "__main__":
    b = benchmark.Benchmark(task.Extract(), data.DEFAULT_DATASET,
                            [field_matching.FieldMatching(), KeysFound()], data.MANIFEST)
    b.grade("runs/<run>")          # a run you already have: graded again, with no call
```

A scorer may also write `prediction(output)`, which reads what it needs of a prediction. Raise
`errors.SystemFailed` there for a prediction it can't use: that document fails, and `score_failure` scores it. Field matching does this for a
prediction that isn't a JSON object.

Scorers aren't part of a run's question, so `grade` adds a `keys_found/` folder beside `field_matching/` in the
runs you already have. The `oeb` command grades with field matching only. omni-bench-core's
[overview](https://github.com/datalab-to/omni-bench-core/blob/main/OVERVIEW.md#scorer) explains scorers.

## What a run writes

```
runs/datalab-20261021-160012-3f9a/       <system>-<YYYYMMDD-HHMMSS>-<4 hex>
  config.yaml                            the config as run, every --set applied and every option written out
  settings.json                          the question it answers, and the launch
  records/<doc_id>.json                  what the system returned for each document
  predictions/<doc_id>.json              each document's extraction alone; all grading reads
  field_matching/
    summary.json                         the grade
    scores.jsonl                         one line per document
    details/<doc_id>.jsonl               one verdict per address, each with its doc_id and suite
    cache/<doc_id>.json                  each document's kept score
```

`settings.json` is written before the first call. A resume is checked against it, and you don't edit it. It holds:

- `run`: the folder's name;
- `system` and `options`: the system's name and every option, defaults included;
- `dataset`: an `hf://datasets/<org>/<name>` address (without its revision) or a folder's full path;
- `manifest`: the manifest's path inside the dataset;
- `task`: `"extract"`;
- `rows`: which documents, such as `{"sample": 20}`; `{}` is every document;
- `dataset_commit`: the commit an `hf://` dataset was downloaded at;
- `timeout_s`: the seconds each document may take, as of the latest launch; and
- `started` and `launch`: when the run started, and the launcher's own record of it.

A record, `records/<doc_id>.json`, holds:

- `system`, `options` and `package`: the system, its options, and the adapter's package and version;
- `output`: the extraction, what predictions/ holds; null on a failure;
- `raw`: the answer as received, so a fixed `read` can read it again with no call;
- `sent`: the schema as the vendor was sent it;
- `error`: null, or the failure's `type` (`SystemFailed`, `SystemTimeout`, `SystemUnreachable`, `Unreadable` or
  `DialectError`), `status`, `transient`, `message` and `traceback`;
- `cost`: `usd`, `credits`, `source`, `tokens_in`, `tokens_out`, `wall_s` and `attempts`;
- `job_id`, `timeout_s` and `captured_at`; and
- `unread`: true when `read` raised a bug on this answer. The document isn't finished, and resuming reads it
  with no call.

A summary holds what omni-bench-core gives every summary (`scorer`, `options`, `run`, `rows`, `failed` by cause,
`coverage`, the share of documents scored, and `partial` when graded with `--partial`), then field matching's
numbers over every document: `headline` (their accuracy), `accuracy`, `precision`, `recall`, one `<error>_rate`
per kind of error, and `per_suite`. A failed document counts with accuracy, precision and recall 0 and every gold
field unfound, so each of these numbers counts it.
