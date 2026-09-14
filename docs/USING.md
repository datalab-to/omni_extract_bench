# Scoring your own predictions

How to run this benchmark against your own extractions, and how to bring your own ground
truth. Companion to `DATA_LAYOUT.md`, which is about how *we* store the published data;
this file is about what you need, which is much less.

---

## The whole thing

```bash
pip install 'omni-extract-bench[benchmark]'

oeb score --predictions preds/ --out run/
```

`--predictions` is a directory of `<doc_id>.json`, each holding the extraction itself. The
corpus is downloaded for you. `--out` writes a run you can query.

That is the entire common case. Everything below is a variation on it.

---

## The contract

A corpus is a directory of document directories, plus an atlas:

```
<corpus>/corpus.parquet            which documents are in the benchmark
<corpus>/<doc_id>/ground_truth.json
<corpus>/<doc_id>/schema.json
```

`oeb build-corpus --corpus DIR` writes the atlas. Anything else in the directory is ignored,
and a document the atlas does not list is not in the benchmark.

**That is the whole contract.** The published benchmark also carries `document.pdf`,
`source.json`, a `corpus.parquet` atlas and some metadata tables, but those are our
operational layer -- caching, provenance and curation. None of them are required of your
corpus, and the same loader reads both. If they were required, scoring your own gold would
mean inventing a "suite" you do not have.

Predictions mirror it, flat:

```
<preds>/<doc_id>.json
```

holding the bare extraction -- no `{"result": ...}` wrapper, no metadata keys.

### The schema is required

It is never inferred, and there is no flag to make it optional. Without a schema an
`additionalProperties` subtree would be graded silently, and a value the model made up cannot
be told from one the schema offered it a slot for. `score.py` refuses a missing schema with
that explanation rather than guessing.

---

## Bring your own ground truth

Point `--corpus` at your own directory. There is no other difference:

```bash
oeb build-corpus --corpus my-benchmark/
oeb score        --corpus my-benchmark/ --predictions preds/ --out run/
```

### Choosing what to run

The atlas is a plain parquet and scoring follows its rows, so filtering it is how you pick a
subset:

```python
import pyarrow.parquet as pq, pyarrow as pa
t = pq.read_table("my-benchmark/corpus.parquet")
keep = [r for r in t.to_pylist() if r["suite"] == "invoices"]
pq.write_table(pa.Table.from_pylist(keep), "my-benchmark/corpus.parquet")
```

Predictions for documents the atlas no longer lists are skipped and counted:

```
12 predictions, 12 documents in the atlas
3 prediction(s) skipped; the atlas does not list them: inv-004, inv-007, inv-009
```

A row naming a file that is **not there** stops the run instead — the atlas is what the run
follows, so it has to resolve. `build-corpus` re-derives it from the tree whenever documents
are added or removed.

Better than overwriting: write the subset to its own file and point at it.

```python
pq.write_table(pa.Table.from_pylist(keep), "my-benchmark/invoices-only.parquet")
```

```bash
oeb score --corpus my-benchmark/invoices-only.parquet --predictions preds/
```

`--corpus` takes a directory (meaning its `corpus.parquet`) or an atlas file (meaning that
file, with the documents beside it). Rows name their files relative to the atlas, so both
describe the same documents.

```bash
oeb verify --corpus my-benchmark/    # what has changed since the atlas was written
```

`verify` reads every document and tells you what it found, so a malformed corpus fails in a
second rather than an hour into a run.

---

## What a run looks like

```
run/
  summary.parquet              one row per (doc_id, prediction_id)
  verdicts/<doc_id>.parquet    every address, with gold and pred
```

Two tables, because they are read on opposite schedules: a leaderboard wants every summary
row and no verdicts, an audit wants one document's verdicts and no summary. The verdict table
is two orders of magnitude larger, so keeping them together would make every leaderboard query
pay for the audit trail.

Partitioned per document so that rescoring one corrected document rewrites one small file.
`doc_id` is both the filename and a column, so a glob query never has to parse filenames.

Sizes, measured: a 660-document run is about **20 MB** of verdicts, roughly 6.4 compressed
bytes per address.

### Exploring it

No library and no API -- these are ordinary parquet files:

```sql
-- every value the model got wrong, across the whole run
select doc_id, address, gold, pred
from 'run/verdicts/*.parquet'
where verdict = 'wrong value';

-- what kind of failure dominates
select verdict, count(*) from 'run/verdicts/*.parquet' group by 1 order by 2 desc;

-- worst documents, with a count of bad addresses
select s.doc_id, round(s.accuracy, 1) acc,
       count(*) filter (where v.verdict <> 'match') bad
from 'run/summary.parquet' s
join 'run/verdicts/*.parquet' v
  on s.doc_id = v.doc_id and s.prediction_id = v.prediction_id
group by 1, 2 order by acc;
```

For one document, without SQL:

```bash
oeb explain --run run/ --doc <doc_id>
```

Verdicts are worth looking at before you trust an accuracy number. On a sample of real
predictions every wrong value was a boundary disagreement rather than a misreading --
`"Glenmere Robotics"` against `"Glenmere Robotics Inc."`, `"14 March 2026"` against
`"Updated 14 March 2026"`. That is invisible in a score and obvious per address.

Verdicts come back from the same pass that computes the score, so they cost almost no extra
time -- 43.8s against 41.4s on an 89,000-leaf document. What they do cost is memory: one
record per address, and the largest document in the corpus has 410,012 of them.
`--no-verdicts` is there for that, not for speed.

---

## Reading the summary

`kind` is `graded` or `unusable`, and it is not decoration.

An **unusable** prediction is one that could not be scored on its merits: an error payload, an
empty object, something that is not an object at all. It records no metrics -- null, not zero.

This matters more than it looks. Zero would be an interpretation, and `mean(accuracy)` without
filtering on `kind` would quietly adopt it. Worse, grading an error blob as though the model
tried and missed every field makes a rate-limited run look like a bad model. An earlier version
of this scorer had two different answers to that question, and a provider read as 100% coverage
while 37 of its 45 outputs were empty.

So: **filter on `kind` before you average.** The CLI reports the mean over graded predictions
and says how many were excluded.

---

## Using the pieces directly

`omni_extract_bench.bench` is the pipeline, and every stage is an iterable of plain data, so
any of them can be replaced without the others noticing:

```python
from omni_extract_bench.bench import cases, documents, predictions, score

for case in cases(documents("corpus/"), predictions("preds/")):
    outcome = score(case)
    print(case.doc.doc_id, outcome.kind, outcome.summary and outcome.summary["accuracy"])
```

Swap `documents` for a loader that reads your database, `predictions` for one that streams
from a bucket, or write the rows as JSONL instead of parquet. The only part that is not
replaceable is the metric.

For a single pair with no benchmark at all:

```python
from omni_extract_bench import grade
grade(prediction, ground_truth, schema)["accuracy"]
```
