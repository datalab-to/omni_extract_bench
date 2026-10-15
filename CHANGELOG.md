# Changelog

## 1.0.0 (unreleased)

Breaking changes to `omni_extract_bench.benchmark`. `score`, `predict` and the command line are
unchanged unless a line below says otherwise.

- `Run` is a NamedTuple, `Run(provider, options, out)`: an address of a run's folder. It has no
  `label`; a run's name is `run.out.name`.
- `Run`'s methods are functions over it: `run.needs(doc)` is `needs(run, doc)`,
  `run.store(doc, result)` is `record(run, doc.doc_id, result)`, `run.score(docs, ...)` is
  `grade(run, docs, ...)`, and `run.outstanding(docs)` is `owed(plan, run)`. New:
  `recorded(run, doc_id)` reads a record, and `prepare(run, dataset, timeout)` makes the folder
  and writes settings.json.
- `needs_run(path)` is gone; use `needs(run, doc)` or `recorded(run, doc_id)`.
- A failed call writes no prediction. `predict`'s `record["result"]` is `None` when
  `record["error"]` is set, and predictions/ holds only answers; the error is in records/.
  Grading reads a failure from the record. A run folder from before still grades the same,
  because its records already carry the error. A folder of your own predictions that marks a
  failure with `{"__error__": ...}` should leave that file out instead: a missing prediction is
  graded as failed.
- `BenchmarkRun` and `ProviderRun` are gone. `BenchmarkRun(providers, **kw).execute()` is
  `execute(plan(providers, **kw))`; `plan` takes the same keyword
  arguments `BenchmarkRun` did and returns a `Plan` (a NamedTuple). `.describe()` is
  `describe(plan)`, `.corpus()` is `corpus(manifest, ...)`, `.predict(docs)` is
  `predict_all(plan)` and `.score(docs)` is `grade_all(plan)`. The old `plan(providers,
  options)`, which returned runs, is `configurations(providers, options)`, which returns
  (provider, options) pairs.
- `run(...)` is gone: it was `execute(plan(...))`. The command line asks before it calls `execute`.
- A folder holding another corpus is refused by `plan`, before anything is read.
- `--score-only` writes no settings.json: it asks nothing.
- `--dataset` and `--manifest` replace `--manifest <file>` and `--data-root`. The dataset is a
  folder or `hf://datasets/<org>/<name>[@<revision>]` (default: ours), and `--manifest` names a
  parquet inside it (default: `manifest.parquet`); paths in it resolve against the dataset's
  folder. So `--manifest hf://datasets/datalab-to/omni_extract_bench/manifests/sample40.parquet`
  is `--manifest manifests/sample40.parquet`, and `--manifest my/corpus/manifest.parquet` is
  `--dataset my/corpus`. A manifest kept outside its corpus moves into it. In Python, `plan`
  takes `dataset` and `manifest`, and `fetch` and `hf_path` are gone (`data.download_dataset`).
- settings.json records `dataset` (without its revision), `commit` and `manifest` in place of
  `corpus` and `corpus_commit`. A run folder is refused only for another dataset, not for
  another manifest of the same one.
- A HuggingFace dataset is downloaded whole, at one revision, into the HuggingFace cache, so its
  documents and gold are always one commit's (fetching only what a run needs is an optimization
  for later).
- Grading is fresh every time: every selected document is graded again, which calls no vendor,
  and scores.jsonl and summary.json are written whole when it finishes, so a run's scores are
  always one grading's, never a mix of two scorers or golds. An interrupted grading writes
  nothing. Each line of scores.jsonl, and summary.json, carries `scorer` (the release and a digest
  of the scoring files). `--rescore` is gone: every grading is one.
- `oeb grade <folder>` grades a folder of your own predictions (`--dataset`, `--manifest`, `--suites`),
  as `opb grade` does: a document left out of it fails.
- `workers_for` is private.
- `grade(run, docs)` writes summary.json itself, so a loop of your own gets the same folder
  `oeb benchmark` writes, and returns the summary of `docs`' lines, as omni-parse-bench's
  `grade` returns a summary. docs/API.md has "A loop of your own".
- One module layout, the same as omni-parse-bench's: `data` (`Doc`, `download_dataset`,
  `read_manifest`, `select`, `dataset_id`), `scorers/field_matching` (was `metric.py`,
  `matching.py`, `values.py` and `recognise.py`: `score`, `gold`, `summarize`; each scorer is a
  package owning its gold shape), `benchmark` (the loop) and `harness`. A `Doc`'s `doc_path` and `gt_path` are full
  local paths, and `Doc.id` is its `doc_id`, the name every benchmark's rows share.
  The old names (`benchmark.Doc`, `fetch`, `read_manifest`, `corpus`, `document`, `hf_dataset`,
  `dataset_id`, `summarise`, and the module `metric`) are gone.
- summary.json, and each of its suites, carries a `headline`: the mean accuracy.
- `score(pred, gold, schema, ...)`: the ground truth argument is `gold`, not `gt`.
- A document never attempted (no prediction and no record, in a run that keeps records) is
  pending: no line, counted in summary.json's `pending`. A recorded failure is failed, and so is
  a document missing from a folder you bring, which keeps no records.
- `grade_all` grades every run over the whole manifest, so `--suites` and `--limit` narrow what
  is predicted, never what is scored.
- `owed(plan, run)` takes the plan first, and `prepare(run, dataset, timeout, manifest)` works out
  a HuggingFace dataset's commit itself; `Plan.root` is gone. `predicted(run, doc_id)` reads a
  prediction.
- `plan(workers=...)` and `--workers` replace `predict_workers` and `--predict-workers`.
- `retry_failed` (and `--retry-failed`) predicts again documents whose recorded failure may not
  recur (`error.transient`).
- `data.read_manifest` is read once per process.
- Internal now: `data._hf_dataset`, and benchmark's `_plural`, `_write_json_atomic`, `_settings`,
  `_head`, `_asked`, `_score_one`, `_write_scores`, `_settings_cell`, `_facts` and `_plan_view`.
  `read_scores(run)` returns a run's score lines, a list in file order.
