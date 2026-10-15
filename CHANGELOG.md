# Changelog

## 0.1.8

Breaking changes to `omni_extract_bench.benchmark`. `score`, `predict` and the command line are
unchanged unless a line below says otherwise.

- `Run` is a NamedTuple, `Run(provider, options, out)`: an address of a run's folder. It has no
  `label`; a run's name is `run.out.name`.
- `Run`'s methods are functions over it: `run.needs(doc)` is `needs(run, doc)`,
  `run.store(doc, result)` is `record(run, doc.doc_id, result)`, `run.score(docs, ...)` is
  `grade(run, docs, ...)`, and `run.outstanding(docs)` is `owed(run, docs)`. New:
  `recorded(run, doc_id)` reads a record, and `prepare(run, corpus, timeout)` makes the folder
  and writes settings.json.
- `needs_run(path)` is gone; use `needs(run, doc)` or `recorded(run, doc_id)`.
- A failed call writes no prediction. `predict`'s `record["result"]` is `None` when
  `record["error"]` is set, and predictions/ holds only answers; the error is in records/.
  Grading reads a failure from the record. A run folder from before still grades the same,
  because its records already carry the error. A folder of your own predictions that marks a
  failure with `{"__error__": ...}` should leave that file out instead: a missing prediction is
  graded as failed.
- `BenchmarkRun` and `ProviderRun` are gone. `BenchmarkRun(providers, **kw).execute()` is
  `execute(plan(providers, **kw))`, or `run(providers, **kw)`; `plan` takes the same keyword
  arguments `BenchmarkRun` did and returns a `Plan` (a NamedTuple). `.describe()` is
  `describe(plan)`, `.corpus()` is `corpus(manifest, ...)`, `.predict(docs)` is
  `predict_all(plan)` and `.score(docs)` is `grade_all(plan)`. The old `plan(providers,
  options)`, which returned runs, is `configurations(providers, options)`, which returns
  (provider, options) pairs.
- `run(...)` takes no `confirm`; the command line asks before it calls `execute`.
- A folder holding another corpus is refused by `plan`, before anything is read.
- `--score-only` writes no settings.json: it asks nothing.
- `--dataset` and `--manifest` replace `--manifest <file>` and `--data-root`. The dataset is a
  folder or `hf://datasets/<org>/<name>[@<revision>]` (default: ours), and `--manifest` names a
  parquet inside it (default: `manifest.parquet`); paths in it resolve against the dataset's
  folder. So `--manifest hf://datasets/datalab-to/omni_extract_bench/manifests/sample40.parquet`
  is `--manifest manifests/sample40.parquet`, and `--manifest my/corpus/manifest.parquet` is
  `--dataset my/corpus`. A manifest kept outside its corpus moves into it. In Python, `plan`
  takes `dataset` and `manifest`, `hf_path` is `hf_dataset`, and `fetch(dataset, manifest)`
  takes the dataset.
- settings.json records `dataset` (without its revision), `commit` and `manifest` in place of
  `corpus` and `corpus_commit`. A run folder is refused only for another dataset, not for
  another manifest of the same one.
- A HuggingFace dataset is downloaded whole, at one revision, into the HuggingFace cache, so its
  documents and gold are always one commit's (fetching only what a run needs is an optimization
  for later).
- A run never mixes scorers or golds. Each line of scores.jsonl carries `scorer` (the release
  and a digest of the scoring files), `gold` (a digest of the document's id, suite, schema and
  gold file) and `prediction` (a digest of its prediction file), and summary.json carries
  `scorer`. A line whose prediction, gold or scorer has changed is graded again, which calls no
  vendor; nothing is refused. A run graded before this release has no stamps, so it is graded
  again once.
- `--rescore` grades every selected document again and drops every other line, so a narrowed
  rescore leaves only what it graded again.
- `grade(run, docs)` writes summary.json itself, so a loop of your own gets the same folder
  `oeb benchmark` writes, and returns the summary of `docs`' lines, as omni-parse-bench's
  `grade` returns a summary. docs/API.md has "A loop of your own".
- One module layout, the same as omni-parse-bench's: `data` (`Doc`, `download_dataset`,
  `read_manifest`, `select`, `dataset_id`), `scorer` (was `metric`: `score`, `gold`,
  `summarize`), `benchmark` (the loop) and `harness`. A `Doc`'s `doc_path` and `gt_path` are full
  local paths, and `Doc.id` is its `doc_id`, the name every benchmark's rows share.
  `benchmark.Doc`, `fetch`, `read_manifest`, `corpus`, `document`, `hf_dataset`, `dataset_id`,
  `summarise` and the module `metric` still work, with a DeprecationWarning, and will be removed
  in a later release.
- summary.json, and each of its suites, carries a `headline`: the mean accuracy.
