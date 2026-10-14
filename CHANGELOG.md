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
- A HuggingFace dataset is fetched lazily: the manifest and gold/ up front, each document's PDF
  only when it is predicted, at the commit the manifest was read at. A `--limit 5` run
  downloads five PDFs, not the corpus. A dataset of your own on HuggingFace keeps its gold
  under gold/ for this; a folder on disk needs nothing.
- A run never mixes scorers or golds. Each row of scores.jsonl carries `scorer` (the release
  and a digest of the scoring files) and `gold` (a digest of the document's id, suite, schema
  and gold file), and summary.json carries `scorer`. A run whose rows another scoring code
  graded, or whose selected documents' gold has changed, is refused by `plan` and by `grade`
  unless `--rescore`. A run graded before this release has no stamps, so it needs one
  `--rescore`; no vendor is called.
- `--rescore` discards every row, not only the selected documents', so a narrowed rescore
  leaves only what it graded again.
- `grade(run, docs)` writes summary.json itself, so a loop of your own gets the same folder
  `oeb benchmark` writes. docs/API.md has "A loop of your own".
