# Changelog

## Unreleased

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
