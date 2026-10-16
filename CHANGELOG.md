# Changelog

## 1.0.0 (unreleased)

**`headline` changes meaning.** It is the mean accuracy over every document, a failed or missing one
counting 0 (as omni-parse-bench's failed page fails its tests), so a system can't rank higher by
failing on its hard documents. It is `accuracy * coverage`; `accuracy`, `precision` and `recall` stay
over the scored documents. The scoring-code digest changes, so every run is graded again once.

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
- `grade(run, docs, score_workers=...)`: grade's scoring processes are `score_workers`, as
  `plan`'s are (it was `workers`). `workers` means calls in flight, and only `plan` takes it.
- A manifest, or a row of one, that breaks the dataset's rules raises `data.ManifestError`, a
  `ValueError`. A malformed `hf://` address is still a plain `ValueError`.
- `record(run, doc_id, result)` returns the record it wrote.
- `data.select(docs, suites, limit, sample, only)` and `plan(sample=..., only=...)`, as
  omni-parse-bench's: `sample` picks that many documents spread over every suite, the same ones each
  time, and `only` keeps those doc ids, each of which must be in the dataset. On the command line,
  `oeb benchmark --sample N` and `--docs FILE` (doc ids, one per line). An unknown suite is now
  refused, naming the dataset's suites; it used to select nothing.
- `plan` warns when `--out` holds a run made at another commit of its dataset, and carries on: the
  run's old predictions are graded against the new gold. Pass another `--out` to keep them apart.
- A `Verdict`'s `verdict` field is `result`, and so is the key in `verdicts/<doc_id>.jsonl`. The
  scoring-code digest changes, so every run is graded again once; no provider is called.
- `--workers` and `plan(workers=...)` are per adapter, as in omni-parse-bench: one number for every
  adapter, or a JSON object of a number per adapter, `'{"reducto": 25, "llm_single_shot": 3}'`
  (`llm_single_shot` is every OpenRouter model id). A key that isn't an adapter is refused, naming the
  adapter it uses. `'reducto=25,datalab=40'` and `"*"` are gone.
- A manifest's `doc_path` and `gt_path` must be relative paths inside the dataset; an absolute path is
  refused, since a dataset is one folder. A gold is named for its document, `<doc_id>.json`, in any
  folder; an input may have any name, and documents may share one (one PDF asked with several
  schemas). A doc id may appear once.
- `read_manifest` keeps the manifest's order; it sorted by suite and doc id. So `--limit N` takes the
  manifest's first N: a run resumed with the same `--limit` may predict other documents than it did.
- `download_dataset` retries a failed download (5 attempts), and refuses a folder that doesn't exist.
- `execute` grades what was predicted even when predicting fails part way, an account failure
  included: a run's unattempted documents are `pending` in its summary. An interrupt (Ctrl-C) grades
  nothing; `--score-only` grades it then.
- Doc ids must be unique ignoring case, since each names files.
- `plan` refuses a run asked for twice and a negative `--limit`, `--sample`, `--workers` or
  `--score-workers`; it used to merge the repeat.
- `grade` refuses a folder with no `predictions/`, as `oeb grade` did.
- `prepare` reads the manifest, so a broken one stops before anything is written.
- `harness.out_name` is exported.
- `predicted` and `read_scores` log a file they can't read.
- `oeb check` says whether a dataset is one oeb can run: its manifest, that every file it names
  exists, and that every gold is a JSON object (`field_matching.check`).
- A document's suite may be null or empty, and is then `"other"` (`data.NO_SUITE`); any other
  non-string is refused.
- `oeb grade <folder>` grades a run against the dataset and manifest its `settings.json` names, and
  refuses another dataset (it would overwrite the run's scores with a grading against gold it never
  saw). Grading a run with `--suites` says it replaces the run's summary. A grading warns of files in
  `predictions/` that match no document, and of predictions it can't read.
- `grade` refuses an empty `docs`, and `oeb check` an empty manifest, rather than writing an empty
  summary over a run's scores.
- `verdicts/` is one grading's, swapped in when it finishes; `--no-verdicts` removes an old one.
- `--retry-failed` follows omni-parse-bench's rule: a timeout, an outage (the new `VendorUnreachable`,
  raised for a failed poll or connection) or a 429 or 5xx (now also 520-524) is predicted again; it
  retried only 429 and 5xx. `record["error"]["transient"]` still says whether `predict` retried
  within the call.
- A manifest's schema must be a non-empty JSON object, and its doc_id a str.
- `score` refuses a prediction that isn't a JSON object (or None), as it does a gold.
- A mistake on the command line is one line, `oeb <command>: <what was wrong>`, and exit 1, instead of
  a traceback; `-q` goes before or after the command. Messages name what was wrong: a `--options` file
  that doesn't exist, JSON that isn't an object, a `--docs` file that doesn't exist, a `--manifest`
  (a path inside `--dataset`), an `hf:/` address with one slash, and which flags selected no documents.
- `select` refuses a bare str of suites, and a negative `limit` or `sample`, and takes any iterable
  for `only`; `plan` refuses a bare str of providers, a timeout that isn't positive, and a `workers`
  that isn't a number or a mapping.
- A bug in an adapter still stops its runs, now naming the document and the adapter.
