# Changelog

## 1.0.0 (unreleased)

Breaking changes throughout, so omni-extract-bench and omni-parse-bench share one layout: the same
modules, the same names, at the same levels, and the same rules. Only the row's type differs (`Doc`
here, `Page` there). `docs/API.md` has the full surface.

`headline` changes meaning: it is the mean accuracy over every document, a failed or missing one
counting 0, so a system can't rank higher by failing on its hard documents (`accuracy * coverage`).

Modules:

- `data` reads a dataset: `Doc`, `download_dataset` (the whole dataset, at one commit), `read_manifest`,
  `select`, `dataset_id`, `commit` and `ManifestError`. It replaces `benchmark.Doc`, `fetch`,
  `read_manifest`, `corpus`, `document`, `hf_dataset`, `dataset_id` and `hf_path`;
- `scorers/field_matching` is the scorer, a package that owns the shape of its gold (was `metric.py`,
  `matching.py`, `values.py` and `recognise.py`): `score`, `gold`, `check`, `summarize`, `Verdict` and
  `GoldError`;
- `errors` (new) holds the two kinds of error a user meets: `UsageError` (a `ValueError`: what was asked
  is wrong) and `Stop` (a run can't go on). `ManifestError` and `GoldError` are `UsageError`s;
  `AccountFailure`, `MissingCredential` and `MissingDependency` are `Stop`s;
- `benchmark` is the loop, as functions over two NamedTuples, `Run(provider, options, out)` and `Plan`:
  `plan`, `configurations`, `prepare`, `needs`, `record`, `recorded`, `predicted`, `grade`, `read_scores`,
  `owed`, `predict_all`, `grade_all`, `execute` and `describe` (`progress.plan_view` is the plan as
  a table), and for a folder, `settled`,
  `declare` and `gradable`; also `is_run`, `restore`, `pools` and `write_json`. `BenchmarkRun`,
  `ProviderRun`, `run(...)`, `needs_run` and `Run`'s methods are gone (`run.needs(doc)` is
  `needs(run, doc)`, and so on);
- `harness` loads adapters declared under the entry-point group `omni_extract_bench.adapters`, files
  ones defined in code with `add_adapter`, and exports `out_name` and `VendorUnreachable`;
  `registry.adapter_names()` lists the names `--workers` takes; and
- the package exports `data`, `scorers`, `gold`, `score`, `summarize`, `check` and `Verdict`; `canon_key`,
  `cmp_leaf` and `match_rows` live in `scorers.field_matching.values` and `.matching`, and `__version__`
  is gone (it is the distribution's, `importlib.metadata`).

Helpers that were public by accident are private: `workers_for`, `_settings` and `_score_one`;
`benchmark` no longer re-exports `predict`. A module imports modules, not names, so it no longer holds
names it only imported: `benchmark.UsageError` is `errors.UsageError`, `scorers.field_matching.canon_key`
(and `cmp_leaf`, `states_nothing`) is `scorers.field_matching.values.canon_key`, and
`harness.schema.resolve_refs` is `scorers.field_matching.resolve_refs`.

The scorer:

- `score(pred, gold, schema, ...)`: the ground truth argument is `gold`, not `gt`; a prediction that
  isn't a JSON object (or None) is refused, as a gold is;
- a `Verdict`'s `verdict` field is `result`, and so is the key in `verdicts/<doc_id>.jsonl`;
- `summarize(lines)` needs nothing but the lines, which carry their suite. A line's status is `scored`
  or `failed` (no usable prediction). The summary, and each suite's block, holds `headline`, `documents`,
  `scored`, `failed`, `coverage`, `accuracy`, `precision`, `recall` and where the errors were; `headline`
  is null for no documents, and the metrics over the scored are null when none was scored;
- a scorer that raises is a bug of ours: grading stops with it, naming the run and the document, and
  writes no scores or summary (the `error` status, the `errors` count and exit 2 are gone); with several runs, those
  graded before the crash keep their new files, and `--score-only` grades again after the fix. An unknown
  `order_matters` name is a `UsageError`;
- `check(docs)` raises a `GoldError` naming the first document whose gold can't be read or isn't a JSON
  object; and
- the scoring-code digest changes, so every run is graded again once; no provider is called.

Grading:

- `grade(run, docs, say=print, *, settings=None, verdicts=True, score_workers=0, progress=NULL)` grades
  the folder against the whole manifest (`grade_all` passes it), so `--suites` and `--limit` narrow what
  is predicted, never what is scored; `oeb grade` has no `--suites`;
- grading is fresh every time: every document is graded again, and `verdicts/`, `scores.jsonl`, then
  `summary.json`, are written when it finishes; an interrupted grading writes no scores or summary. `--rescore` is
  gone. Each line, and the summary, carries `scorer` (the release and a digest of the scoring files);
- a document with no prediction is pending only when a run never attempted it (no record, in a folder
  whose settings.json names a provider); otherwise it fails, so a document left out of a folder of your
  own fails;
- a folder of your own none of whose files names a document of the manifest is refused, writing
  nothing, so a forgotten `--dataset` can't grade it against ours; grading fewer documents than the
  folder's manifest warns that the summary covers only those;
- while any document is pending, `headline` is null, overall and in each suite with pending documents:
  a run has a headline only once it is finished. Each suite's block carries its own `pending`;
- summary.json starts with `run`, `provider` and `settings` (from the folder's settings.json), and holds
  `pending`, `usd` and `credits` (`credits` null when no call reported any); each score line carries
  `run`;
- a prediction is read only under its exact name, case too, so a folder grades the same on every system.
  Grading warns once about files named for no document and predictions that aren't a JSON object
  (graded as failed; the line says why: empty, not UTF-8, not JSON, a list). A UTF-8 byte-order mark is
  read through. `verdicts/` is swapped in only with verdicts; a grading cut short leaves no partial
  verdicts behind;
- a failed call writes no prediction: predictions/ holds only answers, and the error is in records/;
- a gold that can't be read, or isn't a JSON object, is a `GoldError` naming its document (`gold`
  raises it, and `check` is `gold` over every document); grading stops on it, as omni-parse-bench
  does, since it is the dataset's problem, not a scorer bug;
- the record is the source of truth for an answer: a prediction lost or unreadable beside a record of an
  answer is rebuilt from it, with no call, whenever the run is graded (`--score-only` and `oeb grade`
  too);
- a prediction of `{}` is scored, every field the gold states unfound; it was failed as "empty
  prediction"; and
- `owed(plan, run)` counts what grading will grade over the whole manifest, by the same rule as `grade`.

Runs and folders:

- a folder holds one dataset and one manifest: `plan` and `oeb grade` refuse another, in one wording.
  `oeb grade` reads them from the folder's settings.json when not given; a folder of your own gets them
  there once its first grading succeeds (a failed one writes no settings, scores or summary). A folder of
  your own's refusal says its settings.json can be deleted to start over; a run's says to use another
  `--out`, since its records are answers for its own dataset's documents. The rule is per run folder;
  `--out` is only their parent. A manifest's path is spelled one way (`./manifest.parquet` is
  `manifest.parquet`, `data.normal`), and one outside the dataset is refused before anything is compared;
- settings.json records `dataset` (without its revision), `commit` and `manifest` in place of `corpus` and
  `corpus_commit`, and `prepare` writes it whole;
- `record(run, doc_id, result)` writes what `harness.predict` returned and returns it;
- `plan` warns when a run made at one commit of its dataset is planned at another, and carries on; it
  refuses an empty selection, a run asked for twice, `--score-only` into a folder never predicted, a
  timeout that isn't positive and a negative `--workers` or `--score-workers` (a `workers` of the wrong
  type is a `TypeError`);
- `--retry-failed` predicts again a timeout, an outage (`VendorUnreachable`, raised for a failed poll or
  connection) or a 429, 5xx or 52x; a 400 or an answer that couldn't be used is not.
  `record["error"]["transient"]` says only whether `predict` retried within the call;
- a `Stop` stops its adapter's queued documents at once;
- `--workers` and `plan(workers=...)` replace `--predict-workers`: one number for every adapter, or a
  JSON object of a number per adapter (`llm_single_shot` is every model id; an adapter of yours is the
  name it was filed or declared under). A key that isn't an adapter is refused, naming the adapter it
  uses, and 0, overall or for one adapter, is the default; `grade(score_workers=...)` is the scoring
  processes. An option given as a whole number for a float setting is that float, so it names the same
  run; one of the wrong type, NaN or infinity is refused, and an unknown one is named with the options;
- `execute` grades what was predicted when predicting fails part way (an account failure included),
  but not when it is interrupted; `--score-only` grades it then. A bug in an adapter stops its runs,
  naming the run, the document and the adapter; an `extract` that returns something other than an
  `Extraction` says so; and
- summary.json's `usd` and `credits` are each null when no call reported it, so a vendor billed in
  credits doesn't read as free; a record's `cost.billed_out_of_band` is true only for a call that
  succeeded and reported neither.

Datasets:

- `--dataset` and `--manifest` replace `--manifest <file>` and `--data-root`: the dataset is a folder or
  `hf://datasets/<org>/<name>[@<revision>]` (default: ours), downloaded whole into the HuggingFace cache
  and tried again up to five times, each retry logged (a download that would fail the same way every
  time, offline with nothing cached among them, is refused at once), and `--manifest` is a relative path
  inside it (default: `manifest.parquet`). A dataset may hold several manifests;
- a `doc_path` is a relative path inside the dataset to a file, any name, and documents may share one
  (one PDF asked with several schemas); a `gt_path` is a relative path inside the dataset named
  `<doc_id>.json`; a doc_id is a str and a filename, unique ignoring case; a schema is a non-empty JSON
  object, written as a JSON string; a null or empty suite is `"other"` (`data.NO_SUITE`); an empty
  manifest is refused; doc ids that differ only in case are named with both spellings; a refused row is
  named by its doc_id;
- `read_manifest` keeps the manifest's order (it sorted by suite and doc id), so `--limit N` takes the
  manifest's first N, and reads each manifest once per process;
- a `Doc`'s `doc_path` and `gt_path` are full local paths, `Doc.id` is its `doc_id`, and a manifest's
  columns of its own are kept on `doc.extra`; and
- `select(docs, suites, limit, sample, only)` reads its names once, so any iterable works, and refuses
  a bare string or a negative count; `sample` spreads over every suite, and an unknown suite or id is
  refused.

The command line:

- results go to stdout as JSON: `oeb benchmark` and `oeb grade` print each run's whole summary, after a
  `Stop` too; the closing line (`<run>: headline 0.833; 8 of 8 documents graded, 1 failed`) and messages
  go to stderr, untimestamped. There is no `-q`;
- a `UsageError` or a `Stop` prints `oeb <command>: <message>` and exits 1 (`benchmark` adds that a rerun
  resumes); Ctrl-C exits 130; anything else is a traceback, a bug of ours;
- the closing hint counts only the documents `--retry-failed` would predict again;
- `oeb benchmark -y`, and `--score-only`, which spends nothing, print the plan and don't ask; a declined
  plan prints `nothing run`; the question goes to stderr; with no terminal to ask (a pipe, a notebook,
  CI), `oeb benchmark` refuses unless given `-y`; a bug of ours is a traceback and exits 1, as a
  `UsageError` or a `Stop` does; `--sample N` and `--docs FILE` (ids, one per line; blank lines and `#`
  comments skipped) choose documents;
- `oeb grade <folder>` (new) grades a folder of predictions/<doc_id>.json, yours or a run's;
- `oeb check` (new) says whether a dataset is one oeb can run: its manifest, every file it names (naming
  the row of a missing one), and every gold; and
- `oeb score --gold` replaces `--gt` (which says so), and names the flag of a missing, non-JSON or
  non-object file; files given to the command line may start with a UTF-8 byte-order mark; a stray
  prediction named for a document in another case gets "did you mean"; `oeb grade` on the parent of run
  folders names them; a
  bare `--order-matters` is every array; `--options` says when a path doesn't exist; a flag given a
  folder says so; a folder of loose `<doc_id>.json` files is told to move them into `predictions/`.
