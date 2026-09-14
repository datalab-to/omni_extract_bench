# Data layout

How the corpus, the predictions and the scores are stored, and why in that shape.

Companion to `REMOTE_RUNS.md`, which covers running the scorer over this data. This file is
about where the bytes live.

---

## The shape

```
HuggingFace (private)                 R2  datalab-training-pipelines
  corpus.parquet                        vendors/<vendor>.parquet
  tags.parquet                          baselines/<vendor>/<suite>/<doc_id>.json
  <doc_id>/                             scores/<scorer_commit>.parquet
      ground_truth.json
      schema.json
      document.pdf
```

**One rule: parquet holds metadata, files hold payloads.** `scores` is the single exception,
and the next section says why.

---

## The three tables are not the same kind of thing

This is the distinction worth holding, because it decides how much care each one needs.

| table | relationship to truth | cost to rebuild |
| --- | --- | --- |
| `corpus.parquet` | a **view** of the document files | seconds |
| `<vendor>.parquet` | a **view** of the prediction files | minutes |
| `scores/*.parquet` | **primary — nothing else holds it** | *hours of compute* |

Corpus and vendor tables are caches. If one is wrong, delete it and rebuild; a builder bug is
never data loss. **Scores are not.** Nothing else in the system contains them — lose one and
you re-run the scorer over 5,940 documents.

So: rebuild the first two freely, and never overwrite the third.

---

## Why payloads stay as files

Embedding the JSON in parquet was the obvious alternative and compresses beautifully — 242 MB
of predictions to 12.4 MB, about 20x. Two things ruled it out.

**Cell size does not degrade gracefully.** Today the largest ground truth is 13.1 MB and the
largest prediction 29.6 MB. A row group containing one must be decompressed whole to read
anything in it, and there is no partial read of a cell. Ground truth is unbounded in
principle: one pathological document and the table has a 500 MB cell in it. Files have no
such ceiling.

**The bulk-read argument turned out to be hollow.** Embedding would avoid 660 round trips per
vendor — except `snapshot_download` already fetches the whole HF dataset in parallel in one
call, which is how every script here gets the corpus. The problem embedding solved did not
exist on the side where the data lives.

It does exist on the R2 side, where scoring nine vendors is 5,940 individual GETs and there is
no snapshot equivalent. That is a reason to improve the fetch, not to restructure the store.

---

## Why one directory per document

Rather than `ground_truths/<doc_id>.json`, `schemas/<doc_id>.json`, `pdfs/<doc_id>.pdf`.

Selective download works identically either way — `allow_patterns=["*/ground_truth.json"]`
matches per-document paths exactly as `["ground_truths/*"]` matches per-type ones — so
"fetch the 132 MB without the 882 MB of PDFs" is not an argument for either. What remains:

* **New artifact types are additive.** Page images for VLM tagging, an OCR rendering, a
  second annotation pass: each is a file inside the document's directory. Per-type, every one
  is a new top-level directory and the root becomes a list of plurals.
* **The atlas needs one path column, not three.** With `<doc_id>/ground_truth.json` as a
  convention, the atlas stores `doc_id` and the rest is derived. Three URI columns are three
  things that can drift from where the files really are.
* **It is close to what exists.** The corpus is already
  `data/<suite>/<doc_id>/{ground_truth,schema,document}.json`; this drops the suite level and
  keeps the filenames.

Names stay in long form. `ground_truth.json`, not `gt.json` — it appears in error messages and
scripts, and the abbreviation reads as noise to anyone new.

### The suite level goes away deliberately

A path should carry identity, not facts. If files live under `<suite>/<doc_id>/`, the suite
exists both in the path and as an atlas column, and the two can disagree — a file under
`micro1/` whose row says `contextual`. Flat means reclassifying a document is a one-row edit,
which matters while the corpus is still being curated.

The cost is that global uniqueness stops being free. It holds today — 660 doc_ids, all unique,
none unsafe as a filename — but nothing enforces it forward. **The builder must assert it and
fail loudly**: a duplicate `doc_id`, a name containing `/`, or a leading `.` is a build error,
not a silent overwrite. Twenty-eight names contain spaces or brackets
(`research__[zhao25] a survey of LLMs`); legal on a filesystem, awkward in URLs and shells,
and worth normalising while the dataset is still private.

---

## `prediction_id`

**A versioned contract, not an implementation detail.** `scores` joins on it, so changing how
it is computed orphans every historical score, silently. Treat a change as a migration.

### v1

```python
def prediction_id_v1(payload: dict) -> str:
    """Identifies the EXTRACTION, not the run that produced it."""
    result = payload["result"] if "result" in payload else payload
    canon = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]
```

Three properties, all measured on the 5,936 real predictions:

**It survives a re-run.** The envelope carries `_secs`, which differs every time even when the
extraction is byte-identical. Hashing the file or the whole envelope makes a re-run look like
a new prediction, destroying the one property the id exists for:

```
same extraction, different wall time
  raw file bytes   964a87f8...  ->  d719b471...   changes
  whole envelope   cdd0ef9a...  ->  41c2c433...   changes
  result only      db32da34...  ->  db32da34...   stable
```

**It deduplicates.** 5,936 predictions carry only 4,868 distinct ids — 1,068 are exact
duplicates of another, usually several vendors agreeing on an easy document. One is shared by
six vendors. Under a content-addressed id that is one prediction scored once.

**It is not a key on its own.** Three azure-cu ids appear under *different documents*, because
its error payloads are identical whatever the input. **The key is `(doc_id, prediction_id)`.**

Canonical JSON — sorted keys, no whitespace — makes it immune to reserialisation.

---

## Table contents

### `corpus.parquet`

One row per document. Every column is a fact about the file or its shape as JSON, computed
without importing the scorer, so the table is valid for any scorer version and never needs
rebuilding after a scoring change.

```
doc_id            suite             source_id
gt_bytes          schema_bytes      has_pdf
gt_sha256         schema_sha256     max_array_rows      leaf_values
```

`max_array_rows` is the number that drives matching cost — the scorer solves an assignment
over the largest array, and both its time and memory go as the square. `leaf_values`
approximates flattening cost. Both are JSON walks, no scorer import.

**The schema is stored raw.** `strip_benchmark_keys` then `resolve_refs` are the scorer's
business, they change with it, and applying them here would bake one version's opinion into
the corpus. Consumers apply them; 243 of the 660 schemas differ if you forget, silently.

### `tags.parquet`

Separate from `corpus.parquet`, keyed by `doc_id`, stamped with model and date.

VLM-derived tags — "form", "large table", "handwritten" — are model-dependent, dated and not
reproducible. Putting them in `corpus.parquet` would break the property that makes that table
safe: deterministic and cheap to rebuild from the files. A separate table joins just as
easily and keeps the corpus something you can delete without thinking.

### `vendors/<vendor>.parquet`

One row per document. Half of this exists today only inside `_raw/` objects nobody mounts.

```
doc_id            prediction_id     uri              bytes
secs              recovered_after_timeout            usable      error
timeout_s         tier              model            wall_s      http_status
```

`timeout_s`, `tier` and `model` come from `_raw/run_manifest`; it is what made the retry
policy recoverable after the runner itself could not be found. The atlas **summarises**
`_raw/` rather than replacing it — the full records stay in R2.

One table per vendor because vendors are produced independently and vary 29x in size
(datalab 233 MB, gemini 8 MB). Adding one should not rewrite the others, and it is the natural
shard boundary for a remote run.

### `scores/<scorer_commit>.parquet`

Key: `(doc_id, prediction_id, scorer_commit)`. Immutable — one file per scorer version, never
overwritten. That gives version history for free in a store without versioning, and makes
"did the scorer change this?" a file listing rather than a query.

The filename is not the whole provenance. Carry it in the file: platform, numpy and scipy
versions, corpus snapshot, started-at. Same commit on x86 and ARM may not agree, and that is
unverified — see `REMOTE_RUNS.md`.

**The key is what makes the table worth having:**

```
same prediction_id, different scorer_commit, different score   the scorer changed it
different prediction_id                                        the vendor re-ran
same prediction_id AND same scorer_commit, different score     non-determinism: a bug
```

That last line is a test you get for nothing. It would have caught the greedy
order-sensitivity automatically, instead of it surfacing as one document quietly recording
99.2316, then 99.2301, then 99.2328 across a single session with nothing explaining why.

---

## Row groups

`pyarrow` sizes row groups by row COUNT, so content-sizing means slicing the table and making
one `write_table` call per slice on an open `ParquetWriter` — each call starts a new group.

Measured on 660 predictions, same data, same compression:

```
default        12.4 MB file   1 row group    largest 242.6 MB    75.7 ms per single-row read
content-sized  12.4 MB file   9 row groups   largest  33.6 MB     9.8 ms
```

Eight times faster single-row reads at no size cost. It matters wherever cell sizes span
orders of magnitude, which they do everywhere here.

---

## The failure this shape invites

An atlas and its files can drift, and nothing notices. A row pointing at a deleted file, or a
file with no row, stays invisible until something reads it.

`manifest.json` has the same weakness today, but it is 660 entries maintained by hand. An
atlas is generated by tooling and will start looking authoritative, which is worse.

**So `verify` is first-class, not bolted on.** It walks both directions — every row resolves
to a file, every file has a row, every `sha256` matches — and it should run in CI and after
every build. Without it, "the parquet says so" becomes load-bearing and unchecked.

The other standing risk is that nothing rebuilds a vendor atlas when new predictions land. The
runner cannot do it — it is not in any repo we have — so it is a manual step, and manual steps
are forgotten. `verify` catching the drift is the backstop.

---

## Order of work

1. **Freeze `prediction_id` v1.** The only thing expensive to change later; everything
   downstream keys on it.
2. **`corpus.parquet` builder, with `verify`.** Both from the start.
3. **Vendor atlas builder**, including the `_raw` fields.
4. **Scores writer**, once the first two are trusted.
