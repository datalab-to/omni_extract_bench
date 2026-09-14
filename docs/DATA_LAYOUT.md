# Data layout

How the corpus, the predictions and the scores are stored, and why in that shape.

Companion to `REMOTE_RUNS.md`, which covers running the scorer over this data. This file is
about where the bytes live.

---

## The shape

```
HuggingFace (private)                 R2  datalab-training-pipelines
  corpus.parquet   <- the atlas         vendors/<vendor>/predictions.parquet
  metadata.parquet                      vendors/<vendor>/<doc_id>.json
  tags.parquet                          scores/<corpus_version>/<scorer_version>/
  thumbnails.parquet                        summary.parquet
  <doc_id>/                                 verdicts/<doc_id>.parquet
      ground_truth.json
      schema.json
      document.pdf
      source.json
```

**One rule: parquet holds metadata, files hold payloads.** `scores` and `thumbnails` are
the two deliberate exceptions, and each section says why.

---

## The three tables are not the same kind of thing

This is the distinction worth holding, because it decides how much care each one needs.

| table | relationship to truth | cost to rebuild |
| --- | --- | --- |
| `corpus.parquet` | **the benchmark's definition** | seconds, and re-versions the corpus |
| `predictions.parquet` | a **view** of the prediction files | minutes |
| `scores/.../summary.parquet` | **primary — nothing else holds it** | *hours of compute* |

**The corpus atlas is not a cache.** It was, once, and this table used to say so. Since it
became the statement of which documents are in the benchmark -- and the record of what their
files hashed to -- re-deriving it is how you say a change to the data was intended. Cheap to
run, but not a no-op: it is the act that lets a corrected ground truth produce a new corpus
version instead of a silently different score.

The vendor atlas still is a cache. Delete it and rebuild; a builder bug there is never data
loss.

**Scores are neither.** Nothing else in the system contains them — lose one and you re-run the
scorer over the whole corpus. They are never overwritten: a finished table is that corpus and
that scorer's answer, and `--recheck` compares against it rather than replacing it.

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
def prediction_id_v1(result_bytes: bytes) -> str:
    """The extraction as stored, hashed. There is no envelope to see through."""
    return hashlib.sha256(result_bytes).hexdigest()
```

That is the whole definition, and it is only that short because the stored file holds the
bare extraction: no `_secs`, no envelope, so no unwrap, no schema coupling, no
canonicalisation spec. Anyone in any language can reproduce it from the file.

**The stored bytes are the vendor's, not ours.** `json.loads` discards byte offsets, so
parsing a prediction and re-dumping it yields *our* formatting. Take the raw span instead --
decode from the value's start index and keep what the decoder consumed:

```python
dec = json.JSONDecoder()
i = text.index('"result"')
j = text.index(":", i + len('"result"')) + 1
while text[j] in " \t\r\n":
    j += 1
_obj, end = dec.raw_decode(text, j)
verbatim = text[j:end]
```

Measured on all 5,936: every span parses equal to the value the whole file gives, zero
failures, about eight seconds for the corpus. And **100% differ from a reserialised form** --
every vendor writes `", "` where `json.dumps` writes `","`. So this is not an edge case to
guard against, it is every file, and hashing a reserialised form would make the id describe
our formatting rather than the vendor's output.

Two further properties, measured on the same 5,936:

**It survives a re-run.** The envelope carries `_secs`, which differs every time even when
the extraction is byte-identical. Storing only the result means a re-run of the same
extraction gets the same id, which is the property the whole thing exists for.

**It deduplicates, but only byte-for-byte.** 5,936 predictions carry 5,016 distinct ids --
920 are exact duplicates of another, and 343 of those are shared across vendors rather than
within one. Under a content-addressed id that is one prediction scored once.

Hashing a *canonicalised* form instead would find more -- 4,868 distinct, 1,068 duplicates --
because it collapses vendors that agree on content while differing in whitespace. That is the
price of v1 being the vendor's own bytes, and it is the right price: an id that describes what
was actually stored is worth more than 148 extra matches. It does mean the dedup here is
syntactic, and semantically identical predictions can carry different ids.

**It is not a key on its own.** Three azure-cu ids appear under *different documents*, because
its error payloads are identical whatever the input. **The key is `(doc_id, prediction_id)`.**

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

### `source.json`, and why suite is declared

```json
{"suite": "micro1", "source_id": "m1/Manufacturing_Industry_Datasets"}
```

Which collection contributed a document cannot be recovered from the document. Not from the
payloads, which say nothing about it; not from the path, since the suite level is gone; and
**not from the `doc_id` prefix**, which 394 of 660 carry and 266 do not, and which lies as
soon as a document is reclassified. A builder inferring from it would succeed for most and
quietly mislabel the rest -- a plausible wrong answer, which is worse than a stopped build.

So `suite` is **declared, never inferred**, and a document without a `source.json` is a build
error rather than a guess.

Keeping it beside the document rather than in a central manifest buys two things. The atlas
stays derivable from the tree alone -- `--rebuild-atlas` regenerates `corpus.parquet` with no
access to the original dataset, and it reproduces the migrated table exactly, all 660 rows,
every column. And adding a document is one directory rather than an edit to a shared file
that two contributors will conflict over.

It is also where provenance grows. Licence, origin URL, when and by whom a document was
added, whether its ground truth was hand-authored or derived -- none of that is in the
payloads, and `gt_provenance.json` already exists in the current dataset as evidence it is
wanted.

### Metadata tiers, and what decides them

Not everything known about a document belongs in the same table. The line is not cost -- it
is **what you need in order to reproduce it**, because that decides whether a table can be
deleted and rebuilt without thought.

| tier | needs | stamped with | table |
| --- | --- | --- | --- |
| structural | the bytes, and the standard library | nothing -- any Python gives the same answer | `corpus.parquet` |
| measured | a tool that reads the format | the tool and its version | `metadata.parquet` |
| judged | a model | the model and when it ran | `tags.parquet` |
| rendered | a renderer, and real bytes out | the renderer and when | `thumbnails.parquet` |

All keyed on `doc_id`, all joinable, all independently rebuildable.

`corpus.parquet` stays in the first tier on purpose. Its value is that it can be regenerated
from the files with nothing but the standard library, so it is never stale in a way that
matters and never needs a dependency pinned. The moment a column needs a parser, that
property is gone.

### `metadata.parquet`

Measurements that need a tool to read the format, but are otherwise deterministic -- the
same PDF gives the same answer, given the same library.

```
doc_id            page_count
page_width        page_height        (of the first page, in points)
```

`page_count` is the first and the reason the tier exists. It is a fact everyone wants, but
getting it means opening the PDF with a parser: not stdlib, and not perfectly stable either,
since the corpus contains PDFs that make `pypdf` emit `Ignoring wrong pointing object`
warnings and a malformed file can read differently across versions. So it is stamped with the
library that produced it rather than presented as ground truth.

Measured on the corpus: **22,223 pages across 660 documents**, median 22, mean 34, p90 66,
and one document of **878 pages**. 103 documents are over 50 pages and 14 over 200 -- worth
knowing before anything proposes to process per page.

### `tags.parquet`

Judgements from a model -- "form", "large table", "handwritten" -- keyed on `doc_id` and
stamped with the model and date.

Separate from `metadata.parquet` because the provenance differs in kind. A page count is a
measurement anyone with the same library reproduces; a tag is one model's opinion on one day,
and re-running a better model should replace the table rather than appear to correct a fact.
Separate tables make that a swap instead of a merge.

### `thumbnails.parquet`

One rendered image per document, plus `rendered_by` and `rendered_at`.

**This is the one place the rule bends, deliberately.** Payloads live in files, but HF's
dataset viewer only renders images that are *in* the parquet with an `Image()` feature -- a
file in the repo will not preview. The rule exists because cell size is unbounded for ground
truth; a thumbnail is bounded **by construction**, since the resolution is chosen, so the
hazard cannot occur.

Measured before deciding, over 30 documents spread across the corpus:

```
width  format   median   p90     660 documents
  512    WEBP     21 KB   68 KB        13 MB
 1024    WEBP     57 KB  188 KB        36 MB
 1024    JPEG    120 KB  301 KB        77 MB
 1024     PNG    204 KB  658 KB       132 MB
```

WebP at 1024px, ~36 MB, is 4% of the 812 MB of PDFs and legible enough to tell a form from a
table from a filing. PNG is **3.6x larger for identical content** and buys nothing that the
HF viewer or a modern IDE cannot already read.

Rendering costs 13 ms a page: ten seconds for 660 first pages, five minutes for all 22,223.

**One image per document, not per page.** Every page would be 1.2 GB even at the cheapest
setting -- larger than everything else combined, for derived data that regenerates in five
minutes. When VLM tagging needs page images, it should render them in that job rather than
the dataset carrying them forever.

Not built yet. The measurements are recorded so the decision does not have to be made twice.

### `vendors/<vendor>/`

One self-contained directory per vendor, its atlas beside the payloads it describes:

```
vendors/reducto/
    predictions.parquet
    <doc_id>.json          the bare extraction, 660 of them
```

The same shape the corpus uses, where `corpus.parquet` sits beside the document directories.
Deleting a vendor is one `rm -rf`, and `vendors/reducto/` can be handed over as a complete
thing. An earlier split into `vendors/` and `predictions/` meant a vendor lived in two places
and neither half stood alone.

The atlas is `predictions.parquet`, not `metadata.parquet` -- that name is taken at the corpus
level for page counts, and two tables sharing one name is a trap for whoever reads the tree
next.

The directory is the vendor's plain name, `reducto` rather than `reducto_full`. The suffix
names the RUN, and which run this is lives in the stamp; encoding it in the path too would
duplicate a recorded fact, the same reason the corpus dropped its suite level. A second run
wants `vendors/<vendor>/<run>/` or a column, not a suffix. The source directory is kept in
the stamp so a row can be traced back.

Payloads sit flat rather than as `<doc_id>/result.json`. A per-document directory won that
argument for the corpus because a document has several artifacts and will grow more; a
prediction has exactly one, and its run provenance turned out to be per-vendor. If a second
per-prediction artifact ever appears it becomes a directory then, for the same reason.

One row per document the vendor returned. Coverage is ragged and the table says so: azure-cu
and gpt are missing one document each, claude two, because nothing was written for them.

```
doc_id        prediction_id     bytes
secs          recovered_after_timeout      usable      error
```

`prediction_id` doubles as the payload's checksum, so there is no separate hash column and
`verify` needs nothing extra. The atlas is not listed among the payloads it describes --
`verify` globs `*.json`, so a parquet inside its own tree is correctly ignored rather than
read as an orphan.

**Run metadata is a stamp, not columns.**

```
{'timeout_s': '1800', 'model': 'openai/gpt-5.6-sol', 'max_output_tokens': '64000',
 'captured_at': '2026-08-21T22:21:44', 'vendor': 'gpt', 'source_dir': 'gpt_full',
 'prediction_id_version': 'v1', 'rows': '659', 'built': '...'}
```

Those fields sit in `_raw/`, and sampling showed they are run- and vendor-level constants
rather than facts about a document: `timeout_s` is **1800 across all 5,940**, `model` is fixed
per vendor, and `cost.wall_s` duplicates the envelope's `_secs`. So the builder reads **one**
`_raw` object per vendor instead of 660 -- the difference between nine requests and
**1,084 MB**.

`tier` is deliberately absent. It reads like a fact and is not: gpt's records carry two
different hand-written descriptions of the identical model `openai/gpt-5.6-sol`, so it is
prose about a run, not an attribute of one.

### Storing the bare result is about identity, not size

Stripping the envelope saves nothing. Measured across all 5,936: **582 MB in, 582 MB out,
0.0% smaller**, because `{"result": ..., "_secs": N}` is a few dozen bytes against files
averaging 98 KB.

The reason to do it is that a file holding only the extraction has nothing to unwrap, so
`prediction_id` is a plain hash with no rule to agree on and no schema to consult. Two bugs
this week came from unwrap rules guessing wrong; this removes the category.

### `scores/<corpus_version>/<scorer_version>/`

Key: `(doc_id, prediction_id)` within a file; `scorer_version` is the filename. Immutable
— one file per scorer version, never overwritten. That gives version history for free in a store without versioning, and makes
"did the scorer change this?" a file listing rather than a query.

The filename is not the whole provenance. Carry it in the file: platform, numpy and scipy
versions, corpus snapshot, started-at. Same commit on x86 and ARM may not agree, and that is
unverified — see `REMOTE_RUNS.md`.

**The key is what makes the table worth having:**

```
same prediction_id, different scorer_version, different score   the scorer changed it
different prediction_id                                        the vendor re-ran
same prediction_id AND same scorer_version, different score     non-determinism: a bug
```

That last line is a test you get for nothing. It would have caught the greedy
order-sensitivity automatically, instead of it surfacing as one document quietly recording
99.2316, then 99.2301, then 99.2328 across a single session with nothing explaining why. It is
run deliberately with `--recheck`, which rescores and diffs against a stored table rather than
replacing it.

### One row per distinct prediction, not per vendor

A score is a function of the prediction's bytes and the scorer's code, so two vendors that
emitted byte-identical extractions have the same score by construction. **5,936 predictions
are 5,198 distinct `(doc_id, prediction_id)` pairs — 738 scorings, 12.4%, that would be
computing a known answer twice.** 360 of those keys are held by more than one vendor.

That is 5,198 rather than the 5,016 distinct `prediction_id` values counted earlier, because
48 ids recur under different documents and the pair splits them back apart.

Which vendors a row belongs to is not a column. It is recovered by joining the vendor atlases
on `(doc_id, prediction_id)`, where that fact already lives.

The dedup rests on one assumption: files sharing a key are byte-identical. That is what
`prediction_id` means, but it is the kind of assumption that rots quietly, so `--check-dedup`
asserts it across the tree instead of trusting it. Measured: 0 violations.

### What is not in the table

**Predictions that are absent get no row.** The run produced four. They have no file, so no
bytes, so no `prediction_id` — there is nothing to key them on. Absence is a fact about a
vendor's coverage and already shows as a missing row in that vendor's atlas. A scores row for
a prediction that does not exist would put it in the one table meant to describe predictions
that do.

**Metrics on a non-graded row are null, not zero.** 242 of the 5,919 outcomes in the run were
unusable output — error payloads, timeouts, empty 200s. The scorer produced no number for
them and the table does not invent one.

This has a sharp edge worth stating: `mean(accuracy)` without filtering on `kind` silently
ignores 4.1% of the corpus and overstates every vendor. The alternative — storing 0.0 — would
bake an interpretation into the table and be just as silent for anyone who disagreed with it.
`kind` is non-null on every row so a correct aggregation has something to stand on.

### Columns

Every scalar the scorer returns, not the twelve the first run happened to keep: a column is
about eight bytes and re-running to recover a missing one costs hours, so the asymmetry only
points one way. The whole table is **~600 KB for 5,198 rows**.

`matching_exact` says *that* the metric approximated; `approximated` and `skipped_open_maps`
say *where*. Those are the lists you want when a number looks wrong, and they are empty on
almost every row.

### What it costs

One vendor, 660 documents, four workers: **1,161s wall, 4,484s CPU.** Nine vendors, minus the
12.4% the dedup avoids, is **~9.8 CPU-hours, ~2.5h wall at `-j4`**.

That number is almost entirely three documents -- 61% of the CPU, and one of them is 97% of a
single vendor's wall clock, with three workers idle for the last nineteen minutes. Scheduling
hides this across nine vendors and cannot hide it for one. See `TO_LOOK_AT.md` item 15.

### The atlas is the corpus

`corpus.parquet` is not an index of the directory; it is the statement of what the benchmark
contains at the moment it was built. Scoring runs over its rows and nothing else, so a
half-copied document or a scratch directory cannot join a benchmark by being present.

Each row names its files **relative to the atlas** and records their sha256. Paths are stored
rather than implied so a row says what it points at, and they are still required to be
`<doc_id>/ground_truth.json` and `<doc_id>/schema.json` -- a path column that can say anything
is one that can point outside the corpus, or at another corpus.

The hashes make editing data deliberate: a changed ground truth stops the run instead of
quietly producing different numbers, and re-running `build-corpus` is how you record that you
meant it. There is one rebuild verb, and it always describes what is on disk -- which documents
are in the benchmark is decided by which are in the directory, so curating is arranging files.

### The corpus is versioned, not mutated

```
scores/<corpus_version>/<scorer_version>/summary.parquet
scores/<corpus_version>/<scorer_version>/verdicts/<doc_id>.parquet
```

A score means nothing without knowing which corpus produced it, so the corpus version is in
the path. It is a sha256 over the atlas's `(doc_id, gt_sha256, schema_sha256)`, truncated to 16 hex
characters -- computed from the contents rather than taken from a HuggingFace revision, so it
works on a corpus that has never been pushed anywhere, which is every corpus while it is being
built.

Every listed document counts toward it, including ones no vendor has scored yet: the version
identifies the corpus, not the subset that happened to be covered. Curating a row out changes
it too, which is correct -- a filtered corpus is a different benchmark.

**Adding a document or correcting a ground truth therefore produces a new corpus, a new
directory, and a fresh run against it.** Both identifiers are immutable, and a finished table
is never overwritten -- `--recheck` rescores and compares instead, which is the determinism
audit and the only way to rescore a completed version.

This replaced an incremental builder that rescored only what had changed, and the reason is
not that the incremental one was complicated, though it was. It made growing the corpus in
place cheap, and a benchmark that is cheap to grow in place is one where "scored 94.2" means
94.2 against whatever the corpus happened to be that day. Versions are the honest model:
comparable within one, visibly incomparable across two. Growing the benchmark is publishing a
new version of it.

Rows still carry `gt_sha256` and `schema_sha256`, as evidence rather than as a cache key -- so
you can check that a table really scored the corpus its path claims.

The cost is real and worth stating: a one-line correction to a single gold rescores the whole
corpus. That is minutes of fan-out on a cluster and hours on a laptop, so iterate against a
subset directory -- the corpus contract is just a directory, and a small one is a valid one.

### The scorer name has to be a real commit

A working tree with edits cannot be named by a commit without the name lying, and the lie is
not cosmetic — the whole value of the key is that `same prediction_id AND same scorer_version,
different score` means a bug. Two runs from two different dirty trees under one commit would
trip that check forever while nothing was wrong.

So a dirty tree is named `dirty` instead, which cannot be mistaken for a commit.
`scores/dirty/` is scratch: freely overwritten, never published.

Just `dirty`, not `dirty-<timestamp>`. The timestamp looked stricter, and made the whole
feature inert -- every run wrote a new directory, so nothing was ever current and incremental
scoring never engaged once while iterating. **Untracked files count as dirty**: a stray module in
`omni_extract_bench/` changes what gets imported, and `git diff` cannot see it.

### Crash recovery

Hours of compute with the table written only at the end is hours to lose to one crash, and
`scores` is the one thing here nothing else can reproduce cheaply. Each row is appended to a
journal as it lands; the parquet is written once the set is complete.

Only a clean tree resumes — a dirty one gets a fresh name each run and starts over. That is
intended: resuming after an edit would merge rows from two different codebases into one table,
which is the corruption everything above exists to prevent.

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

---

## What building it changed

A throwaway converter was built first -- the whole corpus and one vendor, end to end -- to
find what the design had wrong. Three things.

**Row groups stopped mattering, and that section is gone.** It argued for content-sized row
groups from a real measurement, 8x faster single-row reads, taken when payloads were embedded
in the parquet. Once files hold the payloads the atlases are **35-64 KB** and the machinery
has nothing to do. The probe made it obvious by sizing groups with `gt_bytes` -- a number
describing data the table does not contain.

**Conversion is seconds, not minutes.** 0.4s for 660 documents, 0.1s for a vendor, ~8s to
extract every result span across all 5,936. It had been sized as a batch job worth scheduling;
it is one fast pass and needs no resumability.

**Verbatim extraction applies to every file, not a few.** 100% differ from a reserialised
form, which turns a nicety into the reason the id means anything.

Unchanged by contact: flat filenames (660/660 round-tripped, including the 28 with spaces and
brackets), and `verify` catching all three drift modes -- a row with no file, a file with no
row, and a **one-byte edit**, which is the one nothing else would notice.

### What building the scores writer changed

Probed the same way, and it moved three decisions.

**`unwrap` became an assertion.** The plan was to carry the scorer's envelope-peeling across.
But in this layout both builders store the bare extraction, so an envelope arriving at the
scorer means the builder that wrote it is wrong -- and unwrapping it silently would hide that
while producing a score under a `prediction_id` that hashed the wrapper. Two bugs came from
unwrap rules guessing wrong. A rule that can guess wrong is now a check that cannot.

Measured first, because the assertion is only safe if nothing legitimately trips it: **0 of
660 schemas declare a top-level `result` property, and 0 ground truths are wrapped.** The one
case where `result` is a real field therefore has no example in the corpus, which is exactly
why it is a test rather than an assumption.

**Importing the scorer's `unwrap` pulled in boto3 and huggingface_hub**, because scoring and
downloading live in one file. The writer scores a local tree and should need neither. The rule
it follows instead: reading this layout depends on nothing that fetches it.

**Dedup turned out to be worth doing**, and was not in the plan. 738 of 5,936 scorings (12.4%)
are a second vendor's byte-identical copy, 360 of those keys shared across vendors.

Unchanged by contact: the key, the immutability rule, and the non-determinism check -- which
reproduced 200 documents exactly across a rerun, and agreed between the serial and pooled
paths.
