# Data layout

How the corpus, the predictions and the scores are stored, and why in that shape.

Companion to `REMOTE_RUNS.md`, which covers running the scorer over this data. This file is
about where the bytes live.

---

## The shape

```
HuggingFace (private)                 R2  datalab-training-pipelines
  corpus.parquet                        vendors/<vendor>.parquet
  metadata.parquet                      baselines/<vendor>/<suite>/<doc_id>.json
  tags.parquet                          scores/<scorer_commit>.parquet
  thumbnails.parquet
  <doc_id>/
      ground_truth.json
      schema.json
      document.pdf
```

**One rule: parquet holds metadata, files hold payloads.** `scores` and `thumbnails` are
the two deliberate exceptions, and each section says why.

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
