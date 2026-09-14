# Running the benchmark remotely

A design for scoring the corpus on Modal instead of one laptop. Where the bytes live is `DATA_LAYOUT.md`; this is about running the scorer over them. Two goals, and the second
matters more than the first.

**Fast enough to re-run after every scorer change.** A full run is 5,940 gradings and takes
about two and a half hours locally, so in practice we patch three documents at a time and
reason by hand about which numbers came from which code. That is how a single document
ended up with three different recorded values in one session.

**Reproducible, and traceable to a commit.** Today `r2_scores/*.jsonl` carries no provenance
at all. Nothing in it says which scorer produced it, and `uv run --with` re-resolves
dependencies on every invocation, so a local run is not reproducible even on the same
machine.

---

## What the work actually looks like

Measured, not assumed.

```
660 documents x 9 vendors  = 5,940 gradings
gold + schema needed        = 132 MB   (1,320 files; the PDFs are never read)
predictions, all vendors    = 568 MB   (already in R2, S3-compatible)
```

The cost is extraordinarily skewed:

```
581 of 660 documents are under 200 KB   ~5% of total work,  ~0.3s each
top 20 documents                        ~81% of total work
top 3 documents                         ~49%
the single largest                      ~24%,  1,010s on the exact path
```

Two consequences for the design. Per-document containers would spend all their time on cold
starts for the 581 small ones. And **no amount of parallelism beats the longest single
grading** — one matrix solve cannot be split, so ~17 minutes is the floor.

Local runs are bound by memory, not cores: 18 cores but 24 GB, so about 6 workers, and `-j 4`
was killed by the OS. Remote containers each get their own allocation, which is where the
speed-up comes from.

| | local | Modal |
| --- | --- | --- |
| full run | ~2.5 h, OOM-killed once | ~20-25 min |
| after a scorer change | an afternoon | a coffee break |

---

## Where the data lives, and how it is shaped

Everything below is what exists today, measured against the bucket and the HuggingFace
snapshot on 2026-09-14. A remote runner has to read exactly these and nothing else.

### Predictions: Cloudflare R2, S3-compatible

```
bucket  datalab-training-pipelines
prefix  omni-extract-bench/runs/full/
```

```
omni-extract-bench/runs/full/
├── MANIFEST.json                 1.21 MB   per-provider scores, costs, failures
├── run_full.json                 1.06 MB   per-provider per-document scores from the run
├── eval_sample_full.json        49.5 KB    the frozen document list
├── known_vendor_limits.json      3.83 KB   measured vendor limits, with evidence
├── defective_pairs.json          2.04 KB   documents excluded as defective
├── METRIC_SPEC.md                7.04 KB   the spec as it stood at run time
├── gt_source_fixes.MERGED_INTO_GOLD.json
└── baselines/
    └── <vendor>/
        ├── <suite>/<doc_id>.json          <- THE ONLY THING THE SCORER READS
        └── _raw/<suite>/<doc_id>.json     <- provider debugging; never read
```

Nine vendor directories, all suffixed `_full`:

```
azure-cu_full  claude_full  datalab_full  extend_full  gemini_full
gpt_full       llamaextract_full          mistral_full reducto_full
```

Five suites under each, and every vendor has all 660:

```
extractbench 329   internal 207   micro1 47   longarray 42   contextual 35
```

`<doc_id>` matches the corpus manifest exactly — same string, no translation, and globally
unique across suites, so a flat namespace also works.

**Read `<suite>/` and skip `_raw/`.** They are 660 objects each, and `_raw` is the larger
half: for `datalab_full`, 242.6 MB scored against 263.0 MB raw. It holds
`{raw, provider, subset, doc, llm_calls, cost, vendor_envelope, http, job_ids, schema_sent}`
— useful for diagnosing a vendor, irrelevant to scoring. Mounting it doubles the object count
for nothing.

Bytes per vendor. The first column is what a scorer reads; the second is `_raw`, which it
never does:

```
              scored     _raw        total
datalab       233 MB    263 MB      505.6 MB
llamaextract   88 MB    181 MB      268.7 MB
reducto        93 MB    149 MB      242.3 MB
extend         99 MB    141 MB      240.2 MB
claude         15 MB    121 MB      135.8 MB
azure-cu       13 MB     64 MB       77.4 MB
gpt            13 MB     56 MB       69.0 MB
gemini          8 MB     57 MB       64.8 MB
mistral        11 MB     51 MB       61.5 MB
                                   ------
scored, all nine                    568 MB
```

datalab is 29x gemini on the scored side, entirely because of the `_citations` and `_meta`
sidecars -- about two thirds of its payload is provenance the scorer discards.

`extend_full` additionally carries four `_superseded_reserved_id/` objects. Ignore them; the
authoritative prediction is the one under `<suite>/`.

### Prediction file format

```json
{ "result": { ...the extraction... }, "_secs": 92.3 }
```

`_secs` is wall time for that document. Some carry a third key, `recovered_after_timeout`,
and a runner must not assume the envelope has exactly two:

```
39 of 40 sampled    {result, _secs}
 1 of 40            {result, _secs, recovered_after_timeout}
```

**Unwrap by asking the schema, not by matching key names.** If the schema declares no
top-level `result` property, then a `result` key is the envelope. Keying on metadata names
fails — an attempt that ignored leading-underscore keys still missed
`recovered_after_timeout`, and `cli.py`'s exact-match rule unwraps nothing at all, which makes
every document of every vendor score 0.00 while `usable()` still returns True. See
`TO_LOOK_AT.md` item 1.

A failed extraction is recorded rather than absent:

```json
{ "result": { "__error__": "RuntimeError: LlamaExtract FAILED: ..." }, "_secs": 12.4 }
```

`prediction_io.usable()` rejects those, and they score 0 and stay in the mean.

### Corpus: HuggingFace, `datalab-to/omni_extract_bench`

```
<snapshot>/
├── manifest.json          {suite: [{doc_id, source_id, has_pdf}, ...]}   660 entries
├── exclusions.json        1 document dropped from the frozen 663
├── defective_pairs.json   2 more
└── data/<suite>/<doc_id>/
    ├── ground_truth.json  <- read
    ├── schema.json        <- read
    └── document.pdf       <- NEVER read by the scorer
```

`manifest.json`'s 660 are already the scored set: the excluded and defective documents are
absent from it, so no filtering is needed.

**Stage gold and schema only.** 1,320 files, 132 MB. The full snapshot is 882 MB because of
the PDFs, and staging those wastes 85% of the transfer.

Schemas need two transforms before grading, in this order, and both are required rather than
tidy:

```python
schema = resolve_refs(strip_benchmark_keys(json.loads(raw)))
```

`strip_benchmark_keys` removes `evaluation_config` and `default`, which are harness
annotations rather than part of the task — 243 of the 660 schemas contain them.
`resolve_refs` inlines local `$ref`, because the scorer refuses a schema it cannot see
through. Ground truth needs neither: none of the 660 is enveloped.

### Credentials

```
R2_ENDPOINT_URL   R2_ACCESS_KEY_ID   R2_SECRET_ACCESS_KEY
```

Currently read from `~/datalab/gke_pipelines/.env`. A remote runner should take them from a
Modal Secret instead, and never bake them into an image.

### What a container needs, in total

```
mount  R2 bucket, prefix baselines/<vendor>/<suite>/    read-only, skip _raw
mount  Volume with data/<suite>/<doc_id>/{ground_truth,schema}.json   read-only, 132 MB
write  one JSONL line per document
```

No PDFs, no `_raw`, no network beyond those two.

---

## Shape

### Unit of work: a shard, not a document

One container per (vendor, document) is 5,940 containers, mostly cold start. Instead, sort
each vendor's documents by ground-truth size and pack them into shards of roughly equal
estimated cost — the expensive twenty end up alone, the long tail batches.

```
shard = [(vendor, doc_id), ...]           ~40-60 shards total
```

Longest-first within a shard, as `build_scores.py` does, so a shard's makespan is not set
by scheduling luck.

### Memory: request it, do not guess

The routing commit makes this computable. Exact costs `n * m * 8` bytes and the row counts are
known before dispatch, so a shard can ask for the size it needs:

```
memory = max over shard of (n * m * 8) + 1 GB headroom
```

Greedy is the exception — its cost is unbounded until `_greedy` stores pairs compactly, so any
shard containing a greedy-routed block takes a conservative allocation. `oklahoma` is the only
one in this corpus.

### Data: stage once, mount read-only

- **Gold and schema** (132 MB) into a Modal Volume, keyed by the corpus snapshot SHA. Written
  once, mounted read-only by every container. Never re-downloaded.
- **Predictions** stay in R2 and are read through `CloudBucketMount` — S3-compatible, no copy.
- **PDFs are never staged.** The scorer reads `ground_truth.json` and `schema.json` only.

### Provenance: stamp every row

The part that fixes the real problem. Every result row carries what produced it:

```json
{
  "vendor": "...", "suite": "...", "doc": "...", "accuracy": 99.2327697813,
  "_run": {
    "commit": "67e1fa7...",         "dirty": false,
    "corpus_snapshot": "2216ea5...", "image_id": "im-...",
    "python": "3.11.16", "numpy": "2.4.6", "scipy": "1.17.1",
    "started": "2026-09-14T10:08:13Z"
  }
}
```

The image is built from a pinned commit and refuses to build from a dirty tree, so
`commit` cannot lie. Dependency versions are pinned in the image rather than resolved at
launch, which is what makes two runs of the same commit comparable at all.

With this, "which code produced this number" stops being something to reconstruct by hand.

---

## Launching it

The point is that it is one command.

```bash
modal run remote/score.py                       # HEAD, all vendors, all documents
modal run remote/score.py --vendors datalab,reducto
modal run remote/score.py --commit 30bb927      # re-run an older scorer, unchanged
modal run remote/score.py --docs-over 1000000   # just the expensive tail
```

Results land in a Volume as one JSONL per shard, then a local `collect` merges them into
`r2_scores/<vendor>.jsonl` and writes the leaderboard. Shards are independent, so a failed one
is retried alone rather than restarting the run.

`--commit` is what makes a regression answerable: run two commits over the same corpus and diff
the outputs, instead of arguing from memory about which numbers were current.

---

## What to verify before trusting any of it

**Cross-platform determinism.** Modal is x86, the dev machine is ARM. The costs are
`matched * scale + shared` — integers held in float64, exact below 2^53 — so the solve should
agree. But `linear_sum_assignment` tie-breaking and `canon_key`'s Decimal path are assumptions
until checked. Grade the same ~20 documents both places and diff the full result dicts, not
just accuracy. **This gates everything else.**

**That a shard is not a unit of scoring.** Scores must not depend on how documents were packed.
Run one vendor as a single shard and as ten, and require identical output.

**That the greedy path is still bounded.** Until item 2 lands, a container hitting a dense
greedy block can OOM exactly as the laptop did. The routing commit removed the known cases;
it did not remove the shape.

---

## Cost

Roughly 15 CPU-hours for a full run, so **$5-15** at Modal's rates, and under a dollar for a
three-document re-run. Not a factor in the decision.

---

## Order of work

1. `remote/score.py` — image pinned to a commit, Volume for gold, `CloudBucketMount` for R2,
   one function taking a shard.
2. The determinism check above. If it fails, stop and understand it before building further.
3. Sharding by estimated cost, and memory sized per shard.
4. `collect` — merge shards, verify every (vendor, document) appears exactly once, write the
   leaderboard.
5. Re-run the full corpus at the current commit and diff against `r2_scores/` as it stands.
   Any difference is either a platform issue or something the local runs got wrong.

Step 5 is the real test of the whole thing, and its result is worth recording whichever way
it comes out.
