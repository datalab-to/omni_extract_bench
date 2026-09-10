# To look at

Open questions from scoring datalab's R2 predictions against the Hub ground truth.
Found while lining up the pipeline, not yet explained. Ordered by how likely each is to
be a harness problem rather than a model result.

Sample: 40 documents, 8 per suite, seed 20260910, the three giant documents excluded.
Predictions from `s3://datalab-training-pipelines/omni-extract-bench/runs/full/baselines/`.

Four vendors scored on the identical sample, ~35s each at `-j 6`:

| vendor | score | on returned | notes |
|---|---|---|---|
| reducto | **88.78** | 88.78 | |
| datalab | **88.66** | 88.66 | |
| extend | **82.36** | 82.36 | |
| llamaextract | **71.34** | 73.17 | 1 document absent in R2, scored 0 |

All four use the same `{"result": ..., "_secs": ...}` envelope, so bug 1 below hits every
vendor equally -- nothing here is scoreable through `cli.py` as it stands. Treat these
numbers as a pipeline check, not a leaderboard: 40 of 660 documents, and items 2-4 are
unresolved.

---

## 1. The `{"result": ...}` envelope is not unwrapped by `cli.py`

**Status: understood, fixed only in the scratch script. Needs a real fix.**

`cli.py::_unwrap` unwraps only when the object is *exactly* `{"result": ...}`:

```python
if isinstance(obj, dict) and set(obj) == {"result"}:
```

Every file in this run has at least one sibling key, so nothing is ever unwrapped and the
envelope itself gets graded:

```
39x  {result, _secs}
 1x  {result, _secs, recovered_after_timeout}
```

Run the CLI as it stands over all 660 and **every datalab document scores 0.00** with
thousands of invented fields. `usable()` returns `True` on the envelope, so nothing flags
it -- it reads as total provider failure rather than a harness bug.

Guessing at metadata names does not work. A first attempt ignored leading-underscore keys
and still missed `recovered_after_timeout`. The scratch script now asks the schema: if the
schema declares no top-level `result` property, a `result` key is the envelope, whatever
sits beside it.

This is the exact failure `prediction_io.py`'s own docstring was written about -- two
notions of "usable" disagreeing, with the coverage column paying for it.

---

## 2. Arrays of free-text strings are near-unscoreable

**Status: diagnosed. This is a metric design question, not a bug. Probably the most
important item here.**

Seen on `contextual/research__survey of dimensionality reduction techniques`, which scores
**4.72 for datalab, 4.13 reducto, 3.35 extend, 4.09 llamaextract** -- `found` of 3-5% for
all four. Four independent commercial extractors do not each find 4% of a document.

The top-level structure agrees perfectly. Every field matches, and both sides return
exactly **161 citations**. The array is full-text bibliography entries:

```
GT   "Abdi, H., Lewis-Beck, M.; Bryman, A. & Futing, T. (ed.) (2003). Encyclopedia for
      research methods for the social sciences — Factor rotations in factor analyses.
      Sage, 2003, 792-795."

PRED "[Abdi2003] Abdi, H. Lewis-Beck, M.; Bryman, A. & Futing, T. (ed.) Encyclopedia for
      research methods for the social sciences Factor rotations in factor analyses
      Sage, 2003, 792-795"
```

The same citation. A `[Abdi2003]` key prefix, no parenthesised year, an em-dash dropped,
different terminal punctuation. `canon_key` demands exact equality after canonicalisation,
so the element does not match -- and because these are bare strings in an unordered array
with no sub-fields to pair on, nothing pairs. Each of the 161 is then charged **twice**,
exactly as `_best_pairing` documents: once as a gold row nobody found, once as a row the
model made up. The denominator roughly doubles while the numerator stays near zero.

So a document where all four vendors arguably extracted the bibliography correctly scores
them all near zero. Worth deciding deliberately: is that the intended reading? The options
are all benchmark-design calls, not fixes -- excluding free-text arrays from the leaf
score, scoring them by a similarity threshold rather than equality, or leaving it and
documenting that formatting counts.

Side note: extend returned **329** citations against a gold 161, so it has a duplication
problem on top.

---

## 3. `contextual/10kq__nke_10q_fy2025q2` -- every vendor 48-58

**Status: partly diagnosed. Looks like a ground-truth completeness question.**

| vendor | accuracy | found |
|---|---|---|
| datalab | 53.05 | 57% |
| reducto | 56.69 | 59% |
| extend | 47.88 | 49% |
| llamaextract | 57.66 | 60% |

datalab shows 140 fabricated and 325 invented items. Inspecting what those addresses
actually are:

```
invented item   balance_sheet.goodwill[p1].value          (a row that paired with nothing)
fabricated      balance_sheet.short_term_debt[0].value    (schema offered it, gold is silent)
```

So the models are producing balance-sheet line items the ground truth does not record --
`short_term_debt` has a slot in the schema and no value in the gold, and the models fill
it. Four vendors doing the same thing in the same places is more consistent with a gold
file that captures a subset of the statement than with four models hallucinating the same
line items. Worth checking the gold against the filing before treating this as a vendor
result.

Ruled out while looking: none of the 465 extras are `_citations`/`_meta` sidecars. Those
suffixes appear all over datalab's output and are **not** counted as invented, so they are
not the cause here and are not a scoring problem anywhere.

---

## 4. `contextual` is the worst suite for ALL FOUR vendors

**Status: no longer a datalab question. Systematic.**

| suite | datalab | reducto | extend | llamaextract |
|---|---|---|---|---|
| contextual | 65.42 | 66.08 | 56.71 | 55.64 |
| extractbench | 88.00 | 87.18 | 86.16 | 68.92 |
| internal | 96.95 | 95.33 | 95.01 | 92.94 |
| longarray | 99.81 | 98.61 | 87.78 | 78.00 |
| micro1 | 93.12 | 96.70 | 86.14 | 70.94 |

Every vendor's weakest suite by 20-30 points. Item 2 explains one of the eight documents;
item 3 is a second. Worth confirming the rest of the suite is genuinely hard before the
number is published, since `contextual` drags every headline down equally.

Three of the eight are below 70 for *every* vendor:

```
research__survey of dimensionality reduction    best  4.72   found  3-5%
10kq__nke_10q_fy2025q2                          best 57.66   found 49-60%
resume__Resume-Marketing                        best 65.98   found 36-89%
```

`resume__Resume-Marketing` spreads 22.56 to 65.98 with `found` from 36% to 89%, so that
one looks genuinely hard rather than broken -- the vendors disagree with each other, not
just with the gold.

---

## 5. `micro1/m1__Healthcare_facility_quality_measure_public_reporting` -- 46.71

**Status: understood. A real provider failure, not a harness one. Listed so it is not
re-investigated.**

```
_secs: 1807.2        recovered_after_timeout: true
```

Ran 30 minutes, timed out, returned the scalar fields (`reporting_year`,
`source_table_label`) and dropped the 5,711-row `facilities` array. The 46.71 is a fair
score for that. Note this is the one document carrying the `recovered_after_timeout` key,
so it is also the document that exposed bug 1.

---

## 6. Three documents are scored by the greedy fallback, not optimal matching

**Status: known, quantified, low impact. Two stale comments should be fixed.**

`_exact_ok` sends these to greedy -- the first over `MAX_EXACT`, the other two over
`MAX_CELLS`:

| document | rows | cells |
|---|---|---|
| `extractbench/long__real_oklahoma_unclaimed_2024` | 26,725 | 714 M |
| `micro1/06_19_Government_zoning_and_land_use_geospatial_datasets` | 19,486 | 380 M |
| `micro1/hard__Municipal_continuing_disclosure_..._efis` | 18,494 | 342 M |

Measured cost of greedy on damaged predictions: **at most 0.177 accuracy points**, and it
only ever under-credits. Across a 660-document mean that is ~0.001 points, so raising the
caps is not worth it -- exact on the largest document costs ~23 minutes and ~6 GB against
4.6 minutes and 1.86 GB today.

Two comments in `matching.py` are now false and will mislead whoever next sizes these
constants:

- `force_approximate`: *"The greedy path is unreachable on any realistic input"* -- three
  corpus documents reach it.
- `optimal_pairs`: *"the largest real array (6881 x 6881, 47 million cells)"* -- it is
  26,725 x 26,725, 714 million cells, 15x larger.

---

## 7. `_greedy` has no memory bound, and it is the fallback for running out of memory

**Status: latent. Not hit by this corpus, but the guard is on the wrong side.**

`MAX_CELLS` caps the exact solver's dense matrix; nothing caps `_greedy`, which builds a
Python list holding one `(-w, i, j)` tuple per positive-weight pair at ~100 bytes each,
against 8 bytes per cell for the matrix it replaced. So greedy only wins below ~8%
positive density.

Measured on `long__real_oklahoma_unclaimed_2024`: density **1.74%**, greedy peak 1.86 GB
against ~5.8 GB for exact. Fine here.

But `_best_pairing`'s own docstring describes the bad case -- *"if every row repeats a
value, a currency, a fiscal year, then no pair is ever worth zero"*. At that shape density
approaches 100% and the same document would want **~71 GB**, against exact's 5.32 GB. A
document one shape different from this one walks past the cap into an unbounded fallback.

Cheap fix: store the scored pairs in three numpy arrays plus an `argsort` instead of a list
of tuples -- 12 bytes per pair instead of ~100. Same algorithm, same results, and the
pathological case lands at 8.6 GB rather than 71 GB.

---

## 8. `scripts/test.py` does not strip benchmark keys

**Status: minor, but it makes the self-grade harness disagree with the CLI.**

`cli.py` prepares schemas with `resolve_refs(strip_benchmark_keys(schema))`;
`scripts/test.py` only calls `resolve_refs`. **243 of the 660 schemas** contain
`evaluation_config` or `default`, so the two paths are not grading the same schema.

---

## 9. `pytest tests/` cannot collect this suite at all

**Status: pre-existing, unrelated to any change here, and a hazard for CI.**

`tests/test_capture.py` and `tests/test_cli.py` call `sys.exit()` at import time. Under
pytest that raises `SystemExit` during collection and the whole run dies with
`INTERNALERROR`, reporting **"no tests ran"** -- not a failure, an abort. Every test file
in the suite is written script-style (run directly, exit non-zero on failure), so this is
consistent, but anyone who points CI at `pytest` gets a green-looking nothing.

Either keep the house style and drive the suite with a runner that executes each file, or
guard those two `sys.exit()` calls behind `if __name__ == "__main__"`.

---

## 10. One llamaextract prediction is a recorded vendor error, not a miss

**Status: understood. Noted so the number is read correctly.**

`longarray/cae_v2_10_n742` is present in R2 but its payload is:

```json
{"result": {"__error__": "RuntimeError: LlamaExtract FAILED: An internal service
 error occurred during processing"}, "_secs": ...}
```

`usable()` correctly rejects it, so it scores 0 and stays in the mean -- which is why
llamaextract shows `score 71.34` but `on returned 73.17`. The distinction is working as
designed; it is only worth knowing that the 0 is a vendor-side service failure rather
than a bad extraction. All 40 files exist for all four vendors; this is the only
unusable one.

---

## 11. The 40-document sample cannot separate the top two

**Status: methodological. Do not publish a datalab-vs-reducto ordering from it.**

| vendor | score |
|---|---|
| reducto | 88.78 |
| datalab | 88.66 |
| extend | 82.36 |
| llamaextract | 71.34 |

datalab and reducto sit **0.12 points** apart, and across the similarity sweep they never
separate by more than 0.42 and trade places twice. On 40 documents one document is 2.5%
of the mean, and single documents in this sample differ by 40+ points between vendors --
`m1__Healthcare_facility_quality_measure` alone scored datalab 46.71 because it timed
out. Its presence or absence moves the mean by more than the entire gap.

The extend and llamaextract gaps (~8 and ~16 points) are large and stable at every
threshold, so those are probably real. The top two need the full 660 -- or at minimum a
much larger sample -- before any ordering means anything. The sample also excludes the
three largest documents by construction.

---

## 12. Similarity matching changes the cost model of pairing

**Status: understood, mitigated in the experiment. Relevant if option 2 is ever adopted.**

Strict matching compares canonical values by hash equality, O(1) a pair, inside a fill
that is already O(n*m). Similarity replaces that with a string diff, so a 1,300-row array
goes from 1.7M hash compares to 1.7M diffs -- the first attempt at the sweep had not
finished one pass after ten minutes.

Three guards brought it to ~37% over strict: a length gate (only strings >= 40 chars are
eligible), difflib's own cheap upper bounds (`real_quick_ratio` then `quick_ratio`) before
`ratio()`, and a memo, because the fill prices most pairs twice. Any real implementation
needs all three.
