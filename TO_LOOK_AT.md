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

**Status: unexplained, and systematic across vendors.**

| vendor | accuracy | found |
|---|---|---|
| datalab | 53.05 | 57% |
| reducto | 56.69 | 59% |
| extend | 47.88 | 49% |
| llamaextract | 57.66 | 60% |

datalab's run also shows 140 fabricated and 325 invented. Four vendors clustered in a
ten-point band with `found` around 50-60% is the signature of a shared cause -- the
document, the schema, or the ground truth -- rather than four independent models each
being mediocre in the same way. Not yet diagnosed.

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
