# To look at

Open questions from scoring the R2 predictions against the Hub ground truth. Ordered by
how likely each is to be a harness problem rather than a model result.

**Nothing here is closed.** Every item carries what is known so far and an explicit
**Next**, including the ones where the mechanism is already understood -- knowing why
something happens is not the same as having decided what to do about it. Where a question
has actually been ruled out, it says so inside the item rather than being deleted, so the
same ground does not get covered twice.

Sample: 40 documents, 8 per suite, seed 20260910, the three giant documents excluded.
Predictions from `s3://datalab-training-pipelines/omni-extract-bench/runs/full/baselines/`.

Four vendors scored on the identical sample, ~35s each at `-j 6`:

| vendor | score | on returned | notes |
|---|---|---|---|
| reducto | **88.78** | 88.78 | |
| datalab | **88.66** | 88.66 | |
| extend | **82.36** | 82.36 | |
| llamaextract | **71.34** | 73.17 | 1 document returned a vendor error, scored 0 (item 10) |

All four use the same `{"result": ..., "_secs": ...}` envelope, so bug 1 below hits every
vendor equally -- nothing here is scoreable through `cli.py` as it stands. Treat these
numbers as a pipeline check, not a leaderboard: 40 of 660 documents, and items 2-4 are
unresolved.

---

## 1. The `{"result": ...}` envelope is not unwrapped by `cli.py`

**Known so far:** the cause is nailed down and a working rule exists in the scratch
script. Nothing is fixed in the repo.

**Next:** decide where the rule belongs -- `prediction_io.py` alongside `usable()` is
the obvious home, since this is the same question that module already owns -- then
make `usable()` reject a bare envelope so this can never pass silently again.

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

**Known so far:** the mechanism is fully understood, but nothing is decided. This is
probably the most consequential item in the file.

**Next:** pick one of the three options at the bottom of this section, or reject all
three deliberately. Whichever way it goes, the choice belongs in `METRIC_SPEC.md`,
because right now the spec does not say that formatting counts on free-text arrays.

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

**Known so far:** the extras are identified and they are real balance-sheet line
items, not noise. Whether the gold or the models are wrong is NOT established.

**Next:** open the Nike 10-Q and check whether `short_term_debt` actually has a value
in the filing. If it does, the gold is incomplete and this document is mis-scoring
every vendor. That single check decides the item. If the gold is right, the question
becomes why four vendors invent the same line items.

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

**Known so far:** it is systematic across all four vendors, and items 2 and 3 explain
two of the eight documents. The other six are unexamined.

**Next:** look at the remaining six `contextual` documents the way item 3 was looked
at -- what the extras and misses actually are. Until then it is unknown whether this
suite is genuinely hard or systematically mis-scored.

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

**CORRECTION.** An earlier version of this entry said the document "timed out and dropped
the 5,711-row `facilities` array". That was wrong, and wrong in a way worth recording: a
sketch helper printed only the first six keys of the prediction object, `facilities` was
the seventh, and its absence from a truncated *display* was read as absence from the
*data*. The array is present with 5,552 of 5,711 rows.

**Known so far:** the payload is complete and the failure is extraction quality, not
truncation. Reducto scores **99.73** on the identical document, so the input is
extractable.

```
datalab   found 49.9%   read_right 93.7%   rows paired 5,291 / 5,711
reducto   found 99.7%   read_right 100.0%  rows paired 5,708 / 5,711
```

`read_right` of 93.7% says the values it placed are mostly right; `found` of 49.9% says it
placed half of them under names the gold does not use. Two separate causes, both
systematic:

**It skipped the footnote columns.** The missing addresses are overwhelmingly `*_footnote`
-- 5,587 `asc_11_footnote`, 3,935 `asc_9_footnote`, 3,713 `asc_12_footnote`, and so on
down every measure. In CMS ASC quality data a footnote is the marker that a value is
suppressed or not applicable, so these are not decoration.

**It attributed a whole column block to the wrong measure.** It produced 3,475 rows each of
`asc_11_interval_lower_limit`, `asc_11_interval_upper_limit` and `asc_11_total_cases` --
names the schema never declares -- while leaving the gold's `asc_12_*` equivalents empty.
On a row keyed `facility_id=05C0001831` the gold's `asc_12_rshv_rate` of 11.9 appears in
datalab's `asc_11_rate`. Same number, wrong measure.

17,009 invented/fabricated addresses, and **none** of them are `_citations`/`_meta`
sidecars -- they are all real column names.

**Next:** this is a genuine quality result, so it does NOT belong on the retry list.
The open question is whether the ASC-11/ASC-12 block really is mislabelled or whether the
document carries both and the gold records only one -- the same question as item 3. Worth
deciding once for both, since the answer changes whether these are model errors or gold
gaps.

---

## 6. Three documents are scored by the greedy fallback, not optimal matching

**Known so far:** the cost of greedy is measured (<=0.177 points, always
under-crediting) and raising the caps is not worth it. Two comments in `matching.py`
are provably false and nothing has been changed.

**Next:** surface the greedy count in the run summary so "3 of 660 approximately matched"
is visible at the top rather than per-document. The two false comments are FIXED -- the
"unreachable on any realistic input" claim and the 6881 x 6881 figure, which appeared in
four places and is really 26,725 x 26,725.

**Unmeasured:** the 0.177 figure comes from documents of 532-4,278 rows. Nobody has
measured what greedy costs at 26,725 rows, which needs an exact solve on the biggest
document to compare against (~40 minutes for the pair).

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

Two comments in `matching.py` were false and have been corrected: `force_approximate`
claimed *"The greedy path is unreachable on any realistic input"* when three corpus
documents reach it, and the largest-array figure was given as 6881 x 6881 / 47 million
cells in four places when it is 26,725 x 26,725 / 714 million, 15x larger.

---

## 7. `_greedy` has no memory bound, and it is the fallback for running out of memory

**Known so far:** measured on this corpus greedy is safe (1.74% density, 1.86 GB vs
~5.8 GB for exact). The unbounded case is reasoned, not observed.

**Next:** switch `_greedy` to three numpy arrays plus an `argsort` instead of a list
of tuples. It is a contained change, keeps the algorithm and results identical, and
takes the pathological case from ~71 GB to ~8.6 GB. Worth doing before a provider
hands you a document with a repeated status column.

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

**Known so far:** the two paths prepare schemas differently and 243 of 660 schemas
are affected. Whether it changes any score is NOT established.

**Next:** one-line fix to `scripts/test.py`. Before or after, check whether stripping
actually moves any score -- the scorer only reads `additionalProperties`, so it may
be inert, and knowing which would be useful.

`cli.py` prepares schemas with `resolve_refs(strip_benchmark_keys(schema))`;
`scripts/test.py` only calls `resolve_refs`. **243 of the 660 schemas** contain
`evaluation_config` or `default`, so the two paths are not grading the same schema.

---

## 9. `pytest tests/` cannot collect this suite at all

**Known so far:** reproducible and understood. Pre-existing, unrelated to anything on
this branch.

**Next:** guard the two `sys.exit()` calls behind `if __name__ == "__main__"`, or
commit to the script style and add a runner that executes each file and aggregates
exit codes. Either way CI should not be able to report success on zero tests.

`tests/test_capture.py` and `tests/test_cli.py` call `sys.exit()` at import time. Under
pytest that raises `SystemExit` during collection and the whole run dies with
`INTERNALERROR`, reporting **"no tests ran"** -- not a failure, an abort. Every test file
in the suite is written script-style (run directly, exit non-zero on failure), so this is
consistent, but anyone who points CI at `pytest` gets a green-looking nothing.

Either keep the house style and drive the suite with a runner that executes each file, or
guard those two `sys.exit()` calls behind `if __name__ == "__main__"`.

---

## 10. One llamaextract prediction is a recorded vendor error, not a miss

**Known so far:** fully understood. Recorded so the 0 is read as a vendor-side
service failure rather than a bad extraction.

**Next:** nothing for the harness. When the full run happens, count how many
`__error__` payloads each vendor has -- that is a coverage statistic worth reporting
next to the score, and it is invisible in the mean.

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

**Known so far:** the top two are not separable at this sample size. The extend and
llamaextract gaps are large and stable enough to trust.

**Next:** run the full 660 for all four vendors -- this is the item most other things
are waiting on. Needs item 1 fixed first, or every document scores 0.

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

**Known so far:** the cost is characterised and the three guards that make it
tractable are known and working.

**Next:** only relevant if item 2 lands on the similarity option. If it does, the
guards are not optional and the prefilter belongs in the implementation, not the
caller.

Strict matching compares canonical values by hash equality, O(1) a pair, inside a fill
that is already O(n*m). Similarity replaces that with a string diff, so a 1,300-row array
goes from 1.7M hash compares to 1.7M diffs -- the first attempt at the sweep had not
finished one pass after ten minutes.

Three guards brought it to ~37% over strict: a length gate (only strings >= 40 chars are
eligible), difflib's own cheap upper bounds (`real_quick_ratio` then `quick_ratio`) before
`ratio()`, and a memo, because the fill prices most pairs twice. Any real implementation
needs all three.

---

## 13. Two vendors had a timeout-recovery mechanism the other seven did not

**Known so far:** `recovered_after_timeout` appears in the envelopes of **datalab and
extend only**. Nine datalab documents and four extend documents came back through it, all
at ~1800s -- past the cap the run describes as uniform:

```
timeout at the uniform 1800s limit; the vendor was still processing after 546 polls over 1803s
```

Everyone else hit that wall and was recorded as a failure: azure-cu 42 timeouts, and 58
azure-cu / 59 claude / 22 mistral runs over 1500s with no recovery available.

Two things are now established and worth not re-deriving:

* The recovered payloads are **complete**, not partial. Row counts against gold:
  19,486/19,486, 18,493/18,494, 26,350/26,725, 5,552/5,711, 1,152/1,152, 1,083/1,083.
  An earlier worry that recovery returned truncated output was unfounded.
* Recovery does **not** explain datalab's lead over reducto. Reducto scored at or above
  datalab on nearly every recovered document (99.2, 99.6, 99.9, 99.9, 99.9, 99.8) while
  never needing recovery at all. What recovery bought was documents that claude, gpt,
  gemini, mistral and azure-cu were recorded as LOSING -- up to five vendors on the same
  document.

**FOUND.** The runner is still not on this machine, but it stamped its configuration into
every `_raw/` record in R2, so the policy did not need it. From
`baselines/<vendor>/_raw/<suite>/<doc>.json`:

```
run_manifest.timeout_s      1800          the "uniform 1800s limit" azure-cu's errors quote
run_manifest.max_output_tokens  64000
run_manifest.tier           balanced / extraction_performance / gpt-4.1-mini / ...
cost.wall_s                 the measured time
recovered_after_timeout     true, on 13 documents, and absent otherwise
```

`recovered_after_timeout` is a bare boolean, always `true`, never anything else. Twelve of
the thirteen ran 1801-1812s against the 1800s deadline; the odd one out,
`short__07021-2016-p0016`, took 105s, so the flag is not purely about the clock.

The `http` array says what recovery actually did, and it is less than the name suggests:

```
datalab, normal   (92s)    POST /extract  ->  GET /extract/{id}  ->  GET /extract/{id}
datalab, RECOVERED (1812s) POST /extract  ->  GET /extract/{id}                one poll
extend,  RECOVERED (1804s) POST /extract_runs  ->  5x GET /extract_runs/{id}
```

Every request returned 200. There is no retry, no resubmit, no second POST. The deadline
passed, the harness polled the existing job once more, and the answer was there -- the job
had finished and the runner had stopped waiting. All four recovered records also carry
`conventions_applied: false` against `true` on a normal one, so the post-processing step was
skipped on the late payload.

So the asymmetry is about thirty seconds of patience, and it is worth 9 documents to datalab
and 4 to extend. The other seven vendors hit 1800s and were recorded as failures.

**Next:** decide whether a document that finished just past the deadline counts. Re-running
with a strict uniform timeout costs datalab those 9 and extend those 4; re-running with a
post-deadline poll for everyone would give azure-cu's 42 timeouts the same chance. Either is
defensible; the current state -- two vendors getting it and seven not -- is not. Note this
does not need the runner: `timeout_s` and the poll behaviour are both recoverable from
`_raw/`, which is also the only provenance we have for how any prediction was produced.

---

## 14. The full nine-vendor run, and the 185-document retry list

**Known so far:** 660 documents x 9 vendors, exact matching, **zero harness failures** --
every loss is a recorded vendor error, with no empty `{}` and no malformed responses.

```
datalab       91.75   660/660        claude        83.04   613/660
reducto       91.67   660/660        gpt           82.15   654/660
extend        88.08   660/660        llamaextract  79.58   656/660
gemini        72.21   559/660        mistral       70.19   614/660
azure-cu      54.14   607/660
```

Coverage, not quality, drives the bottom half. On documents actually returned, gemini is
**85.25** -- third -- against 72.21 on the full corpus. Claude loses 6.4 points to zeros,
mistral 5.3, azure-cu 4.7.

Splitting the losses by whether re-running could change them gives **185 retry / 72 real
limits** (`retry_list.csv` in the session scratchpad):

```
  91  200 with an empty body                    42  hit the harness's own 1800s cap
  40  gateway error envelope, not the vendor's  22  prompt exceeds the context window
  20  5xx backend error                          7  hard 400 invalid request
  18  200 with only keep-alive padding
```

The left column is infrastructure wearing a vendor's name; the right is real capability
and re-running changes nothing. The blank-200 signature appears for gemini (61), claude
(25) and gpt (5) -- three unrelated vendors sharing one symptom -- and claude's error
bodies mention Cloudflare while the 400s carry a `"Provider returned error"` envelope that
is a proxy's, not Anthropic's or Google's.

**Next:** retry the 185 under whatever policy item 13 settles on, then rebuild the board.
Gemini stands to move ~13 points and would reorder the middle of the table. Until then
this is a coverage measurement as much as a quality one, and should be reported as both.

---

## 15. `order_matters` is never passed, and 109 documents ask for ordering

**Known so far:** the capability exists and is exercised by the tests; no scoring path
supplies it. The candidate list is measured but NOT triaged, and it has visible false
positives.

**Next:** triage the 50 + 56 arrays below by hand -- the regex cannot tell "ordered
list" from "ordered/requisitioned line item" -- then decide where the surviving names
live. They are per-document, so they belong beside the corpus rather than in the scorer;
the schema is the natural home, since it is already what the harness reads.

`grep order_matters` finds it in `score.py` (17), `METRIC_SPEC.md` (4) and five test
files. It appears in **neither `cli.py` nor `scripts/score_r2.py`**, so every array in
every published run has been scored order-insensitively, including the ones whose ground
truth explicitly asks for an order.

Scanning all 660 schemas (refs resolved) for ordering language in an array's own
`description` or its `items.description`: **151 arrays across 109 documents.** They split
three ways, and the split is the whole point -- only one group actually needs the flag.

| group | arrays | docs | what it means |
| --- | --- | --- | --- |
| object rows, **no positional field** | 50 | 48 | order is unrecoverable from content -- the real candidates |
| **scalar arrays** | 56 | 45 | order ignored *and* a near-miss double-charged (§8) |
| object rows **with** a positional field | 45 | 40 | `rank`, `line_no`, `subject_number` already carry it; pairing recovers the order without the flag |

Examples from the first group:

```
holdings          "one Holding object per row, in the order they appear in the table"
creditors         "Every creditor row in document order"
items             "Every genuine catalog line item ... in reading order"
authors           "Complete, ordered list of authors as printed in the byline"
tables[*].rows    "One entry per printed data line of the schedule, top to bottom"
```

The scalar-array group is the more consequential one, because there position is the only
identity an element has:

```
tables[*].column_headers        "The ordered list of LEAF data-column labels, left to right"
creditors[*].mailing_address    "Mailing-address lines in order (street, city/state/zip)"
risk_management_cycle_stages    "Names of the stages in the council's risk-management cycle diagram, in order"
section_letters                 "... section headers in the paper body, in their order of appearance"
```

A model that returns a correct mailing address with the city line first scores it perfect
today. One that returns the right column headers left-to-right gets no credit for the
ordering it was asked for.

**Two false positives are already visible in the candidate list**, which is why this is a
triage job and not a patch: `line_items` matches on "Every **ordered**/requisitioned line
item", where "ordered" means purchased; and `age_groups` matches on "the **ranked**
results for that group", which describes its children rather than itself -- the ranking is
real one level down at `age_groups[*].results`, and that array carries a `rank` field, so
it is already recoverable and needs nothing.

Full scan, with every path, phrase and description, is in the session scratchpad as
`order_scan.json`.

**Not a scorer bug.** §5 says the default is order-free deliberately, because the order
rows appear in a document is usually an accident of layout. The gap is that the escape
hatch was built, documented, tested -- and then never wired to a run.

---

## 16. What the long-text values actually are, and where item 2 really applies

**Known so far:** the exposure is measured and it is far narrower than item 2 implies.
Nothing is decided; item 2's three options are still open.

**Next:** item 2 can be scoped to bibliographies before choosing between them. Whatever
is chosen must NOT touch category 4 below, where exact match is correct.

Threshold first, because item 12's similarity sweep used a 40-char gate and a different
number would make the two measurements incomparable. **100 characters is the better cut,
and the corpus says why:** at >=100, 6,448 of 6,478 values (99.5%) are eight words or
more, so length alone isolates prose; at >=40 only 52% are, and the rest are URLs, codes
and concatenated identifiers where exact match is entirely fair. Using 40 would double the
apparent exposure with values that are not the problem.

For scale: 66.8% of gold scalars are strings and their **median length is 8 characters**
(p90 25, p99 73). Long text is genuinely the tail -- 6,478 values, 0.22% of the corpus.

| suite | gold values | >=100ch | share | in scalar arrays | docs with any |
| --- | --- | --- | --- | --- | --- |
| **contextual** | 12,364 | 1,803 | **14.58%** | **1,722 (95.5%)** | 23/35 |
| internal | 10,473 | 201 | 1.92% | 30 | 44/207 |
| extractbench | 606,916 | 3,015 | 0.50% | 75 | 62/329 |
| micro1 | 1,962,615 | 1,395 | 0.07% | 0 | 10/47 |
| longarray | 293,139 | 64 | 0.02% | 0 | 13/42 |

**Of the 1,827 long values that sit inside scalar arrays -- the double-charge case -- 1,721
are one field, `citations`, in six `contextual` research papers.** Item 2's "arrays of
free-text strings are near-unscoreable" is not a corpus-wide property. It is bibliographies,
in six documents, in the suite §7 weights at a fifth of the headline. That also answers
item 4's question about `contextual`: five of its 35 documents are 69-93% long-text-in-
scalar-arrays by value count.

The long values are four different things, and they do not want the same treatment:

1. **Bibliographies.** `citations` (n=1,721, median 201 chars, 100% in scalar arrays),
   schema: *"List of works cited by this paper."* The only category with the double charge.
2. **Prose commentary**, named fields, charged once. `facts.text` (406), `directives_comment`
   (1,238 -- near-identical OFAC boilerplate), `communications.text`, `inspection_notes`.
3. **Boilerplate and delimited lists.** `standing_offer_description` (491),
   `trade_agreements` (358, median **502 chars**) -- the latter is a comma-joined list
   crammed into one scalar, asked for "as printed in the column". A model returning the same
   agreements in a different order fails outright, and no scoring rule fixes that; the schema
   is asking for a serialization rather than a fact.
4. **Long structured identifiers, where exact match is correct.** `holdings.security_name`
   (326), e.g. `Abry Liquid Credit CLO Ltd., Series 2025-2A, Class C, (3-mo. CME Term SOFR +
   2.10%), 5.78%, 01/15/39`. Every token is content: two tranches differing in one digit are
   different securities. Also `aliases.name`, `creditors.creditor_name`, `parties.address`.
   A similarity threshold here would be actively wrong.

A rule aimed at "long free text" in general would loosen 6,478 values to fix 1,721, and
would loosen category 4 along the way.

Noticed while measuring, and now item 15: `facts` is described as *"Every numbered fact
paragraph **in order**"*, and we score that array order-insensitively.

---

## 17. The published board scored two vendors on 660 documents and seven on 657

**Known so far:** the defect is confirmed and the three documents identified. The
per-vendor effect is measured for the three vendors re-scored so far and is NOT uniform
in sign, so the published ordering cannot be corrected by arithmetic -- it needs the
complete run.

**Next:** rebuild the board from a run where every vendor covers the same 660, and make
`score_r2.py` refuse to write a summary whose document count disagrees with the manifest.
The summaries claimed `660 / 660` while the file held 657, which is the same class of
silent-coverage bug as item 1.

`r2_scores/*.jsonl` holds **657** rows for seven of the nine vendors; only `datalab` and
`reducto` have 660. The missing documents are **the same three in every one of the seven**,
and they are exactly item 6's three giants:

```
extractbench/long__real_oklahoma_unclaimed_2024                  26,725 rows
micro1/06_19_Government_zoning_and_land_use_geospatial_datasets  19,486 rows
micro1/hard__Municipal_continuing_disclosure_..._efis            18,494 rows
```

The untracked `r2_reducto.log` and `r2_last2.log` fit the sequence: the nine-vendor run
stopped short of the giants, then datalab and reducto were re-run to completion and the
other seven were not. Every `*.summary.txt` nonetheless reports `documents 660`,
`scored 660` -- the count came from the manifest rather than from the rows actually written.

**The effect is not a uniform inflation, which was the first guess and it was wrong.**
These documents are easy for the top two and catastrophic for at least one other vendor:

| vendor | with the three | without | effect | scores on the three |
| --- | --- | --- | --- | --- |
| datalab | 91.75 | 91.72 | **+0.030** | 98.16, 97.23, 99.87 |
| reducto | 92.09 | 92.06 | **+0.027** | 94.72, 99.23, 100.00 |
| extend | 88.09 | 88.17 | **-0.080** | **24.92**, 99.95, 87.02 |

So excluding them *helped* extend and would have *hurt* datalab and reducto. They are
strongly discriminating documents -- the top two score 94-100 while extend collapses to
24.92 on the Oklahoma array -- and dropping them removed signal rather than adding a
constant.

Which means the published item 14 ordering below the top two cannot be repaired by adding
an offset; each of the remaining six has to be re-scored on the full corpus before the
board means anything. That is already in flight.

**Watch for this when reading any old-vs-new comparison**: `extend` appears to fall
88.16 -> 88.09, but 88.16 was its mean over 657 documents. Like-for-like on the 657 it
shares with the new run, it *rose* by +0.006, and the three giants account for the -0.08.

---

## 18. Four documents are partial responses scored as quality results, and none is on the retry list

**Known so far:** four cases are identified across two vendors, and the evidence points
vendor-side in each. `retry_list.csv` cannot contain them by construction: it is built
from empty and error payloads, and these are structurally valid 200s.

**Next:** re-run these four. For the extend one, re-run it twice -- once as configured
and once with the array strategy unset -- because the mode involved is one WE selected
(below), and that is the only way to tell a transient chunk failure from a deterministic
one. Then add a completeness check to `usable()` or beside it, since `retry_list.csv`
will keep missing this class until something looks at row counts.

| vendor | document | score | read_right | what came back |
| --- | --- | --- | --- | --- |
| extend | `extractbench/long__real_oklahoma_unclaimed_2024` | 24.92 | 97.8% | 7,142 of 26,725 rows |
| llamaextract | `extractbench/short__sec_13f_0009_coatue_management` | 0.55 | 100.0% | 14 addresses; peer median 2,174 |
| llamaextract | `extractbench/short__sec_13f_0019_soros_fund_management` | 0.79 | 100.0% | 27 addresses; peer median 2,907 |
| llamaextract | `micro1/f47be8a4__aaq-mntrpt-2005-vic-report-final` | 11.16 | 99.7% | 816 addresses; peer median 6,676 |

The `read_right` column is the tell. In all four, essentially everything returned is
CORRECT. These are not extraction failures; they are coverage failures wearing a quality
score, and the mean cannot tell the difference.

**The extend case, because the mechanism is fully visible.** Its 7,142 rows form exactly
**15 contiguous blocks, one per page, in ascending page order** -- pages 2, 3, 6, 12, 15,
16, 18-22, 38, 57, 58, 59 -- and every one of those pages is complete (436/441, 445/445,
459/459, 472/474). The other 43 pages returned zero rows. That is concatenated per-chunk
output with 43 chunks contributing nothing; a harness-side loss would give partial pages
or interleaving, not whole-page blocks in sorted order.

Ruled out on our side: `capture.py` has no truncation path; the file is valid JSON that
parses in full (a truncated write would be malformed); the envelope is a plain
`{result, _secs}` at 1,083s with no `recovered_after_timeout`, so it is not the 1,800s cap;
and extend returned the OTHER two giants complete -- 19,486 rows against a gold 19,486, and
18,493 against 18,494 -- so there is no size ceiling on either side.

**What is ours** is `omni_extract_bench/harness/providers/extend_provider.py:52`:

```python
ARRAY_STRATEGY = os.environ.get("EXTEND_ARRAY_STRATEGY", "large_array_max_context")
```

sent as `advancedOptions.arrayStrategy.type`. Per the comment there, this is Extend's MAX
array mode and the vendor flagged that we were benchmarking without it. So the dropout
happened in a non-default mode we selected on the vendor's own advice, on the largest array
in the corpus -- which is precisely the case the mode exists for. Nothing in the stored
artifacts says whether the default would have done better, and the re-run is the only way
to find out. If it reproduces, it is a real result and worth telling Extend.

**The gap this exposes.** `usable()` asks whether a payload parses and is non-empty. All
four pass. `retry_list.csv`'s 258 rows are drawn from empty bodies, gateway errors, 5xx,
timeouts and 400s -- every category assumes the failure is visible in the envelope. A
response missing three quarters of its rows is invisible to all of it. Two cheap detectors
would have caught all four before they reached a board: rows returned against the peer
median for that document, or pages covered against the document's own page span. Neither
needs ground truth, so both could run at capture time.
