# Corpus observations

Findings about the published corpus rather than about the scorer. Recorded so they are not
rediscovered. Measured 2026-09-11 against the 660-document snapshot in the HuggingFace cache
(`datalab-to/omni_extract_bench`), whose subset sizes match §7: 329 / 207 / 47 / 42 / 35.

Nothing here is a scorer bug. Each is a schema- or ground-truth-side observation, and the
scorer behaves exactly as `METRIC_SPEC.md` specifies on all of it.

## Scalar arrays: widely declared, thinly populated

| | schemas declaring ≥1 scalar array | scalar-array nodes | array-of-object nodes |
| --- | --- | --- | --- |
| all 660 | **448 (67.9%)** | 1370 | 718 |

Two-thirds of schemas declare one, across 404 distinct node names, and they outnumber
array-of-object nodes nearly 2:1. But they hold only **1.4%** of actual gold values
(41,074 of 2,885,507), so the double-charge described in `METRIC_SPEC.md` §8 is diluted
to near-irrelevance corpus-wide.

The exception is the tail: **35 documents have more than half their addresses inside scalar
arrays**, and they concentrate in `contextual` (12 of 35 documents, 34%) and `internal`
(15 of 207). Since §7 weights subsets equally, `contextual` is a fifth of UNIFIED.

| share | scalar | addrs | subset | doc |
| --- | --- | --- | --- | --- |
| 97.4% | 1087 | 1116 | contextual | `research__[zhao25] a survey of LLMs` |
| 93.6% | 276 | 295 | contextual | `research__[li25] vlm survey` |
| 91.2% | 426 | 467 | extractbench | `short__census_statab_12s0711` |
| 90.9% | 502 | 552 | extractbench | `short__census_statab_employment` |

## `citations` measures transcription, not extraction

The `contextual` scalar arrays split into two populations:

- **Atomic tokens** — `keywords`, `skills`, `languages`, `lenders`, `socialLinks`; median
  7–27 characters. Exactly what the metric is built for. No concern.
- **Prose** — `citations`; median 198–216 characters, max 1292. 1793 across the subset, 1081
  in one document.

Because §2 folds typography and nothing else, a `citations` leaf measures verbatim
reproduction. Sensitivity on `research__[zhao25]`, perfect except one variance per entry:

| variance | accuracy |
| --- | --- |
| smart quotes / whitespace / trailing period / PDF hyphenation | 100.00 |
| drop trailing arXiv or DOI | 91.42 |
| elide authors to "et al." | 8.35 |
| truncate to 120 characters | 5.73 |

At 97.4% citations, that document's score is essentially a bibliography-string reproduction
test. In aggregate it is ~2.9% of UNIFIED (1793 of `contextual`'s 12,364 addresses, and
`contextual` is 1/5), so the risk is concentration rather than magnitude: **one document in
`contextual` measures a different task than the other 34, at equal weight.**

The fix is schema-side — model `citations` as an array of objects (`authors`, `title`,
`venue`, `year`). Unlike a tag list, a citation genuinely has several attributes, so the row
pairs on whatever is right and wrong fields cost one `misread` instead of two slots. See
`METRIC_SPEC.md` §8 for why the same move does nothing for a list of bare tags.

## Ground-truth artifacts in `contextual` citations

PDF text-extraction damage survives into gold. Two kinds, with opposite outcomes:

- **Hyphenation across line breaks** (`intelli- gence`, `Computa- tional`) — **costs nothing.**
  `canon_key` folds it; an extractor that correctly rejoins scores 100.00. Verified, not
  assumed.
- **Broken combining accents** (`Garc´ıa` for "García", `Dess\`ı` for "Dessì") — **does not
  fold.** The dotless ı is a distinct character. An extractor that reads the name correctly
  scores zero on that entry and is charged twice.

**25 of 1793 `contextual` citations (1.4%)** carry the second kind. Candidates for the §6
corrections overlay, which is the right channel — the corpus is never edited in place.

## Reproducing

All figures come from flattening `ground_truth.json` with `score.flatten` and walking
`schema.json` with `values.unwrap_schema`. Address counts are taken before
`drop_empty_gt_rows` and before open-map skipping, so they run slightly above the grader's
real denominators — fine for distributions, not exact.
