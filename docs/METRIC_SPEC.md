# Metric specification

The complete definition of the benchmark score. Everything the grader does is here; anything
not here is a bug. Properties are stated formally and each is enforced by a test in
`tests/test_metric_properties.py`.

## 1. Objects

- **Document** `d` with ground truth `G` and schema `S`. A vendor returns prediction `P`.
- **Leaf**: a scalar reachable in `G` or `P` by walking objects and arrays. `null` on both
  sides is *not* a leaf (it asserts nothing). Objects and arrays are never leaves themselves.
- **Row**: an object element of an array. Arrays of scalars are compared as multisets.

## 2. Leaf comparison — `cmp(p, g) ∈ {0, 1}`

Deterministic and type-directed. No model participates.

| gold type | rule |
| --- | --- |
| boolean | canonical equality |
| integer / identifier-like | exact after canonicalisation — IDs, phone numbers and account numbers never fuzzy-match |
| decimal number | equal within `1e-6 · max(1, |g|)`; **sign notation** folded (`(98.2)`, `−98.2`, `98.2-` all parse to −98.2) but **sign value preserved** (`−98.2 ≠ +98.2`) |
| date-like string | equal if both parse to the same calendar date under any supported format |
| other string | canonical equality (case, whitespace, punctuation, smart quotes, unicode fractions) |

Canonicalisation is a single shared function applied identically to `p` and `g`, so the
comparison is symmetric by construction.

## 3. Array semantics — optimal assignment

For an array with predicted rows `P₁…Pₙ` and gold rows `G₁…Gₘ`:

1. **Weight** `w(Pᵢ, Gⱼ)` = number of leaves that match if the two are paired. The weight uses
   **the same equivalences as `cmp`** (dates, float precision, unicode fractions, sign
   notation). It must: when the field that *distinguishes* two rows is a date, a weight that
   only accepted literal equality paired the rows arbitrarily and a correct extraction scored
   50 — reintroducing the very format penalty §2 removes.
2. **Pairing** = the assignment maximising `Σ w` over all matchings — solved exactly
   (Hungarian, `omni_extract_bench/matching.py`). Pairs with `w = 0` are discarded: two rows sharing nothing
   are not a pair.
3. Leaves inside paired rows are scored recursively. Leaves of unpaired rows count as misses —
   on either side, so both omission and over-production are penalised.
4. **Blocking**: rows may be partitioned by fields compared exactly; rows in different blocks
   can never pair, so per-block optimal is globally optimal.
5. **Exactness budget**: the assignment is solved by `scipy.optimize.linear_sum_assignment`,
   so what bounds exactness is the **cost matrix**, which is `O(n·m)` whatever solves it —
   the largest gold array here is 6881 rows, 47 million cells, 0.38 GB as float64, only ~12%
   under the ceiling. Beyond `MAX_CELLS` (or `MAX_EXACT` on the smaller dimension) the solver
   falls back to greedy and *reports* `exact=False` rather than claiming optimality. Greedy
   costs 0.2–0.8% of assignment weight, and it is charged only to providers returning very
   large tables, so it is surfaced per grade (`matching_exact`, `greedy_blocks`) and never
   silent.

*Why exact matching matters:* the previous key-inference heuristic lost 2.6 points on a single
10-Q, and 28% of array-cell misses on one subset were the right value attached to the wrong row.

## 4. Document score

    leaf_accuracy(d) = 100 · (matched leaves) / (total leaves)

`total leaves` counts every gold leaf plus every spurious predicted leaf. Recall and precision
over rows are reported separately and are never folded into `leaf_accuracy`.

## 5. Aggregation

    subset_score  = mean of leaf_accuracy over that subset's documents
    UNIFIED       = mean of the subset scores

Equal weight per subset, because subsets differ ~10× in size; leaf- or document-weighting
would let the largest subset decide the benchmark and would silently re-weight it whenever a
subset grew. A document a vendor returned nothing usable for scores **0** — excluding failures
would reward fragility. Coverage is reported beside the score, never inside it.

## 6. Ground truth adjustments

Applied uniformly, before scoring, to every vendor alike:

- **Placeholder rows dropped**: a gold row whose payload fields are all `null` asserts no fact.
  Charging a vendor for omitting it penalises everyone for an unstated convention.
- **Verified corrections**: applied as an overlay only where a human read the source and the
  evidence is recorded (`GT_LEDGER.md`). Benchmark corpora are never edited in place.

## 7. Properties

| # | property | statement |
| --- | --- | --- |
| P1 | Identity | `score(G, G) = 100` for every `G` |
| P2 | Permutation invariance | reordering any array in `P` or `G` leaves the score unchanged |
| P3 | Monotonicity | correcting one wrong leaf never lowers the score |
| P4 | No double counting | each gold leaf contributes exactly 1 to the denominator |
| P5 | Determinism | identical inputs produce identical output, always |
| P6 | Coverage honesty | a missing prediction scores 0 and is never dropped from the mean |
| P7 | Symmetry of canonicalisation | `cmp(p, g) = cmp(g, p)` |
| P8 | Optimality | no pairing of array rows yields a higher score than the one chosen |
| P9 | Null neutrality | `null` on both sides adds nothing to numerator or denominator |
| P10 | Sign fidelity | sign notation is free; a wrong sign is never free |
| P11 | Nesting invariance | wrapping a document in an extra level changes neither score nor denominator |
| P12 | Omission is charged | returning a subset of the gold rows never scores 100 |
| P13 | Scalar arrays are scored | an array of scalars contributes leaves at every depth |
| P14 | Pairing/scoring agreement | any equivalence `cmp` honours is honoured when choosing the pairing |
| P15 | No silent approximation | a grade that used the greedy fallback says so |

P1, P2, P5–P7, P9, P10 were already enforced. P3, P4 and P8 were added when this spec was
written — the plan flagged monotonicity and no-double-counting as unasserted, and optimality
was claimed before it was tested.

**P11–P15 are regressions, not hypotheticals.** Each corresponds to a defect that reached the
leaderboard before it was caught; the README summarises them. P11/P12 cover a defect that made omission
free for top-level arrays — one provider scored 100.0 on a document where it returned almost
nothing. P11 is the strongest of these: if depth cannot change the score, no depth-dependent
scoring path can exist, which is the whole class rather than the instance.

## 8. Deliberate non-goals

- No LLM judge in scoring. Measured worth ≈ 0.2 points on one subset; not worth
  nondeterminism in a benchmark.
- No per-subset scoring paths. One grader or the comparison is meaningless.
- No substring acceptance and no sign flipping to accommodate ground-truth conventions.
  Conventions are written into the schema instead (`schema_overlay.py`) so vendors are told,
  not guessed at.


## 9. Open maps are not evaluated

A schema node declaring `additionalProperties` leaves the property names to the document: the
extractor must invent them by reading headings off the page. This benchmark does not evaluate
that shape, and the reason is delivery rather than taste — `dialects.STRICT_ALLOWED_KEYS` does
not forward the keyword, so a strict vendor receives a bare `{"type": "object"}` and has
nothing to answer with. Grading the node would score a request the harness never made.

Such a subtree is skipped on **both** sides: it contributes to neither the numerator nor the
denominator, exactly as a `null` does. The rest of the document scores normally, and the skip
is reported on the grade (`ignored_open_maps`) so an ungraded region can never pass unnoticed.
Detection reads *explicit* presence of the keyword as intent; under JSON Schema semantics
`additionalProperties` defaults to true, which would make every object qualify. With no schema
supplied nothing is skipped, because nothing can be identified.

Support can be added later if the corpus needs it. The natural shape is an array of
`{name, value}` rows, which every vendor can produce and which the array machinery already
grades — the heading becomes a value rather than an address.

### Object keys are addresses

An object's keys are matched literally. A prediction is generated against the schema, so its
property names are the schema's property names, which are also ground truth's — a predicted
key that is not exactly a gold key names a field the extractor invented, and is charged as
spurious while gold's unmatched key is charged as missing.

Keys were briefly compared by canonical form so that a key differing only in case still
joined. That existed solely for open maps, which §9 now excludes, so the folding had no case
left to serve. It also required a collision guard — folding can merge two distinct keys of one
object and silently discard a value — and it was never consistent: `canonical` strips
`, - . / ( )` and whitespace but keeps the underscore, so it forgave `Invoice_No` and not
`invoice no`.

## 10. `null` asserts nothing, and is scored that way

**The rule.** A `null` is not a leaf and gets no address. The metric scores *facts*, so an
assertion is scored and an absence is not. Everything below follows from that one sentence.

| gold | prediction | verdict | accuracy | denominator |
| --- | --- | --- | --- | --- |
| `{a: 1, b: null}` | `{a: 1, b: null}` | — | 100.0 | 1 |
| `{a: 1, b: null}` | `{a: 1}` | — | 100.0 | 1 |
| `{a: 1, b: null}` | `{a: 1, b: 5}` | `b` spurious | 50.0 | 2 |
| `{a: 1, b: 5}` | `{a: 1, b: null}` | `b` missing | 50.0 | 2 |
| `{a: 1, b: 5}` | `{a: 1}` | `b` missing | 50.0 | 2 |

Read the first two rows together: agreeing that a field is empty earns nothing, and neither
does omitting it. Read the last two together: emitting `null` where gold has a value is
charged exactly as omitting it is, and the two are currently **indistinguishable** — both
report `missing`, so a diagnostic cannot separate a model that declined from one that never
tried. Recovering that distinction means carrying the prediction's `null` addresses through
alignment, which nothing does today.

**Why absence-agreement earns nothing.** The loud argument is that it would pay for laziness:
on a schema of 20 fields where gold fills 3, a prediction consisting of nothing but nulls
would score **85.0** instead of 0.0. The quieter argument is the one that decides it —
counting absence-agreement makes the score depend on **schema width instead of document
content**. Add fifty optional fields to a schema and every provider's score rises, with no
document and no extraction changed. Two benchmarks over the same corpus with differently
verbose schemas would stop being comparable, which is the problem this metric exists to fix.
Not charging gold-`null`/prediction-absent is the same argument mirrored: there is no fact
there to find, so failing to find it costs nothing.

This is property **P9**.

**A prediction that says `null` scores exactly as one that omits the key**, and that is a
comparability guarantee rather than a convenience. `dialects.to_strict_dialect` rewrites
properties as `["string", "null"]` because strict vendors require every declared property to
be present and use `null` to mean "no value"; permissive vendors simply omit the key. The
harness therefore *causes* the two conventions to coexist across providers. If the two scored
differently, a vendor's score would move with the serialization convention this harness
imposed on it rather than with how well it read the document.

That is the same argument as schema width, one level down. Absence-agreement cannot count, or
schema width would move scores; absence-notation cannot matter, or dialect would. Both reduce
to one rule: **score the facts in the document, and nothing about the shape of the request.**

The price is specific. A `null` may be a considered decline while an absent key may be a
truncated response, and this decision makes them indistinguishable — so this benchmark cannot
report whether a model abstains honestly. Recovering that would mean carrying the
*prediction's* `null` addresses through alignment. Note that the fabrication measure in §11
is unaffected: it keys off **gold's** nulls, and gold is never renumbered.

**Rows that are entirely null are dropped, on both sides.** A gold row whose payload is all
`null` asserts no fact, so charging a vendor for omitting it would penalise everyone for an
unstated convention (§6). The same filter runs over the prediction, which means an *invented*
all-null row is free — a model may pad its output with empty rows without penalty. That is
consistent with scoring facts, an empty row being no claim at all, but it does mean output
bloat is not measured here.

**Inside an ordered array, position is the address, so a `null` occupies one.** With
`order_matters` naming an array, a gold `null` becomes a placeholder the prediction has to
keep:

| gold | prediction | accuracy | why |
| --- | --- | --- | --- |
| `[a, null, c]` | `[a, null, c]` | 100.0 | `c` is at index 2 on both sides |
| `[a, null, c]` | `[a, b, c]` | 66.7 | `b` is spurious, but `c` stays at index 2 |
| `[a, null, c]` | `[a, c]` | 33.3 | closing the gap moves `c` to index 1: spurious *and* missing |

Note the incentive in the last two rows: filling the empty slot with junk scores **higher**
than omitting it, because the junk preserves the alignment of everything after it. That falls
out of "the index is the address", which is the whole meaning of the flag, and it is recorded
here rather than fixed — but it is a reason to name an array in `order_matters` only when its
order genuinely carries meaning. Order-free arrays have no such trap: `[a, null, c]` and
`[a, c]` both score 100.0, because the null was never an address to begin with.

## 11. False assertions, split three ways

`1 - precision` is the rate at which a prediction asserts something untrue. That single
number hides three different bugs, so `grade` reports them separately:

| count | meaning | what it points at |
| --- | --- | --- |
| `misread` | the document has a value at this address; the model read it wrongly | OCR, units, sign, date format |
| `fabricated` | **the schema offered this slot, the document is silent, the model filled it** | schema pressure — the model will not leave a field empty |
| `invented_item` | an array element that paired with nothing | over-segmentation: a header or subtotal read as data, a row emitted twice |
| `invented_field` | a name the schema never declared | schema non-adherence: an invented key, or a synonym for a declared one |

Rows and fields are separated because their magnitudes differ. One invented row contributes
a leaf for every column it has; one invented field contributes one. Lumped together, a single
hallucinated twelve-column row is indistinguishable from twelve invented field names, and the
two have different causes and different fixes.

`fabricated` is the sharpest hallucination signal available from this data: the document does
not say this, and the model said it anyway, in a slot the schema held open.

### The authority is the schema, not gold's `null`s

A gold field written `null` and a gold field left out entirely mean the same thing (§10). So
classifying by gold's `null`s would sort two semantically identical ground truths into
different buckets. `_schema_leaves` reads the slots off the schema instead, which is the only
authority that does not move. Nothing inside an `additionalProperties` object counts as a
slot, because that subtree is not graded at all — a value nobody asked for cannot be a filled
slot.

This is why **the schema is required**. Without one, `fabricated` could only be reported as
zero, which would read as "this model never fabricates" — a false claim rather than a missing
measurement. A missing schema also silently disables open-map detection (§9), so requiring it
closes both holes at once.

### The identities

    asserted = matched + misread + fabricated + invented_item + invented_field
    gold     = matched + misread + unfound
    total    = matched + misread + unfound + fabricated + invented_item + invented_field

    precision = matched / asserted
    recall    = matched / gold
    accuracy  = matched / total
    f1        = 2·precision·recall / (precision + recall)

Note that `misread` is counted in both `asserted` and `gold`, but only **once** in `total`.
That is the entire reason `accuracy` and `f1` differ: a value read wrongly is one gold fact
you failed to recover (charged once), and simultaneously one false thing you asserted plus
one true thing you missed (charged twice).

These counts are exactly the histogram of the verdicts `explain` returns, so the two surfaces
cannot drift apart. Both are asserted by `tests/test_false_assertions.py`, over the worked
example and over 2400 generated gradings.

### A scalar array has no cells to misread

Elements of a scalar array compare as a multiset, so they have no identity. A value read
wrongly there is therefore **not** a `misread`: it is one gold element nobody produced
(`unfound`) plus one element the model produced that is not in the document
(`invented_item`). It is charged on both sides, which costs more than the same error in a
named field:

| error | verdicts | denominator | accuracy |
| --- | --- | --- | --- |
| `name` read wrongly | 1 `misread` | 3 | 66.7 |
| `q[1]` read wrongly | 1 `unfound` + 1 `invented_item` | 4 | 50.0 |

This is not new behaviour — the multiset denominator has always been
`len(gold) + len(pred) − matches` — but it was undocumented. It follows from the array being
order-free: with no cell identity there is nothing for a value to be *wrong about*, only
content that is present or absent. Naming the array in `order_matters` gives its positions
meaning and restores `misread` for it.
