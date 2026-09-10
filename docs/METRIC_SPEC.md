# Metric specification

The complete definition of the benchmark score. Anything the grader does that is not here is
a bug. Every property in §9 is enforced by a test, and the tests are named where they matter.

## The idea

Give every value in a document an address, then compare addresses.

Object keys are addresses already. The model was handed the schema, so it uses the key names
the ground truth uses: `invoice.total` means the same thing in both documents.

Array indices are not addresses. The model emits rows in whatever order it read them off the
page, so its row 0 may be the ground truth's row 2. That is the only real problem to solve:
work out which predicted row goes with which gold row, renumber the prediction to match, and
then every value has an address that means the same thing on both sides.

Scoring is then set arithmetic on two sets of addresses.

## 1. Addresses

- **Document** `d` with ground truth `G`, prediction `P`, and schema `S`. The schema is
  **required** — §5 and §4 both depend on it.
- **Value**: a scalar at a complete path. Objects and arrays are never values; addresses pass
  through them. `{"a": {"b": [{"c": 1}]}}` holds exactly one value, at `a.b[0].c`.
- **Address**: the path to a value, as tagged steps — a key or an index. The tag matters,
  because a document may contain the key `"0"` and it must not join index `0`.
- **Row**: an object element of an array. Arrays of scalars are compared as multisets, so
  their elements are values, not rows.
- **Node key**: an address with the index numbers blanked, so `books[0].chapters` and
  `books[7].chapters` share one name. Written `books[*].chapters`. This is how an array is
  named in `order_matters`.

## 2. Comparing one value

Deterministic and type-directed. No model participates. `canon_key` is the single comparison
rule, used by leaf scoring and by row pairing alike, so the two cannot disagree.

| gold type | rule |
| --- | --- |
| boolean | canonical equality |
| integer / identifier-like | exact after canonicalisation — IDs, phone numbers and account numbers never fuzzy-match |
| decimal number | rounded to **7 significant digits**; **sign notation** folded (`(98.2)`, `−98.2`, `98.2-` all parse to −98.2) but **sign value preserved** (`−98.2 ≠ +98.2`) |
| date-like string | equal if both parse to the same calendar date under any supported format. A timestamp at **midnight** counts as its date, because that is how vendors spell a date; any other time of day is kept and compared, so two different times still differ |
| other string | canonical equality (case, whitespace, punctuation, smart quotes, unicode fractions) |

Canonicalisation is applied identically to `p` and `g`, so comparison is symmetric by
construction.

**Seven significant digits is a bucket, not a tolerance.** Values are compared by turning each
into a key and testing the keys for equality — that is what lets the scorer do set arithmetic
on addresses instead of comparing pairs. The cost is boundary effects: `0.99999994` and
`1.00000004` differ by one part in ten million and land in different buckets, so they are
scored as a disagreement. A relative tolerance would call them equal, but a tolerance cannot
be expressed as a key. Anything agreeing to seven significant digits without straddling a
boundary matches.

**Format differences are free only where the table above says so.** These are folded, and
are tested by `tests/test_comparison_surface.py`:

    1,234 = 1234        $1,234.56 = 1234.56      12.34% = 12.34      1 234.56 = 1234.56
    5.00 = 5            1.0 = 1                  (1,234.56) = -1234.56
    1234.56- = -1234.56    −1234.56 = -1234.56
    31/10/2024 = 2024-10-31    10/31/24 = 2024-10-31    Oct 31, 2024 = 2024-10-31
    2024-10-31T00:00:00Z = 2024-10-31        a timestamp at MIDNIGHT is a date
    2024-10-31T09:00:00Z = 2024-10-31 09:00:00     two spellings of one instant
    ACME CORP = Acme Corp      "Acme  Corp." = "Acme Corp"      TRUE = true
    "N/A" = "-" = "" (all canonicalise to empty)

These are **not**, and will score as disagreements even though a human would call them
formatting:

    USD 1234.56 ≠ 1234.56        1234.56 USD ≠ 1234.56       currency codes are not stripped
    1.234,56 ≠ 1234.56           European decimal notation
    2024-10-31T09:00:00Z ≠ 2024-10-31        a timestamp with a REAL time is not a date
    2024-10-31T09:00:00Z ≠ 2024-10-31T17:00:00Z    time of day is scored, not discarded
    "yes" ≠ true                 only true/false spellings are booleans

So a disagreement the grader reports is a real one *for the formats above*, and the second
list is where to look first if a provider's misses look like formatting. It is a list of
candidate scorer gaps, not of vendor errors.

## 3. Aligning arrays

For an array with predicted rows `P₁…Pₙ` and gold rows `G₁…Gₘ`:

1. **Price** every candidate pair: `w(Pᵢ, Gⱼ)` is the number of values that would match. It
   uses the same equivalences as §2. Pricing descends — two rows can be identical apart from
   a nested array — so pricing a pair means solving the pairing of the arrays inside it, one
   array level per hop.
2. **Pair** by the assignment maximising `Σ w`, solved exactly with
   `scipy.optimize.linear_sum_assignment`. Ties are broken by shared addresses. Without that
   tie-break, two solvers returned different equal-weight assignments and so different scores
   for one input — 50.60 against 51.85 — because the objective maximised matched values while
   the score divided by a union denominator. P2 and P8 are what hold this now.
3. **Discard any pair worth zero.** The bar is one matching value, and it decides the
   denominator. Clearing it means the row's values are compared against its partner's.
   Failing it means every gold value in the row is charged as missing *and* everything the
   prediction put there is charged as invented — so neither omission nor invention is free.
   Clearing the bar does not make the rest of a row free: anything it asserts that gold does
   not have still adds to the denominator, which is why a row of one readable field and five
   wrongly-guessed list items can score below omitting it (§8).
4. **Renumber** the prediction onto the gold row it paired with. A predicted row that paired
   with nothing keeps an index of its own, written `lines[p2]`, so it can never be mistaken
   for a gold row and cannot be silently dropped.
5. **Exactness budget.** What bounds exactness is the cost matrix, `O(n·m)` whatever solves
   it. Past `MAX_CELLS` (250 million) or `MAX_EXACT` (20 000 on the smaller dimension), the
   solver falls back to greedy and reports `matching_exact: false`. The largest gold array in
   this corpus is 6881 rows — 47 million cells, 19% of the cap — and solves exactly.

Order is free by default, because the order rows appear in a document is usually an artefact
of layout. Naming an array in `order_matters` makes its index an address again; see §8 for
what that trades away.

*Why exact matching matters:* the previous key-inference heuristic lost 2.6 points on a single
10-Q, and 28% of array-cell misses on one subset were the right value attached to the wrong
row.

## 4. What is reported

Once the prediction is renumbered, every address falls into exactly one bucket. These are the
verdicts `explain()` returns, one per address, and `grade()` reports their counts.

| bucket | meaning |
| --- | --- |
| `matched` | address on both sides, values agree |
| `misread` | address on both sides, values differ — the document has it, the model read it wrongly |
| `unfound` | gold address the prediction never used |
| `fabricated` | the schema offered this slot, the document is silent, the model asserted a value |
| `invented_item` | an array element that paired with nothing |
| `invented_field` | a name the schema never declared |

`fabricated` is the sharpest hallucination signal available: the document does not say this,
and the model said it anyway, in a slot the schema held open. It keys off the **schema**, not
gold's `null`s — a gold field written `null` and one left out mean the same thing (§5), so
keying off gold would sort two identical ground truths into different buckets.

`invented_item` is separate from `invented_field` because their magnitudes differ. One
invented row contributes a value for every column it has; one invented field contributes one.
Lumped together, a hallucinated twelve-column row is indistinguishable from twelve invented
key names, and those have different causes and different fixes.

**Everything else is arithmetic on those six numbers:**

    asserted  = matched + misread + fabricated + invented_item + invented_field
    gold      = matched + misread + unfound
    total     = matched + misread + unfound + fabricated + invented_item + invented_field

    accuracy   = 100 · matched / total       how much of the document you recovered
    precision  = matched / asserted          how much of what you said was true
    recall     = matched / gold
    f1         = 2·precision·recall / (precision + recall)

    found      = (matched + misread) / total did you look in the right places
    read_right = matched / (matched + misread)  ...and read them correctly
                 found · read_right = accuracy / 100

A detection is a keypath-and-value, so a value found at the right address but read wrongly is
charged on both sides — as a box with the right location and the wrong class would be.

### `accuracy` is the score; `f1` is the check on it

The two divide by different things, and the difference is worth stating exactly, because they
can rank two predictions oppositely. Write the buckets as `m` matched, `w` misread, `u`
unfound, `x` everything asserted that gold has no address for. Then

    gold      G = m + w + u
    asserted  P = m + w + x
    total     T = m + w + u + x

    accuracy = m / T          = m / (m + w + u + x)
    f1       = 2m / (G + P)   = 2m / (2m + 2w + u + x)

**`f1` is the textbook measure and charges a misread twice.** Under the detection framing,
`FP = P − m = w + x` and `FN = G − m = w + u`, so `w` is in both: a value at the right address
with the wrong content is a thing you asserted that is untrue *and* a gold fact you did not
recover. Object detection does the same with a right-place/wrong-class box. `f1` is exactly
the Dice coefficient over `(address, value)` pairs.

**`accuracy` deviates, deliberately, and charges a misread once.** Jaccard over those pairs
would be `m / (m + 2w + u + x)`, because a misread contributes a distinct gold pair and a
distinct predicted pair. `accuracy` is short by exactly `w`. That deviation *is* **P4** — each
gold value contributes exactly 1 to the denominator — and it is what makes the headline
literally interpretable: "you recovered 60% of this document". Charge a misread twice and the
denominator exceeds the number of facts in the document, and the sentence stops being true.

**The price of the deviation is that the two can disagree about which prediction is better.**
When no value is misread, `accuracy` is Jaccard and `f1` is `2J/(1+J)`, a strictly increasing
function of it, so their ordering is identical — measured over 3302 pairs, **zero**
disagreements. Introduce misreads and the monotone relationship breaks: over 3234 pairs where
one prediction misreads and the other omits and invents, they rank **2.1%** of them
oppositely. `accuracy` is systematically kinder to a model that misreads; `f1` is kinder to
one that omits and invents.

So, to be unambiguous: **`accuracy` is the score and decides any ranking.** `f1` is not a
tie-breaker for it. The signature of a model filling in fields it cannot read is `accuracy`
holding up while **`precision`** falls — not `f1`, which nets out when a good guess is mixed
with a hopeless one (§8). Report `accuracy`, `precision` and `recall` together: they are a
complete basis, from which `f1`, `found` and `read_right` all follow.

| | `misread` | accuracy | f1 |
| --- | --- | --- | --- |
| perfect | 0 | 100.00 | 100.00 |
| one value misread | 1 | 50.00 | 50.00 |
| one value missing | 0 | 50.00 | **66.67** |
| one value invented | 0 | 66.67 | **80.00** |

A misread is where they agree; a one-sided address is what splits them.

**Row counts.** `gt_rows`, `pred_rows` and `matched_rows` count object-valued array elements
at every depth. They support "returned 44 of 349 rows"; they do not feed precision or recall.
`matched_rows` is not row *correctness* — a value repeated on every row (a currency, a fiscal
year) pairs rows that are otherwise entirely wrong. For row correctness, group `explain()`'s
verdicts by the address up to the last index step.

**Honesty flags.** `matching_exact` and `approximated` say when an array was too large to
solve exactly. `skipped_open_maps` says which subtrees were not graded at all (§5).

Asserted by `tests/test_false_assertions.py`, over a worked example and 2400 generated
gradings, including that `grade`'s counts equal `explain`'s verdict histogram.

## 5. What is not scored

One rule with three instances: **the metric scores facts, so anything asserting no fact is
not scored, on either side.**

**`null` is not a value and gets no address.** Agreeing that a field is empty earns nothing;
so does omitting it. Asserting a value where gold is silent is charged. A prediction that
says `null` scores exactly as one that omits the key.

That last point is a comparability guarantee, not a convenience. `dialects.to_strict_dialect`
rewrites properties as `["string","null"]` because strict vendors must emit every declared
property and use `null` for "no value", while permissive vendors omit the key. The harness
therefore *causes* both conventions to exist across providers. If they scored differently, a
vendor's score would move with the serialization convention the harness imposed on it.

The deeper reason absence cannot earn credit: it would make the score depend on **schema width
instead of document content**. Add fifty optional fields and every provider's score rises,
with no document and no extraction changed — and two benchmarks over the same corpus with
differently verbose schemas would stop being comparable, which is the problem this metric
exists to fix. Both rules reduce to one: *score the facts in the document, and nothing about
the shape of the request.*

**`additionalProperties` objects are not graded.** Such a node leaves the property names to
the document, so the model must invent them by reading headings off the page. The reason for
skipping is delivery rather than taste: `dialects.STRICT_ALLOWED_KEYS` does not forward the
keyword, so a strict vendor receives a bare `{"type":"object"}` and has nothing to answer
with. Grading it would score a request the harness never made. Skipped on both sides,
reported in `skipped_open_maps`, and detected from *explicit* presence of the keyword.

**All-null rows are dropped, on both sides.** A gold row whose payload is entirely `null`
asserts no fact, so charging a vendor for omitting it would penalise everyone for an unstated
convention. The same filter runs over the prediction, so an invented empty row is free —
consistent with scoring facts, but it does mean output bloat is not measured here.

**What this does and does not cost.** `{"b": null}` and `{}` are indistinguishable — but they
are the same behaviour, so nothing is lost. Abstaining *is* producing no value at an address,
however it is spelled, and that is measured: a model that declines rather than guesses shows
higher `precision`, and §8's `f1` threshold prices the choice.

    declines 5 fields with null      acc 87.50   f1 93.33   precision 100.00
    omits the same 5 fields          acc 87.50   f1 93.33   precision 100.00
    guesses those 5 fields instead   acc 87.50   f1 87.50   precision  87.50

What is *not* recoverable is the intent behind one absence: whether the model weighed a field
and declined, or never reached it. Even that is usually legible from the shape of what is
missing — declining scatters across the fields a model finds hard, truncation leaves a
contiguous tail across every field of the last rows.

Asserted by `tests/test_spec_null_semantics.py`.

## 6. Ground truth adjustments

Applied uniformly, before scoring, to every vendor alike.

- **Placeholder rows dropped** — see §5.
- **Verified corrections** applied as an overlay only where a human read the source and the
  evidence was recorded. Benchmark corpora are never edited in place. Both the corrected
  corpus and its ledger are published with the data rather than with this scorer, which
  ships no benchmark data — so neither is in this repository.

## 7. Aggregation

    subset_score = mean of accuracy over that subset's documents
    UNIFIED       = mean of the subset scores

> **Not yet implemented.** `cli.py` takes a flat mean over documents, which is the thing this
> section says must not happen. Doing it properly needs each document's subset, which the run
> manifest carries. Until then a leaderboard printed by this repository is not UNIFIED.

Equal weight per subset, because the subsets really do differ by about 10× — 329, 207, 47, 42
and 35 documents in the full sample — so leaf- or document-weighting
would let the largest subset decide the benchmark and would silently re-weight it whenever a
subset grew. A document a vendor returned nothing usable for scores **0** — excluding failures
would reward fragility. Coverage is reported beside the score, never inside it.

## 8. What the metric pays for

A benchmark is a set of incentives. Three rules, for anyone building against this:

1. **Emit a row if you can read any part of it, carrying only the parts you can read.** Those
   values sit at gold's own addresses, so the row costs exactly what omitting it would have
   cost and earns whatever it got right. Never truncate to be safe.
2. **Only guess when you are more likely right than wrong.** Leaving a field empty never
   costs *more* than a wrong value, and inside an array it costs less — and `null`, `[]` and
   omitting the key are all the same thing. But a value you get right always beats silence,
   so this is a question about your hit-rate, not about caution. The threshold is below.
3. **Do not invent a row you cannot read at all.** It is charged twice.

Together: **report everything you can read. Beyond that, a guess pays only if it is more
often right than wrong.** Rule 1 is the floor and is guaranteed; rule 2 is the judgement call
on top of it.

Ten gold rows of four fields, the first five read perfectly, the last five either omitted or
attempted with *j* of 4 right:

| the last five rows | accuracy | f1 |
| --- | --- | --- |
| omitted | 50.0 | 66.7 |
| emitted, 0 of 4 right | 33.3 | 50.0 |
| emitted, 1 of 4 right | **62.5** | 62.5 |
| emitted, 2 of 4 right | 75.0 | 75.0 |
| emitted, 4 of 4 right | 100.0 | 100.0 |

Note what rule 1 does **not** say. A row that pairs is not automatically better than an
omitted one — fill its unreadable fields with guesses and the denominator grows. Take a row
of one readable field and five unreadable list items: guessed, it scores **41.2** where
omitting it scores 50.0, because the row pairs on the readable field but each wrong list
element costs two denominator slots rather than one (§4). Emitting the row with only its
readable field scores **58.3**, beating both. A strict schema cannot take that option away,
since `null` and `[]` score exactly as an omitted key does.

**Where the two headline numbers disagree, deliberately.** At one of four right, `accuracy`
prefers attempting and `f1` prefers omitting. Both are correct: the model recovered a real
fact, *and* most of what it said was false. So neither should be quoted alone.

### `accuracy` is not gameable; it is indifferent

Searched exhaustively over every strategy on a four-field document — each field omitted, filled
correctly, or filled wrongly — the highest reachable `accuracy` is monotone in the number of
correct values produced, and **no strategy scores above one that produced more correct
values**:

| correct values produced | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| best `accuracy` | 0.00 | 25.00 | 50.00 | 75.00 | 100.00 |

So `accuracy` cannot be raised except by being right more often. Guessing well raises it
because guessing well produces correct values, which is the metric working rather than an
exploit. A model that outputs `"US"` by reading the page and one that outputs it by knowing
the column is usually `US` have produced the same output, and `accuracy` — which scores
output, not process — says so. No measure that looks only at the output can separate them.

Nor does corpus diversity separate them. Measured against a model that abstains, always
guessing the modal value gains **+43.25** where the base rate is 90%, **+25.25** across a
mixed corpus averaging 53%, and **+5.00** even where the prior is inverted. Variance does not
defeat the strategy; it prices the guess at the corpus-average hit rate, which is the right
price for it.

**What `accuracy` genuinely does not do is distinguish a wrong value from a blank.** Both cost
the same, because the gold address is in the denominator either way. That is not an
inflation — nothing is gained — but it matters to whoever consumes the output, who would act
on a wrong value and would know to ignore a blank.

**`precision` is what resolves that**, and it is why the two travel together:

    accuracy    how much of the document did you recover
    precision   how much of what you said can I trust

`precision` falls for *every* wrong assertion, monotonically, so unlike `f1` it cannot be
masked by mixing a good guess with a hopeless one. `f1` answers a third question — *was that
particular guess worth making* — improving only when a guess is right more often than roughly
half the current `f1`. That is a real abstention threshold, and `accuracy` has none: its
gradient at a hopeless guess is exactly zero.

**Order-freedom is a trade, not a gift.** A wrong value costs more inside a scalar array
(50.0) than in a named field (66.7), because array elements have no identity for a value to be
*wrong about* — a wrong element is a missing item plus an extra one. What the array buys is
that reordering is free, where reordering three named fields scores 0. You cannot have both;
`order_matters` chooses. Comparability is untouched, because the shape is fixed across every
provider — unlike schema width, which varied within a comparison.

Asserted by `tests/test_incentives.py`, including that the figures above are the ones the
scorer produces.

## 9. Properties

| # | property | statement |
| --- | --- | --- |
| P1 | Identity | `score(G, G) = 100` for every `G` |
| P2 | Permutation invariance | reordering any array in `P` or `G` leaves the score unchanged |
| P3 | Monotonicity | correcting one wrong value never lowers the score |
| P4 | No double counting | each gold value contributes exactly 1 to the denominator |
| P5 | Determinism | identical inputs produce identical output, always |
| P6 | Coverage honesty | a missing prediction scores 0 and is never dropped from the mean |
| P7 | Symmetry of comparison | `cmp(p, g) = cmp(g, p)` |
| P8 | Optimality | no pairing of array rows yields a higher score than the one chosen |
| P9 | Null neutrality | `null` on both sides adds nothing to numerator or denominator |
| P10 | Sign fidelity | sign notation is free; a wrong sign is never free |
| P11 | Nesting invariance | wrapping a document in an extra level changes neither score nor denominator |
| P12 | Omission is charged | returning a subset of the gold rows never scores 100 |
| P13 | Scalar arrays are scored | an array of scalars contributes values at every depth |
| P14 | Pairing/scoring agreement | the pairing that earned the score is the pairing that is committed |
| P15 | No silent approximation | a grade that used the greedy fallback says so |
| P16 | Bucket completeness | the six buckets partition every address, and equal `explain`'s verdicts |
| P17 | Partial extraction beats omission | a row emitted with only the values read correctly scores above omitting that row |
| P18 | Configuration is checked | an `order_matters` name fitting no array raises, rather than silently applying to nothing |
| P19 | Schema is required | a grade without one raises, rather than reporting `fabricated: 0` |
| P20 | `accuracy` and `f1` agree on ranking when nothing is misread | with `misread = 0` on both sides, `accuracy` is Jaccard and `f1` is `2J/(1+J)`, so their ordering is identical |
| P21 | `accuracy` is not gameable | no prediction scores above one that produced more correct values; `accuracy` rises only by being right more often |

P11–P15 are regressions, not hypotheticals — each corresponds to a defect that reached a
leaderboard before it was caught. P11 is the strongest: if depth cannot change the score, no
depth-dependent scoring path can exist, which is the whole class rather than the instance.
P16–P19 were added with the false-assertion split; P18 replaced a silent no-op in which a
mistyped configuration left the array unordered and the run finished looking fine.

## 10. Deliberate non-goals

- **No LLM judge in scoring.** Measured worth ≈ 0.2 points on one subset; not worth
  nondeterminism in a benchmark.
- **No per-subset scoring paths.** One grader, or the comparison is meaningless.
- **No substring acceptance and no sign flipping** to accommodate ground-truth conventions.
  Conventions belong in the schema, so vendors are told rather than guessed at.
- **No separate credit for declining.** It needs none: abstaining is producing no value, and
  `precision` already distinguishes a model that declines from one that guesses (§5). A
  bonus for silence would be a second, gameable path to a good score.
