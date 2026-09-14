# Metric specification

The complete definition of the benchmark score. Anything the grader does that is not here is
a bug. Every property in §9 is enforced by a test, and the tests are named where they matter.

§11 is the practical end of it: what to check before trusting a number, and what the metric
is measuring that no part of the scorer decides. Observations about the published corpus
rather than about the scorer are in `CORPUS_NOTES.md`.

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
  `books[7].chapters` share one name. It is internal; what you pass to `order_matters` is its
  printed form, `"books[*].chapters"` — the same string `explain` shows you.

## 2. Comparing one value

Deterministic and type-directed. No model participates. `canon_key` is the single comparison
rule, used by leaf scoring and by row pairing alike, so the two cannot disagree.

| gold type | rule |
| --- | --- |
| boolean | canonical equality |
| integer / identifier-like | exact after canonicalisation — IDs, phone numbers and account numbers never fuzzy-match |
| decimal number | integer part exact, **fraction rounded to 7 significant digits**; **sign notation** folded (`(98.2)`, `−98.2`, `98.2-` all parse to −98.2) but **sign value preserved** (`−98.2 ≠ +98.2`) |
| date-like string | equal if both parse to the same calendar date under any supported format. A timestamp at **midnight** counts as its date, because that is how vendors spell a date; any other time of day is kept and compared, so two different times still differ |
| other string | canonical equality (case, whitespace, punctuation, smart quotes, unicode fractions) |

Canonicalisation is applied identically to `p` and `g`, so comparison is symmetric by
construction.

**What sorts a value into "integer" or "decimal" is whether its text contains a `.`** — that
and nothing else. `8303911426` is never parsed as a number and so is compared exactly, which
is what stops an account number fuzzy-matching. `8303911426.0` *is* parsed — and because an
integral float keys as its integer, the two agree.

**Rounding is the one rule here that folds values rather than spellings.** Everything else
folds two ways of writing one value — `"5.00"` and `5`, `(98.2)` and `−98.2`, `01/15/2024` and
`2024-01-15`. `33.33333333` and `33.3333333` are two *different* values, and only rounding
calls them equal; it is there to forgive ground truth re-derived at a different precision than
the page printed. The integer part is compared exactly and the fraction keeps 7 significant
digits of its own, so the rule is **inert for any fraction of 7 significant digits or fewer** —
cents, prices, quantities and rates to six places are compared exactly, at every magnitude.
`canon_key` in `omni_extract_bench/values.py` carries the rest of the argument, including why
nothing gentler than rounding can serve as a key.

**Format differences are free only where the table above says so.** These are folded, and
are tested by `tests/test_comparison_surface.py`:

    1,234 = 1234        $1,234.56 = 1234.56      12.34% = 12.34      1 234.56 = 1234.56
    5.00 = 5            1.0 = 1                  (1,234.56) = -1234.56
    1234.56- = -1234.56    −1234.56 = -1234.56
    31/10/2024 = 2024-10-31    10/31/24 = 2024-10-31    Oct 31, 2024 = 2024-10-31
    2024-10-31T00:00:00Z = 2024-10-31        a timestamp at MIDNIGHT is a date
    2024-10-31T09:00:00Z = 2024-10-31 09:00:00     two spellings of one instant
    ACME CORP = Acme Corp      "Acme  Corp." = "Acme Corp"      TRUE = true
    "N/A" = "-" = "none" = "na"    the placeholder WORDS fold into each other

`""` is NOT in that list, and the difference is the point: a placeholder word is something
the page printed, so it has an address and compares like any other value. An empty string is
what a BLANK cell serialises to, so it has no address at all (§5) -- gold `"N/A"` against a
predicted `""` is a miss, not a match.

These are **not** folded, and will score as disagreements even though a human would call them
formatting:

    USD 1234.56 ≠ 1234.56        1234.56 USD ≠ 1234.56       currency codes are not stripped
    1.234,56 ≠ 1234.56           European decimal notation
    2024-10-31T09:00:00Z ≠ 2024-10-31        a timestamp with a REAL time is not a date
    2024-10-31T09:00:00Z ≠ 2024-10-31T17:00:00Z    time of day is scored, not discarded
    "yes" ≠ true                 only true/false spellings are booleans

So a disagreement the grader reports is a real one *for the formats above*, and the second
list is where to look first if a provider's misses look like formatting. It is a list of
candidate scorer gaps, not of vendor errors.

### These rules fold spelling, not wording

Comparison is `canon_key(p) == canon_key(g)`, an equality test, so there is no partial credit
for a string and one character decides it; §10 rules out the two mechanisms that could soften
that. For a name, code, date or amount, spelling is the whole job. For prose it is not — the
realistic disagreements on a long string are edits. **So a prose-valued leaf measures verbatim
transcription.** On a 1081-entry bibliography list, perfect but for one variance per entry
(`CORPUS_NOTES.md`):

    smart quotes, whitespace, trailing period, PDF hyphenation     100.00
    a trailing arXiv or DOI dropped                                 91.42
    authors elided to "et al."                                       8.35
    truncated to 120 characters                                      5.73

Fix it in the schema, not the scorer: decompose prose into fields so agreement can be partial.
§8 covers what that changes, and why it does nothing for a list of bare tags.

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
   Clearing the bar does not make the rest of the row free either: anything it asserts that
   gold does not have still adds to the denominator (§8).
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
| `invented_item` | a value under an array element that paired with nothing |
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

    accuracy   = 100 · matched / total       how much you recovered, net of inventions
    precision  = matched / asserted          how much of what you said was true
    recall     = matched / gold              how much you recovered, inventions ignored
    f1         = 2·precision·recall / (precision + recall)

    found      = (matched + misread) / total did you look in the right places
    read_right = matched / (matched + misread)  ...and read them correctly
                 found · read_right = accuracy / 100

A detection is a keypath-and-value, so a value found at the right address but read wrongly is
charged on both sides — as a box with the right location and the wrong class would be.

### Why `accuracy` and not `f1` or Jaccard

`accuracy` reads the alignment; the others cannot. `f1` and Jaccard are functions of
`matched`, `|gold|` and `|asserted|` alone, so these two are identical to them:

    gold {a:1, b:2}   pred {a:1, b:99}     found b, misread it
    gold {a:1, b:2}   pred {a:1, c:99}     missed b, invented c

Both give `matched = 1`, `|gold| = 2`, `|asserted| = 2`. `f1` scores both 50.00 and Jaccard
both 33.33. `accuracy` scores them 50.00 and 33.33, because it knows the first pair shares an
address. **That distinction is kept on purpose:** a misread means the extractor *located* the
field, which is a different problem from not finding it.

`f1` is reported too, and equals `accuracy` exactly when both documents use the same
addresses. Where they differ they can rank two predictions oppositely (2.1% of contrasting
pairs), so **`accuracy` decides a ranking** and `f1` is not a tie-breaker.

### What `accuracy` charges, and the one thing it does not

**It cannot be inflated.** `total = |gold| + invented`, so `accuracy ≤ matched / |gold|` — the
ceiling rises only by being right more often. And every unpaired row adds all of its values to
the denominator, so proposing all 27 combinations of a three-key row to guarantee one hit
scores **3.70**, not 100. At scale the same arithmetic bites harder: ten gold rows returned
perfectly score 100.00, and those same ten rows with a thousand invented ones alongside them
score **0.99**.

**It is indifferent in exactly one place: at an address where gold has a value, a wrong value
costs what a blank costs.** Both recovered nothing there, which is the right answer to *how
much of this document did you recover*. The other question — *can I trust what it did say* —
belongs to `precision`, which charges every wrong assertion.

Asserting where the document is **silent** is not in that exemption. It creates an address
gold does not have, so `accuracy` charges it — and the bucket table above counts it as
`fabricated`.

**Row counts.** `gt_rows`, `pred_rows` and `matched_rows` count object-valued array elements
at every depth. They support "returned 44 of 349 rows"; they do not feed precision or recall.
`matched_rows` is not row *correctness* — a value repeated on every row (a currency, a fiscal
year) pairs rows that are otherwise entirely wrong. For row correctness, group `explain()`'s
verdicts by the address up to the last index step.

**The split is stable; the pairing is not unique.** Several pairings can be equally optimal —
the objective pins the optimal *value*, not which rows achieved it — and the split reads that
choice. Rows are therefore sorted by a content key before the solve. Without it, 35 of 600
documents reported a different split under a permutation, while `accuracy`, `precision`,
`recall` and `f1` never moved. P22, `tests/test_pairing_determinism.py`.

**Honesty flags.** `matching_exact` and `approximated` say when an array was too large to
solve exactly. `skipped_open_maps` says which subtrees were not graded at all (§5); nothing
else in the output reveals that.

Asserted by `tests/test_false_assertions.py`, over a worked example and 2400 generated
gradings, including that `grade`'s counts equal `explain`'s verdict histogram.

## 5. What is not scored

One rule with three instances: **the metric scores facts, so anything asserting no fact is
not scored, on either side.**

**`null` is not a value and gets no address.** Agreeing that a field is empty earns nothing;
so does omitting it. Asserting a value where gold is silent is charged. A prediction that
says `null` scores exactly as one that omits the key.

**The empty string is a third spelling of the same thing.** A document can *print* `N/A`; it
cannot print emptiness, so `""` is what a blank cell becomes on the way into JSON — exactly
what `null` means. `null`, `""`, whitespace and an absent key therefore all score alike, on
both sides -- at every level, including inside an array row (see below).

The rule stops there, and the corpus is why. The placeholder *words* are ink on the page:
`N/A`, `None`, `-` and `not applicable` appear as real gold values **24,980 times**, and 67
documents hold both those strings and `null` in the same file — the annotation distinguishes
"the page printed N/A" from "the page is silent", so folding them would delete answers. `""`
carries no such risk: all **1,369** gold empty strings in the corpus are blank cells, 1,196 of
them a single empty column in one check register.

Without this rule the score moved with a vendor's serialization habit rather than with what it
read: one provider's house style of `""` for blank cost it **7.92 points on `longarray`** while
a provider writing `null` for the same blanks paid nothing.

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

**Rows are never deleted; only leaves are.** There is no row-level filter, and there used to
be. `drop_empty_gt_rows` removed any array row whose *payload* fields all asserted nothing,
where payload meant every key that did not look like a dimension — decided by matching a
hardcoded list of substrings (`period`, `id`, `name`, `date`, `unit`, …) against the field
NAME. It was introduced for a real reason: the 10-Q ground truth carries rows like
`{"data_period": "FY2025 Q2", "segment_type": "company", "value": null}`, which state no fact
and which no extractor produces, so scoring them charged every vendor for an unstated
annotation convention.

It was removed anyway, for two reasons.

*It contradicted this very section.* The payload set was read off the row in hand, so
`{"id": "1"}` had no payload at all and survived as an "all-dimension row", while
`{"id": "1", "action": null}` had a payload asserting nothing and was deleted — **taking a
correct `id` with it**. The same prediction scored 62.50 or 25.00 depending only on which of
the three equivalent spellings of absence it used. 132 of the 174 rows the rule dropped had
exactly one payload key, so a single `null` deleted them; 116 of them carried five other
values that went with the row.

*It was unnecessary.* `flatten` already skips every leaf that asserts nothing, so a row
asserting nothing contributes no addresses, and a row with no addresses cannot be matched,
missed or charged. Blank rows are inert on either side with no rule at all: a blank gold row
scores 100 against a prediction that omits it, and a blank predicted row scores 100 against
gold that omits it. The row rule was solving a problem the leaf rule had already solved, and
paying for it with a field-name heuristic that §5.2 and P4 exist to forbid.

**What that costs, and where the cost belongs.** 174 gold rows across 43 documents now
contribute their coordinate leaves. Measured over nine vendors this is −0.03 to −0.06 corpus
mean for each, uniform enough to change no ordering, though individual documents move up to
±5. Those rows are a GROUND TRUTH question, not a scoring one: if a 10-Q truly has no such
line, the gold is wrong and belongs in the audit that `TO_LOOK_AT.md` items 19–21 describe,
where it can be fixed once instead of being hidden by a scorer rule on every run. Note also
that an invented empty row in a prediction remains free — it asserts nothing — so output bloat
is still not measured here.

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

### 5.1 Rendering is folded; content is not

§2 lists which spellings fold. This says what the list is FOR, because the question keeps
arriving as "the schema said *as printed*, so why is a reformatted date a match?"

`17 APR 2020` and `2020-04-17` compare equal. `NIKE, Inc.` and `Nike` do not. The test is not
how alike the strings look -- it is whether the fact survives the round trip. A date rewritten
in another format still names the same day and can be recovered from either spelling. A company
name with its legal suffix removed has dropped something no rule recovers.

| | fact recoverable? | |
| --- | --- | --- |
| `17 APR 2020` vs `2020-04-17` | yes -- same day | folded |
| `""` vs `null` vs an absent key | yes -- the cell is blank either way | folded (§5) |
| `NIKE, Inc.` vs `Nike` | no -- the suffix is gone | charged |
| `null` vs `"n/a"` | no -- a page printing "N/A" is not a page that is silent | charged (§5) |

**A field description asking for a particular rendering does not change this**, and the reason
matters more than the rule. A `description` is a JSON Schema *annotation*: it constrains
nothing, no validator checks it, and no vendor is obliged to follow it. Charging a model for
rewriting a date it demonstrably read correctly would be scoring whether its serializer happens
to match this corpus's annotation style.

**What it costs, stated plainly.** A few fields want the rendering *as* the fact -- one asks for
a clerk's date stamp "transcribed verbatim as written", because there the characters are the
evidence. Those get leniency they explicitly declined, and separating them would take a
per-field scoring rule. This metric does not take that trade: `canon_key` stays a function of
two values and never of two values and a field path (see `values.py`). The unfairness is real,
bounded, and preferred to the alternative.

**It does not run the other way.** Nothing here licenses the GROUND TRUTH to normalise. A gold
value that drops a legal suffix, a title, or an article is wrong against its own schema, and
`TO_LOOK_AT.md` items 19 and 20 record 28 such cases found by vendor consensus.

### 5.2 Only what was asked is scored

**The scorer enforces nothing the model was not shown.** `harness/dialects.py` states the delivery half
of this -- a dialect transform "may change how a constraint is *encoded*, never what is *asked
for*" -- and this is the scoring half.

It has teeth because dialects are applied PER VENDOR. `STRICT_ALLOWED_KEYS` is
`type, enum, properties, items, required, description`; a strictly-validating vendor receives
only those, while a permissive one may receive the whole schema. So `pattern`, `format`,
`minimum` and `maxLength` reach some vendors and provably not others. Scoring against them
would penalise vendors in proportion to how strict their API is -- a fact about the harness,
reported as a fact about the vendor, which is the failure `harness/dialects.py` exists to prevent.

Two consequences worth naming:

* **`enum` is the exception, and it is underused.** It is the one validating keyword on the
  allowlist: it reaches the vendor, constrains the output, and can be checked against the gold.
  For a closed set it beats describing the options in prose, which does none of the three.
* **Constraints belong on the annotation side.** `pattern`, `format` and `minimum` are worth
  carrying in the schema so the GROUND TRUTH can be linted against them -- a field declaring
  `XXX-XX-XXXX` and a gold value with five X's is a mechanical catch, and that exact error
  occurs sixteen times in this corpus. Their value is in checking the answer key, not in
  scoring the answer.

### 5.3 Where the fold is lenient on purpose, and what that costs

§5.1 says rendering folds and content does not. This says where that line was drawn *generously*,
because a reader who finds `5.2.1.5` scoring equal to `5215` deserves to find it written down
here rather than discover it in a number.

**The policy.** Normalisation changes only where the current behaviour is egregious. Everywhere
else this metric folds what the upstream metric folds, and the benefit of the doubt goes to the
model. The reason is that this benchmark measures extraction, and a fold that is wrong in
principle but harmless in this corpus costs the reader nothing while its removal costs real
matches. Only one fold met the bar for removal:

* **Leading zeros are kept.** Upstream strips them inside every digit run, making `INV-007` the
  same as `INV-7`, `02000` the same as `2000`, and `wenqifan03@gmail.com` the same as
  `wenqifan3@gmail.com`. Zero-padding is how a document says which identifier it means, so this
  one is not a rendering difference at all -- it is content, and it is the one divergence.

**What stays lenient, measured.** Whitespace, commas, hyphens and periods fold from anywhere in a
value. That is wrong in principle -- two section identifiers can differ only by their dots -- and
right here. Measured over all 660 documents against nine vendors:

| | |
| --- | --- |
| value matches the punctuation fold recovers | 1,607 |
| ...that differ by punctuation **alone** (`PO BOX 125` / `P.O. BOX 125`) | 1,532 |
| ...that are the same bibliography entry with a `[4] ` citation number | 75 |
| ...that credit a **wrong** value as right | 0 |
| gold values it merges inside one field | 215 |
| ...whose digit strings differ | 0 |

The hyphen half of that was found the expensive way: keeping hyphens broke 526 matches in a
single Schedule I return, where the gold writes EINs and ZIP+4s bare (`311440073`) and every
model hyphenates them (`31-1440073`), costing three vendors roughly fifteen points on that
document for nothing.

**The bill.** Five collisions, enumerated in `tests/test_canon_properties.py` under
`ACCEPTED_LENIENCY`, where a test asserts they still behave as priced:

`5.2.1.5` = `5215` · `1.1%w/w` = `11%w/w` · `#30-2` = `#302` · `RR-2` = `RR2` · `90-94` = `9094`

**How little P1 this actually gives up** is the reason it is affordable. `1:00.50` and `1:50`
stay distinct; so do `0.11%w/w` and `11%w/w`, `COM PAR $.001` and `COM PAR $.01`, and
`arXiv:2405.06211v3` and `arXiv:2405.6211v3`. None of them is protected by punctuation. All of
them are protected by keeping leading zeros -- which is why that divergence earns its place and
the punctuation ones did not.

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

**The two weightings compose.** Every value carries `1/total` inside its document; every
document carries equal weight across the means above. So a gold value's weight in UNIFIED is

    1 / (values in its document × documents in its subset × number of subsets)

**inversely proportional to the size of the document holding it.** Within a document, the
largest table decides the score: a 500-row, 4-column table beside a 5-field header leaves the
header at 0.25% of that document's number. Across documents, on the published corpus (2 to
410,012 values per document) the heaviest single gold value carries **57,029×** the weight of
the lightest (`CORPUS_NOTES.md`).

Both follow from decisions argued above — an address per value, equal weight per subset — but
the composition is not neutral, and it is why document shape is a first-order input to a
leaderboard. A corpus needing a different trade makes it in how subsets are built.

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
correct values produced:

| correct values produced | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| best `accuracy` | 0.00 | 25.00 | 50.00 | 75.00 | 100.00 |

**That search is scoped, and the scope is load-bearing.** A document with no arrays has a fixed
denominator, so `accuracy` there is simply proportional to the count of correct values. Add an
array and the denominator moves with the pairing, and the stronger reading — *no prediction
scores above one that produced more correct values* — is false: a prediction that recovers
**every** gold value scores 23.08 when it also emits ten invented rows, against 66.67 for one
that emits three rows with a field wrong in each. That is the metric working, not failing. A
metric with the stronger property would make spamming rows free, because volume would never
cost anything. What holds is the pair stated in P21.

So `accuracy` cannot be raised except by being right more *often* — not by being right more
*times*. Guessing well raises it
because guessing well produces correct values, which is the metric working rather than an
exploit. A model that outputs `"US"` by reading the page and one that outputs it by knowing
the column is usually `US` have produced the same output, and `accuracy` — which scores
output, not process — says so. No measure that looks only at the output can separate them.

Nor does corpus diversity separate them. Measured against a model that abstains, always
guessing the modal value gains **+43.25** where the base rate is 90%, **+25.25** across a
mixed corpus averaging 53%, and **+5.00** even where the prior is inverted. Variance does not
defeat the strategy; it prices the guess at the corpus-average hit rate, which is the right
price for it.

Indifference is the other half, and §4 states it: a wrong value costs what a blank costs.
Nothing is gained by that, but it matters to whoever consumes the output, who would act on the
wrong value and would know to ignore the blank. **`precision` is what resolves it**, and it is
why the two travel together:

    accuracy    how much of the document did you recover, net of what you invented
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

The charge scales with list length, and naming an array trades a substitution penalty for an
alignment one. On a five-element list, one wrong element *in place* improves 66.7 → 80.0; one
*dropped* element shifts everything after it, 80.0 → **0.0**. Omission is the commoner
failure, so the trade usually runs the wrong way. Use `order_matters` only where position is
content — page numbers, ranked results, time periods.

**Wrapping scalars in objects does not buy it back.** `["a","b"]` and
`[{"tag":"a"},{"tag":"b"}]` score identically: the bar in §3 is one matching value, and a
one-field row's only value is the one that is wrong, so the pair prices at zero. It helps only
when the element carries a *second* field to anchor the pairing — then the row pairs and the
wrong field is one misread. Hence decomposing prose (§2) works where wrapping a tag list does
not.

Asserted by `tests/test_incentives.py`, including that the figures above are the ones the
scorer produces.

## 9. Properties

Eight, kept under their original ids so the tests and the history still line up. Each is
either a defect that reached a leaderboard before it was caught, or a guarantee a reader would
reasonably doubt. Anything that follows from these, or that holds of any metric at all, is
stated where it belongs instead: comparison rules in §2, aggregation in §7, incentives in §8.

| # | property | statement |
| --- | --- | --- |
| P2 | Order never matters | permuting any array in `P` or `G` changes neither the score nor the diagnostics |
| P8 | Optimality | no pairing of array rows yields a higher score than the one chosen |
| P11 | Nesting invariance | wrapping a document in an extra level changes neither score nor denominator |
| P13 | Scalar arrays are scored | an array of scalars contributes values at every depth |
| P14 | Pairing/scoring agreement | the pairing that earned the score is the pairing that is committed |
| P15 | No silent shortfall | a greedy pairing and a skipped open map are both reported, never dropped (the tests call the second half P23) |
| P16 | Bucket completeness | the six buckets partition every address, and equal `explain`'s verdicts |
| P21 | Right is not punished, volume is not rewarded | correcting one value never lowers the score; output that produces no correct value never raises it, and strictly lowers it whenever the score was above zero |

P11, P13, P14 and P15 are regressions, not hypotheticals — each corresponds to a defect that
reached a leaderboard before it was caught. P11 is the strongest: if depth cannot change the
score, no depth-dependent scoring path can exist, which is the whole class rather than the
instance. P2 now carries what P22 stated separately, because two vendors returning identical
rows in a different order were once given different hallucination numbers while the score
itself looked fine.

**P21 replaces a claim that was both unproven and undesirable.** It read: *no prediction scores
above one that produced more correct values.* That is false — a prediction recovering every
gold value scores **23.08** when it also emits ten invented rows, against **66.67** for one
emitting three rows with a field wrong in each — and it is false in the direction that matters,
because a metric with that property would make spamming rows free. The evidence behind it was
an exhaustive search over a four-field document with no arrays, which is the one setting where
the denominator cannot move and so the attack cannot exist.

The two clauses are monotonicity in two different arguments, over the two things a prediction
can do — edit an assertion, or add one:

    hold what you emit fixed, make more of it right    the score cannot fall
    hold what is right fixed, emit more                the score cannot rise

Neither implies the other, and the first is not free. Correcting one value can re-pair rows and
move the denominator, which it does in 17% of the generated cases, and the score still never
falls.

**Where the rest went.** P1 identity and P12 omission follow from P16; P4 is P16 restated. P5
determinism is P2 in the only case where it could fail, ties. P7 symmetry belongs to the
comparison surface (§2), as do P9 null neutrality and P10 sign fidelity. P6 coverage honesty is
an aggregation decision (§7). P17 and P20 are incentives (§8). P18 and P19 are configuration
hygiene: a name fitting no array raises, and a missing schema raises. Their tests stay. They
are simply not what distinguishes this metric from any other.

### One example each

Minimal cases, one per property. The scorer produces every figure; the properties themselves
are enforced over generated documents, not over these.

    P2   gold l:[{k:x},{k:y}]    pred those two rows, reversed     100.00, and one diagnostic split
    P8   gold l:[{k:x,v:1},{k:y,v:2}]    pred both rows swapped    matched 4 of 4
    P11  gold {a:1, b:2} and the same document inside {w:{…}}      50.00 and total 2, both ways
    P13  gold {t:[1,2,3]}    pred {t:[1,2,9]}                      total 4, 50.00
    P14  explain() renumbers the swap to l[0], l[1]                its 4 matches are grade()'s 4
    P15  12 rows against a shrunk budget                           matching_exact false, [12]
         an additionalProperties subtree                           absent from total, listed in skipped
    P16  gold {a:1, b:2}    pred {a:1, b:9, c:3}                   1 match, 1 wrong, 1 fabricated
                                                                   — the histogram sums to total
    P21  one value corrected, over 4000 documents                  never lower; denominator moved in 672
         rows that share nothing appended, over 3000               never higher; 0.00 stays 0.00

## 10. Deliberate non-goals

- **No LLM judge in scoring.** Measured worth ≈ 0.2 points on one subset; not worth
  nondeterminism in a benchmark.
- **No per-subset scoring paths.** One grader, or the comparison is meaningless.
- **No substring acceptance and no sign flipping** to accommodate ground-truth conventions.
  Conventions belong in the schema, so vendors are told rather than guessed at.
- **No separate credit for declining.** It needs none: abstaining is producing no value, and
  `precision` already distinguishes a model that declines from one that guesses (§5). A
  bonus for silence would be a second, gameable path to a good score.
- **No key, and no bucketing rows before matching.** Every predicted row is priced against
  every gold row; correspondence is decided by how much the rows agree, never by their
  agreement on one chosen column. The alternative — partition rows on a key, pair only
  within a partition — is cheaper and is what the metric this benchmark is compared against
  does, so it is worth saying why we do not.

  A partition is sound only if rows disagreeing on the key genuinely cannot correspond, and
  nothing establishes that. The key is either supplied by a caller with no basis for picking
  one, or inferred from the data, and inferring it fails in three ways we have measured on a
  four-row invoice:

  | prediction | under a key | here |
  | --- | --- | --- |
  | labels right, amounts rounded to the dollar | P 0.000, R 0.000, cells 0.0 — identical to submitting nothing | 50.0 |
  | amounts right, every label wrong | P 1.000, R 1.000 | 50.0 |

  Both read four of eight cells correctly. Under a key the first scores zero because its
  rows land in partitions the gold rows are not in, and each is then charged twice — once
  as a row nobody found, once as a row the model invented — while its correct cells are
  dropped from the tally rather than scored. Precision and recall under a key measure
  agreement on the key column and nothing else; where no key can be inferred they measure
  the row count alone, and a submission of empty objects scores 1.000 on both.

  The cost of not partitioning is `O(n*m)`. That is handled by solving exactly up to a
  memory ceiling and falling back to greedy past it — which **reports itself**
  (`matching_exact`, `approximated`). A partition is an approximation that cannot: it
  removes pairings and the score still reads as exact. P15 is the rule it would break.

## 11. Reading a score

**Quote three.** `accuracy` is how much came back, `precision` how much of it can be trusted,
`f1` whether a guess was worth making. They can rank two predictions oppositely (§8).

**Check four things before comparing providers.**

- **Identical schema on both runs.** `fabricated` keys off the schema (§5), so a wider schema
  moves the split. The scorer cannot check this for you.
- **`matching_exact`** — false means an array was paired greedily (§3).
- **`skipped_open_maps`** — non-empty means part of the document was not scored (§5). A
  document can report 100.00 with every value inside a skipped subtree wrong.
- **UNIFIED or not** — §7 specifies a mean of subset means; `cli.py` takes a flat mean over
  documents, so this repository's output is not UNIFIED.

**What the scorer does not decide.** Document shape sets the weights (§7). Schema leaf
granularity sets what counts as equal: a paragraph in a leaf is all-or-nothing (§2), a bare
list is scored without identity at double the cost per error (§8). Both are fixed before the
grader runs.

## 12. Comparing vendors that fail differently

§8 says what a model should emit. This says what to expect when two of them are wrong equally
often but in different shapes. One rule produces all of it:

**An error costs one address if its element pairs, and two if it does not.** A paired element
puts both readings at the same address, so the error is one `misread`. An unpaired element
shares no address, so gold's leaves are `unfound` *and* the model's are `invented_item`, and
the union grows by both (§4).

**Same error count, three shapes.** Ten wrong field-values on a ten-row, five-field table;
only the distribution differs.

| the ten errors | accuracy | recall | precision | f1 | misread | unfound | invented_item | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one wrong field in each of ten rows | 80.00 | 0.800 | 0.800 | 0.800 | 10 | 0 | 0 | 50 |
| two rows wrong in all five fields | 66.67 | 0.800 | 0.800 | 0.800 | 0 | 10 | 10 | 60 |
| two rows omitted entirely | 80.00 | 0.800 | 1.000 | 0.889 | 0 | 10 | 0 | 50 |

**`accuracy` is the only one of the three that sees concentration.** Rows one and two have the
same `matched`, the same gold size and the same asserted size, so every ratio built from those
is identical; they differ only in `total`, which the ten unpaired addresses inflate. Rank by
`accuracy` and the spread failure leads by thirteen points; rank by `f1` and they tie.
`precision` does the opposite job, separating row one from row three where `accuracy` cannot.
That is the concrete reason §11 says quote three.

**The cost is a step, not a slope.** The same table, varying how much of one row is right:

| of that row's five fields, correct | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| accuracy | 81.82 | 92.00 | 94.00 | 96.00 | 98.00 | 100.00 |
| total | 55 | 50 | 50 | 50 | 50 | 50 |

The first correct value in a row is worth ten points and each one after it two, because the
first is what clears the pairing bar in §3. Row-level pairing failure, not cell-level error,
is what dominates a bad score.

**A repeated value rescues the pairing.** Identical schema, identical document, the same five
fields corrupted — the only difference is whether that row still carries the one field that
repeats down the table:

    last row keeps the shared date  ->  pairs     accuracy 91.67   misread 5            total 60
    last row also loses the date    ->  no pair   accuracy 81.82   unfound 6, invented 6  total 66

A filing date, fund name or currency code repeated in every row clears the §3 bar on its own,
halving what a fully wrong row costs. That is a property of the schema, not of the model, and
it belongs with the pre-comparison checks in §11.

**Magnitudes are shape-dependent.** The directions above are structural. The numbers come from
a ten-by-five table and will move with table width, row count, and nesting depth.

Not yet asserted. Every figure here is the scorer's, but no property in §9 pins the claim: P22
covers order-stability of the diagnostics, not distribution-sensitivity of the score.
