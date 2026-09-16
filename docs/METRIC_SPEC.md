# Metric specification

The definition of the benchmark score. Anything the grader does that is not here is a bug.
Every property in §9 is enforced by a test.

Figures quoted below were measured on the 660-document snapshot of 2026-09-11, which the
620-document corpus later replaced. They are evidence for the rules, not claims about what is
published now: a rule stands or falls on the argument, and the numbers say what it cost when it
was decided.

**The idea.** Give every value an address, then compare addresses. Object keys are addresses
already — the model was handed the schema, so it uses the same key names. Array indices are
not: the model emits rows in whatever order it read them. So the only real problem is working
out which predicted row goes with which gold row; renumber, and scoring is set arithmetic.

## 1. Addresses

- **Value**: a scalar at a complete path. Objects and arrays are never values; addresses pass
  through them. `{"a": {"b": [{"c": 1}]}}` holds one value, at `a.b[0].c`.
- **Address**: the path to a value, as tagged steps — a key or an index. The tag matters: a
  document may contain the key `"0"`, and it must not join index `0`.
- **Row**: an object element of an array. Arrays of scalars are compared as multisets.
- The schema is **required**; §4 and §5 both depend on it.

## 2. Comparing one value

Deterministic and type-directed. No model participates. `canon_key` in
`omni_extract_bench/values.py` is the single comparison rule, used by leaf scoring and row
pairing alike, so the two cannot disagree.

| gold type | rule |
| --- | --- |
| boolean | canonical equality |
| integer / identifier-like | exact after canonicalisation — IDs, phone and account numbers never fuzzy-match |
| decimal number | integer part exact, fraction rounded to 7 significant digits; sign *notation* folded (`(98.2)`, `−98.2`, `98.2-`), sign *value* preserved (`−98.2 ≠ 98.2`) |
| date-like string | equal if both parse to the same calendar date. A timestamp at midnight is its date; any other time of day is kept and compared |
| other string | canonical equality (case, whitespace, punctuation, smart quotes, unicode fractions) |

What sorts a value into integer or decimal is whether its text contains a `.` — that and
nothing else, which is what stops an account number fuzzy-matching.

Folded, and tested in `tests/test_comparison_surface.py`:

    1,234 = 1234      $1,234.56 = 1234.56      12.34% = 12.34      5.00 = 5
    (1,234.56) = -1234.56      1234.56- = -1234.56      −1234.56 = -1234.56
    31/10/2024 = 2024-10-31    Oct 31, 2024 = 2024-10-31
    2024-10-31T00:00:00Z = 2024-10-31          midnight is a date
    ACME CORP = Acme Corp      "Acme  Corp." = "Acme Corp"      TRUE = true

Not folded, and scored as disagreements — the first place to look when a provider's misses
look like formatting, since this is a list of candidate scorer gaps rather than vendor errors:

    USD 1234.56 ≠ 1234.56      currency codes are not stripped
    1.234,56 ≠ 1234.56         European decimal notation
    2024-10-31T09:00:00Z ≠ 2024-10-31          a real time of day is not a date
    "yes" ≠ true               only true/false spellings are booleans

`""` is deliberately absent from both lists: a placeholder *word* is ink the page printed and
compares like any other value, while an empty string is what a blank cell serialises to and
gets no address at all (§5).

**These rules fold spelling, not wording.** Comparison is equality, so one character decides a
string and there is no partial credit; §10 rules out the mechanisms that would soften it. For a
name, code, date or amount, spelling is the whole job. For prose it is not, so a prose-valued
leaf measures verbatim transcription — on a 1081-entry bibliography, dropping a trailing DOI
from every entry scores 91.42 and eliding authors to "et al." scores 8.35. Fix that in the
schema by decomposing prose into fields, not in the scorer.

## 3. Aligning arrays

1. **Price** every candidate pair: `w(Pᵢ, Gⱼ)` is the number of values that would match, using
   §2's equivalences. Pricing descends, one array level per hop.
2. **Pair** by the assignment maximising `Σ w`, solved exactly with
   `scipy.optimize.linear_sum_assignment`, ties broken by shared addresses. Without that
   tie-break two solvers returned different equal-weight assignments, and so different scores.
3. **Discard any pair worth zero.** The bar is one matching value. Failing it charges every
   gold value in the row as missing *and* everything the prediction put there as invented, so
   neither omission nor invention is free.
4. **Renumber** onto the paired gold row. A predicted row that paired with nothing keeps an
   index of its own, written `lines[p2]`, so it can never be mistaken for a gold row.
5. **Exactness budget.** Past `MAX_CELLS` (250 million) or `MAX_EXACT` (20 000 on the smaller
   dimension) the solver falls back to greedy and reports `matching_exact: false`. The largest
   gold array measured was 6881 rows — 19% of the cap — and solves exactly.

Order is free by default, because row order is usually an artefact of layout. Naming an array
in `order_matters` makes its index an address again; §8 covers what that trades away.

## 4. What is reported

Every address falls into exactly one bucket. These are the verdicts `explain()` returns and
the counts `grade()` reports:

| bucket | meaning |
| --- | --- |
| `matched` | address on both sides, values agree |
| `misread` | address on both sides, values differ |
| `unfound` | gold address the prediction never used |
| `fabricated` | the schema offered the slot, the document is silent, the model asserted a value |
| `invented_item` | a value under an array element that paired with nothing |
| `invented_field` | a name the schema never declared |

`fabricated` keys off the **schema**, not gold's `null`s — a gold field written `null` and one
left out mean the same thing (§5), so keying off gold would sort two identical ground truths
differently. `invented_item` is separate from `invented_field` because one invented row
contributes a value per column while one invented field contributes one.

    asserted  = matched + misread + fabricated + invented_item + invented_field
    gold      = matched + misread + unfound
    total     = matched + misread + unfound + fabricated + invented_item + invented_field

    accuracy   = 100 · matched / total       how much you recovered, net of inventions
    precision  = matched / asserted          how much of what you said was true
    recall     = matched / gold              how much you recovered, inventions ignored

**Why `accuracy` and not `f1` or Jaccard.** Those are functions of `matched`, `|gold|` and
`|asserted|` alone, so they cannot tell *misread* from *missed and invented elsewhere* — both
score 50.00 under `f1`. `accuracy` scores them 50.00 and 33.33, because it knows the first
pair shares an address, and locating a field is a different problem from not finding it.

**It cannot be inflated.** `total = |gold| + invented`, so `accuracy ≤ matched / |gold|`.
Proposing all 27 combinations of a three-key row to guarantee one hit scores 3.70. Ten gold
rows returned perfectly score 100.00; the same ten with a thousand invented ones score 0.99.

**It is indifferent in exactly one place:** where gold has a value, a wrong value costs what a
blank costs. Both recovered nothing there. *Can I trust what it said* is `precision`'s question.

**Honesty flags.** `matching_exact` and `approximated` say when an array was too large to solve
exactly; `skipped_open_maps` says which subtrees were not graded (§5). Asserted by
`tests/test_false_assertions.py`, including that `grade`'s counts equal `explain`'s histogram.

## 5. What is not scored

One rule, three instances: **the metric scores facts, so anything asserting no fact is not
scored, on either side.**

**`null`, `""`, whitespace and an absent key all score alike**, at every level. A page can print
`N/A` but cannot print emptiness, so `""` is what a blank cell becomes in JSON. The placeholder
*words* are ink and remain ordinary values — `N/A`, `None` and `-` appear as real gold 24,980
times, and 67 documents hold both those strings and `null` in one file.

This is a comparability guarantee, not a convenience: `harness/dialects.py` makes strict vendors
emit `null` where permissive ones omit the key, so scoring them differently would move a score
with the serialisation convention the harness imposed. The deeper reason absence cannot earn
credit is that it would make the score depend on **schema width instead of document content** —
add fifty optional fields and every provider's score rises with nothing extracted.

**A fold may trim a value; it may never consume one**, stated as an invariant on the one public
entry point:

> `canon_key(v) == ""` **implies** `states_nothing(v)`

Seventeen spellings once violated it. `[1]` through `[14]` as reference numbers all keyed
empty, so reversing every reference scored 100.00. Asserted by construction over 128 probe
values in `tests/test_canon_properties.py`, not by a list — a list is what let them sit.

**`additionalProperties` objects are not graded.** The keyword is not forwarded to strict
vendors, so grading it would score a request the harness never made. Skipped on both sides and
reported in `skipped_open_maps`.

**Rows are never deleted; only leaves are.** A row asserting nothing contributes no addresses
and so cannot be matched, missed or charged — the leaf rule already does the work a row filter
would, without a field-name heuristic that §5.2 forbids.

Asserted by `tests/test_spec_null_semantics.py`.

### 5.1 Rendering is folded; content is not

`17 APR 2020` and `2020-04-17` compare equal. `NIKE, Inc.` and `Nike` do not. The test is not
how alike the strings look — it is whether the fact survives the round trip. A date in another
format still names the same day; a company name with its legal suffix removed has dropped
something no rule recovers. A field description asking for a particular rendering does not
change this, and nothing here licenses the ground truth to normalise.

### 5.2 Only what was asked is scored

**The scorer enforces nothing the model was not shown.** No rule may read a field's name,
description or position to decide how to compare its value; the comparison depends on the value
alone. Otherwise the benchmark would grade a question it never asked.

### 5.3 Where the fold is lenient on purpose, and what that costs

Whitespace, commas, hyphens and periods fold from anywhere in a string, which merges a few
values that are not the same: `1:00.50` and `1:50`, and the three in `KNOWN_COLLISIONS` nobody
chose — `½` = `1/2` = `12`. Twelve such merges are enumerated in `tests/test_canon_properties.py`.
The policy is that normalisation changes only where current behaviour is egregious, because
each tightening is a score change that has to be paid for.

### 5.4 Reading a comparison back

`explain()` returns both sides twice, raw and canonical, so a reader can see *why* two values
counted as equal without recomputing the rule and risking a second opinion about equality.

## 6. Ground truth adjustments

Corrections change the gold, never the scorer. A scorer rule that exists to hide a bad
annotation makes every future run wrong in the same way.

## 7. Aggregation

    SUBJECT   = mean over the documents of one subset
    UNIFIED   = mean of the subset scores

Equal weight per subset, not per document, so a large subset cannot dominate. The two
weightings compose: every value carries `1/total` inside its document, so a value's weight is
**inversely proportional to the size of the document holding it**.

> **Not yet implemented.** `run_score.py` takes a flat mean over documents, which is the thing
> this section says must not happen. Doing it properly needs each document's subset, which a
> manifest can carry as an ordinary column. Until then a leaderboard printed by this repository
> is not UNIFIED.

## 8. What the metric pays for

Recovering a value is the only thing that raises a score. Two consequences worth stating:

**Omission is charged.** Returning 44 of 349 rows scores about 12, not 100. A metric that lets
an extractor skip rows for free ranks a truncating system above a complete one.

**Volume is not rewarded.** A document whose every gold value is returned scores 23.08 when it
also emits ten invented rows, against 66.67 for one. Output that produces no correct value
never raises a score and strictly lowers it whenever the score was above zero (P21).

**Order-freedom is a trade.** A wrong value costs more inside a scalar array than in an object,
because the array has no key to hold the address still. Wrapping scalars in objects does not
buy it back. Asserted by `tests/test_incentives.py`.

## 9. Properties

Each is enforced by `tests/test_metric_properties.py` unless noted.

| | | |
| --- | --- | --- |
| P2 | Order never matters | permuting any array changes neither score nor diagnostics |
| P8 | Optimality | no pairing of rows yields a higher score than the one chosen |
| P11 | Nesting invariance | wrapping a document in an extra level changes neither score nor denominator |
| P13 | Scalar arrays are scored | an array of scalars contributes values at every depth |
| P14 | Pairing/scoring agreement | the pairing that earned the score is the pairing committed |
| P15 | No silent shortfall | a greedy pairing and a skipped open map are reported, never dropped |
| P16 | Bucket completeness | the six buckets partition every address and equal `explain`'s verdicts |
| P21 | Right is not punished, volume is not rewarded | correcting one value never lowers the score |
| P22 | Split determinism | the diagnostic split does not move under a permutation (`tests/test_pairing_determinism.py`) |

P1 identity and P12 omission follow from P16; P4 is P16 restated.

## 10. Deliberate non-goals

No partial credit for strings, no per-field or per-vendor modes, no model in the comparison
loop, no confidence weighting. Each would make two runs of the same scorer on the same data
answer differently, or would require reading the field to decide how to compare it (§5.2).

## 11. Reading a score

**Quote three.** `accuracy` is how much came back, `precision` how much of it can be trusted,
and coverage how many documents were graded at all.

**Check four things before comparing providers:** coverage, `matching_exact`,
`skipped_open_maps`, and whether the misses are in §2's second list — a harness gap reads
exactly like a vendor error.

**What the scorer does not decide.** Document shape sets the weights (§7), and schema width
sets how many addresses exist. Both are properties of the benchmark, not of the extractor.

## 12. Comparing vendors that fail differently

An error costs one address if its element pairs and two if it does not, so the same error count
scores differently depending on how it is distributed. Ten wrong values scattered across ten
rows cost ten addresses; concentrated into two rows they can break the pairing and cost far
more. `accuracy` is the only one of the three that sees that concentration, and the cost is a
step rather than a slope: a repeated value — a currency, a fiscal year — can rescue a pairing
that would otherwise fail. Directions are structural; magnitudes are shape-dependent.
