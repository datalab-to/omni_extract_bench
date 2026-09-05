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
| other string | canonical equality under the string fold below |

**The string fold** (`normalize.canonical`): lowercase; NFKD with combining marks dropped; smart
quotes and dashes to ASCII; placeholder markers (`n/a`, `none`, `-`, `..`) to empty; short footnote
markers (`[1]`, `[a]`) removed; whitespace collapsed to one space; punctuation stripped from the
**edges** of the value only; leading zeros inside digit runs dropped; unicode fractions expanded.
Punctuation **between** characters is content and is kept.

| equal | distinct |
| --- | --- |
| `Acme Inc.` / `Acme Inc` | `1/2` / `12` |
| `ABN AMRO Bank N.V.` / `ABN AMRO BANK N.V.,` | `Section 2.1` / `Section 21` |
| `"quoted"` / `quoted` | `v1.2` / `v12` |
| `[1] Y. Bengio` / `Y. Bengio` | `Inst itutional` / `Institutional` |
| `Table  B-1.` / `Table B-1` | `UBS AG, Stamford Branch` / `UBS AG (Stamford Branch)` |

An earlier fold deleted every internal period, slash, hyphen and space, which merged the
right-hand column. Measured on the reference corpus, the narrower rule costs every provider
0.3–0.6 points about equally and changes no rank.

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
5. **Exactness budget**: the solver is `O(n²·m)` in the *smaller* dimension `n`. Blocks are
   solved exactly while `n²·m ≤ MAX_WORK`; beyond that the solver falls back to greedy and
   *reports* `exact=False` rather than claiming optimality. The budget is on work, not row
   count: a truncating provider produces a cheap rectangular problem (44×349 ≈ 7e5 operations),
   so gating on the larger dimension pushed exactly the cases that most need accurate scoring
   onto the approximate path. Greedy costs 0.2–0.8% of assignment weight, and it is charged
   only to providers returning very large tables, so it is surfaced per grade
   (`matching_exact`, `greedy_blocks`) and never silent.

*Why exact matching matters:* the previous key-inference heuristic lost 2.6 points on a single
10-Q, and 28% of array-cell misses on one subset were the right value attached to the wrong row.

## 4. Document score

    leaf_accuracy(d) = 100 · (matched leaves) / (total leaves)

`total leaves` counts every gold leaf plus every spurious predicted leaf. Recall and precision
over rows are reported separately and are never folded into `leaf_accuracy`.

## 5. Aggregation

    score         = mean of leaf_accuracy over all documents
    subset_score  = mean of leaf_accuracy over that subset's documents

The headline is the document mean; subset means are reported beside it. A document a vendor
returned nothing usable for scores **0** — excluding failures would reward fragility. Coverage
is reported beside the score, never inside it.

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


### Object keys are values

An object's keys are compared by the same canonical form as its values, not literally. This
matters only where the keys come from the DOCUMENT rather than the schema — an open
`additionalProperties` map, whose keys are headings the extractor read off the page.
Schema-declared property names are unaffected: both sides spell them the way the schema does.

The reason is that ground truth is not reliably verbatim about case. In this corpus a document
prints a heading in capitals, gold records it title-cased, and an extractor that transcribed it
faithfully scored zero for that entire group — penalised for being closer to the document than
the gold file. A benchmark cannot ask for verbatim transcription and then grade it against a
normalised answer.

If canonicalisation would merge two distinct keys of the same object (`Total` and `TOTAL`),
that object falls back to literal pairing: merging them would silently discard one side's
value, which is a worse failure than the one being fixed.

## 9. Run protocol

The metric is only fair if the inputs to it were produced the same way. Every published run
follows these rules, and each is recorded per document in the stored raw record so it can be
audited rather than trusted:

- **One timeout for everyone** (1800 s per document). A document that exceeds it scores 0
  for that provider unless the provider's job completed server-side and can be fetched by its
  job id after the deadline — in which case the result counts and the timeout is *also*
  reported. Recovery measures accuracy, not latency; the timeout count is published beside the
  score. The rule is applied to every provider; whether it can benefit depends on whether the
  vendor retains results, and that difference is stated.
- **Maximum tier for everyone.** Each provider runs at its highest-accuracy setting. Any
  exception is disclosed in the same sentence as that provider's score.
- **Same schema for everyone.** Benchmark-only keys are stripped before a schema is sent;
  a conventions overlay, where used, is applied to the same documents for every provider.
- **Every raw response is kept** — status, body, headers, request and job ids, vendor-reported
  usage — so any score can be recomputed, and any intervention checked, without re-running.
- **Interventions are rules, not edits.** A re-run happens only when a stored result was
  produced by a harness fault (a capture crash, an account-level stop, an empty 200) and
  never because a score looked wrong. Each rule and the documents it touched are listed with
  the results.
- **Exclusions are declared** and applied to every provider, including already-scored ones.
- **Coverage is published beside every score**, per §5, together with the count of documents
  recovered after the deadline.
