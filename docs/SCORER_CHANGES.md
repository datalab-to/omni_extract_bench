# What changed, and why: `grading.py` → `score.py`

For review. Every number below is reproducible from the tests named at the end.

## In one paragraph

The old scorer walked the two documents in step, pairing array rows as it met them. The new
one gives every value an address, works out which predicted row goes with which gold row,
renumbers the prediction to match, and then compares two sets of addresses. That change is
what fixes the bugs below — a row is now a thing with a name, rather than wherever the walk
happened to put it.

## The bugs it fixes

### 1. A correct extraction could score zero

Two rows that differ only in their nested content:

```
gold  books: [ {chapters: [a, b]}, {chapters: [c, d]} ]
pred  books: [ {chapters: [c, d]}, {chapters: [a, b]} ]     the same answer, rows swapped
```

|  | old | new |
| --- | --- | --- |
| this document | **0.00** | 100.00 |
| 200 permuted-but-perfect documents | **200 wrong** | 0 wrong |

The old scorer decided which rows to pair *without looking inside them*. Two rows with no
distinguishing fields of their own therefore tied, the matcher broke the tie arbitrarily, and
every chapter landed under the wrong book. This is a violation of two properties the spec
already claimed: identity (P1) and permutation invariance (P2).

The new scorer prices a candidate pair by descending into it, so the chapters decide which
book is which. That is the single most important difference.

### 2. Recall was meaningless for nested tables

An invoice with ten line items, of which the model returned three. The invoice itself has a
number, so it pairs on its own content and both scorers agree on the score:

```
gold  invoices: [ {no: "INV-1", lines: [10 items]} ]
pred  invoices: [ {no: "INV-1", lines: [3 items]} ]
```

|  | old | new |
| --- | --- | --- |
| score | 36.36 | 36.36 |
| `gt_rows` | **1** | 11 |
| `recall` | **1.00** | 0.36 |

The score is right and the diagnostics are not. The old scorer counted rows only in arrays the
*schema declared at the top level*, so it counted the invoice — one row, returned — and
reported **recall 1.00** for a document missing seven of its ten lines. Most real tables are
nested (`invoice.lines`), so this was the common case rather than an edge case.

Row counts now come from the documents at every depth, and `recall` is over values rather than
rows, so it agrees with the score.

Worth noting: had the invoice *not* carried a number of its own, the old scorer would have
scored the whole document **0.00** — that is bug 1 again, and the two compound. Nested tables
were where it hurt most.

### 3. Precision could not see invented content

Two invented top-level fields, in a document with no array in it:

|  | old | new |
| --- | --- | --- |
| `precision` | **0.00** | 0.50 |

Old precision was computed over *rows in top-level arrays*. A document without such an array
had no rows, so precision and recall were both reported as `0.00` — indistinguishable from a
provider that returned nothing at all. Invented fields were invisible to it no matter how many
there were.

### 4. A missing schema degraded silently

|  | old | new |
| --- | --- | --- |
| `grade(pred, gold, {})` | `gt_rows=0, recall=0.00`, **and still prints `leaf_accuracy=50.00`** | refuses: *a schema is required* |

`cli._score_one` passed `{}` whenever the schema file was absent, so a run without
`--schema-dir` produced a full-looking report in which every recall was zero. Nothing said so.

### 5. A `$ref` schema was mis-graded rather than refused

|  | old | new |
| --- | --- | --- |
| schema containing `$ref` | **scores 100.00** | refuses: *resolve it first* |

Neither scorer follows a `$ref`. So an `additionalProperties` object written behind a pointer
was **graded**, while the identical object written inline was **skipped** — the same document
scored two ways depending on how its schema was spelled. The new scorer refuses the schema and
names `dialects.resolve_refs`; the CLI now applies it.

### 6. A document whose root is an array scored zero on identity

|  | old | new |
| --- | --- | --- |
| `grade(G, G)` where `G = [{...}]` | **0.00**, denominator 0 | refuses, explaining why |

`prep_ground_truth` discards a non-object document, so everything vanished before scoring.
Both were wrong; the new one is loud about it. (A root-level array cannot be sent to the
vendors this benchmark measures anyway — OpenAI's structured outputs rejects one — so such a
document means the harness is misconfigured, not that a provider failed.)

### Fixed in both, earlier in the same work

These were repaired in `grading.py` first and carried across, so they are not differences
between the two files — but they are part of the same change and worth knowing:

- **Shape disagreement erased one side.** A gold table returned as `{}` scored 100.00 over a
  denominator of 1. Both sides are now charged.
- **Mixed arrays dropped their scalars** — an array holding both objects and values scored only
  the objects.
- **Arrays of arrays were compared by string form**, so `[[15, 25]]` matched `[[1.5, 2.5]]`
  (both canonicalise to `[1525]`) and scored 100.
- **`null` in a scalar array took a denominator slot**, so a correct document could not reach
  100.

## What the new scorer adds

| | what it gives you |
| --- | --- |
| `explain()` | one verdict per address — `match`, `wrong value`, `missing`, `fabricated`, `invented item`, `invented field`. There was no equivalent before. This is what makes error mining possible. |
| `order_matters` | per-array order-dependence, named the way `explain` prints it (`"invoice.lines"`, `"matrix[*]"`), and **validated** — a name fitting no array raises rather than silently applying to nothing. |
| `f1` | reported alongside `accuracy`, because accuracy alone cannot charge a hopeless guess: a wrong value at a gold address costs exactly what a blank costs. |
| the false-assertion split | `misread` (read it wrongly) vs `fabricated` (the schema offered a slot, the document is silent) vs `invented_item` / `invented_field` (structure nobody asked for). `1 − precision` used to hide all four. |
| `found` / `read_right` | accuracy factorised into *did you look in the right places* × *did you read them correctly*. They multiply to the score, so every result splits into a coverage problem and a reading problem. |

## What was dropped

- **Blocking.** The old scorer could partition rows by fields compared exactly, so rows in
  different blocks never paired. It was a heuristic and it is unnecessary — pricing is exact
  without it, and the partition itself was a source of bugs (rows differing only in
  capitalisation once landed in different blocks and scored 0.00).
- **`pair_object_keys`.** Reduced to literal key pairing and inlined. Keys are addresses: a
  predicted key that is not exactly a gold key names a field the model invented.

## The trap when comparing two runs

Only six key names are shared, and **two of them mean different things**:

| shared name | old meaning | new meaning |
| --- | --- | --- |
| `precision` | matched rows / predicted rows, top-level arrays only | matched values / values asserted |
| `recall` | matched rows / gold rows, top-level arrays only | matched values / gold values |
| `gt_rows`, `pred_rows` | top-level, schema-declared arrays | every array, every depth |
| `matched` | matched **rows** | matched **values** |
| `matching_exact` | same | same |

The headline score was `leaf_accuracy` and is now `accuracy`. Numbers from an old run and a new
run are not comparable field-by-field.

## How to check any of this

    python3 tests/test_score_equivalence.py    # classifies every divergence from the old scorer
    python3 tests/test_score_properties.py     # P1-P19, generative
    python3 tests/test_incentives.py           # what the metric pays a model to do
    python3 tests/test_false_assertions.py     # the misread/fabricated/invented split
    python3 tests/test_cli.py                  # the command line, end to end
    python3 tests/test_spec_claims.py          # every figure printed in METRIC_SPEC.md

`test_score_equivalence.py` is the one to read first: it grades 1600 documents with both
scorers and requires every disagreement to fall into a *named* class. Today there are two —
nested-row pairing, where the new scorer is right and the old one violates P1/P2, and ties
between equal-weight assignments, where both are correct.
