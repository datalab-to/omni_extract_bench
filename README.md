# Omni Extract Bench

One scorer for document-extraction benchmarks.

Extraction benchmarks tend to ship their own grader, so scores are not comparable across them
and format differences get scored as errors. This is the scoring half: a single metric, applied
identically to every subset and every provider, with the properties it claims written down and
tested rather than asserted.

It grades a predicted JSON object against a ground-truth object and a JSON Schema. It does not
run extractors, and it ships no benchmark data.

## Install

```bash
pip install -e .
```

Python 3.9+. No dependencies beyond the standard library.

## Use

```python
from omni_extract_bench import grade

result = grade(prediction, ground_truth, schema)
result["leaf_accuracy"]   # 0-100, the headline number
result["recall"]          # gold rows matched / gold rows
result["precision"]       # gold rows matched / predicted rows
result["matching_exact"]  # False if an array was too large to solve exactly
```

From the command line:

```bash
omni-extract-bench score      --pred p.json --gt g.json --schema s.json
omni-extract-bench score-dir  --pred-dir preds/ --gt-dir gt/ --schema-dir schemas/
omni-extract-bench leaderboard --pred-root baselines/ --gt-dir gt/ --schema-dir schemas/
```

`leaderboard` scores every provider directory under `--pred-root` over the same document list.

## What the metric does

**Leaf value accuracy.** Every scalar in the ground truth is one point. The score is the
fraction matched, counting spurious predicted leaves against you as well as missing ones.

**Arrays are matched optimally.** Rows are paired by maximum-weight bipartite matching
(Hungarian / Jonker-Volgenant, pure stdlib), not by index or by a guessed key, so a provider is
never punished for row order. Where a document is too large to solve exactly, the fallback is
approximate and **says so** in `matching_exact` — an approximate score is never reported as
though it were exact.

**Format is free; content is not.** `10/31/2024` equals `2024-10-31`; `5`, `5.0` and `"5.00"`
agree; `(98.2)` equals `-98.2`. But `-98.2` never equals `98.2`, and ID-like integers stay
exact so `8303911426` never equals `8303511426`.

**Omission is charged.** Returning 44 of 349 rows scores about 12, not 100. This is the single
most important property: a metric that lets an extractor skip rows for free will rank a
truncating system above a complete one.

**No output scores zero.** A document a provider failed to return is not dropped from its mean,
or a system that fails on hard documents outranks one that attempts them. Coverage is reported
alongside, and a "score on returned documents only" view separates *processes documents badly*
from *silently fails on hard documents*.

**One definition of equality.** `canon_key(value)` is the only comparison rule, used by leaf
scoring, row-pair weighting, and blocking alike. There is no per-field or per-vendor mode.

Full specification, including all sixteen properties: [`docs/METRIC_SPEC.md`](docs/METRIC_SPEC.md).

## Why the properties are tested

Most of them are regressions. Each of these reached a leaderboard before it was caught:

| property | the bug it prevents |
| --- | --- |
| P11 nesting invariance | omission was free for top-level arrays but charged when nested — 44 of 349 rows scored **100.0** |
| P13 scalar arrays scored | a top-level array of strings had denominator 0 |
| P14 pairing/scoring agree | pairing demanded literal equality while scoring accepted date formats, so a correct extraction scored **50.0** |
| P16 blocking equality | blocking used a third notion of equality, so rows differing only in **capitalisation** scored **0.0** |

Four of these were one root cause: several implementations of "are these two values equal?"
that had to agree, and did not. They are now one function, which makes a disagreement
unrepresentable rather than something tests have to catch.

```bash
python tests/test_metric_properties.py        # P1-P16, generative
python tests/test_metric_structural_audit.py  # wrapping invariance over 300 generated documents
python tests/test_grader_invariants.py        # identity, determinism, traps
```

The structural audit is the strongest of these: if wrapping a document in an extra level cannot
change its score, no depth-dependent scoring path can exist.

## Providers

`providers/extend_provider.py` is a reference adapter showing the shape a provider integration
takes. Provider APIs change; treat it as an example rather than a maintained client.

## Licence and attribution

Apache 2.0 — see [`LICENSE`](LICENSE).

Value canonicalisation builds on the `longextract_bench` grader (MIT, © Micro1), vendored under
`omni_extract_bench/vendor/` with its licence intact. See [`NOTICE`](NOTICE).
