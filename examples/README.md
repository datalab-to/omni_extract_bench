# A worked example

Three invoices and three predictions, chosen so each one teaches something you need before
trusting a number. Run from the repository root — `uv run` builds the package for you, so
there is nothing to install first.

```bash
uv run oeb build-corpus --corpus examples/corpus
uv run oeb score --corpus examples/corpus --predictions examples/predictions
```

```
scorer 0.1.0, source predictions
3 predictions, 3 documents in the atlas
invoice-a                 100.00
invoice-b                  55.56
invoice-c               unusable  rate limited after 3 retries

{'graded': 2, 'unusable': 1}
mean accuracy over the 2 graded: 77.78
1 unusable (the provider returned nothing scoreable), excluded from that mean
```

## invoice-a scores 100 although nothing matches literally

The gold says `"2024-01-15"` and `1250.00`. The prediction says `"01/15/2024"` and the *string*
`"1,250.00"`, and lists the two line items in the opposite order. None of that is an extraction
error, so none of it is charged: format is free, and rows are matched optimally rather than by
position — a vendor that emits a table bottom-to-top has not made a mistake.

## invoice-b scores 55.56, and `explain` says why

```bash
uv run oeb explain --corpus examples/corpus --predictions examples/predictions --doc invoice-b
```

```
lines[0].qty                                wrong value     gold=3  pred=8
lines[1].price                              missing         gold=120.5  pred=None
lines[1].qty                                missing         gold=1  pred=None
lines[1].sku                                missing         gold="B-8"  pred=None

accuracy 55.56
```

Five of nine leaves are right. The dropped line item costs three of them, not one: **omission is
charged**, because a metric that lets an extractor skip rows for free ranks a truncating system
above a complete one.

## invoice-c is `unusable`, not zero

The model returned an error, so there is nothing to score on its merits. It records null metrics
and stays out of the mean — averaging it in as a zero would make a rate-limited run look like a
bad model. **Filter on `kind` before you average.**

# Keeping the results

```bash
uv run oeb score --corpus examples/corpus --predictions examples/predictions --out /tmp/run
```

```
/tmp/run/summary.parquet              one row per prediction
/tmp/run/verdicts/<doc_id>.parquet    every address, with gold and pred
```

Ordinary parquet, so exploring needs no library:

```sql
select doc_id, address, gold, pred
from '/tmp/run/verdicts/*.parquet'
where verdict <> 'match';
```

A run is written once. To score the same thing again, name a different `--out`; comparing two
runs is a join on `(doc_id, prediction_id)`.

Add `--jobs N` to score documents in parallel — one worker per document.

# The corpus

```
examples/corpus/corpus.parquet            the atlas: what is in the benchmark
examples/corpus/<doc_id>/ground_truth.json
examples/corpus/<doc_id>/schema.json
```

Two files per document. `build-corpus` writes the atlas from whatever is in the directory, so
adding or removing a document means changing the directory and re-running it.

A schema is required and never inferred: without one, an `additionalProperties` subtree would be
graded silently, and a value the model invented could not be told from one the schema offered it
a slot for.

## Running a subset

The atlas is a plain table and scoring follows its rows, so a subset is a filtered atlas written
**beside** the full one — not over it:

```python
import pyarrow as pa, pyarrow.parquet as pq

t = pq.read_table("examples/corpus/corpus.parquet")
keep = [r for r in t.to_pylist() if r["doc_id"] != "invoice-c"]
pq.write_table(pa.Table.from_pylist(keep), "examples/corpus/graded-only.parquet")
```

```bash
uv run oeb score --corpus examples/corpus/graded-only.parquet --predictions examples/predictions
```

```
2 predictions, 2 documents in the atlas
1 prediction(s) skipped; the atlas does not list them: invoice-c
```

`--corpus` takes a directory, meaning its `corpus.parquet`, or an atlas file, meaning that file
with the documents beside it. Rows name their files relative to the atlas, so both describe the
same documents and the full list stays intact.

A prediction for a document the atlas does not list is skipped and counted — that is what
filtering is. A prediction matching nothing at all is an error, with near matches suggested. A
row naming a file that is **not there** stops the run: the atlas is what the run follows, so it
has to resolve.

## Adding metadata

Anything else in the atlas is yours, and is what makes it worth querying — which collection a
document came from, its page count, how big its ground truth is:

```sql
select suite, count(*) docs, sum(leaf_values) leaves
from 'corpus.parquet' group by 1 order by docs desc;
```

`build-corpus` carries columns it does not understand through a rebuild, so they survive.
