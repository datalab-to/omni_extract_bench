# A worked example

Three invoices, three predictions, and the three things worth understanding before you trust a
number.

```bash
oeb build-corpus --corpus corpus
oeb score        --corpus corpus --predictions predictions
```

```
invoice-a                 100.00
invoice-b                  55.56
invoice-c               unusable  rate limited after 3 retries

{'graded': 2, 'unusable': 1}
mean accuracy over the 2 graded: 77.78
1 unusable (the provider returned nothing scoreable), excluded from that mean
```

### invoice-a scores 100 although nothing matches literally

The gold says `"2024-01-15"` and `1250.00`; the prediction says `"01/15/2024"` and the *string*
`"1,250.00"`, and lists the two line items in the opposite order. None of that is an extraction
error, so none of it is charged. Format is free, and rows are matched optimally rather than by
position — a vendor that emits a table bottom-to-top has not made a mistake.

### invoice-b scores 55.56, and `explain` says why

```bash
oeb explain --corpus corpus --predictions predictions --doc invoice-b
```

```
lines[0].qty      wrong value   gold=3      pred=8
lines[1].price    missing       gold=120.5  pred=None
lines[1].qty      missing       gold=1      pred=None
lines[1].sku      missing       gold="B-8"  pred=None
```

Five of nine leaves are right. The dropped line item costs three points, not one: **omission is
charged**, because a metric that lets an extractor skip rows for free ranks a truncating system
above a complete one.

### invoice-c is `unusable`, not zero

The model returned an error, so there is nothing to score on its merits. It records null
metrics and is kept out of the mean — averaging it in as a zero would make a rate-limited run
look like a bad model. **Filter on `kind` before you average.**

## Keeping the results

```bash
oeb score --corpus examples/corpus --predictions examples/predictions --out /tmp/run
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

## Making it yours

The corpus is a directory of document directories plus an atlas:

```
corpus/corpus.parquet            what is in the benchmark
corpus/<doc_id>/ground_truth.json
corpus/<doc_id>/schema.json
```

Add or remove a document by changing what is in the directory, then re-run `build-corpus`. The
corpus version changes, because a different set of documents is a different benchmark.

Edit a ground truth and the next run stops:

```
invoice-a: ground_truth.json has changed since the atlas was written.
  If that was intended, record it:  oeb build-corpus --corpus examples/corpus
```

That is the point: data cannot move underneath a benchmark by accident.
