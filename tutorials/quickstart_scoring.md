# Quickstart: scoring

Scoring takes one table. Each row corresponds to a document, where its ground truth and prediction
live, and the schema it was asked for.

```bash
git clone git@github.com:datalab-to/omni_extract_bench.git
cd omni_extract_bench && pip install -e ".[benchmark]"      # add [s3] for s3:// paths
```

## The example

`tutorials/data/quickstart_scoring/` holds one invoice, the schema it was asked for, and one
prediction.

```json
// gold.json — what the document says
{
  "invoice_id": "INV-77",
  "total": "1,240.00",
  "line_items": [
    {"sku": "AX-9910", "qty": 2},
    {"sku": "BX-2201", "qty": 1}
  ]
}

// pred.json — one sku misread, one row missing, one row invented
{
  "invoice_id": "INV-77",
  "total": "1240.00",
  "line_items": [
    {"sku": "AX-9919", "qty": 2},
    {"sku": "ZZ-0000", "qty": 9}
  ]
}
```

## Create table

```bash
python - <<'PY'
import pathlib
import pyarrow as pa, pyarrow.parquet as pq

data = pathlib.Path("tutorials/data/quickstart_scoring")

rows = [{"doc_id": "invoice-77",                          # unique within the table
         "gt_path": str(data / "gold.json"),              # local, s3:// or gs://
         "pred_path": str(data / "pred.json"),
         "schema": (data / "schema.json").read_bytes(),   # inline, not a path
         "vendor": "my-model"}]                           # extra columns ride through

pq.write_table(pa.Table.from_pylist(rows), "jobs.parquet")
PY
```

Four columns are required: `doc_id`, `gt_path`, `pred_path`, `schema`. The two documents are
paths because an extraction is unbounded (the document could be thousands of pages); the schema is inline.

## Score 

You point to the table and where you want the outputs to live.

```bash
oeb score --manifest jobs.parquet --out run/
```

Two parquet datasets land in `run/`:
- `scores/` has one row per manifest row
- `verdicts/` one row per address after alignment. 

They are split because they operate at different levels of granularity. 

## Read the scores

```bash
python - <<'PY'
import pyarrow.parquet as pq

for r in pq.read_table("run/scores").select(
        [
          "doc_id", 
          "status", 
          "accuracy", 
          "matched", 
          "misread",
         "unfound", 
         "invented_item"
        ]).to_pylist():
    print("  ".join(f"{k}={v}" for k, v in r.items()))
PY
```

```
doc_id=invoice-77  status=scored  accuracy=37.5  matched=3  misread=1  unfound=2  invented_item=2
```

Every row comes back, including failures, because coverage is only visible if failures
occupy rows. A row is either `scored` or `error`, and an `error` row says why in `error`: the
traceback where something raised, or the system's own words where the prediction recorded a
failure of its own, like `{"__error__": "context length exceeded"}`.

Metrics on those rows are **null**, never zero — a zero claims the model tried and missed
every field. So when you average a corpus, filter on `status == "scored"` and say how many
documents that was.

## Read the verdicts

Every address carries exactly one verdict, so the six verdict columns partition the score.

```bash
python - <<'PY'
import pyarrow.parquet as pq

for r in pq.read_table("run/verdicts").to_pylist():
    print(f"{r['address']:<20} {r['verdict']:<14} gold={str(r['gold_raw']):<12}"
          f" pred={str(r['pred_raw']):<12} canon={r['gold_canon']}/{r['pred_canon']}")
PY
```

```
invoice_id           matched        gold="INV-77"     pred="INV-77"     canon=inv77/inv77
line_items[0].qty    matched        gold=2            pred=2            canon=2/2
line_items[0].sku    misread        gold="AX-9910"    pred="AX-9919"    canon=ax9910/ax9919
line_items[1].qty    unfound        gold=1            pred=None         canon=1/None
line_items[1].sku    unfound        gold="BX-2201"    pred=None         canon=bx2201/None
line_items[p1].qty   invented_item  gold=None         pred=9            canon=None/9
line_items[p1].sku   invented_item  gold=None         pred="ZZ-0000"    canon=None/zz0000
total                matched        gold="1,240.00"   pred="1240.00"    canon=1240/1240
```

Things to point out:


- **`line_items[p1]`** is the row the prediction invented. Rows are matched by content, not by
  position, so an unpaired predicted row gets a made-up label and is counted against meaning `p1` is
  not gold's index 1. 
- **`line_items[1]`** is gold's second row, which the prediction never produced.

| verdict | |
|---|---|
| `matched` | address and value both agree |
| `misread` | address in both, values differ |
| `unfound` | a gold address the prediction never produced |
| `fabricated` | the schema offered the slot, gold is silent, the model filled it |
| `invented_item` | an address under an array row that paired with nothing |
| `invented_field` | an address the schema never offered |

## On Modal

Same scorer, one container per batch, for when one machine is the bottleneck or you want to
close the laptop. Once:

```bash
pip install -e ".[modal]" && modal token new
modal secret create oeb-s3 \
    AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  FSSPEC_S3_ENDPOINT_URL=https://...
```

`FSSPEC_S3_ENDPOINT_URL` is only for R2 or another S3-compatible store. Then, with a manifest
whose paths are all `s3://` and which lives in the bucket itself:

```bash
modal run --detach -m omni_extract_bench.run_score_modal \
    --manifest s3://bucket/jobs.parquet --out s3://bucket/run --rows 16
```

It waits and prints the tally at the end. `--detach` is what lets you close the laptop
meanwhile: Modal keeps the driver alive when your client goes away, and without it the run
is stopped the moment the command exits.
Your machine never needs bucket credentials — it passes the paths as strings and only the
containers read and write. Follow it with `modal app logs <id>`; parts appear in the bucket as
each container finishes, and read back exactly as a local run's do:

```python
import fsspec, pyarrow.parquet as pq
scores = pq.read_table("bucket/run/scores", filesystem=fsspec.filesystem("s3"))
```
