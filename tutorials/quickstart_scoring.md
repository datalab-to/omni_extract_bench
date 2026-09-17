# Quickstart: scoring

Scoring is one function over one document: the prediction, the ground truth, and the schema
the prediction was asked for.

```bash
git clone git@github.com:datalab-to/omni_extract_bench.git
cd omni_extract_bench && uv pip install -e .
```

That is the whole install — `score` is pure Python and scipy.

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

## Score it

```bash
python - <<'PY'
import json, pathlib
from omni_extract_bench import score

data = pathlib.Path("tutorials/data/quickstart_scoring")
load = lambda name: json.loads((data / name).read_text())

result = score(load("pred.json"), load("gold.json"), load("schema.json"))
print("  ".join(f"{k}={result[k]}" for k in
                ("accuracy", "matched", "misread", "unfound", "invented_item")))
PY
```

```
accuracy=0.375  matched=3  misread=1  unfound=2  invented_item=2
```

Or from a terminal, with no Python around it:

```bash
oeb score --gt tutorials/data/quickstart_scoring/gold.json \
          --pred tutorials/data/quickstart_scoring/pred.json \
          --schema tutorials/data/quickstart_scoring/schema.json
```

## Read the verdicts

Every address carries exactly one verdict, so the six verdict counts partition the score.
Ask for them with `verdicts=True`, which reuses the alignment the score already paid for.

```bash
python - <<'PY'
import json, pathlib
from omni_extract_bench import score
from omni_extract_bench.metric import show

data = pathlib.Path("tutorials/data/quickstart_scoring")
load = lambda name: json.loads((data / name).read_text())

result = score(load("pred.json"), load("gold.json"), load("schema.json"), verdicts=True)
for v in result["verdicts"]:
    print(f"{show(v.address):<20} {v.verdict:<14} gold={str(v.gold_raw):<12}"
          f" pred={str(v.pred_raw):<12} canon={v.gold_canon}/{v.pred_canon}")
PY
```

```
invoice_id           matched        gold=INV-77       pred=INV-77       canon=inv77/inv77
line_items[0].qty    matched        gold=2            pred=2            canon=2/2
line_items[0].sku    misread        gold=AX-9910      pred=AX-9919      canon=ax9910/ax9919
line_items[1].qty    unfound        gold=1            pred=None         canon=1/None
line_items[1].sku    unfound        gold=BX-2201      pred=None         canon=bx2201/None
line_items[p1].qty   invented_item  gold=None         pred=9            canon=None/9
line_items[p1].sku   invented_item  gold=None         pred=ZZ-0000      canon=None/zz0000
total                matched        gold=1,240.00     pred=1240.00      canon=1240/1240
```

Things to point out:

- **`line_items[p1]`** is the row the prediction invented. Rows are matched by content, not by
  position, so an unpaired predicted row gets a made-up label — `p1` is not gold's index 1.
- **`line_items[1]`** is gold's second row, which the prediction never produced.
- **`total`** matched across `1,240.00` and `1240.00`. Both sides are carried raw *and*
  canonical so you can see that a fold did that work, and disagree with it;
  `values.canon_trace(raw)` names the step responsible.

| verdict | |
|---|---|
| `matched` | address and value both agree |
| `misread` | address in both, values differ |
| `unfound` | a gold address the prediction never produced |
| `fabricated` | the schema offered the slot, gold is silent, the model filled it |
| `invented_item` | an address under an array row that paired with nothing |
| `invented_field` | an address the schema never offered |

## Scoring a corpus

There is no runner here, and no table format to put your corpus into. A corpus is a loop, and
it stays yours — which means the decisions in it stay yours too:

```python
import collections, json, pathlib
from omni_extract_bench import score

rows = []
for doc in pathlib.Path("corpus").iterdir():
    pred = json.loads((doc / "pred.json").read_text())
    # A prediction can be a RECORDED FAILURE rather than an answer. Decide explicitly which
    # it is: scoring `{}` as 0.0 says the model tried and missed every field, and a corpus
    # where that is silently true of a third of the documents reports the wrong thing.
    if not pred or "__error__" in pred:
        rows.append({"doc_id": doc.name, "status": "error", "accuracy": None})
        continue
    result = score(pred, json.loads((doc / "gold.json").read_text()),
                   json.loads((doc / "schema.json").read_text()))
    rows.append({"doc_id": doc.name, "status": "scored", "subset": doc.parent.name, **result})

scored = [r for r in rows if r["status"] == "scored"]
print(f"{len(scored)}/{len(rows)} documents scored")

# Mean per subset, then mean of those -- NOT a flat mean over documents, which lets a large
# subset dominate. See docs/METRIC_SPEC.md section 7.
by_subset = collections.defaultdict(list)
for r in scored:
    by_subset[r["subset"]].append(r["accuracy"])
unified = sum(sum(v) / len(v) for v in by_subset.values()) / len(by_subset)
print(f"UNIFIED {unified:.2f}")
```

Two things that loop does on purpose, and that a runner would have decided for you:

- **Failures occupy rows.** Coverage is only visible if a document that produced nothing is
  still counted. Its metrics are `None`, never `0` — so say how many documents an average
  was over.
- **The aggregation is named.** Equal weight per subset rather than per document is a claim
  about what the benchmark measures, and it belongs where you can read it.
