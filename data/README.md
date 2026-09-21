# Example data

Two examples, for the two things you can run without a corpus.

## `gold.json` / `pred.json` / `schema.json` — scoring

A made-up invoice, written to exercise every verdict the metric produces: a matched value, a
misread one, a missing one, a fabricated one, an invented row and an invented field. Used by
the README's `oeb score` walkthrough. No PDF, because scoring never needs one.

```bash
oeb score --gt data/gold.json --pred data/pred.json --schema data/schema.json
```

## `h5filing.*` — predicting

A real benchmark document, so `oeb predict` can be tried against a live vendor without
downloading the corpus. The smallest in the set at 9.4 KB, which makes it the cheapest thing to
send: a two-page Texas Railroad Commission H-5 injection-well filing, 46 fields.

```bash
oeb predict --provider datalab --doc data/h5filing.pdf --schema data/h5filing.schema.json
```

`h5filing.gold.json` is its ground truth, so the two verbs compose:

```bash
oeb predict --provider datalab --doc data/h5filing.pdf \
            --schema data/h5filing.schema.json | jq .result > /tmp/pred.json
oeb score --pred /tmp/pred.json --gt data/h5filing.gold.json --schema data/h5filing.schema.json
```

### Provenance

`h5filing.*` is `short__h5Filing-46780-1` from the `extractbench` suite of
[datalab-to/omni_extract_bench](https://huggingface.co/datasets/datalab-to/omni_extract_bench),
which is Apache-2.0 — the same licence as this repository. The full corpus is 620 documents;
`oeb benchmark` downloads it.
