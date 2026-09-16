---
license: TBD                      # the PDFs, not the code -- see Licence below
task_categories:
  - visual-question-answering
  - table-question-answering
tags:
  - document-extraction
  - structured-extraction
  - json-schema
  - benchmark
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files: manifest.parquet
---

# Omni Extract Bench

620 documents, each with a JSON Schema it was asked for and the ground-truth extraction a
human agreed it should produce. The point of the dataset is that the three things ship
together: a schema is what makes an extraction gradeable, and a ground truth without the
schema it answers cannot be scored against a model that was asked for something else.

Scored by [`omni-extract-bench`](https://github.com/datalab-to/omni_extract_bench) — one
metric, applied identically to every document and every provider.

## Layout

```
manifest.parquet        one row per document; the schema is in it
pdfs/<doc_id>.pdf       the document
gold/<doc_id>.json      the ground-truth extraction
```

| column | |
|---|---|
| `doc_id` | opaque, stable, unique; the name the files carry |
| `doc_path` | `pdfs/<doc_id>.pdf`, relative to the dataset root |
| `gt_path` | `gold/<doc_id>.json`, relative to the dataset root |
| `schema` | the JSON Schema itself, inline as bytes |

**Paths are relative, and that is deliberate.** The scorer reads a relative path from a
`--root` you give it and an absolute path or a URI as written, so this table works wherever
you unpack it — and a `pred_path` you add pointing at your own predictions is absolute and is
read from where it actually is. Nothing here has to be rewritten after download.

The schema rides in the table rather than in a file because it is what makes the row
self-describing: hand someone one row and they can reproduce the grading.

## Get it

```bash
uv pip install 'omni-extract-bench[benchmark]' polars   # polars is for the examples below
huggingface-cli download datalab-to/omni-extract-bench --repo-type dataset --local-dir benchmark
```

Everything below assumes `benchmark/` is that directory, and that you are working next to it,
not inside it. Each command names it with `--root`; export it once instead if you prefer:

```bash
export OEB_ROOT=$PWD/benchmark
```

## Score your own predictions

One JSON file per document, named however you like. Join them onto the manifest by `doc_id`
and hand the result to the scorer:

```python
import pathlib
import polars as pl

mine = pathlib.Path("my-predictions").resolve()      # absolute: these are yours, not the corpus's
files = sorted(mine.glob("*.json"))                  # once, so the two columns stay aligned
preds = pl.DataFrame({"doc_id": [f.stem for f in files],
                      "pred_path": [str(f) for f in files]})

(pl.read_parquet("benchmark/manifest.parquet")
   .join(preds, on="doc_id")                         # an inner join scores the subset you have
   .write_parquet("jobs.parquet"))
```

```bash
oeb score --root benchmark --manifest jobs.parquet --out run/ --jobs 8
```

Resolve `pred_path` to an absolute path, as above. A relative one would be read from `--root`,
which is where the corpus lives and your predictions do not.

## Or predict with the harness

If the provider is one the harness speaks, there is nothing to join — its output table *is* a
score manifest, with `gt_path` carried through from this one:

```bash
oeb predict --root benchmark --manifest benchmark/manifest.parquet \
            --out preds/ --provider datalab
oeb score   --root benchmark --manifest preds/manifest.parquet --out run/
```

Providers: `azure-cu`, `claude`, `datalab`, `datalab-accurate`, `extend`, `gemini`, `gpt`,
`gpt-pro`, `llamaextract`, `mistral`, `reducto`. Each needs its own credentials in the
environment, and the adapters live behind an extra: `uv pip install 'omni-extract-bench[harness]'`.

## Read the results

Two tables land in `run/`: `scores.parquet/` has a row per document, `verdicts.parquet/` a row
per address.

```python
import polars as pl

scores = pl.read_parquet("run/scores.parquet")
graded = scores.filter(pl.col("status") == "scored")
print(f"{graded['accuracy'].mean():.2f} over {len(graded)} of {len(scores)} documents")
```

Filter on `status`, and say how many documents you averaged. A document the provider could not
answer comes back with null metrics and an `error`, not a zero — a zero would claim it tried
and missed every field, which is a different result and a better-looking one.

## The metric

Both extractions are flattened to addresses mapped to scalar values, values are normalised, and
arrays are Hungarian-matched on content so that row order is not scored. Every unique address
then gets exactly one verdict:

| verdict | |
|---|---|
| `matched` | paired, and the values agree |
| `misread` | paired, and they do not |
| `unfound` | the ground truth has it, the prediction does not |
| `fabricated` | the schema offered it, the ground truth is silent, the prediction is not |
| `invented_item` | inside an array element that paired with nothing |
| `invented_field` | an address the schema never declared |

Accuracy is `matched` over every address; precision and recall fall out of the same verdicts.
Full specification: [`docs/METRIC_SPEC.md`](https://github.com/datalab-to/omni_extract_bench/blob/main/docs/METRIC_SPEC.md).

## Licence

**TBD** — the PDFs are the question here, not the code; the scorer is Apache 2.0.

## Citation

```bibtex
@misc{omni_extract_bench,
  title  = {Omni Extract Bench},
  author = {Datalab},
  year   = {2026},
  url    = {https://github.com/datalab-to/omni_extract_bench}
}
```
