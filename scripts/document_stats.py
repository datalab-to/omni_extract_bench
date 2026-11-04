#!/usr/bin/env python3
"""Describe each document of a manifest: `data.DOCUMENT_STATS`, one column each.

    uv run --extra benchmark --with pymupdf python scripts/document_stats.py --out build/manifest.parquet
    uv run --extra benchmark --with pymupdf python scripts/document_stats.py --dataset my-dataset/ \
        --manifest manifests/sample.parquet --out build/sample.parquet

Every column is computed, with no model: two are read off the PDF, the rest counted from the gold. The script
reads one manifest of the dataset (`--manifest`, by default manifest.parquet) and writes it to `--out`, one
parquet file, with the columns added (or replaced). A dataset with several manifests is described one manifest
at a time. The dataset itself is never written to; replacing or uploading the manifest is a separate step.

The columns:

    pages           the PDF's page count
    scanned         whether the PDF has no text layer: under 50 characters a page on its first three pages
    gold_chars      the gold as compact JSON, in characters
    table_rows      the rows of each array of objects in the gold, largest first: [540, 12], or []
                    when there are none. A nested array (rows[*].items) is one entry, its rows
                    summed over its parents. In DuckDB, the largest is list_max(table_rows) and the
                    count of tables len(table_rows)
    num_values      leaf values in the gold, nulls included
    num_null        those that are null
    max_depth       keys on the path to the deepest value, array indices not counted: 1 for a top-level
                    field or tags[0], 2 for rows[0].x, 3 for rows[0].items[0].y
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from omni_extract_bench import data  # noqa: E402

SCANNED_CHARS = 50  # text-layer characters a page, below which the PDF counts as scanned
PAGES_SAMPLED = 3


def pdf_stats(pdf: Path) -> dict[str, Any]:
    """`pages` and `scanned`."""
    import pymupdf

    with pymupdf.open(pdf) as doc:
        n = min(PAGES_SAMPLED, len(doc))
        chars = sum(len(doc[i].get_text().strip()) for i in range(n))
        return {"pages": len(doc), "scanned": chars < SCANNED_CHARS * n}


def gold_stats(gold: Any) -> dict[str, Any]:
    """Every column read off the gold.

    >>> gold_stats({"a": 1, "rows": [{"x": None, "items": [{"y": 2}]}, {"x": 3, "items": []}]})
    {'gold_chars': 64, 'table_rows': [2, 1], 'num_values': 4, 'num_null': 1, 'max_depth': 3}
    """
    tables: dict[str, int] = {}
    out = {"num_values": 0, "num_null": 0, "max_depth": 0}

    def walk(x: Any, path: str, depth: int) -> None:
        if isinstance(x, dict):
            for k, v in x.items(): walk(v, f"{path}.{k}" if path else k, depth + 1)
        elif isinstance(x, list):
            if x and all(isinstance(r, dict) for r in x): tables[path] = tables.get(path, 0) + len(x)
            for v in x: walk(v, f"{path}[*]", depth)
        else:
            out["num_values"] += 1
            out["num_null"] += x is None
            out["max_depth"] = max(out["max_depth"], depth)

    walk(gold, "", 0)
    return {"gold_chars": len(json.dumps(gold, ensure_ascii=False, separators=(",", ":"))),
            "table_rows": sorted(tables.values(), reverse=True), **out}


def stats(doc: data.Doc) -> dict[str, Any]:
    """`data.DOCUMENT_STATS` for one document, in that order."""
    got = pdf_stats(doc.doc_path) | gold_stats(json.loads(Path(doc.gt_path).read_text()))
    assert set(got) == set(data.DOCUMENT_STATS), set(got) ^ set(data.DOCUMENT_STATS)
    return {k: got[k] for k in data.DOCUMENT_STATS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default=data.DEFAULT_DATASET)
    p.add_argument("--manifest", default=data.MANIFEST, help="a manifest inside the dataset (default: %(default)s)")
    p.add_argument("--out", type=Path, required=True, help="the parquet file to write the described manifest to")
    args = p.parse_args()

    root, docs = data.read_manifest(args.dataset, args.manifest)
    src = Path(root) / args.manifest
    if args.out.resolve() == src.resolve():
        raise SystemExit(f"--out is the manifest itself, {src}; write the new manifest somewhere else, then replace "
                         "the old one by hand")
    t = pq.read_table(src)
    assert t.column("doc_id").to_pylist() == [d.doc_id for d in docs]
    got = []
    for i, d in enumerate(docs):
        got.append(stats(d))
        if i % 50 == 0: print(f"{i}/{len(docs)} {d.doc_id}", flush=True)
    t = t.drop_columns([c for c in data.DOCUMENT_STATS if c in t.column_names])
    for c in data.DOCUMENT_STATS:
        xs = [x[c] for x in got]
        # Note: typed, since a manifest whose tables are all empty would otherwise infer list<null>.
        t = t.append_column(c, pa.array(xs, type=pa.list_(pa.int64())) if c == "table_rows" else pa.array(xs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, args.out)
    print(f"wrote {args.out}: {t.num_rows} rows, {t.num_columns} columns")


if __name__ == "__main__":
    main()
