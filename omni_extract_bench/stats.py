"""What describes each document of a manifest: `data.DOCUMENT_STATS`, computed without a model.

`describe` reads one manifest and returns it as a table with one column per stat added (or replaced); `oeb
stats` writes that table to a new parquet file. Two stats are read off the PDF, with pypdf; the rest are
counted from the gold:

    pages           the PDF's page count
    scanned         whether the PDF has no text layer: under 50 characters a page on its first three pages
    gold_chars      the gold as compact JSON, in characters
    table_rows      the rows of each array of objects in the gold, largest first: [540, 12], or [] when there
                    are none. A nested array (rows[*].items) is one entry, its rows summed over its parents
    num_values      leaf values in the gold, nulls included
    num_null        those that are null
    max_depth       keys on the path to the deepest value, array indices not counted: 1 for a top-level field
                    or tags[0], 2 for rows[0].x, 3 for rows[0].items[0].y
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import data

SCANNED_CHARS = 50  # text-layer characters a page, below which the PDF counts as scanned
PAGES_SAMPLED = 3


def pdf_stats(doc: data.Doc) -> dict[str, Any]:
    """`pages` and `scanned`, read off the document's PDF."""
    try:
        import pypdf
    except ImportError:
        raise ImportError("oeb stats reads PDFs with pypdf; install it with the benchmark extra: "
                          "pip install 'omni-extract-bench[benchmark]'") from None
    try:
        pages = pypdf.PdfReader(doc.doc_path).pages
        n = min(PAGES_SAMPLED, len(pages))
        chars = sum(len((pages[i].extract_text() or "").strip()) for i in range(n))
    except Exception as exc:  # noqa: BLE001 - pypdf raises many kinds for a damaged PDF; each means the same
        raise ValueError(f"{doc.doc_id}: its PDF {doc.doc_path} can't be read ({type(exc).__name__}: {exc}); fix or "
                         "replace the file") from None
    return {"pages": len(pages), "scanned": chars < SCANNED_CHARS * n}


def gold_stats(gold: Any) -> dict[str, Any]:
    """Every stat counted from the gold.

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
    got = pdf_stats(doc) | gold_stats(json.loads(Path(doc.gt_path).read_text()))
    assert set(got) == set(data.DOCUMENT_STATS), set(got) ^ set(data.DOCUMENT_STATS)
    return {k: got[k] for k in data.DOCUMENT_STATS}


def describe(dataset: str | Path = data.DEFAULT_DATASET, manifest: str = data.MANIFEST):
    """The manifest's path, and the manifest as a pyarrow table with `data.DOCUMENT_STATS` added, after its
    other columns; columns of those names it already had are replaced."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, docs = data.read_manifest(dataset, manifest)
    src = Path(root) / manifest
    t = pq.read_table(src)
    assert t.column("doc_id").to_pylist() == [d.doc_id for d in docs]
    got = [stats(d) for d in docs]
    t = t.drop_columns([c for c in data.DOCUMENT_STATS if c in t.column_names])
    for c in data.DOCUMENT_STATS:
        xs = [x[c] for x in got]
        # Note: typed, since a manifest whose tables are all empty would otherwise infer list<null>.
        t = t.append_column(c, pa.array(xs, type=pa.list_(pa.int64())) if c == "table_rows" else pa.array(xs))
    return src, t
