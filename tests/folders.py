"""Small datasets and folders for the pytest-style tests.

    from tests.folders import SCHEMA, dataset, oeb, predictions
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from omni_extract_bench import cli, data

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def dataset(root: Path, n: int = 2, **row) -> Path:
    """A dataset folder of `n` documents, each answering {"a": "x"}; `row` overrides every row's columns."""
    (root / "pdfs").mkdir(parents=True)
    (root / "gold").mkdir()
    rows = []
    for i in range(n):
        (root / "pdfs" / f"d{i}.pdf").write_bytes(b"%PDF")
        (root / "gold" / f"d{i}.json").write_text(json.dumps({"a": "x"}))
        rows.append({"doc_id": f"d{i}", "suite": "s", "doc_path": f"pdfs/d{i}.pdf",
                     "gt_path": f"gold/d{i}.json", "schema": json.dumps(SCHEMA)} | row)
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.Table.from_pylist(rows[:1]).schema if rows else
                                        pa.schema([(c, pa.string()) for c in data.COLUMNS])),
                   root / "manifest.parquet")
    return root


def predictions(folder: Path, answers: dict[str, object]) -> Path:
    """`folder` with predictions/ holding each named answer (a str is written as it is)."""
    (folder / "predictions").mkdir(parents=True, exist_ok=True)
    for name, x in answers.items():
        (folder / "predictions" / name).write_text(x if isinstance(x, str) else json.dumps(x))
    return folder


def oeb(capsys, *argv: str) -> tuple[int, str, str]:
    """`oeb argv...`: its exit code, stdout and stderr."""
    code = cli.main(list(argv))
    got = capsys.readouterr()
    return code, got.out, got.err
