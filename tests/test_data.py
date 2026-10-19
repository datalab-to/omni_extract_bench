"""`data`: what an extraction manifest may hold.

omni-bench-core checks what every dataset shares (ids, suites, paths, gold file names); these check that its
checks reach an oeb dataset, and oeb's own check, that each schema is a JSON object.

    pytest tests/test_data.py
"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omni_bench_core.errors import UsageError

from omni_extract_bench import data
from tests.folders import dataset


@pytest.mark.parametrize("schema", ["null", "[]", "{}"])
def test_a_schema_that_cant_be_sent_is_refused(tmp_path, schema):
    with pytest.raises(UsageError, match="d0") as got:
        data.read_manifest(dataset(tmp_path / "ds", schema=schema))
    assert repr(schema) in str(got.value)


def test_a_schema_that_isnt_json_is_refused(tmp_path):
    with pytest.raises(UsageError, match="not a JSON string"):
        data.read_manifest(dataset(tmp_path / "ds", schema="{nope"))


def test_a_doc_id_that_isnt_a_str_is_refused(tmp_path):
    ds = dataset(tmp_path / "ds")
    rows = pq.read_table(ds / "manifest.parquet").to_pylist()
    pq.write_table(pa.Table.from_pylist([{**r, "doc_id": i} for i, r in enumerate(rows)]), ds / "manifest.parquet")
    with pytest.raises(UsageError, match="doc_id 0"):
        data.read_manifest(ds)


def test_an_empty_manifest_is_refused(tmp_path):
    with pytest.raises(UsageError, match="names no rows"):
        data.read_manifest(dataset(tmp_path / "ds", n=0))


def test_a_manifest_is_a_relative_path_inside_the_dataset(tmp_path):
    ds = dataset(tmp_path / "ds")
    for manifest in (str(ds / "manifest.parquet"), "../ds/manifest.parquet"):
        with pytest.raises(UsageError, match="inside the dataset"):
            data.read_manifest(ds, manifest)


def test_ids_differing_only_in_case_show_both_spellings(tmp_path):
    ds = dataset(tmp_path / "ds")
    (ds / "gold" / "D0.json").write_text((ds / "gold" / "d0.json").read_text())
    rows = pq.read_table(ds / "manifest.parquet").to_pylist()
    pq.write_table(pa.Table.from_pylist([rows[0], rows[0] | {"doc_id": "D0", "gt_path": "gold/D0.json"}]),
                   ds / "manifest.parquet")
    with pytest.raises(UsageError) as e:
        data.read_manifest(ds)
    assert "d0" in str(e.value) and "D0" in str(e.value)


def test_a_doc_is_its_row_with_full_paths_and_its_schema_read(tmp_path):
    ds = dataset(tmp_path / "ds", lang="en")
    [d0, d1] = data.read_manifest(ds)[1]
    assert d0.id == "d0" and d0.doc_path == ds / "pdfs" / "d0.pdf" and d0.gt_path == ds / "gold" / "d0.json"
    assert d0.schema["type"] == "object" and d1.extra == {"lang": "en"}
