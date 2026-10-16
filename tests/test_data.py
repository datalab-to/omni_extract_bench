"""`data`: what a manifest may hold, and how documents are chosen.

    pytest tests/test_data.py
"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from omni_extract_bench import data
from omni_extract_bench.errors import UsageError
from tests.folders import dataset, oeb


@pytest.mark.parametrize("schema", ["null", "[]", "{}"])
def test_a_schema_that_cant_be_sent_is_refused(tmp_path, schema):
    with pytest.raises(data.ManifestError, match="d0") as got:
        data.read_manifest(dataset(tmp_path / "ds", schema=schema))
    assert repr(schema) in str(got.value)


def test_a_doc_id_that_isnt_a_str_is_refused(tmp_path):
    ds = dataset(tmp_path / "ds")
    rows = pq.read_table(ds / "manifest.parquet").to_pylist()
    pq.write_table(pa.Table.from_pylist([{**r, "doc_id": i} for i, r in enumerate(rows)]), ds / "manifest.parquet")
    with pytest.raises(data.ManifestError, match="doc_id 0"):
        data.read_manifest(ds)


def test_an_empty_manifest_is_refused(tmp_path):
    with pytest.raises(data.ManifestError, match="names no documents"):
        data.read_manifest(dataset(tmp_path / "ds", n=0))


def test_a_manifest_is_a_relative_path_inside_the_dataset(tmp_path):
    ds = dataset(tmp_path / "ds")
    for manifest in (str(ds / "manifest.parquet"), "../ds/manifest.parquet"):
        with pytest.raises(data.ManifestError, match="--manifest manifest.parquet|without '..'"):
            data.read_manifest(ds, manifest)
    with pytest.raises(data.ManifestError, match="ds/manifest.parquet"):
        data.read_manifest(ds, "ds/manifest.parquet")


def test_a_one_slash_hf_address_is_named():
    with pytest.raises(UsageError, match="with two slashes"):
        data.read_manifest("hf:/datasets/datalab-to/omni_extract_bench")


def test_select_reads_its_names_once_and_refuses_a_bare_str(tmp_path):
    docs = data.read_manifest(dataset(tmp_path / "ds"))[1]
    with pytest.raises(TypeError, match="'s'"):
        data.select(docs, "s")
    assert len(data.select(docs, (x for x in ["s"]))) == 2, "a generator is read once"
    assert [d.doc_id for d in data.select(docs, only=["d1"])] == ["d1"]
    with pytest.raises(UsageError, match=r"\['t'\]"):
        data.select(docs, ["t"])


@pytest.mark.parametrize("kw, said", [({"limit": -1}, "--limit is -1"), ({"sample": -2}, "--sample is -2")])
def test_select_refuses_a_negative_count(tmp_path, kw, said):
    with pytest.raises(UsageError, match=said):
        data.select(data.read_manifest(dataset(tmp_path / "ds"))[1], **kw)


def test_check_names_the_row_of_a_missing_file(tmp_path, capsys):
    ds = dataset(tmp_path / "ds")
    (ds / "gold" / "d1.json").unlink()
    code, _, err = oeb(capsys, "check", "--dataset", str(ds))
    assert code == 1 and "d1 gt_path" in err


def test_ids_differing_only_in_case_show_both_spellings(tmp_path):
    ds = dataset(tmp_path / "ds")
    (ds / "gold" / "D0.json").write_text((ds / "gold" / "d0.json").read_text())
    rows = pq.read_table(ds / "manifest.parquet").to_pylist()
    pq.write_table(pa.Table.from_pylist([rows[0], rows[0] | {"doc_id": "D0", "gt_path": "gold/D0.json"}]),
                   ds / "manifest.parquet")
    with pytest.raises(data.ManifestError) as e:
        data.read_manifest(ds)
    assert "'d0'" in str(e.value) and "'D0'" in str(e.value)


def test_a_dataset_offline_and_uncached_is_refused_at_once(monkeypatch):
    import huggingface_hub
    from huggingface_hub import errors as hf_errors

    tries = []

    def offline(*a, **k):
        tries.append(1)
        raise hf_errors.LocalEntryNotFoundError("x")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    with pytest.raises(UsageError, match="cache"):
        data.download_dataset("hf://datasets/someone/nowhere@nope")
    assert len(tries) == 1
