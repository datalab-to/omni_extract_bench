"""What a user gets wrong is refused, or said, in one line naming the value and the fix.

    pytest tests/test_guardrails.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from omni_extract_bench import benchmark, cli, data
from omni_extract_bench.scorers import field_matching

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


def oeb(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    got = capsys.readouterr()
    return code, got.out, got.err


def predictions(folder: Path, answers: dict[str, object]) -> Path:
    (folder / "predictions").mkdir(parents=True, exist_ok=True)
    for name, x in answers.items():
        (folder / "predictions" / name).write_text(x if isinstance(x, str) else json.dumps(x))
    return folder


# A run folder is graded against the dataset it was predicted on.

def test_grade_takes_the_runs_own_dataset_and_refuses_another(tmp_path, capsys):
    ds, other = dataset(tmp_path / "ds"), dataset(tmp_path / "other")
    run = predictions(tmp_path / "run", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}})
    (run / "settings.json").write_text(json.dumps({"provider": "mine", "dataset": data.dataset_id(ds),
                                                   "manifest": "manifest.parquet"}))
    code, out, _ = oeb(capsys, "grade", str(run), "--score-workers", "1")
    assert code == 0 and json.loads(out)["headline"] == 1.0
    code, _, err = oeb(capsys, "grade", str(run), "--dataset", str(other), "--score-workers", "1")
    assert code == 1 and "a run never mixes datasets" in err and json.loads((run / "summary.json").read_text())["headline"] == 1.0


def test_grading_nothing_is_refused_rather_than_an_empty_summary(tmp_path):
    run = benchmark.Run("mine", {}, predictions(tmp_path / "run", {}))
    with pytest.raises(ValueError, match="no documents to grade"):
        benchmark.grade(run, [], score_workers=1)


def test_check_refuses_an_empty_manifest(tmp_path, capsys):
    code, _, err = oeb(capsys, "check", "--dataset", str(dataset(tmp_path / "ds", n=0)))
    assert code == 1 and "names no documents" in err


def test_grade_says_which_files_it_could_not_use(tmp_path, caplog):
    ds = dataset(tmp_path / "ds")
    run = predictions(tmp_path / "run", {"d0.json": {"a": "x"}, "d1.json": "[1]", "d0.pdf.json": {"a": "x"}})
    (run / "predictions" / "nested").mkdir()
    with caplog.at_level(logging.WARNING, logger="omni_extract_bench"):
        benchmark.grade(benchmark.Run("mine", {}, run), list(data.read_manifest(ds)[1]), score_workers=1)
    said = caplog.text
    assert "2 files in predictions/ match no document" in said and "d0.pdf.json" in said and "nested/" in said
    assert "1 predictions can't be read" in said and "d1" in said


# A grading's verdicts are its own.

def test_verdicts_are_one_gradings_and_go_with_no_verdicts(tmp_path):
    ds = dataset(tmp_path / "ds")
    docs = list(data.read_manifest(ds)[1])
    run = benchmark.Run("mine", {}, predictions(tmp_path / "run", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}}))
    (run.out / "verdicts").mkdir()
    (run.out / "verdicts" / "stale.jsonl").write_text("")
    benchmark.grade(run, docs, score_workers=1)
    assert sorted(p.name for p in (run.out / "verdicts").iterdir()) == ["d0.jsonl", "d1.jsonl"]
    assert not (run.out / ".verdicts.partial").exists()
    benchmark.grade(run, docs, verdicts=False, score_workers=1)
    assert not (run.out / "verdicts").exists()


# --retry-failed predicts again what may go the other way: omni-parse-bench's rule.

@pytest.mark.parametrize("error, again", [
    ({"type": "VendorTimeout", "status": None}, True),
    ({"type": "VendorUnreachable", "status": None}, True),
    ({"type": "VendorError", "status": 429}, True),
    ({"type": "VendorError", "status": 524}, True),
    ({"type": "VendorError", "status": 400}, False),
    ({"type": "VendorError", "status": None}, False),
    ({"type": "DialectError", "status": None}, False),
])
def test_retry_failed_follows_omni_parse_bench(tmp_path, error, again):
    run = benchmark.Run("mine", {}, tmp_path / "run")
    run.records.mkdir(parents=True)
    (run.records / "d.json").write_text(json.dumps({"result": None, "error": error}))
    doc = data.Doc("d", "s", tmp_path / "d.pdf", tmp_path / "d.json", SCHEMA)
    assert benchmark.needs(run, doc, retry_failed=True) is again and benchmark.needs(run, doc) is False


# A manifest row oeb can't run is refused, naming the row.

@pytest.mark.parametrize("row, said", [
    ({"schema": "null"}, "non-empty JSON object"), ({"schema": "[]"}, "non-empty JSON object"),
    ({"schema": "{}"}, "non-empty JSON object"),
])
def test_a_schema_that_cant_be_sent_is_refused(tmp_path, capsys, row, said):
    code, _, err = oeb(capsys, "check", "--dataset", str(dataset(tmp_path / "ds", **row)))
    assert code == 1 and said in err and "d0" in err


def test_a_doc_id_that_isnt_a_str_is_refused(tmp_path):
    ds = dataset(tmp_path / "ds")
    rows = pq.read_table(ds / "manifest.parquet").to_pylist()
    pq.write_table(pa.Table.from_pylist([{**r, "doc_id": i} for i, r in enumerate(rows)]), ds / "manifest.parquet")
    with pytest.raises(data.ManifestError, match="not a filename"):
        data.read_manifest(ds)


def test_a_missing_manifest_says_it_is_a_path_inside_the_dataset(tmp_path, capsys):
    ds = dataset(tmp_path / "ds")
    code, _, err = oeb(capsys, "check", "--dataset", str(ds), "--manifest", "ds/manifest.parquet")
    assert code == 1 and "a path inside the dataset (--dataset)" in err


def test_a_one_slash_hf_address_is_named(capsys):
    code, _, err = oeb(capsys, "check", "--dataset", "hf:/datasets/datalab-to/omni_extract_bench")
    assert code == 1 and "with two slashes" in err


# Mistakes on the command line are one line and exit 1.

def test_options_name_what_was_wrong(tmp_path, capsys):
    ds = str(dataset(tmp_path / "ds"))
    code, _, err = oeb(capsys, "benchmark", "--providers", "datalab", "--dataset", ds, "--options", "opts.jsn", "-y")
    assert code == 1 and "no file of that name" in err
    code, _, err = oeb(capsys, "benchmark", "--providers", "datalab", "--dataset", ds, "--options", "[1]", "-y")
    assert code == 1 and "must be a JSON object" in err


def test_docs_naming_a_missing_file_says_so(tmp_path, capsys):
    ds = str(dataset(tmp_path / "ds"))
    code, _, err = oeb(capsys, "benchmark", "--providers", "datalab", "--dataset", ds, "--docs", "nope.txt", "-y")
    assert code == 1 and err.startswith("oeb benchmark: --docs nope.txt: no such file")


def test_an_unknown_provider_is_one_line(tmp_path, capsys):
    code, _, err = oeb(capsys, "benchmark", "--providers", "nope", "--dataset", str(dataset(tmp_path / "ds")), "-y")
    assert code == 1 and err.startswith("oeb benchmark: unknown provider 'nope'") and "Traceback" not in err


def test_score_refuses_a_prediction_that_isnt_an_object(tmp_path, capsys):
    for name, x in (("p.json", [1]), ("g.json", {"a": "x"}), ("s.json", SCHEMA)):
        (tmp_path / name).write_text(json.dumps(x))
    code, _, err = oeb(capsys, "score", "--pred", str(tmp_path / "p.json"), "--gt", str(tmp_path / "g.json"),
                       "--schema", str(tmp_path / "s.json"))
    assert code == 1 and "--pred" in err and "JSON object" in err
    with pytest.raises(TypeError, match="prediction must be a JSON object"):
        field_matching.score([1], {"a": "x"}, SCHEMA)


def test_a_model_ids_options_hint_never_suggests_model(capsys):
    code, out, _ = oeb(capsys, "providers", "openai/gpt-5.6-sol")
    assert code == 0 and '"model"' not in out.split("oeb benchmark")[-1]


def test_quiet_goes_after_the_command_too(tmp_path, capsys):
    code, _, _ = oeb(capsys, "check", "--dataset", str(dataset(tmp_path / "ds")), "-q")
    assert code == 0


# The library refuses what it would misread.

def test_a_bare_str_of_names_is_refused(tmp_path):
    docs = data.read_manifest(dataset(tmp_path / "ds"))[1]
    with pytest.raises(TypeError, match="pass a list"):
        data.select(docs, "s")
    with pytest.raises(TypeError, match="pass a list"):
        benchmark.plan("datalab", dataset=str(tmp_path / "ds"))
    assert [d.doc_id for d in data.select(docs, only=["d1"])] == ["d1"]


@pytest.mark.parametrize("kw, said", [({"limit": -1}, "limit is -1"), ({"sample": -2}, "sample is -2")])
def test_select_refuses_a_negative_count(tmp_path, kw, said):
    with pytest.raises(ValueError, match=said):
        data.select(data.read_manifest(dataset(tmp_path / "ds"))[1], **kw)


@pytest.mark.parametrize("kw, err, said", [
    ({"timeout": 0}, ValueError, "--timeout is 0"), ({"timeout": -1}, ValueError, "--timeout is -1"),
    ({"workers": "3"}, TypeError, "workers is '3'"),
    ({"workers": -3}, ValueError, "0 means the default per vendor"),
])
def test_plan_refuses_what_it_would_misread(tmp_path, kw, err, said):
    with pytest.raises(err, match=said):
        benchmark.plan(["datalab"], out=tmp_path / "runs", dataset=str(dataset(tmp_path / "ds")), **kw)


def test_no_documents_selected_names_the_flags_given(tmp_path):
    ds = str(dataset(tmp_path / "ds"))
    with pytest.raises(ValueError, match="no documents selected by --docs"):
        benchmark.plan(["datalab"], out=tmp_path / "runs", dataset=ds, only=frozenset())


def test_an_adapter_bug_stops_its_runs_naming_the_document(tmp_path, monkeypatch):
    import types

    from omni_extract_bench.harness import document

    real = document.adapter

    def broken(provider):
        a = real(provider)
        def extract(pdf, schema, *, timeout, config):
            raise KeyError("result")
        return types.SimpleNamespace(__name__=a.__name__, Config=a.Config, prepare_schema=a.prepare_schema,
                                     extract=extract)
    monkeypatch.setattr(document, "adapter", broken)
    monkeypatch.setenv("DATALAB_API_KEY", "x")
    p = benchmark.plan(["datalab"], out=tmp_path / "runs", dataset=str(dataset(tmp_path / "ds", n=1)),
                       score_workers=1, workers=1)
    with pytest.raises(KeyError) as got:
        benchmark.predict_all(p)
    assert any("while predicting d0 with the datalab adapter" in n and "a bug" in n for n in got.value.__notes__)
