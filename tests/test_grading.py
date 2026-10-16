"""`benchmark.grade` and `oeb grade`: what a grading reads, writes, refuses and says.

    pytest tests/test_grading.py
"""
from __future__ import annotations

import json
import logging

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from omni_extract_bench import benchmark, data
from omni_extract_bench.errors import UsageError
from tests.folders import dataset, oeb, predictions

QUIET = lambda m: None


def test_a_run_folder_is_graded_on_its_own_dataset_and_refuses_another(tmp_path, capsys):
    ds, other = dataset(tmp_path / "ds"), dataset(tmp_path / "other")
    run = predictions(tmp_path / "run", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}})
    (run / "settings.json").write_text(json.dumps({"provider": "mine", "dataset": data.dataset_id(ds),
                                                   "manifest": "manifest.parquet"}))
    code, out, _ = oeb(capsys, "grade", str(run), "--score-workers", "1")
    assert code == 0 and json.loads(out)["headline"] == 1.0
    code, _, err = oeb(capsys, "grade", str(run), "--dataset", str(other), "--score-workers", "1")
    assert code == 1 and data.dataset_id(other) in err and data.dataset_id(ds) in err
    assert json.loads((run / "summary.json").read_text())["headline"] == 1.0, "a refused grading writes nothing"


def test_your_folder_remembers_what_it_was_graded_on(tmp_path, capsys):
    ds = dataset(tmp_path / "ds")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}})
    code, _, _ = oeb(capsys, "grade", str(mine), "--dataset", str(tmp_path / "nope"))
    assert code == 1 and not (mine / "settings.json").exists(), "a failed grading writes nothing"
    assert oeb(capsys, "grade", str(mine), "--dataset", str(ds), "--score-workers", "1")[0] == 0
    said = json.loads((mine / "settings.json").read_text())
    assert said["dataset"] == data.dataset_id(ds) and said["manifest"] == "manifest.parquet"
    code, out, _ = oeb(capsys, "grade", str(mine), "--score-workers", "1")
    assert code == 0 and json.loads(out)["documents"] == 2, "graded again on its own dataset, not ours"
    with pytest.raises(SystemExit):
        oeb(capsys, "grade", str(mine), "--suites", "s")


def test_a_folder_holds_one_manifest(tmp_path, capsys):
    ds = dataset(tmp_path / "ds")
    (ds / "manifests").mkdir()
    pq.write_table(pq.read_table(ds / "manifest.parquet").slice(0, 1), ds / "manifests" / "one.parquet")
    p = benchmark.plan(["datalab"], out=tmp_path / "runs", dataset=str(ds))
    benchmark.prepare(p.runs[0], str(ds), 30, "manifest.parquet")
    with pytest.raises(UsageError, match="manifest manifest.parquet, not manifests/one.parquet"):
        benchmark.plan(["datalab"], out=tmp_path / "runs", dataset=str(ds), manifest="manifests/one.parquet")
    code, _, err = oeb(capsys, "grade", str(p.runs[0].out), "--manifest", "manifests/one.parquet")
    assert code == 1 and "manifests/one.parquet" in err


def test_grade_says_once_which_files_it_could_not_use(tmp_path, caplog):
    ds = dataset(tmp_path / "ds")
    run = predictions(tmp_path / "run", {"d0.json": {"a": "x"}, "d1.json": "[1]", "d0.pdf.json": {"a": "x"}})
    (run / "predictions" / "nested").mkdir()
    with caplog.at_level(logging.WARNING, logger="omni_extract_bench"):
        benchmark.grade(benchmark.Run("mine", {}, run), data.read_manifest(ds)[1], QUIET, score_workers=1)
    assert "d0.pdf.json" in caplog.text and "nested/" in caplog.text
    assert caplog.text.count("d1") == 1, "an unreadable prediction is said once"


def test_a_name_that_differs_in_case_is_stray_everywhere(tmp_path, caplog):
    ds = dataset(tmp_path / "ds")
    run = predictions(tmp_path / "run", {"d0.json": {"a": "x"}, "D1.json": {"a": "x"}})
    with caplog.at_level(logging.WARNING, logger="omni_extract_bench"):
        s = benchmark.grade(benchmark.Run("mine", {}, run), data.read_manifest(ds)[1], QUIET, score_workers=1)
    assert (s["scored"], s["failed"], s["coverage"]) == (1, 1, 0.5) and "D1.json" in caplog.text
    lines = benchmark.read_scores(benchmark.Run("mine", {}, run))
    assert [(x["doc_id"], x["status"]) for x in lines] == [("d0", "scored"), ("d1", "failed")]
    assert all(x["run"] == "run" for x in lines)


def test_verdicts_are_one_gradings_and_go_with_no_verdicts(tmp_path):
    docs = data.read_manifest(dataset(tmp_path / "ds"))[1]
    run = benchmark.Run("mine", {}, predictions(tmp_path / "run", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}}))
    (run.out / "verdicts").mkdir()
    (run.out / "verdicts" / "stale.jsonl").write_text("")
    benchmark.grade(run, docs, QUIET, score_workers=1)
    assert sorted(p.name for p in (run.out / "verdicts").iterdir()) == ["d0.jsonl", "d1.jsonl"]
    assert not (run.out / ".verdicts.partial").exists()
    benchmark.grade(run, docs, QUIET, verdicts=False, score_workers=1)
    assert not (run.out / "verdicts").exists()


def test_a_run_with_pending_documents_has_no_headline_yet(tmp_path):
    docs = data.read_manifest(dataset(tmp_path / "ds", n=3))[1]
    run = benchmark.Run("mine", {}, tmp_path / "run")
    run.records.mkdir(parents=True)
    run.predictions.mkdir()
    benchmark.record(run, "d0", {"result": {"a": "x"}, "error": None})
    said = []
    s = benchmark.grade(run, docs, said.append, score_workers=1)
    assert s["headline"] is None and s["pending"] == 2 and s["accuracy"] == 1.0
    assert s["per_suite"]["s"]["pending"] == 2 and s["per_suite"]["s"]["headline"] is None
    assert said == [("run: headline — (2 pending: a headline needs every document); 1 of 3 documents "
                     "graded, 0 failed")]
    for d in ("d1", "d2"): benchmark.record(run, d, {"result": {"a": "x"}, "error": None})
    s = benchmark.grade(run, docs, said.append, score_workers=1)
    assert s["headline"] == 1.0 and s["pending"] == 0 and s["per_suite"]["s"]["pending"] == 0
    assert said[-1] == "run: headline 1.000; 3 of 3 documents graded, 0 failed"


def test_a_summary_says_what_graded_it_and_what_it_cost(tmp_path):
    docs = data.read_manifest(dataset(tmp_path / "ds"))[1]
    run = benchmark.Run("mine", {}, tmp_path / "run")
    run.records.mkdir(parents=True)
    run.predictions.mkdir()
    benchmark.record(run, "d0", {"result": {"a": "x"}, "error": None, "cost": {"usd": 0.25, "credits": None}})
    benchmark.record(run, "d1", {"result": None, "error": {"type": "VendorError", "message": "400", "status": 400},
                                 "cost": {"usd": 0.5, "credits": 3}})
    s = benchmark.grade(run, docs, QUIET, settings={"settings": {"mode": "fast"}}, score_workers=1)
    assert (s["run"], s["provider"], s["settings"]) == ("run", "mine", {"mode": "fast"})
    assert (s["failed"], s["usd"], s["credits"]) == (1, 0.75, 3)


def test_a_scorer_bug_is_left_out_and_exits_2(tmp_path, capsys, monkeypatch):
    from omni_extract_bench.scorers import field_matching

    ds = dataset(tmp_path / "ds")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}})
    real = field_matching.score
    monkeypatch.setattr(field_matching, "score", lambda pred, gold, schema, **kw:
                        (_ for _ in ()).throw(RuntimeError("a bug")) if pred.get("a") == "y" else real(pred, gold,
                                                                                                    schema, **kw))
    (mine / "predictions" / "d1.json").write_text(json.dumps({"a": "y"}))
    code, out, err = oeb(capsys, "grade", str(mine), "--dataset", str(ds), "--score-workers", "1")
    s = json.loads(out)
    assert code == 2 and "raised in our code" in err
    assert (s["errors"], s["documents"], s["headline"], s["coverage"]) == (1, 1, 1.0, 1.0)


def test_grading_the_predictions_folder_itself_points_at_its_parent(tmp_path, capsys):
    run = predictions(tmp_path / "run", {"d0.json": {"a": "x"}})
    code, _, err = oeb(capsys, "grade", str(run / "predictions"))
    assert code == 1 and f"pass its parent, {run}" in err


def test_a_file_of_a_suite_is_never_stray(tmp_path, caplog):
    ds = dataset(tmp_path / "ds")
    rows = pq.read_table(ds / "manifest.parquet").to_pylist()
    pq.write_table(pa.Table.from_pylist([rows[0], rows[1] | {"suite": "t"}]), ds / "manifest.parquet")
    run = predictions(tmp_path / "run", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}})
    with caplog.at_level(logging.WARNING, logger="omni_extract_bench"):
        s = benchmark.grade(benchmark.Run("mine", {}, run), data.read_manifest(ds)[1], QUIET, score_workers=1)
    assert "named for no document" not in caplog.text and set(s["per_suite"]) == {"s", "t"}
