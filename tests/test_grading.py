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
    said, a_run = [], {"provider": "mine"}   # Note: a run's settings name its provider (`is_run`).
    s = benchmark.grade(run, docs, said.append, settings=a_run, score_workers=1)
    assert s["headline"] is None and s["pending"] == 2 and s["accuracy"] == 1.0
    assert s["per_suite"]["s"]["pending"] == 2 and s["per_suite"]["s"]["headline"] is None
    assert said == [("run: headline — (2 pending: a headline needs every document); 1 of 3 documents "
                     "graded, 0 failed")]
    for d in ("d1", "d2"): benchmark.record(run, d, {"result": {"a": "x"}, "error": None})
    s = benchmark.grade(run, docs, said.append, settings=a_run, score_workers=1)
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


def test_a_scorer_bug_stops_grading_naming_the_document_and_writes_nothing(tmp_path, capsys, monkeypatch):
    """A bug of ours: raised as it is, since a summary with the document left out would be wrong."""
    from omni_extract_bench.scorers import field_matching

    ds = dataset(tmp_path / "ds")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}, "d1.json": {"a": "y"}})
    real = field_matching.score

    def score(pred, gold, schema, **kw):
        if pred.get("a") == "y": raise RuntimeError("a bug")
        return real(pred, gold, schema, **kw)
    monkeypatch.setattr(field_matching, "score", score)
    with pytest.raises(RuntimeError) as e:
        oeb(capsys, "grade", str(mine), "--dataset", str(ds), "--score-workers", "1")
    assert "d1" in " ".join(e.value.__notes__)
    assert sorted(f.name for f in mine.iterdir()) == ["predictions"], "nothing written"


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


def test_a_gold_that_cant_be_read_stops_grading_naming_its_document(tmp_path, capsys):
    """A bad gold is the dataset's problem, as in omni-parse-bench: refused (exit 1), not a scorer bug."""
    ds = dataset(tmp_path / "ds")
    (ds / "gold" / "d1.json").write_text("[]")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}})
    code, _, err = oeb(capsys, "grade", str(mine), "--dataset", str(ds), "--score-workers", "1")
    assert code == 1 and "d1" in err and not (mine / "summary.json").exists()


def test_a_prediction_of_bytes_is_graded_as_failed(tmp_path, capsys):
    ds = dataset(tmp_path / "ds")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}})
    (mine / "predictions" / "d1.json").write_bytes(b"\xff\xfe\x00")
    code, out, _ = oeb(capsys, "grade", str(mine), "--dataset", str(ds), "--score-workers", "1")
    assert code == 0 and json.loads(out)["failed"] == 1


def test_an_answer_whose_prediction_was_lost_is_rebuilt_not_paid_for_again(tmp_path):
    """The record is the source of truth for an answer: a lost or corrupt prediction is written from it."""
    run = benchmark.Run("mine", {}, tmp_path / "run")
    run.predictions.mkdir(parents=True), run.records.mkdir()
    doc = data.Doc("d0", "s", tmp_path / "d0.pdf", tmp_path / "d0.json", {})
    benchmark.record(run, "d0", {"result": {"a": "x"}, "error": None})
    (run.predictions / "d0.json").unlink()
    assert not benchmark.needs(run, doc), "it has an answer"
    assert benchmark.restore(run, "d0") and json.loads((run.predictions / "d0.json").read_text()) == {"a": "x"}
    (run.predictions / "d0.json").write_text("{not json")
    assert benchmark.restore(run, "d0") and not benchmark.restore(run, "d0"), "rebuilt once, then left"
    benchmark.record(run, "d1", {"result": None, "error": {"type": "VendorError", "status": 400}})
    assert not benchmark.restore(run, "d1"), "a failure has no answer to rebuild"


def test_a_grading_that_matches_no_document_is_refused_and_writes_nothing(tmp_path, capsys):
    ds = dataset(tmp_path / "ds")
    for k, answers in enumerate(({"invoice.json": {"a": "x"}}, {})):
        mine = predictions(tmp_path / f"mine{k}", answers)
        code, _, err = oeb(capsys, "grade", str(mine), "--dataset", str(ds), "--score-workers", "1")
        assert code == 1 and "--dataset" in err
        assert sorted(f.name for f in mine.iterdir()) == ["predictions"], "nothing written, nothing locked"


def test_no_verdicts_after_an_interrupted_grading_leaves_no_verdicts(tmp_path):
    docs = data.read_manifest(dataset(tmp_path / "ds"))[1]
    run = benchmark.Run("mine", {}, predictions(tmp_path / "run", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}}))
    (run.out / ".verdicts.partial").mkdir()
    (run.out / ".verdicts.partial" / "d0.jsonl").write_text("{}")
    benchmark.grade(run, docs, QUIET, verdicts=False, score_workers=1)
    assert not (run.out / "verdicts").exists() and not (run.out / ".verdicts.partial").exists()


def test_a_folder_of_your_own_with_records_is_yours(tmp_path, capsys):
    """A run is a folder whose settings.json names a provider; records/ alone doesn't make one."""
    ds = dataset(tmp_path / "ds")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}})
    (mine / "records").mkdir()
    code, out, _ = oeb(capsys, "grade", str(mine), "--dataset", str(ds), "--score-workers", "1")
    s = json.loads(out)
    assert code == 0 and (s["documents"], s["failed"], s["pending"]) == (2, 1, 0)


@pytest.mark.parametrize("raw, why", [(b"", "empty"), (b"\xff\xfe", "UTF-8"), (b"{not", "isn't JSON"),
                                      (b"[1]", "list"), (b"null", "null")])
def test_a_prediction_that_isnt_an_object_says_why(tmp_path, raw, why):
    docs = data.read_manifest(dataset(tmp_path / "ds"))[1]
    run = benchmark.Run("mine", {}, predictions(tmp_path / "run", {"d0.json": {"a": "x"}}))
    (run.predictions / "d1.json").write_bytes(raw)
    benchmark.grade(run, docs, QUIET, score_workers=1)
    (line,) = [x for x in benchmark.read_scores(run) if x["doc_id"] == "d1"]
    assert line["status"] == "failed" and why in line["error"]


def test_a_prediction_with_a_byte_order_mark_reads_as_its_json(tmp_path):
    docs = data.read_manifest(dataset(tmp_path / "ds"))[1]
    run = benchmark.Run("mine", {}, predictions(tmp_path / "run", {"d0.json": {"a": "x"}}))
    (run.predictions / "d1.json").write_bytes(b"\xef\xbb\xbf" + json.dumps({"a": "x"}).encode())
    assert benchmark.grade(run, docs, QUIET, score_workers=1)["headline"] == 1.0


def test_with_nothing_scored_the_metrics_are_null(tmp_path):
    docs = data.read_manifest(dataset(tmp_path / "ds"))[1]
    run = benchmark.Run("mine", {}, predictions(tmp_path / "run", {"d0.json": "[]", "d1.json": "[]"}))
    s = benchmark.grade(run, docs, QUIET, score_workers=1)
    assert (s["headline"], s["coverage"], s["accuracy"], s["misread_rate"]) == (0.0, 0.0, None, None)
    assert s["per_suite"]["s"]["accuracy"] is None


def test_grading_part_of_a_folders_manifest_says_so(tmp_path, caplog):
    ds = dataset(tmp_path / "ds", n=3)
    mine = predictions(tmp_path / "mine", {f"d{i}.json": {"a": "x"} for i in range(3)})
    (mine / "settings.json").write_text(json.dumps({"dataset": data.dataset_id(ds), "manifest": "manifest.parquet"}))
    with caplog.at_level(logging.WARNING):
        benchmark.grade(benchmark.Run("mine", {}, mine), data.read_manifest(ds)[1][:1], QUIET, score_workers=1)
    assert "1 document of its manifest's 3" in caplog.text


def test_a_manifest_spelled_another_way_is_the_same_manifest(tmp_path, capsys):
    ds = dataset(tmp_path / "ds")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}})
    assert oeb(capsys, "grade", str(mine), "--dataset", str(ds), "--manifest", "./manifest.parquet",
               "--score-workers", "1")[0] == 0
    assert json.loads((mine / "settings.json").read_text())["manifest"] == "manifest.parquet"
    assert oeb(capsys, "grade", str(mine), "--manifest", "manifest.parquet", "--score-workers", "1")[0] == 0


def test_loose_predictions_are_told_where_to_go(tmp_path, capsys):
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "d0.json").write_text("{}")
    code, _, err = oeb(capsys, "grade", str(flat))
    assert code == 1 and "flat/predictions" in err


def test_every_grading_rebuilds_a_lost_prediction_from_its_record(tmp_path, capsys):
    """--score-only, oeb grade and a library grade all score an answer its record holds, rebuilt."""
    ds = dataset(tmp_path / "ds")
    run = benchmark.Run("mine", {}, tmp_path / "run")
    run.predictions.mkdir(parents=True), run.records.mkdir()
    for d in ("d0", "d1"): benchmark.record(run, d, {"result": {"a": "x"}, "error": None})
    benchmark.declare(run.out, str(ds), "manifest.parquet", {"run": "run", "provider": "mine"})
    (run.predictions / "d1.json").unlink()
    code, out, _ = oeb(capsys, "grade", str(run.out), "--score-workers", "1")
    assert code == 0 and json.loads(out)["failed"] == 0 and (run.predictions / "d1.json").exists()
    (run.predictions / "d1.json").write_text("[]")
    s = benchmark.grade(run, data.read_manifest(ds)[1], QUIET, score_workers=1)
    assert s["failed"] == 0 and json.loads((run.predictions / "d1.json").read_text()) == {"a": "x"}


def test_the_subset_warning_reads_the_folders_own_settings_and_calls_no_ungraded_file_stray(tmp_path, caplog):
    ds = dataset(tmp_path / "ds", n=3)
    mine = predictions(tmp_path / "mine", {f"d{i}.json": {"a": "x"} for i in range(3)})
    (mine / "settings.json").write_text(json.dumps({"dataset": data.dataset_id(ds), "manifest": "manifest.parquet"}))
    with caplog.at_level(logging.WARNING):
        benchmark.grade(benchmark.Run("mine", {}, mine), data.read_manifest(ds)[1][:1], QUIET,
                        settings={"dataset": data.dataset_id(ds)}, score_workers=1)
    assert "1 document of its manifest's 3" in caplog.text and "named for no document" not in caplog.text


def test_a_folder_of_your_own_is_not_called_a_run(tmp_path, capsys):
    ds, other = dataset(tmp_path / "ds"), dataset(tmp_path / "other")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}})
    assert oeb(capsys, "grade", str(mine), "--dataset", str(ds), "--score-workers", "1")[0] == 0
    code, _, err = oeb(capsys, "grade", str(mine), "--dataset", str(other), "--score-workers", "1")
    assert code == 1 and "holds a run" not in err and "settings.json" in err


def test_usd_is_null_when_no_call_reported_one(tmp_path):
    ds = dataset(tmp_path / "ds")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}})
    s = benchmark.grade(benchmark.Run("mine", {}, mine), list(data.read_manifest(ds)[1]), QUIET,
                        score_workers=1)
    assert s["usd"] is None and s["credits"] is None


def test_an_unreadable_record_is_warned_about_once(tmp_path, caplog):
    ds = dataset(tmp_path / "ds")
    run = predictions(tmp_path / "run", {"d0.json": {"a": "x"}})
    (run / "records").mkdir()
    (run / "records" / "d1.json").write_text("{not json")
    (run / "settings.json").write_text(json.dumps({"provider": "datalab"}))
    with caplog.at_level(logging.WARNING):
        benchmark.grade(benchmark.Run("datalab", {}, run), list(data.read_manifest(ds)[1]), QUIET,
                        score_workers=1)
        benchmark.recorded(benchmark.Run("datalab", {}, run), "d1")
    assert sum("isn't readable JSON" in r.getMessage() for r in caplog.records) == 1


def test_grade_on_the_parent_of_run_folders_names_them(tmp_path):
    (tmp_path / "runs" / "datalab-1" / "predictions").mkdir(parents=True)
    with pytest.raises(UsageError, match="datalab-1"):
        benchmark.gradable(tmp_path / "runs")
