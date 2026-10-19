"""Omni Extract Bench end to end, offline: the extraction task, field matching and the `oeb` command on
omni-bench-core.

A `GoldEcho` adapter answers each document with its own gold, so field matching scores 1.0, and anything
lower is a bug in how oeb reads, predicts or grades, not in a vendor.

    pytest tests/test_flow.py
"""
from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from typing import ClassVar

import pytest
from omni_bench_core import adapter, errors

from omni_extract_bench import benchmark, harness
from omni_extract_bench.harness import contract
from tests.folders import SCHEMA, dataset, oeb, predictions

CALLS: list[str] = []
LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True)
class GoldEcho(contract.Extractor):
    """Answers a document at `<dataset>/pdfs/<id>.pdf` with its gold at `<dataset>/gold/<id>.json`, and refuses
    the ids in `refuse` with a 400."""

    name: ClassVar[str] = "gold-echo"
    refuse: tuple[str, ...] = ()

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        with LOCK: CALLS.append(pdf.stem)
        if pdf.stem in self.refuse: raise errors.VendorError("refused", status=400)
        return adapter.Call(raw={"extraction": json.loads((pdf.parent.parent / "gold" / f"{pdf.stem}.json").read_text())})


def test_gold_echo_scores_one_and_a_resumed_run_calls_nothing(tmp_path):
    b = benchmark.default(dataset(tmp_path / "ds"))
    [(name, g)] = b.run([GoldEcho()], tmp_path / "runs").items()
    s = g["scorers"]["field_matching"]
    assert s["headline"] == 1.0 and s["graded"] == 2 and s["failed"] == 0, s
    folder = tmp_path / "runs" / name
    assert (folder / "field_matching" / "details" / "d0.jsonl").exists()
    rec = json.loads((folder / "records" / "d0.json").read_text())
    assert rec["sent"] == SCHEMA and rec["output"] == {"a": "x"}
    CALLS.clear()
    b.run([GoldEcho()], tmp_path / "runs")
    assert CALLS == []


def test_a_refused_document_counts_as_zero(tmp_path):
    [g] = benchmark.default(dataset(tmp_path / "ds")).run([GoldEcho(refuse=("d1",))], tmp_path / "runs").values()
    s = g["scorers"]["field_matching"]
    assert s["failed"] == 1 and s["headline"] == 0.5 and s["accuracy"] == 1.0, s


def test_a_sample_leaves_the_rest_pending(tmp_path):
    [g] = benchmark.default(dataset(tmp_path / "ds")).run([GoldEcho()], tmp_path / "runs", limit=1).values()
    s = g["scorers"]["field_matching"]
    assert s["pending"] == 1 and s["headline"] is None, s


def test_the_schema_is_sent_without_benchmark_keys(tmp_path):
    annotated = {**SCHEMA, "evaluation_config": "array_llm"}
    b = benchmark.default(dataset(tmp_path / "ds", schema=json.dumps(annotated)))
    [name] = b.run([GoldEcho()], tmp_path / "runs")
    rec = json.loads((tmp_path / "runs" / name / "records" / "d0.json").read_text())
    assert "evaluation_config" not in json.dumps(rec["sent"])


def test_a_model_id_is_the_llm_adapter_and_names_its_folder():
    a = harness.system("openai/gpt-5.6-sol", {"attempts": 2})
    assert a.model == "openai/gpt-5.6-sol" and a.folder_name.startswith("openai__gpt-5.6-sol-")
    with pytest.raises(errors.UsageError, match="provider name"):
        harness.system("openai/gpt-5.6-sol", {"model": "other/one"})
    with pytest.raises(errors.UsageError, match="unknown provider"):
        harness.system("nope")


def test_oeb_run_grade_and_score(tmp_path, capsys, monkeypatch):
    ds = dataset(tmp_path / "ds")
    monkeypatch.setitem(harness.ADAPTERS, "gold-echo", GoldEcho)
    code, out, err = oeb(capsys, "run", "--providers", "gold-echo", "--dataset", str(ds), "--out",
                         str(tmp_path / "runs"), "-y")
    assert code == 0, err
    [g] = json.loads(out).values()
    assert g["scorers"]["field_matching"]["headline"] == 1.0

    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}})
    code, out, err = oeb(capsys, "grade", str(mine), "--dataset", str(ds))
    assert code == 0, err
    s = json.loads(out)["scorers"]["field_matching"]
    assert s["failed"] == 1 and s["headline"] == 0.5, s

    for name, x in (("p.json", {"a": "x"}), ("g.json", {"a": "x"}), ("s.json", SCHEMA)):
        (tmp_path / name).write_text(json.dumps(x))
    code, out, err = oeb(capsys, "score", "--pred", str(tmp_path / "p.json"), "--gold", str(tmp_path / "g.json"),
                         "--schema", str(tmp_path / "s.json"))
    assert code == 0 and json.loads(out)["accuracy"] == 1.0

    code, _, err = oeb(capsys, "run", "--providers", "nope", "--dataset", str(ds), "-y")
    assert code == 1 and err.startswith("oeb run: unknown provider")


def test_oeb_predict_prints_the_record_and_fails_on_a_failed_one(tmp_path, capsys, monkeypatch):
    ds = dataset(tmp_path / "ds")
    (tmp_path / "s.json").write_text(json.dumps({**SCHEMA, "evaluation_config": "x"}))
    monkeypatch.setitem(harness.ADAPTERS, "gold-echo", GoldEcho)
    code, out, err = oeb(capsys, "predict", "--provider", "gold-echo", "--doc", str(ds / "pdfs" / "d0.pdf"),
                         "--schema", str(tmp_path / "s.json"))
    rec = json.loads(out)
    assert code == 0 and rec["output"] == {"a": "x"} and rec["sent"] == SCHEMA, err
    code, out, _ = oeb(capsys, "predict", "--provider", "gold-echo", "--doc", str(ds / "pdfs" / "d0.pdf"),
                       "--schema", str(tmp_path / "s.json"), "--options", '{"refuse": ["d0"]}')
    rec = json.loads(out)
    assert code == 1 and rec["error"]["status"] == 400 and rec["sent"] == SCHEMA
    code, _, err = oeb(capsys, "predict", "--provider", "gold-echo", "--doc", str(tmp_path / "no.pdf"),
                       "--schema", str(tmp_path / "s.json"))
    assert code == 1 and "no such file" in err


def test_oeb_check_counts_a_good_dataset_and_names_a_missing_file(tmp_path, capsys):
    ds = dataset(tmp_path / "ds")
    code, out, err = oeb(capsys, "check", "--dataset", str(ds))
    got = json.loads(out)
    assert code == 0 and (got["rows"], got["suites"]) == (2, 1), err
    (ds / "pdfs" / "d1.pdf").unlink()
    code, _, err = oeb(capsys, "check", "--dataset", str(ds))
    assert code == 1 and "d1" in err


def test_oeb_run_docs_predicts_only_those(tmp_path, capsys, monkeypatch):
    ds = dataset(tmp_path / "ds")
    (tmp_path / "ids.txt").write_text("# just one\nd1\n")
    monkeypatch.setitem(harness.ADAPTERS, "gold-echo", GoldEcho)
    CALLS.clear()
    code, out, err = oeb(capsys, "run", "--providers", "gold-echo", "--dataset", str(ds), "--out",
                         str(tmp_path / "runs"), "--docs", str(tmp_path / "ids.txt"), "-y")
    assert code == 0 and CALLS == ["d1"], (err, CALLS)
    [g] = json.loads(out).values()
    assert g["scorers"]["field_matching"]["pending"] == 1



def test_old_commands_still_work_and_say_what_replaced_them(tmp_path, capsys, monkeypatch):
    """`oeb benchmark`, `--score-only`, `--score-workers` and `--gt` work as before, with a deprecation line."""
    ds = dataset(tmp_path / "ds")
    monkeypatch.setitem(harness.ADAPTERS, "gold-echo", GoldEcho)
    code, _, err = oeb(capsys, "benchmark", "--providers", "gold-echo", "--dataset", str(ds), "--out",
                       str(tmp_path / "runs"), "-y")
    assert code == 0 and "`oeb benchmark` is deprecated" in err, err
    code, out, err = oeb(capsys, "run", "--providers", "gold-echo", "--dataset", str(ds), "--out",
                         str(tmp_path / "runs"), "--score-only", "--score-workers", "2")
    assert code == 0 and "--score-only is deprecated" in err and "--score-workers is deprecated" in err, err
    [g] = json.loads(out).values()
    assert g["scorers"]["field_matching"]["headline"] == 1.0
    for name, v in (("p.json", {"a": "x"}), ("s.json", SCHEMA)):
        (tmp_path / name).write_text(json.dumps(v))
    code, out, err = oeb(capsys, "score", "--pred", str(tmp_path / "p.json"), "--gt", str(tmp_path / "p.json"),
                         "--schema", str(tmp_path / "s.json"))
    assert code == 0 and "--gt is deprecated" in err and json.loads(out)["accuracy"] == 1.0, err
