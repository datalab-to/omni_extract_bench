"""The `oeb` command: what it prints, and a mistake as one line naming the value and the fix.

    pytest tests/test_cli.py
"""
from __future__ import annotations

import json

import pytest

from omni_extract_bench import benchmark, cli
from omni_extract_bench.errors import UsageError
from omni_extract_bench.scorers import field_matching
from tests.folders import SCHEMA, dataset, oeb, predictions


def test_options_name_what_was_wrong(tmp_path, capsys):
    ds = str(dataset(tmp_path / "ds"))
    code, _, err = oeb(capsys, "benchmark", "--providers", "datalab", "--dataset", ds, "--options", "opts.jsn", "-y")
    assert code == 1 and "'opts.jsn'" in err and "nor a file that exists" in err
    code, _, err = oeb(capsys, "benchmark", "--providers", "datalab", "--dataset", ds, "--options", "[1]", "-y")
    assert code == 1 and "must be a JSON object" in err


def test_docs_naming_a_missing_file_says_so(tmp_path, capsys):
    ds = str(dataset(tmp_path / "ds"))
    code, _, err = oeb(capsys, "benchmark", "--providers", "datalab", "--dataset", ds, "--docs", "nope.txt", "-y")
    assert code == 1 and err.startswith("oeb benchmark: --docs nope.txt: no such file")


def test_a_docs_file_skips_blank_lines_and_comments(tmp_path):
    (f := tmp_path / "ids.txt").write_text("# the hard ones\nd0  # a note\n\nd1\n")
    assert cli.read_ids(f, "--docs") == frozenset({"d0", "d1"})


def test_an_unknown_provider_or_bad_workers_is_one_line(tmp_path, capsys):
    ds = str(dataset(tmp_path / "ds"))
    code, _, err = oeb(capsys, "benchmark", "--providers", "nope", "--dataset", ds, "-y")
    assert code == 1 and err.startswith("oeb benchmark: unknown provider 'nope'") and "Traceback" not in err
    code, _, err = oeb(capsys, "benchmark", "--providers", "datalab", "--dataset", ds, "--workers", "abc", "-y")
    assert code == 1 and err.startswith("oeb benchmark: --workers") and "'abc'" in err


@pytest.mark.parametrize("raised, code, said", [
    (KeyboardInterrupt(), 130, "oeb benchmark: interrupted; rerun to resume"),
    ("account", 1, "oeb benchmark: AccountFailure: out of credits; fix it and rerun to resume"),
])
def test_an_account_failure_or_ctrl_c_is_one_line(tmp_path, capsys, monkeypatch, raised, code, said):
    from omni_extract_bench.harness import errors

    exc = errors.AccountFailure("out of credits") if raised == "account" else raised

    def boom(p, say):
        raise exc
    monkeypatch.setattr(benchmark, "execute", boom)
    got = oeb(capsys, "benchmark", "--providers", "datalab", "--dataset", str(dataset(tmp_path / "ds")),
              "--out", str(tmp_path / "runs"), "-y")
    assert got[0] == code and said in got[2] and "Traceback" not in got[2]


def test_an_interrupt_says_resume_only_for_benchmark(monkeypatch, capsys):
    def interrupted(args): raise KeyboardInterrupt
    monkeypatch.setattr(cli, "cmd_check", interrupted)
    assert cli.main(["check"]) == 130
    assert capsys.readouterr().err == "oeb check: interrupted\n"


def test_score_names_the_flag_and_says_null(tmp_path, capsys):
    (g := tmp_path / "g.json").write_text(json.dumps({"a": "x"}))
    (s := tmp_path / "s.json").write_text(json.dumps(SCHEMA))
    (n := tmp_path / "n.json").write_text("null")
    (l := tmp_path / "l.json").write_text("[1]")
    for pred, flag, said in ((n, "--pred", "a JSON null"), (l, "--pred", "a JSON list")):
        code, _, err = oeb(capsys, "score", "--pred", str(pred), "--gold", str(g), "--schema", str(s))
        assert code == 1 and flag in err and said in err
    code, _, err = oeb(capsys, "score", "--pred", str(g), "--gold", str(tmp_path / "x.json"), "--schema", str(s))
    assert code == 1 and "--gold" in err and "no such file" in err
    with pytest.raises(TypeError, match="prediction must be a JSON object"):
        field_matching.score([1], {"a": "x"}, SCHEMA)


def test_a_bare_order_matters_is_every_array(tmp_path, capsys):
    schema = {"type": "object", "properties": {"xs": {"type": "array", "items": {"type": "string"}}}}
    (g := tmp_path / "g.json").write_text(json.dumps({"xs": ["a", "b", "c"]}))
    (p := tmp_path / "p.json").write_text(json.dumps({"xs": ["c", "b", "a"]}))
    (s := tmp_path / "s.json").write_text(json.dumps(schema))
    loose = json.loads(oeb(capsys, "score", "--pred", str(p), "--gold", str(g), "--schema", str(s))[1])
    strict = json.loads(oeb(capsys, "score", "--pred", str(p), "--gold", str(g), "--schema", str(s),
                            "--order-matters")[1])
    assert loose["accuracy"] == 1.0 and strict["accuracy"] < 1.0


def test_a_model_ids_options_hint_never_suggests_model(capsys):
    code, out, _ = oeb(capsys, "providers", "openai/gpt-5.6-sol")
    assert code == 0 and '"model"' not in out.split("oeb benchmark")[-1]


def test_grade_prints_the_summary_and_a_closing_line(tmp_path, capsys):
    ds = dataset(tmp_path / "ds")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}})
    code, out, err = oeb(capsys, "grade", str(mine), "--dataset", str(ds), "--score-workers", "1")
    assert code == 0 and "headline 1.000" in err and json.loads(out)["headline"] == 1.0
    with pytest.raises(SystemExit):
        oeb(capsys, "grade", str(mine), "-q")


def test_check_describes_a_dataset_that_can_run(tmp_path, capsys):
    code, out, _ = oeb(capsys, "check", "--dataset", str(dataset(tmp_path / "ds")))
    assert code == 0 and json.loads(out) == {"ok": True, "manifest": "manifest.parquet", "documents": 2,
                                             "suites": 1, "inputs": 2}


def test_library_misuse_says_how_to_call_it(tmp_path):
    from omni_extract_bench import data

    doc = data.read_manifest(dataset(tmp_path / "ds"))[1][0]
    with pytest.raises(TypeError, match=r"pass \[doc\]"):
        field_matching.check(doc)
    with pytest.raises(ValueError, match="summarize takes score lines"):
        field_matching.summarize([field_matching.score({"a": "x"}, {"a": "x"}, SCHEMA)])
    with pytest.raises(UsageError, match="no folder at"):
        benchmark.read_scores(benchmark.Run("mine", {}, tmp_path / "nope"))


def test_a_flag_given_a_folder_is_one_line(tmp_path, capsys):
    (tmp_path / "s.json").write_text(json.dumps(SCHEMA))
    code, _, err = oeb(capsys, "score", "--pred", str(tmp_path), "--gold", str(tmp_path / "s.json"),
                       "--schema", str(tmp_path / "s.json"))
    assert code == 1 and "--pred" in err and "folder" in err


def test_an_unknown_order_matters_name_is_one_line(tmp_path, capsys):
    pred = tmp_path / "p.json"
    pred.write_text(json.dumps({"a": "x"}))
    (tmp_path / "s.json").write_text(json.dumps(SCHEMA))
    code, _, err = oeb(capsys, "score", "--pred", str(pred), "--gold", str(pred), "--schema",
                       str(tmp_path / "s.json"), "--order-matters", "lineitems")
    assert code == 1 and "lineitems" in err


def test_a_stop_still_prints_the_summaries_it_wrote(tmp_path, capsys, monkeypatch):
    from omni_extract_bench.harness.errors import AccountFailure

    def execute(p, say):
        for r in p.runs:
            r.out.mkdir(parents=True, exist_ok=True)
            (r.out / "summary.json").write_text(json.dumps({"headline": None, "pending": 2}))
        raise AccountFailure("out of credits")
    monkeypatch.setattr(benchmark, "execute", execute)
    code, out, err = oeb(capsys, "benchmark", "--providers", "datalab", "--dataset", str(dataset(tmp_path / "ds")),
                         "--out", str(tmp_path / "runs"), "-y")
    (s,) = json.loads(out).values()
    assert code == 1 and s["pending"] == 2 and "AccountFailure" in err and "resume" in err
