"""Omni Extract Bench end to end, offline: the extraction task, field matching and the `oeb` command on
omni-bench-core.

A `GoldEcho` adapter answers each document with its own gold, so field matching scores 1.0, and anything
lower is a bug in how oeb reads, predicts or grades, not in a vendor.

    pytest tests/test_flow.py
"""
from __future__ import annotations

import dataclasses
import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import ClassVar

import pytest
from omni_bench_core import adapter, errors

from omni_extract_bench import benchmark, cli, harness
from omni_extract_bench.harness import contract
from omni_extract_bench.scorers import field_matching
from tests.folders import SCHEMA, configured, dataset, oeb, predictions, raised

ROOT = Path(__file__).resolve().parent.parent
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
        if pdf.stem in self.refuse: raise errors.SystemFailed("refused", status=400)
        return adapter.Call(raw={"extraction": json.loads((pdf.parent.parent / "gold" / f"{pdf.stem}.json").read_text())})


@dataclasses.dataclass(frozen=True)
class Broke(GoldEcho):
    """An account that can't pay: every call is a 402."""

    name: ClassVar[str] = "broke"

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        raise errors.SystemFailed("out of credits", status=402)


@dataclasses.dataclass(frozen=True)
class Buggy(GoldEcho):
    """An adapter with a bug in its own code."""

    name: ClassVar[str] = "buggy"

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        return {}["missing"]


@dataclasses.dataclass(frozen=True)
class Offline(GoldEcho):
    """A server of your own that isn't running: the connection is refused, and the adapter reports it as the
    document's failure."""

    name: ClassVar[str] = "offline"

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        raise errors.SystemUnreachable("ConnectError") from ConnectionRefusedError(61, "Connection refused")


READ_FIXED = threading.Event()


@dataclasses.dataclass(frozen=True)
class ReadBug(GoldEcho):
    """`GoldEcho` with a bug in `read` until `READ_FIXED` is set, so a run with the fix is the same command."""

    name: ClassVar[str] = "read-bug"

    def read(self, raw, q):
        if not READ_FIXED.is_set(): return raw["no such field"]
        return super().read(raw, q)


@dataclasses.dataclass(frozen=True)
class CrashingPrediction(field_matching.FieldMatching):
    """Field matching with a bug in `prediction`."""

    def prediction(self, output):
        return output["no such field"]


@dataclasses.dataclass(frozen=True)
class KeyedEcho(GoldEcho):
    """`GoldEcho`, needing a key that no test sets."""

    name: ClassVar[str] = "keyed-echo"
    keys: ClassVar[tuple[str, ...]] = ("OEB_TEST_NEVER_SET_KEY",)


def test_gold_echo_scores_one_and_a_resumed_run_calls_nothing(tmp_path):
    b = benchmark.default(dataset(tmp_path / "ds"))
    g = b.run(GoldEcho(), tmp_path / "runs")
    name, s = g["run"], g["scorers"]["field_matching"]
    assert s["headline"] == 1.0 and s["rows"] == 2 and s["failed"] == 0, s
    assert s["scored"]["headline_is"] == "accuracy" and "partial" not in s and "documents" not in s, s
    folder = tmp_path / "runs" / name
    assert (folder / "field_matching" / "details" / "d0.jsonl").exists()
    rec = json.loads((folder / "records" / "d0.json").read_text())
    assert rec["sent"] == SCHEMA and rec["output"] == {"a": "x"}
    CALLS.clear()
    b.run(GoldEcho(), tmp_path / "runs", resume=name)
    assert CALLS == []


def test_a_refused_document_counts_as_zero(tmp_path):
    g = benchmark.default(dataset(tmp_path / "ds")).run(GoldEcho(refuse=("d1",)), tmp_path / "runs")
    s = g["scorers"]["field_matching"]
    assert s["failed"] == 1 and s["headline"] == 0.5 and s["scored"]["accuracy"] == 1.0, s
    # the breakdown is over the scored documents only; the failed one is counted at the top
    assert s["scored"]["per_suite"]["s"]["accuracy"] == 1.0 and s["scored"]["per_suite"]["s"]["rows"] == 1, s


def test_a_bug_in_read_stops_the_run_and_resuming_reads_the_kept_answer_with_no_call(tmp_path):
    b = benchmark.default(dataset(tmp_path / "ds"))
    READ_FIXED.clear()
    CALLS.clear()
    with pytest.raises(KeyError) as got:
        b.run(ReadBug(), tmp_path / "runs", workers=1)
    assert "ReadBug raised this while reading the answer" in got.value.__notes__[0], got.value.__notes__
    [folder] = (tmp_path / "runs").iterdir()
    kept = {f.stem: json.loads(f.read_text()) for f in (folder / "records").glob("*.json")}
    assert kept and all(r["unread"] is True and r["output"] is None and r["error"] is None and r["raw"]
                        for r in kept.values()), kept
    CALLS.clear()
    READ_FIXED.set()
    try:
        g = b.run(ReadBug(), tmp_path / "runs", workers=1, resume=folder.name)
    finally:
        READ_FIXED.clear()
    assert not set(CALLS) & set(kept), f"a kept answer was paid for again: {CALLS}"
    assert g["run"] == folder.name and g["scorers"]["field_matching"]["headline"] == 1.0, g
    for id in kept:
        rec = json.loads((folder / "records" / f"{id}.json").read_text())
        assert "unread" not in rec and rec["output"] == {"a": "x"}, rec


def test_an_unusable_prediction_counts_zero_with_its_traceback(tmp_path):
    ds = dataset(tmp_path / "ds")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}, "d1.json": ["not", "an", "object"]})
    s = benchmark.default(ds).grade(mine)["scorers"]["field_matching"]
    assert s["failed"] == 1 and s["headline"] == 0.5, s
    lines = {x["id"]: x for x in map(json.loads, (mine / "field_matching" / "scores.jsonl").read_text().splitlines())}
    assert "isn't a JSON object" in json.dumps(lines["d1"]), lines["d1"]
    assert "SystemFailed" in lines["d1"]["traceback"] and "prediction" in lines["d1"]["traceback"], lines["d1"]


def test_an_unexpected_exception_in_prediction_stops_grading(tmp_path):
    from omni_bench_core import benchmark as core_benchmark

    from omni_extract_bench import task

    ds = dataset(tmp_path / "ds")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}, "d1.json": {"a": "x"}})
    b = core_benchmark.Benchmark(task.Extract(), ds, [CrashingPrediction()])
    with pytest.raises(KeyError) as got:
        b.grade(mine)
    assert got.value.__notes__ == ["CrashingPrediction raised this while reading the prediction of d0"], got.value


def test_a_limit_is_its_own_run_graded_over_its_own_rows(tmp_path):
    b = benchmark.default(dataset(tmp_path / "ds"))
    small = b.run(GoldEcho(), tmp_path / "runs", limit=1)
    s = small["scorers"]["field_matching"]
    assert s["rows"] == 1 and s["headline"] == 1.0, s
    assert b.run(GoldEcho(), tmp_path / "runs")["run"] != small["run"]


def test_the_schema_is_sent_without_benchmark_keys(tmp_path):
    annotated = {**SCHEMA, "evaluation_config": "array_llm"}
    b = benchmark.default(dataset(tmp_path / "ds", schema=json.dumps(annotated)))
    name = b.run(GoldEcho(), tmp_path / "runs")["run"]
    rec = json.loads((tmp_path / "runs" / name / "records" / "d0.json").read_text())
    assert "evaluation_config" not in json.dumps(rec["sent"])


def test_a_model_id_is_the_llm_adapter_and_names_its_folder():
    a = harness.system("openai/gpt-5.6-sol", {"attempts": 2})
    assert a.model == "openai/gpt-5.6-sol" and a.label == "openai__gpt-5.6-sol"
    with pytest.raises(ValueError, match="its `model` is its name"):
        harness.system("openai/gpt-5.6-sol", {"model": "other/one"})
    with pytest.raises(ValueError, match="unknown system"):
        harness.system("nope")


def test_every_shipped_vendor_declares_its_key():
    want = {"datalab": ("DATALAB_API_KEY",), "reducto": ("REDUCTO_API_KEY",), "extend": ("EXTEND_API_KEY",),
            "llamaextract": ("LLAMA_CLOUD_API_KEY",), "mistral": ("MISTRAL_API_KEY",),
            "azure-cu": ("AZURE_CU_KEY",)}
    assert {a.name: a.keys for a in harness.SHIPPED} == want
    assert harness.LLM.keys == ("OPENROUTER_API_KEY",)


def test_an_older_key_name_stands_in(monkeypatch):
    for k in ("LLAMA_CLOUD_API_KEY", "LLAMAPARSE_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    llama, llm = harness.system("llamaextract"), harness.system("openai/gpt-5.6-sol")
    assert llama.missing_keys() == ["LLAMA_CLOUD_API_KEY"] and llm.missing_keys() == ["OPENROUTER_API_KEY"]
    monkeypatch.setenv("LLAMAPARSE_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert llama.missing_keys() == [] and llm.missing_keys() == []


def test_oeb_run_grade_and_score(tmp_path, capsys, monkeypatch):
    ds = dataset(tmp_path / "ds")
    monkeypatch.setitem(harness.ADAPTERS, "gold-echo", GoldEcho)
    code, out, err = oeb(capsys, "run", "--config", configured(capsys, ds, "gold-echo"), "--out",
                         str(tmp_path / "runs"), "-y")
    g = json.loads(out)
    assert code == 0 and g["run"].startswith("gold-echo-") and g["scorers"]["field_matching"]["headline"] == 1.0, err

    for name, x in (("p.json", {"a": "x"}), ("g.json", {"a": "x"}), ("s.json", SCHEMA)):
        (tmp_path / name).write_text(json.dumps(x))
    code, out, err = oeb(capsys, "score", "--pred", str(tmp_path / "p.json"), "--gold", str(tmp_path / "g.json"),
                         "--schema", str(tmp_path / "s.json"))
    assert code == 0 and json.loads(out)["accuracy"] == 1.0 and json.loads(out)["gold_rows"] == 0

    assert raised(ValueError, capsys, "init", "nope").startswith("unknown system")


def test_oeb_grade_refuses_an_unfinished_folder_unless_partial(tmp_path, capsys):
    ds = dataset(tmp_path / "ds")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}})
    err = raised(ValueError, capsys, "grade", str(mine), "--dataset", str(ds))
    assert "isn't finished" in err and "d1" in err and "--partial" in err, err
    assert not (mine / "settings.json").exists()
    code, out, err = oeb(capsys, "grade", str(mine), "--dataset", str(ds), "--partial")
    s = json.loads(out)["scorers"]["field_matching"]
    assert code == 0 and s["failed"] == 1 and s["headline"] == 0.5, err
    assert s["partial"] == {"no_answer": 1, "not_scored": 0, "e.g.": ["d1"]}, s


def test_oeb_grade_refuses_a_run_with_a_lost_document_unless_partial(tmp_path, capsys, monkeypatch):
    ds = dataset(tmp_path / "ds")
    monkeypatch.setitem(harness.ADAPTERS, "gold-echo", GoldEcho)
    code, out, err = oeb(capsys, "run", "--config", configured(capsys, ds, "gold-echo"), "--out",
                         str(tmp_path / "runs"), "-y")
    name = json.loads(out)["run"]
    run = tmp_path / "runs" / name
    for sub in ("records", "predictions"): (run / sub / "d1.json").unlink()
    err = raised(ValueError, capsys, "grade", str(run))
    assert "never predicted" in err and "d1" in err, err
    code, out, err = oeb(capsys, "grade", str(run), "--partial")
    assert code == 0 and json.loads(out)["scorers"]["field_matching"]["partial"]["e.g."] == ["d1"], err


def test_oeb_run_raises_an_unfinished_refusal(tmp_path, capsys, monkeypatch):
    """A document the scorer couldn't score leaves the run unfinished: no grading is printed, and the refusal is
    raised, a `ValueError`."""
    def score(self, pred, gold):
        raise errors.Unscored("no judge")

    ds = dataset(tmp_path / "ds")
    monkeypatch.setitem(harness.ADAPTERS, "gold-echo", GoldEcho)
    # Note: one worker, so the patched score runs in this process rather than in a fresh one.
    monkeypatch.setattr(field_matching.FieldMatching, "workers", 1)
    monkeypatch.setattr(field_matching.FieldMatching, "cache", False)
    monkeypatch.setattr(field_matching.FieldMatching, "score", score)
    err = raised(ValueError, capsys, "run", "--config", configured(capsys, ds, "gold-echo"), "--out",
                 str(tmp_path / "runs"), "-y")
    assert "not scored (no judge)" in err, err


def test_a_missing_key_is_refused_before_any_call(tmp_path, capsys, monkeypatch):
    ds = dataset(tmp_path / "ds")
    monkeypatch.delenv("OEB_TEST_NEVER_SET_KEY", raising=False)
    monkeypatch.setitem(harness.ADAPTERS, "keyed-echo", KeyedEcho)
    CALLS.clear()
    err = raised(ValueError, capsys, "run", "--config", configured(capsys, ds, "keyed-echo"), "--out",
                 str(tmp_path / "runs"), "-y")
    assert CALLS == [] and not (tmp_path / "runs").exists(), err
    assert "export OEB_TEST_NEVER_SET_KEY=..." in err, err
    err = raised(ValueError, capsys, "predict", "--system", "keyed-echo", "--doc", str(ds / "pdfs" / "d0.pdf"),
                 "--schema", str(ds / "gold" / "d0.json"))
    assert CALLS == [] and "export OEB_TEST_NEVER_SET_KEY=..." in err, err


def test_oeb_predict_prints_the_record_and_fails_on_a_failed_one(tmp_path, capsys, monkeypatch):
    ds = dataset(tmp_path / "ds")
    (tmp_path / "s.json").write_text(json.dumps({**SCHEMA, "evaluation_config": "x"}))
    monkeypatch.setitem(harness.ADAPTERS, "gold-echo", GoldEcho)
    code, out, err = oeb(capsys, "predict", "--system", "gold-echo", "--doc", str(ds / "pdfs" / "d0.pdf"),
                         "--schema", str(tmp_path / "s.json"))
    rec = json.loads(out)
    assert code == 0 and rec["output"] == {"a": "x"} and rec["sent"] == SCHEMA, err
    for options in ('{"refuse": ["d0"]}', '{"gold-echo": {"refuse": ["d0"]}}'):
        code, out, _ = oeb(capsys, "predict", "--system", "gold-echo", "--doc", str(ds / "pdfs" / "d0.pdf"),
                           "--schema", str(tmp_path / "s.json"), "--options", options)
        rec = json.loads(out)
        assert code == 1 and rec["error"]["status"] == 400 and rec["sent"] == SCHEMA, options
    err = raised(ValueError, capsys, "predict", "--system", "gold-echo", "--doc", str(ds / "pdfs" / "d0.pdf"),
                 "--schema", str(tmp_path / "s.json"), "--options", '{"gold-echo": [{}, {}]}')
    assert "one set of options" in err, err
    err = raised(ValueError, capsys, "predict", "--system", "gold-echo", "--doc", str(tmp_path / "no.pdf"),
                 "--schema", str(tmp_path / "s.json"))
    assert "no such file" in err


def test_oeb_check_counts_a_good_dataset_and_names_a_missing_file(tmp_path, capsys):
    ds = dataset(tmp_path / "ds")
    code, out, err = oeb(capsys, "check", "--dataset", str(ds))
    got = json.loads(out)
    assert code == 0 and got["rows"] == 2 and "suites" not in got, err
    (ds / "pdfs" / "d1.pdf").unlink()
    assert "d1" in raised(ValueError, capsys, "check", "--dataset", str(ds))


def test_oeb_run_only_predicts_and_grades_only_those(tmp_path, capsys, monkeypatch):
    ds = dataset(tmp_path / "ds")
    monkeypatch.setitem(harness.ADAPTERS, "gold-echo", GoldEcho)
    CALLS.clear()
    code, out, err = oeb(capsys, "run", "--config", configured(capsys, ds, "gold-echo"), "--set", "rows.only=[d1]",
                         "--out", str(tmp_path / "runs"), "-y")
    assert code == 0 and CALLS == ["d1"], (err, CALLS)
    assert json.loads(out)["scorers"]["field_matching"]["rows"] == 1


def test_oeb_systems_names_the_key(capsys):
    code, out, _ = oeb(capsys, "systems", "datalab")
    assert code == 0 and "export DATALAB_API_KEY=..." in out and "mode" in out


def test_version_abbreviations_and_no_command(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == f"oeb {cli.__version__}"
    with pytest.raises(SystemExit):
        cli.main(["run", "--config", "c.yaml", "--conf", "c.yaml"])
    assert "unrecognized arguments: --conf" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["run"])
    assert "the following arguments are required: --config" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main([])
    assert "required: command" in capsys.readouterr().err


def test_old_commands_still_work_and_say_what_replaced_them(tmp_path, capsys, monkeypatch):
    """The commands and flags from before 1.0 work as before, each with a deprecation line."""
    ds = dataset(tmp_path / "ds")
    monkeypatch.setitem(harness.ADAPTERS, "gold-echo", GoldEcho)
    for name, v in (("p.json", {"a": "x"}), ("s.json", SCHEMA)):
        (tmp_path / name).write_text(json.dumps(v))
    code, out, err = oeb(capsys, "score", "--pred", str(tmp_path / "p.json"), "--gt", str(tmp_path / "p.json"),
                         "--schema", str(tmp_path / "s.json"))
    assert code == 0 and "--gt is deprecated" in err and json.loads(out)["accuracy"] == 1.0, err
    code, out, err = oeb(capsys, "score", "--pred", str(tmp_path / "p.json"), "--gold", str(tmp_path / "p.json"),
                         "--schema", str(tmp_path / "s.json"), "--verdicts")
    assert code == 0 and "--verdicts is deprecated" in err and "verdicts" in json.loads(out), err
    code, out, err = oeb(capsys, "predict", "--provider", "gold-echo", "--doc", str(ds / "pdfs" / "d0.pdf"),
                         "--schema", str(tmp_path / "s.json"))
    assert code == 0 and "--provider is deprecated" in err, err
    code, out, err = oeb(capsys, "providers")
    assert code == 0 and "`oeb providers` is deprecated" in err and "datalab" in out, err


# Note: run in a fresh interpreter whose imports of the [harness] and hub packages fail, as on a base install.
BASE_INSTALL = """
import importlib.abc, json, sys
BLOCKED = {"httpx", "requests", "openai", "huggingface_hub"}

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED: raise ImportError(f"No module named {name!r}", name=name)

sys.meta_path.insert(0, Block())
from omni_extract_bench import benchmark, cli, task
ds, mine = sys.argv[1:]
assert cli.main(["check", "--dataset", ds]) == 0
assert cli.main(["grade", mine, "--dataset", ds, "--partial"]) == 0
assert not BLOCKED & {m.split(".")[0] for m in sys.modules}, sorted(BLOCKED & set(sys.modules))
"""


def test_the_base_install_grades_without_the_harness(tmp_path):
    ds = dataset(tmp_path / "ds")
    mine = predictions(tmp_path / "mine", {"d0.json": {"a": "x"}})
    got = subprocess.run([sys.executable, "-c", BASE_INSTALL, str(ds), str(mine)], cwd=ROOT, capture_output=True,
                         text=True, check=False)
    assert got.returncode == 0, got.stderr
    assert json.loads(got.stdout[got.stdout.index("{\n  \"run\""):])["scorers"]["field_matching"]["headline"] == 0.5


def test_oeb_run_is_its_config_and_its_folder_keeps_it(tmp_path, capsys, monkeypatch):
    """`--set` changes the config's values; the run folder's config.yaml, given back, asks the same question."""
    ds = dataset(tmp_path / "ds")
    monkeypatch.setitem(harness.ADAPTERS, "gold-echo", GoldEcho)
    c = configured(capsys, ds, "gold-echo")
    code, out, err = oeb(capsys, "run", "--config", c, "--set", "system.gold-echo.refuse=[d1]", "--set", "workers=2",
                         "--out", str(tmp_path / "runs"), "-y")
    g = json.loads(out)
    assert code == 0 and g["scorers"]["field_matching"]["headline"] == 0.5, err
    kept = str(tmp_path / "runs" / g["run"] / "config.yaml")
    assert "workers: 2" in Path(kept).read_text() and "- d1" in Path(kept).read_text()
    code, out, _ = oeb(capsys, "run", "--config", kept, "--out", str(tmp_path / "runs"), "-y", "--resume", g["run"])
    assert code == 0 and json.loads(out)["run"] == g["run"]
    err = raised(ValueError, capsys, "run", "--config", c, "--set", "system.gold-echo.refuze=[d1]", "-y")
    assert "did you mean system.gold-echo.refuse?" in err, err
    err = raised(ValueError, capsys, "run", "--config", c, "--set", "workers=two", "-y")
    assert "workers:" in err and "'two'" in err, err


def test_a_model_ids_config_runs_again_from_its_folder(tmp_path, capsys):
    """An OpenRouter model id's options name its model; a config naming the same model is the same system."""
    ds = dataset(tmp_path / "ds")
    c = configured(capsys, ds, "openai/gpt-5.6-sol")
    assert "model: openai/gpt-5.6-sol" in Path(c).read_text()
    assert harness.system("openai/gpt-5.6-sol", {"model": "openai/gpt-5.6-sol"}) == harness.system("openai/gpt-5.6-sol")
    err = raised(ValueError, capsys, "run", "--config", c, "--set", "system.openai/gpt-5.6-sol.model=x/y", "-y")
    assert "its `model` is its name" in err, err


def test_oeb_predict_raises_a_bug_as_it_is(tmp_path, capsys, monkeypatch):
    ds = dataset(tmp_path / "ds")
    monkeypatch.setitem(harness.ADAPTERS, "buggy", Buggy)
    with pytest.raises(KeyError) as got:
        oeb(capsys, "predict", "--system", "buggy", "--doc", str(ds / "pdfs" / "d0.pdf"),
            "--schema", str(ds / "gold" / "d0.json"))
    assert got.value.__notes__[0].startswith("Buggy raised this while predicting"), got.value.__notes__


def test_a_refused_connection_stops_the_run(tmp_path, capsys, monkeypatch):
    ds = dataset(tmp_path / "ds")
    monkeypatch.setitem(harness.ADAPTERS, "offline", Offline)
    err = raised(errors.SystemUnreachable, capsys, "run", "--config", configured(capsys, ds, "offline"), "--out",
                 str(tmp_path / "runs"), "-y")
    assert "the connection was refused" in err, err
    assert not list((tmp_path / "runs").glob("*/records/*.json"))
    err = raised(errors.SystemUnreachable, capsys, "predict", "--system", "offline", "--doc",
                 str(ds / "pdfs" / "d0.pdf"), "--schema", str(ds / "gold" / "d0.json"))
    assert "start it" in err, err


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize("typed", ["n\n", ""], ids=["no", "ctrl-d"])
def test_no_at_the_prompt_runs_nothing(tmp_path, capsys, monkeypatch, typed):
    ds = dataset(tmp_path / "ds")
    monkeypatch.setitem(harness.ADAPTERS, "gold-echo", GoldEcho)
    monkeypatch.setattr(sys, "stdin", Tty(typed))
    CALLS.clear()
    c = configured(capsys, ds, "gold-echo")
    code, out, err = oeb(capsys, "run", "--config", c, "--out", str(tmp_path / "runs"))
    assert code == 1 and out == "" and CALLS == [] and err.endswith("proceed? [y/N] " + "\n" * (not typed)
                                                                    + "oeb run: not run\n"), err
