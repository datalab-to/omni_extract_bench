"""`benchmark.plan`, `needs` and the harness's registry: what a run will do, checked before any call.

    pytest tests/test_plan.py
"""
from __future__ import annotations

import json
import types

import pytest

from omni_extract_bench import benchmark, data
from omni_extract_bench.errors import Stop, UsageError
from tests.folders import SCHEMA, dataset


def test_a_users_mistake_is_a_usage_error_and_a_stopped_run_a_stop():
    from omni_extract_bench.harness import errors

    assert issubclass(data.ManifestError, UsageError) and issubclass(UsageError, ValueError)
    for e in (errors.AccountFailure, errors.MissingCredential, errors.MissingDependency):
        assert issubclass(e, Stop) and not issubclass(e, UsageError)
    assert issubclass(errors.MissingDependency, ImportError)


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


@pytest.mark.parametrize("kw, err, value", [
    ({"timeout": 0}, UsageError, "--timeout is 0"), ({"timeout": -1}, UsageError, "--timeout is -1"),
    ({"workers": "3"}, TypeError, "'3'"), ({"workers": -3}, UsageError, "--workers is -3"),
])
def test_plan_refuses_what_it_would_misread(tmp_path, kw, err, value):
    with pytest.raises(err, match=value):
        benchmark.plan(["datalab"], out=tmp_path / "runs", dataset=str(dataset(tmp_path / "ds")), **kw)


def test_plan_refuses_a_bare_str_of_providers(tmp_path):
    with pytest.raises(TypeError, match="'datalab'"):
        benchmark.plan("datalab", dataset=str(dataset(tmp_path / "ds")))


def test_no_documents_selected_names_the_flags_given(tmp_path):
    with pytest.raises(UsageError, match="--docs chose none of the manifest's 2 documents"):
        benchmark.plan(["datalab"], out=tmp_path / "runs", dataset=str(dataset(tmp_path / "ds")), only=frozenset())


def test_score_only_on_a_folder_never_predicted_is_refused(tmp_path):
    with pytest.raises(UsageError, match="never predicted"):
        benchmark.plan(["datalab"], out=tmp_path / "runs", dataset=str(dataset(tmp_path / "ds")), score_only=True)


def test_an_adapter_bug_stops_its_runs_naming_the_document(tmp_path, monkeypatch):
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


def test_an_adapter_returning_the_wrong_type_says_so(tmp_path, monkeypatch):
    from omni_extract_bench.harness import document

    real = document.adapter
    monkeypatch.setattr(document, "adapter", lambda provider: types.SimpleNamespace(
        __name__=real(provider).__name__, Config=real(provider).Config,
        prepare_schema=real(provider).prepare_schema, extract=lambda pdf, schema, **kw: {"a": "x"}))
    (pdf := tmp_path / "d.pdf").write_bytes(b"%PDF")
    with pytest.raises(TypeError, match="extract must return an Extraction"):
        document.predict("datalab", pdf, SCHEMA)


def test_an_added_adapter_is_keyed_by_the_name_it_was_filed_under(monkeypatch):
    from omni_extract_bench.harness import registry

    monkeypatch.setitem(registry.ADAPTERS, "mine", types.ModuleType("plug.adapter"))
    assert registry.resolve("mine") == "mine" and registry.resolve("datalab") == "datalab"
    assert registry.resolve("openai/gpt-5") == "llm_single_shot"
    assert "mine" in registry.adapter_names() and benchmark._per_adapter({"mine": 2}) == {"mine": 2}
    with pytest.raises(UsageError, match="'plug.adapter'"):
        benchmark._per_adapter({"plug.adapter": 2})


def test_an_option_given_as_another_type_is_one_run_or_refused():
    from omni_extract_bench.harness import registry

    assert registry.out_name("datalab", {"poll_interval": 5}) == registry.out_name("datalab")
    with pytest.raises(UsageError, match="poll_interval is 'x'"):
        registry.config_for("datalab", {"poll_interval": "x"})


def test_a_folders_unreadable_settings_json_is_named(tmp_path):
    (tmp_path / "settings.json").write_text("{nope")
    with pytest.raises(UsageError, match="settings.json isn't readable JSON"):
        benchmark.settled(tmp_path)
    assert benchmark.settled(tmp_path / "nothing") == {}


def test_the_package_exports_the_scorer_and_the_dataset():
    import omni_extract_bench as oeb

    assert {"data", "scorers", "gold", "score", "summarize", "check", "Verdict"} <= set(oeb.__all__)
    assert not hasattr(oeb, "__version__"), "the version is the distribution's: importlib.metadata"
