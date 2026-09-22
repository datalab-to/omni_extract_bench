#!/usr/bin/env python3
"""A RUN IS A PROVIDER PLUS ITS OPTIONS, and each one is stored on its own.

`--options` changes what is measured. When it did not change WHERE the answer was stored,
`needs_run` found the other configuration's records and skipped every document -- so a run
that asked for `mode=fast` reported `balanced` results and never called the vendor at all.
A silently wrong number, arrived at by a resume doing its job.

Run: python3 tests/test_run_layout.py
"""
import dataclasses
import io
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import types

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import omni_extract_bench.benchmark as B                                       # noqa: E402
import omni_extract_bench.harness.document as D
import omni_extract_bench.harness.registry as V                                  # noqa: E402

_REAL_ADAPTER = V.adapter  # the real lookup, for Config and __name__


def as_adapter(lookup):
    """A stub `provider -> extract` as an adapter: the three names `Adapter` asks for.

    `Config` and `prepare_schema` come from the real adapter, so a test that means to fake
    only the vendor call is not also faking the declaration or the schema it is sent.
    """
    def wrapped(provider):
        real = _REAL_ADAPTER(provider)
        # __name__ included: an adapter is a MODULE, and `resolve` reads it to file the run
        # under the right vendor. A double that omits it is not standing in for an adapter.
        return types.SimpleNamespace(__name__=real.__name__, Config=real.Config,
                                     prepare_schema=real.prepare_schema,
                                     extract=lookup(provider))
    return wrapped


from omni_extract_bench.harness.contract import Cost, Extraction
from omni_extract_bench.harness.errors import AccountFailure
from omni_extract_bench.harness.registry import out_name  # noqa: E402

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
FAILS = []


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


def named(provider, **opts):
    return out_name(provider, opts)


def _refused(call):
    try:
        call()
        return False
    except ValueError:
        return True


print("\nA NAME IS A FUNCTION OF THE SETTINGS")
report("a run is named for the vendor and a digest of what it was sent",
       re.fullmatch(r"datalab-[0-9a-f]{8}", named("datalab")) is not None, named("datalab"))
report("an option set to the stock value names the same run",
       named("datalab", mode="balanced") == named("datalab"))
report("a model id still gets one directory, not a nested pair",
       named("openai/gpt-5.6-sol").startswith("openai__gpt-5.6-sol-"),
       named("openai/gpt-5.6-sol"))
report("a steered run is a different directory, not a suffix on the stock one",
       named("datalab", mode="accurate") != named("datalab"),
       named("datalab", mode="accurate"))

print("\nDISTINCT STEERINGS GET DISTINCT DIRECTORIES")
apart = [named("datalab", mode="accurate"), named("datalab", mode="fast"),
         named("datalab", mode="fast", poll_interval=2.5),
         named("reducto", system_prompt="a b"),
         named("reducto", system_prompt="a_b"),
         named("reducto", system_prompt="L" * 300),
         named("reducto", system_prompt="M" * 300)]
report("every one of them is unique", len(set(apart)) == len(apart), str(apart))

print("\nAND NOT ON WHAT THE DEFAULT HAPPENED TO BE THAT DAY")
def _datalab_defaulting_to(tier):
    module = types.ModuleType("fake_datalab")

    @dataclasses.dataclass(frozen=True)
    class Config:
        mode: str = tier
        base_url: str = "https://www.datalab.to"
        poll_interval: float = 5.0

    module.Config = Config
    module.prepare_schema = lambda schema: schema
    module.extract = lambda *a, **k: None
    return lambda provider: module


_real_adapter = V.adapter
try:
    V.adapter = _datalab_defaulting_to("balanced")
    was_stock, was_pinned = named("datalab"), named("datalab", mode="accurate")
    V.adapter = _datalab_defaulting_to("accurate")
    now_stock, now_pinned = named("datalab"), named("datalab", mode="balanced")
finally:
    V.adapter = _real_adapter
report("the same name never means two different settings", was_stock != now_stock,
       f"{was_stock} vs {now_stock}")
report("...and the same settings always mean the same name", was_stock == now_pinned,
       f"{was_stock} vs {now_pinned}")
report("...however the defaults moved around them", was_pinned == now_stock,
       f"{was_pinned} vs {now_stock}")

print("\nAND THE NAME DOES NOT DEPEND ON HOW IT WAS WRITTEN")
report("option order does not change it",
       named("datalab", mode="fast", poll_interval=2.5)
       == named("datalab", poll_interval=2.5, mode="fast"))
seeds = {subprocess.run(
    [sys.executable, "-c",
     "from omni_extract_bench.harness.registry import out_name;"
     "print(out_name('datalab', {'mode': 'fast'}))"],
    capture_output=True, text=True,
    env={"PYTHONHASHSEED": str(n), "PATH": _os.environ.get("PATH", ""),
         "PYTHONPATH": str(ROOT)}).stdout.strip() for n in (0, 1, 2)}
report("and neither does the interpreter's hash seed", len(seeds) == 1, str(seeds))

print("\nAND IT STAYS A DIRECTORY NAME")
report("a huge value cannot become a huge path",
       len(named("reducto", system_prompt="x" * 500)) < 80,
       named("reducto", system_prompt="x" * 500))
report("nothing unsafe reaches the filesystem",
       not (set(named("reducto", system_prompt="a/b c:d")) & set("/\\:*?\"<>|")),
       named("reducto", system_prompt="a/b c:d"))
report("...including the model id, suffix and all",
       not (set(named("mistralai/mistral-medium-3-5:batch")) & set("/\\:*?\"<>|")),
       named("mistralai/mistral-medium-3-5:batch"))

print("\nA STEERED RUN DOES NOT INHERIT THE STOCK RUN'S RECORDS")
out = pathlib.Path(tempfile.mkdtemp())
docs = []
for i in range(3):
    g = out / f"g{i}.json"; g.write_text(json.dumps({"a": "x"}))
    pdf = out / f"p{i}.pdf"; pdf.write_bytes(b"%PDF")
    docs.append(B.Doc(f"d{i}", "s", pdf, g,
                      {"type": "object", "properties": {"a": {"type": "string"}}}))
B.fetch = lambda *_: out
B.read_manifest = lambda *a, **k: docs
called = []
D.adapter = as_adapter(lambda prov: lambda pdf, schema, *, timeout, config: (
    called.append(config.mode),
    Extraction(result={"a": config.mode}, cost=Cost(usd=0.1)))[1])

B.run(["datalab"], out=out, score_workers=1, predict_workers={"*": 1})
report("the stock run calls the vendor at its stock setting", called == ["balanced"] * 3,
       str(called))

called.clear()
B.run(["datalab"], out=out, score_workers=1, predict_workers={"*": 1},
      options={"datalab": {"mode": "fast"}})
report("the steered run calls the vendor again, at the asked-for setting",
       called == ["fast"] * 3, f"{called} -- empty means it reused the stock records")
steered_dir = named("datalab", mode="fast")
report("...into a directory of its own",
       (out / steered_dir).is_dir() and (out / named("datalab")).is_dir(), steered_dir)

asked = json.loads((out / steered_dir / "settings.json").read_text())
report("the run says what it was asked, in the directory, from the start",
       (asked["provider"], asked["settings"]["mode"]) == ("datalab", "fast"), str(asked))

stock = json.loads((out / named("datalab") / "predictions" / "d0.json").read_text())
steered = json.loads((out / steered_dir / "predictions" / "d0.json").read_text())
report("the two answers are kept apart", (stock["a"], steered["a"]) == ("balanced", "fast"),
       f"{stock['a']} / {steered['a']}")
rec = json.loads((out / steered_dir / "records" / "d0.json").read_text())
report("...and the record states what it was sent",
       rec["run_manifest"]["settings"]["mode"] == "fast")

print("\nA VENDOR'S OWN TIERS, COMPARED IN ONE INVOCATION")
out2 = pathlib.Path(tempfile.mkdtemp())
B.fetch = lambda *_: out2
summary = B.run(["datalab"], out=out2, score_workers=1, predict_workers={"*": 1},
                options={"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]})
report("each configuration is its own row in the summary",
       sorted(summary) == sorted([named("datalab", mode="balanced"),
                                  named("datalab", mode="accurate")]),
       str(sorted(summary)))
report("...matching its directory exactly",
       sorted(summary) == sorted(d.name for d in out2.iterdir() if d.is_dir()),
       str(sorted(d.name for d in out2.iterdir() if d.is_dir())))
steered_row = summary[named("datalab", mode="accurate")]
report("the row names the vendor it came from", steered_row["provider"] == "datalab")
report("...and what it was sent", steered_row["settings"]["mode"] == "accurate")
report("the stock row carries its own settings",
       summary[named("datalab")]["settings"]["mode"] == "balanced")
report("both ran the same vendor",
       {r["provider"] for r in summary.values()} == {"datalab"})

for one in summary.values():
    on_disk = json.loads((out2 / one["run"] / "summary.json").read_text())
    report(f"{one['run']}: its directory carries its own row", on_disk == one)
report("and there is no table across them to go stale",
       not (out2 / "summary.json").exists())

B.read_manifest = lambda *a, **k: docs[:1]
narrow = B.run(["datalab"], out=out2, score_workers=1, predict_workers={"*": 1},
               options={"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]})
B.read_manifest = lambda *a, **k: docs
kept = json.loads((out2 / named("datalab") / "summary.json").read_text())
report("a narrowed resume leaves the directory's own summary whole",
       kept["documents"] == len(docs), f'{kept["documents"]} of {len(docs)}')
report("...while the return value answers what this invocation ran",
       narrow[named("datalab")]["documents"] == 1,
       str(narrow[named("datalab")]["documents"]))

print("\nAND A RUN THAT IS GRADED SURVIVES THE ONE AFTER IT NOT BEING")
out3 = pathlib.Path(tempfile.mkdtemp())
B.fetch = lambda *_: out3
real_score, graded = B.Run.score, {"n": 0}


def interrupted_on_the_second(self, *a, **k):
    graded["n"] += 1
    if graded["n"] == 2:
        raise KeyboardInterrupt
    return real_score(self, *a, **k)


B.Run.score = interrupted_on_the_second
try:
    B.run(["datalab"], out=out3, score_workers=1, predict_workers={"*": 1},
          options={"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]})
    report("the interrupt reaches the caller", False, "it was swallowed")
except KeyboardInterrupt:
    report("the interrupt reaches the caller", True)
finally:
    B.Run.score = real_score

first, second = named("datalab", mode="balanced"), named("datalab", mode="accurate")
report("the run that finished grading published its row",
       (out3 / first / "summary.json").exists())
report("the one that did not, did not", not (out3 / second / "summary.json").exists())
report("...so aggregating the directories finds one finished run, not a half-counted second",
       sorted(p.parent.name for p in out3.glob("*/summary.json")) == [first])

print("\nA RUN IS A PROVIDER PLUS ITS OPTIONS, AND THE CAP IS STILL THE VENDOR'S")
for bad in ({"datalb": {"mode": "accurate"}},
            {"reducto": {"agentic_table_mode": "default"}}):
    try:
        B.plan(["datalab"], bad)
        report(f"--options {list(bad)[0]!r} is refused", False, "it was accepted")
    except ValueError as exc:
        report(f"--options {list(bad)[0]!r} is refused", "silently ignored" in str(exc))
try:
    B.plan(["datalab"], {"datalab": []})
    report("an empty option list is refused", False, "it was accepted")
except ValueError as exc:
    report("an empty option list is refused", "would not run at all" in str(exc))

import threading as _threading                                                 # noqa: E402
import time as _time                                                           # noqa: E402
from omni_extract_bench.benchmark import WORKERS                               # noqa: E402

wide = pathlib.Path(tempfile.mkdtemp())
many = []
for i in range(20):
    g = wide / f"g{i}.json"; g.write_text(json.dumps({"a": "x"}))
    pdf = wide / f"p{i}.pdf"; pdf.write_bytes(b"%PDF")
    many.append(B.Doc(f"d{i}", "s", pdf, g,
                      {"type": "object", "properties": {"a": {"type": "string"}}}))
B.fetch = lambda *_: wide
B.read_manifest = lambda *a, **k: many
live = {"n": 0, "peak": 0}
guard = _threading.Lock()


def counting(prov):
    def extract(pdf, schema, *, timeout, config):
        with guard:
            live["n"] += 1
            live["peak"] = max(live["peak"], live["n"])
        _time.sleep(0.02)
        with guard:
            live["n"] -= 1
        return Extraction(result={"a": "x"}, cost=Cost(usd=0.1))
    return extract


D.adapter = as_adapter(counting)
B.run(["datalab"], out=wide, score_workers=1,
      options={"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]})
report("two runs of one vendor share its concurrency cap, not double it",
       live["peak"] <= WORKERS["datalab"],
       f'peak {live["peak"]} in flight against a cap of {WORKERS["datalab"]}')
report("...and the whole cap is in use, not half of it each",
       live["peak"] > WORKERS["datalab"] // 2,
       f'peak {live["peak"]}, which a cap split two ways could not exceed '
       f'{WORKERS["datalab"] // 2}')

print("\nAND SAYS WHAT A RESUME WOULD COST, BEFORE IT COSTS IT")
from omni_extract_bench.benchmark import BenchmarkRun                          # noqa: E402
from rich.console import Console                                               # noqa: E402

seen_dir = pathlib.Path(tempfile.mkdtemp())
plan = BenchmarkRun(["datalab"], out=seen_dir,
                    options={"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]})
report("the plan is available before anything is fetched or called",
       len(plan.runs) == 2 and len(plan.providers) == 1
       and plan.providers[0].adapter == "datalab", repr(plan))
report("...and two model ids land under one adapter, on one budget",
       len(BenchmarkRun(["openai/gpt-5.6-sol", "anthropic/claude-opus-5"]).providers) == 1)
report("a bad option is refused by the constructor, before any download",
       _refused(lambda: BenchmarkRun(["datalab"], options={"datalb": {"mode": "fast"}})))

bal, acc = plan.runs
bal.records.mkdir(parents=True, exist_ok=True)
for d in docs[:2]:
    B.write_json_atomic(bal.records / f"{d.doc_id}.json", {"done": True})
(bal.out).mkdir(parents=True, exist_ok=True)
(bal.out / "scores.jsonl").write_text(
    "".join(json.dumps({"doc_id": d.doc_id, "suite": "s", "status": "scored"}) + "\n"
            for d in docs))
# A graded document is a row AND its verdicts, because that is what grading writes by default.
(bal.out / "verdicts").mkdir(parents=True, exist_ok=True)
for d in docs:
    (bal.out / "verdicts" / f"{d.doc_id}.jsonl").write_text("")
report("a run counts what it would still predict, and still grade",
       bal.outstanding(docs) == (len(docs) - 2, 0), str(bal.outstanding(docs)))
report("...a run with nothing on disk owes everything",
       acc.outstanding(docs) == (len(docs), len(docs)), str(acc.outstanding(docs)))
report("--rescore owes every grade again, however many are on disk",
       bal.outstanding(docs, rescore=True) == (len(docs) - 2, len(docs)),
       str(bal.outstanding(docs, rescore=True)))
stated = plan.describe(docs)
row = next(line for line in stated.splitlines() if named("datalab") in line)
cells = [c.strip() for c in row.split("\u2502")]
report("the plan states it per run, once the corpus is known",
       cells[-3:-1] == [str(len(docs) - 2), "0"], row)
report("...and nothing is cut off, whatever the widest cell is",
       "\u2026" not in stated, stated)

print("\nAND ASKS BEFORE IT SPENDS")
ask = pathlib.Path(tempfile.mkdtemp())
B.fetch = lambda *_: ask
D.adapter = as_adapter(counting)
shown, called = [], {"n": 0}


def counting_adapter(prov):
    inner = counting(prov)

    def extract(pdf, schema, *, timeout, config):
        called["n"] += 1
        return inner(pdf, schema, timeout=timeout, config=config)
    return extract


D.adapter = as_adapter(counting_adapter)
declined = B.BenchmarkRun(["datalab"], out=ask, score_workers=1,
                          predict_workers={"*": 1}).execute(
    confirm=lambda plan: (shown.append(plan), False)[1])
report("a declined plan calls no vendor at all", called["n"] == 0, f'{called["n"]} calls')
report("...and gives back nothing rather than a half summary", declined == {})
# The callback is handed a rich renderable, not text: the terminal gets colour and column
# widths a log cannot. Rendered here to read it back.
def as_text(renderable):
    console = Console(file=io.StringIO(), width=100, no_color=True)
    console.print(renderable)
    return console.file.getvalue()


report("...having shown what it would have done",
       named("datalab") in as_text(shown[0]) and "predict" in as_text(shown[0]),
       as_text(shown[0]) if shown else "")
report("...and wrote no summary.json", not list(ask.glob("*/summary.json")))

called["n"] = 0
accepted = B.BenchmarkRun(["datalab"], out=ask, score_workers=1,
                          predict_workers={"*": 1}).execute(confirm=lambda plan: True)
report("an approved plan runs", called["n"] == len(B.read_manifest()) and len(accepted) == 1,
       f'{called["n"]} calls, {len(accepted)} rows')

print("\nEACH RUN IS TIMED FOR ITSELF, NOT FOR THE QUEUE IT SHARES")
from omni_extract_bench.progress import Progress                               # noqa: E402

timed = pathlib.Path(tempfile.mkdtemp())
B.fetch = lambda *_: timed
landed = {}
_t0 = _time.monotonic()


def uneven(prov):
    def extract(pdf, schema, *, timeout, config):
        _time.sleep(0.02 if config.mode == "balanced" else 0.5)
        with guard:
            landed[config.mode] = _time.monotonic() - _t0
        return Extraction(result={"a": "x"}, cost=Cost(usd=0.1))
    return extract


D.adapter = as_adapter(uneven)
# `setdefault`, because a benchmark raises two displays -- one for predicting and one for
# grading -- and it is the predicting one whose per-leg elapsed this is about.
_exit, elapsed = Progress.__exit__, {}
Progress.__exit__ = lambda self, *a: ([elapsed.setdefault(n, s.elapsed)
                                       for n, s in self.stats.items()], _exit(self, *a))[1]
try:
    B.run(["datalab"], out=timed, score_workers=1,
          options={"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]})
finally:
    Progress.__exit__ = _exit
quick, slow = named("datalab", mode="balanced"), named("datalab", mode="accurate")
report("the run that finished first says so", elapsed[quick] < elapsed[slow] / 2,
       f'{elapsed[quick]:.2f}s against the vendor\'s {elapsed[slow]:.2f}s')
report("...and it matches when its last document actually landed",
       abs(elapsed[quick] - landed["balanced"]) < 0.1,
       f'reported {elapsed[quick]:.2f}s, landed at {landed["balanced"]:.2f}s')

print("\nAND THE CAP BELONGS TO THE SERVICE, NOT TO THE NAME YOU TYPED")
models = pathlib.Path(tempfile.mkdtemp())
B.fetch = lambda *_: models
live["n"] = live["peak"] = 0
D.adapter = as_adapter(counting)
ids = ["openai/gpt-5.6-sol", "anthropic/claude-opus-5", "google/gemini-3.7-flash"]
B.run(ids, out=models, score_workers=1)
report("three model ids share one budget, not one each",
       live["peak"] <= WORKERS["llm_single_shot"],
       f'peak {live["peak"]} against a cap of {WORKERS["llm_single_shot"]}; '
       f'a pool per name gives {len(ids) * WORKERS["llm_single_shot"]}')
report("...while each still runs its own model, into its own directory",
       len([d for d in models.iterdir() if d.is_dir()]) == len(ids),
       str(sorted(d.name for d in models.iterdir() if d.is_dir())))
report("a name people type still sets the cap",
       B.workers_for(["openai/gpt-5.6-sol"], {"openai/gpt-5.6-sol": 2}) == 2)
report("...and where several names in one group disagree, the smallest wins",
       B.workers_for(ids, {"openai/gpt-5.6-sol": 2, "anthropic/claude-opus-5": 20}) == 2)
report("...falling back to the adapter's own limit when nobody says",
       B.workers_for(ids, None) == WORKERS["llm_single_shot"])

print("\nAN ACCOUNT FAILURE IS ABOUT THE ACCOUNT, SO IT STOPS EVERY RUN OF THAT VENDOR")
acct = pathlib.Path(tempfile.mkdtemp())
B.fetch = lambda *_: acct
tried = []


def broke(prov):
    def extract(pdf, schema, *, timeout, config):
        with guard:
            tried.append(config.mode)
        raise AccountFailure("out of credits")
    return extract


D.adapter = as_adapter(broke)
try:
    B.run(["datalab"], out=acct, score_workers=1,
          options={"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]})
    report("the account failure reaches the caller", False, "it was swallowed")
except AccountFailure:
    report("the account failure reaches the caller", True)
report("...and the other tier is not asked to pay it again",
       len(tried) <= WORKERS["datalab"],
       f'{len(tried)} documents attempted across both tiers: {sorted(set(tried))}')

print(f"\n{'RUNS DO NOT MIX' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
