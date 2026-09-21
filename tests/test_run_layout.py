#!/usr/bin/env python3
"""A RUN IS A PROVIDER PLUS ITS OPTIONS, and each one is stored on its own.

`--options` changes what is measured. When it did not change WHERE the answer was stored,
`needs_run` found the other configuration's records and skipped every document -- so a run
that asked for `mode=fast` reported `balanced` results and never called the vendor at all.
A silently wrong number, arrived at by a resume doing its job.

Run: python3 tests/test_run_layout.py
"""
import dataclasses
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
import omni_extract_bench.harness.vendor as V                                  # noqa: E402
from omni_extract_bench.harness.extraction import (AccountFailure, Cost,       # noqa: E402
                                                   Extraction)
from omni_extract_bench.harness.vendor import out_name                          # noqa: E402

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
# Every name carries a digest of the WHOLE resolved settings, so a directory holds one
# configuration. The readable half is the difference from the defaults and is decoration.
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
# The whole point. Two of them sharing a directory is the mixing this exists to stop, so the
# digest -- not the readable prefix -- is what carries the guarantee.
apart = [named("datalab", mode="accurate"), named("datalab", mode="fast"),
         named("datalab", mode="fast", poll_interval=2.5),
         named("reducto", system_prompt="a b"),      # the name shows none of these, so
         named("reducto", system_prompt="a_b"),      # only the digest keeps them in four
         named("reducto", system_prompt="L" * 300),  # directories -- which is why it
         named("reducto", system_prompt="M" * 300)]  # covers every field and not a few
report("every one of them is unique", len(set(apart)) == len(apart), str(apart))

print("\nAND NOT ON WHAT THE DEFAULT HAPPENED TO BE THAT DAY")
# THE FAILURE THIS REPLACED. The digest used to cover only what DIFFERED from the adapter's
# defaults, so the name held within a version of the code and broke across one: move the
# `Config` default the day a vendor's maximum tier moves, and the new stock run is stored
# under the same plain `datalab` as the old one. `needs_run` finds records, skips every
# document, and two tiers are averaged into one published number.
def _datalab_defaulting_to(tier):
    module = types.ModuleType("fake_datalab")

    @dataclasses.dataclass(frozen=True)
    class Config:
        mode: str = tier
        base_url: str = "https://www.datalab.to"
        poll_interval: float = 5.0

    module.Config = Config
    return lambda provider: module


_real_module = V._module
try:
    V._module = _datalab_defaulting_to("balanced")
    was_stock, was_pinned = named("datalab"), named("datalab", mode="accurate")
    V._module = _datalab_defaulting_to("accurate")
    now_stock, now_pinned = named("datalab"), named("datalab", mode="balanced")
finally:
    V._module = _real_module
report("the same name never means two different settings", was_stock != now_stock,
       f"{was_stock} vs {now_stock}")
# And the other direction, which is what naming the RESOLVED settings buys over naming the
# difference from them: a run pinned to the old value is the same run, so it resumes into its
# own directory instead of being bought a second time under a new name.
report("...and the same settings always mean the same name", was_stock == now_pinned,
       f"{was_stock} vs {now_pinned}")
report("...however the defaults moved around them", was_pinned == now_stock,
       f"{was_pinned} vs {now_stock}")

print("\nAND THE NAME DOES NOT DEPEND ON HOW IT WAS WRITTEN")
report("option order does not change it",
       named("datalab", mode="fast", poll_interval=2.5)
       == named("datalab", poll_interval=2.5, mode="fast"))
# `hash()` is salted per process, so it would name the same run differently tomorrow.
seeds = {subprocess.run(
    [sys.executable, "-c",
     "from omni_extract_bench.harness.vendor import out_name;"
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
# The PROVIDER half too, which went unsanitised while only the readable half was checked: an
# OpenRouter suffix is part of the id, and `mistral-medium-3-5:batch` is not a filename on
# every filesystem. Two ids that sanitise alike still differ -- the model is one of the
# settings the digest covers.
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
B.fetch = lambda root: out
B.read_manifest = lambda *a, **k: docs
called = []
# The adapter is handed its `Config`, so a stub reads the setting off a field.
V.adapter = lambda prov: (lambda pdf, schema, *, timeout, config: (
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

# A DIRECTORY SAYS WHAT IT IS WITHOUT BEING FINISHED. The name is a digest, `summary.json`
# is written at the end and an interrupted run never reaches it, and a record states the
# settings only once a document has landed.
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
# `--options` may give one provider a LIST, and each entry is its own run, directory and
# summary row -- rather than a second provider name minted per tier.
out2 = pathlib.Path(tempfile.mkdtemp())
B.fetch = lambda root: out2
summary = B.run(["datalab"], out=out2, score_workers=1, predict_workers={"*": 1},
                options={"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]})
# The summary is keyed by the RUN, not the vendor: a measured configuration is (provider,
# how it was steered), and keyed by the vendor alone a steered row silently replaces a stock
# one. The key is the same string that named the directory.
report("each configuration is its own row in the summary",
       sorted(summary) == sorted([named("datalab", mode="balanced"),
                                  named("datalab", mode="accurate")]),
       str(sorted(summary)))
report("...matching its directory exactly",
       sorted(summary) == sorted(d.name for d in out2.iterdir() if d.is_dir()),
       str(sorted(d.name for d in out2.iterdir() if d.is_dir())))
# Carried in the row so a published table can be labelled without parsing the key apart.
steered_row = summary[named("datalab", mode="accurate")]
report("the row names the vendor it came from", steered_row["provider"] == "datalab")
report("...and what it was sent", steered_row["settings"]["mode"] == "accurate")
report("the stock row carries its own settings",
       summary[named("datalab")]["settings"]["mode"] == "balanced")
report("both ran the same vendor",
       {r["provider"] for r in summary.values()} == {"datalab"})

# EACH RUN CARRIES ITS OWN ROW. Grading is serial, so waiting for the last vendor to publish
# the first one leaves an interrupted invocation with directories of answers and nothing that
# reads them.
for one in summary.values():
    on_disk = json.loads((out2 / one["run"] / "summary.json").read_text())
    report(f"{one['run']}: its directory carries its own row", on_disk == one)
report("and there is no table across them to go stale",
       not (out2 / "summary.json").exists())

# THE FILE DESCRIBES THE DIRECTORY, NOT THE INVOCATION. A narrower resume regrades nothing and
# must not leave a two-document summary sitting on a full run -- `runs/*/summary.json` is what
# gets aggregated, and a table built from those would be wrong by the ratio of the two.
B.read_manifest = lambda *a, **k: docs[:1]          # what `--limit 1` would select
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
B.fetch = lambda root: out3
real_score_all, graded = B.score_all, {"n": 0}


def interrupted_on_the_second(*a, **k):
    graded["n"] += 1
    if graded["n"] == 2:
        raise KeyboardInterrupt
    return real_score_all(*a, **k)


B.score_all = interrupted_on_the_second
try:
    B.run(["datalab"], out=out3, score_workers=1, predict_workers={"*": 1},
          options={"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]})
    report("the interrupt reaches the caller", False, "it was swallowed")
except KeyboardInterrupt:
    report("the interrupt reaches the caller", True)
finally:
    B.score_all = real_score_all

first, second = named("datalab", mode="balanced"), named("datalab", mode="accurate")
report("the run that finished grading published its row",
       (out3 / first / "summary.json").exists())
report("the one that did not, did not", not (out3 / second / "summary.json").exists())
report("...so aggregating the directories finds one finished run, not a half-counted second",
       sorted(p.parent.name for p in out3.glob("*/summary.json")) == [first])

print("\nA RUN IS A PROVIDER PLUS ITS OPTIONS, AND THE CAP IS STILL THE VENDOR'S")
# `--options` keyed to a name that is not being run drops the steering silently, and the run
# then reports as stock -- a steered measurement wearing a stock label.
for bad in ({"datalb": {"mode": "accurate"}},        # typo
            {"reducto": {"agentic_table_mode": "default"}}):   # not in --providers
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

# THE CAP IS THE VENDOR'S AND IT IS SHARED. Two runs of one provider go at once, so a cap
# applied to each puts twice as many documents in flight as the vendor tolerates.
import threading as _threading                                                 # noqa: E402
import time as _time                                                           # noqa: E402
from omni_extract_bench.harness import WORKERS                                 # noqa: E402

wide = pathlib.Path(tempfile.mkdtemp())
many = []
for i in range(20):
    g = wide / f"g{i}.json"; g.write_text(json.dumps({"a": "x"}))
    pdf = wide / f"p{i}.pdf"; pdf.write_bytes(b"%PDF")
    many.append(B.Doc(f"d{i}", "s", pdf, g,
                      {"type": "object", "properties": {"a": {"type": "string"}}}))
B.fetch = lambda root: wide
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


V.adapter = counting
B.run(["datalab"], out=wide, score_workers=1,
      options={"datalab": [{"mode": "balanced"}, {"mode": "accurate"}]})
report("two runs of one vendor share its concurrency cap, not double it",
       live["peak"] <= WORKERS["datalab"],
       f'peak {live["peak"]} in flight against a cap of {WORKERS["datalab"]}')
# AND SHARE IT DYNAMICALLY. Dividing the cap between the runs also keeps the count legal --
# and leaves half of it idle whichever run has work. One queue per vendor uses the lot.
report("...and the whole cap is in use, not half of it each",
       live["peak"] > WORKERS["datalab"] // 2,
       f'peak {live["peak"]}, which a cap split two ways could not exceed '
       f'{WORKERS["datalab"] // 2}')

print("\nAND SAYS WHAT A RESUME WOULD COST, BEFORE IT COSTS IT")
# The plan a `BenchmarkRun` can state without a download or a vendor call, and -- once the
# corpus is known -- what each run still owes. `--rescore` is why the two halves are counted
# separately: every prediction can be on disk and every grade still outstanding.
from omni_extract_bench.benchmark import BenchmarkRun                          # noqa: E402

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
report("a run counts what it would still predict, and still grade",
       bal.outstanding(docs) == (len(docs) - 2, 0), str(bal.outstanding(docs)))
report("...a run with nothing on disk owes everything",
       acc.outstanding(docs) == (len(docs), len(docs)), str(acc.outstanding(docs)))
# `grading_split` is the one definition, so this cannot promise 0 and then grade 3.
report("--rescore owes every grade again, however many are on disk",
       bal.outstanding(docs, rescore=True) == (len(docs) - 2, len(docs)),
       str(bal.outstanding(docs, rescore=True)))
report("the plan states it per run, once the corpus is known",
       f"{len(docs) - 2:>5} to predict,     0 to grade" in plan.describe(docs),
       plan.describe(docs))

print("\nEACH RUN IS TIMED FOR ITSELF, NOT FOR THE QUEUE IT SHARES")
# One queue serves every run of a vendor, so finishing them together reported the whole
# vendor's wall time on every line -- two reducto tiers both "done in 4m44s" while one of them
# had been finished for minutes.
from omni_extract_bench.progress import Progress                               # noqa: E402

timed = pathlib.Path(tempfile.mkdtemp())
B.fetch = lambda root: timed
landed = {}
_t0 = _time.monotonic()


def uneven(prov):
    def extract(pdf, schema, *, timeout, config):
        _time.sleep(0.02 if config.mode == "balanced" else 0.5)
        with guard:
            landed[config.mode] = _time.monotonic() - _t0
        return Extraction(result={"a": "x"}, cost=Cost(usd=0.1))
    return extract


V.adapter = uneven
_exit, elapsed = Progress.__exit__, {}
Progress.__exit__ = lambda self, *a: (elapsed.update(
    {n: s.elapsed for n, s in self.stats.items()}), _exit(self, *a))[1]
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
# Every `org/model` id is one `llm_single_shot`, one OpenRouter endpoint and one key. Keyed by
# provider name each of them got a pool of its own -- measured, three models put 15 against a
# budget of 5 -- and no name-keyed table could ever cover them, since model ids are open-ended.
models = pathlib.Path(tempfile.mkdtemp())
B.fetch = lambda root: models
live["n"] = live["peak"] = 0
V.adapter = counting
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
# The key is the same for both tiers, so the second cannot pay either. Given a pool per run it
# found that out for itself, a wave of documents later; given one per vendor it is told.
acct = pathlib.Path(tempfile.mkdtemp())
B.fetch = lambda root: acct
tried = []


def broke(prov):
    def extract(pdf, schema, *, timeout, config):
        with guard:
            tried.append(config.mode)
        raise AccountFailure("out of credits")
    return extract


V.adapter = broke
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
