#!/usr/bin/env python3
"""A steered run is stored apart from a stock one.

`--options` changes what is measured. While it did not change WHERE the answer was stored,
`needs_run` found the stock run's records and skipped every document -- so a run that asked
for `mode=fast` reported `balanced` results, with no override recorded anywhere, and never
called the vendor at all. A silently wrong number, arrived at by a resume doing its job.

`datalab-accurate` was the workaround: a second provider name minted so that one case had
somewhere else to live. A run being (provider, options) rather than a provider is the general
form of it, and the preset is gone.

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
from omni_extract_bench.harness.extraction import Cost, Extraction             # noqa: E402
from omni_extract_bench.harness.vendor import out_name                          # noqa: E402

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
FAILS = []


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


def named(provider, **opts):
    return out_name(provider, opts)


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
# What `datalab-accurate` used to be for. A run is (provider, options), so `--options` may
# give one provider a LIST and each entry is its own run, directory and summary row.
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

print(f"\n{'RUNS DO NOT MIX' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
