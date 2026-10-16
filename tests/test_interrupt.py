#!/usr/bin/env python3
"""Ctrl-C stops the run, and leaves it resumable.

THE SUBTLETY THAT MAKES THIS WORTH A TEST. Python delivers a KeyboardInterrupt to the MAIN
thread. Adapters run concurrently, so each adapter's `drain` is in a worker -- where a
`except KeyboardInterrupt` can never fire. The handler has to be in `predict_all`, on the main thread,
and it has to set a flag every provider shares; otherwise the interrupt escapes into
`ThreadPoolExecutor.__exit__`, which waits for every provider to work through the whole corpus.
Measured on a toy two-level pool that is 8.1s against 0.0s, and on the real corpus it is hours:
a Ctrl-C that appears to do nothing at all.

It went undetected once because the obvious ways to test it do not deliver the signal: a bash
background job has SIGINT set to SIG_IGN, and a pty without a controlling terminal never turns
^C into a signal. `subprocess.send_signal` is what a terminal actually does. An ignored SIGINT
is also inherited, and Python installs no KeyboardInterrupt handler over one, so each child
script restores the handler itself: otherwise a CI runner that ignores SIGINT fails this test.

Run: python3 tests/test_interrupt.py
"""
import json
import pathlib
import signal
import subprocess
import types
import sys
import tempfile
import threading
import time

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import omni_extract_bench.benchmark as B                                       # noqa: E402
from omni_extract_bench.scorers import field_matching as scorer                # noqa: E402
import omni_extract_bench.data as DS
from omni_extract_bench.harness.registry import out_name  # noqa: E402

FAILS = []
ROOT = pathlib.Path(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


out = pathlib.Path(tempfile.mkdtemp())

print("\nAND A REAL SIGINT STOPS A RUN, ON A DOCUMENT BOUNDARY")
script = out / "run_it.py"
script.write_text(f'''
import sys, json, pathlib, signal, time, types
signal.signal(signal.SIGINT, signal.default_int_handler)
sys.path.insert(0, {str(ROOT)!r})
import omni_extract_bench.benchmark as B
import omni_extract_bench.data as DS
import omni_extract_bench.harness.document as D
import omni_extract_bench.harness.registry as V

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

out = pathlib.Path(sys.argv[1])
docs = []
for i in range(40):
    g = out / f"g{{i}}.json"; g.write_text(json.dumps({{"a": "x"}}))
    p = out / f"p{{i}}.pdf"; p.write_bytes(b"%PDF")
    docs.append(DS.Doc(f"d{{i:02d}}", "s", p, g, {{"type": "object",
                "properties": {{"a": {{"type": "string"}}}}}}))
DS.download_dataset = lambda *_: out
DS.read_manifest = lambda *a, **k: (DS.download_dataset(), tuple(docs))
D.adapter = as_adapter(lambda prov: lambda pdf, schema, *, timeout, **o:
    time.sleep(0.4) or Extraction(result={{"a": "x"}}, cost=Cost(usd=0.5)))
B.execute(B.plan(["datalab", "reducto"], out=out, score_workers=1, workers=2))
''')
work = pathlib.Path(tempfile.mkdtemp())
proc = subprocess.Popen([sys.executable, str(script), str(work)],
                        stderr=subprocess.DEVNULL)
time.sleep(2.5)
proc.send_signal(signal.SIGINT)
sent = time.time()
try:
    proc.communicate(timeout=45)
    stopped = time.time() - sent
except subprocess.TimeoutExpired:
    proc.kill(); proc.communicate(); stopped = 999.0

report("the run stops promptly rather than draining the corpus", stopped < 15,
       f"{stopped:.1f}s after the signal -- it worked through the queue instead")
report("...and exits as interrupted", proc.returncode in (-signal.SIGINT, 130, 1),
       str(proc.returncode))

print("\nWHAT IT LEAVES BEHIND IS RESUMABLE")
for provider in ("datalab", "reducto"):
    run = work / out_name(provider)
    recs = run / "records"
    preds = run / "predictions"
    done = sorted(p.stem for p in recs.glob("*.json")) if recs.exists() else []
    have = {p.stem for p in preds.glob("*.json")} if preds.exists() else set()
    report(f"{provider}: it stopped part way, not at the end", 0 < len(done) < 40,
           f"{len(done)} of 40")
    report(f"{provider}: every record has its prediction beside it",
           not (set(done) - have), f"orphans: {sorted(set(done) - have)}")
    report(f"{provider}: every record parses, so none is re-run for being corrupt",
           all(B.recorded(B.Run(provider, {}, run), d) is not None for d in done))

report("no half-written file survives anywhere",
       not list(work.rglob("*.partial*")), str(list(work.rglob("*.partial*"))))

print("\nONE VENDOR'S BILLING PROBLEM IS NOT EVERY VENDOR'S")
import collections                                                             # noqa: E402
import omni_extract_bench.harness.document as D
import omni_extract_bench.harness.registry as V                                  # noqa: E402
from omni_extract_bench.harness.contract import Cost, Extraction
from omni_extract_bench.harness.errors import AccountFailure

_REAL_ADAPTER = V.adapter


def as_adapter(lookup):
    """A stub `provider -> extract` as an adapter: the three names `Adapter` asks for.

    `Config` and `prepare_schema` come from the real adapter, so a test that means to fake
    only the vendor call is not also faking the declaration or the schema it is sent.
    """
    def wrapped(provider):
        real = _REAL_ADAPTER(provider)
        # __name__ too: `resolve` reads it to file the run under the right vendor.
        return types.SimpleNamespace(__name__=real.__name__, Config=real.Config,
                                     prepare_schema=real.prepare_schema,
                                     extract=lookup(provider))
    return wrapped

acct = pathlib.Path(tempfile.mkdtemp())
pair = []
for i in range(6):
    g = acct / f"g{i}.json"; g.write_text(json.dumps({"a": "x"}))
    pdf = acct / f"p{i}.pdf"; pdf.write_bytes(b"%PDF")
    pair.append(DS.Doc(f"d{i}", "s", pdf, g, {"type": "object",
                                             "properties": {"a": {"type": "string"}}}))
calls = collections.Counter()


def broke_adapter(prov):
    def extract(pdf, schema, *, timeout, config):
        calls[prov] += 1
        if prov == "datalab":
            raise AccountFailure("out of credits")
        return Extraction(result={"a": "x"}, cost=Cost(usd=0.1))
    return extract


saved_adapter, saved_fetch, saved_manifest = D.adapter, DS.download_dataset, DS.read_manifest
D.adapter = as_adapter(broke_adapter)
DS.download_dataset = lambda *_: acct
DS.read_manifest = lambda *a, **k: (DS.download_dataset(), tuple(pair))
try:
    B.execute(B.plan(["datalab", "reducto"], out=acct, score_workers=1, workers=1))
    report("an account failure stops the run, and says so", False, "execute returned normally")
except AccountFailure:
    report("an account failure stops the run, and says so", True)
finally:
    V.adapter, DS.download_dataset, DS.read_manifest = saved_adapter, saved_fetch, saved_manifest
report("the healthy vendor still ran every document", calls["reducto"] == 6,
       f"{calls['reducto']} of 6 -- a shared flag stops it after one")
_sums = {s.parent.name: json.loads(s.read_text()) for s in acct.glob("*/summary.json")}
report("...and what was predicted is graded: the healthy run whole",
       _sums.get(out_name("reducto"), {}).get("documents") == 6, str(_sums.get(out_name("reducto"))))
report("...and the broke run's documents pending, so its summary doesn't claim to be whole",
       _sums.get(out_name("datalab"), {}).get("pending") == 6, str(_sums.get(out_name("datalab"))))
report("...while its predictions are on disk for --score-only",
       len(list((acct / out_name("reducto") / "predictions").glob("*.json"))) == 6)

print("\nA GRADING WRITES WHAT IT GRADED, WHOLE")
narrow = pathlib.Path(tempfile.mkdtemp())
(narrow / "predictions").mkdir()
five = []
for i in range(5):
    g = narrow / f"g{i}.json"; g.write_text(json.dumps({"a": "x"}))
    (narrow / "predictions" / f"d{i}.json").write_text(json.dumps({"a": "x"}))
    five.append(DS.Doc(f"d{i}", "s", narrow / "x.pdf", g,
                      {"type": "object", "properties": {"a": {"type": "string"}}}))
B.grade(B.Run("datalab", {}, narrow), five, score_workers=1)
lines = lambda: len((narrow / "scores.jsonl").read_text().strip().splitlines())   # noqa: E731
report("a full run writes every row", lines() == 5, str(lines()))
back = B.grade(B.Run("datalab", {}, narrow), five[:2], score_workers=1)
report("a narrowed grading's files are that grading's: the two it graded", lines() == 2, str(lines()))
report("...and so is its summary", back["documents"] == 2, str(back["documents"]))

print("\nA FULL DISK STOPS THE VENDOR INSTEAD OF BUYING THE REST")
disk = pathlib.Path(tempfile.mkdtemp())
twenty = []
for i in range(20):
    g = disk / f"g{i}.json"; g.write_text(json.dumps({"a": "x"}))
    pdf = disk / f"p{i}.pdf"; pdf.write_bytes(b"%PDF")
    twenty.append(DS.Doc(f"d{i}", "s", pdf, g,
                        {"type": "object", "properties": {"a": {"type": "string"}}}))
paid = collections.Counter()
D.adapter = as_adapter(lambda prov: lambda pdf, schema, *, timeout, **o: (
    paid.update([prov]), Extraction(result={"a": "x"}, cost=Cost(usd=0.1)))[1])
real_write_json = B.write_json
DS.read_manifest = lambda *a, **k: (disk, tuple(twenty))
B.write_json = lambda path, obj, **kw: (
    real_write_json(path, obj, **kw) if path.name == "settings.json"
    else (_ for _ in ()).throw(OSError("No space left on device")))
try:
    B.predict_all(B.Plan(runs=(B.Run("datalab", {}, disk / "x"),), docs=tuple(twenty),
                         dataset=str(disk), manifest="manifest.parquet", timeout=5,
                         workers=4, score_workers=1, verdicts=True,
                         score_only=False))
    report("a failed write stops the vendor", False, "it returned normally")
except OSError:
    report("a failed write stops the vendor", True)
finally:
    B.write_json = real_write_json
    D.adapter, DS.read_manifest = saved_adapter, saved_manifest
report("...after a couple of documents, not all twenty", paid["datalab"] <= 8,
       f"{paid['datalab']} of 20 paid for with nothing stored")

print("\nA FAILURE INSIDE ONE DOCUMENT IS ONE ROW, NOT THE PASS")
vd = pathlib.Path(tempfile.mkdtemp())
(vd / "predictions").mkdir()
some = []
for i in range(5):
    g = vd / f"g{i}.json"; g.write_text(json.dumps({"a": "x"}))
    (vd / "predictions" / f"d{i}.json").write_text(json.dumps({"a": "x"}))
    some.append(DS.Doc(f"d{i}", "s", vd / "x.pdf", g,
                      {"type": "object", "properties": {"a": {"type": "string"}}}))
real_write = pathlib.Path.write_text


def failing_write(self, *a, **k):
    if "verdicts" in str(self) and self.stem == "d4":
        raise OSError("no space left on device")
    return real_write(self, *a, **k)


pathlib.Path.write_text = failing_write
try:
    B.grade(B.Run("datalab", {}, vd), some, score_workers=1, verdicts=True)
    report("a grading that can't write its verdicts stops", False, "it returned normally")
except OSError:
    report("a grading that can't write its verdicts stops, not a row passed off as the vendor's", True)
finally:
    pathlib.Path.write_text = real_write
report("...and writes nothing: no scores, no summary, no verdicts half-made",
       not any((vd / f).exists() for f in ("scores.jsonl", "summary.json", "verdicts", ".verdicts.partial")))

print("\nAND AN INTERRUPT WHILE SCORING KEEPS WHAT IT GRADED")
score_script = out / "score_it.py"
score_script.write_text(f"""
import sys, json, pathlib, signal
signal.signal(signal.SIGINT, signal.default_int_handler)
sys.path.insert(0, {str(ROOT)!r})
import omni_extract_bench.benchmark as B
import omni_extract_bench.data as DS

def main():
    out = pathlib.Path(sys.argv[1])
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    schema = {{"type": "object", "properties": {{"rows": {{"type": "array", "items":
              {{"type": "object", "properties": {{"a": {{"type": "string"}},
                                                 "b": {{"type": "number"}}}}}}}}}}}}
    docs = []
    for i in range(60):
        gt = {{"rows": [{{"a": f"v{{j}}", "b": j}} for j in range(1500)]}}
        g = out / f"g{{i}}.json"; g.write_text(json.dumps(gt))
        (out / "predictions" / f"d{{i:02d}}.json").write_text(
            json.dumps({{"rows": list(reversed(gt["rows"]))}}))
        docs.append(DS.Doc(f"d{{i:02d}}", "s", out / "x.pdf", g, schema))
    B.grade(B.Run("datalab", {{}}, out), docs, score_workers=4)

if __name__ == "__main__":
    main()
""")
work2 = pathlib.Path(tempfile.mkdtemp())
proc = subprocess.Popen([sys.executable, str(score_script), str(work2)],
                        stderr=subprocess.DEVNULL)
time.sleep(2.0)
proc.send_signal(signal.SIGINT)
sent = time.time()
try:
    proc.communicate(timeout=60); stopped = time.time() - sent
except subprocess.TimeoutExpired:
    proc.kill(); proc.communicate(); stopped = 999.0

sj = work2 / "scores.jsonl"
kept = [json.loads(l) for l in sj.read_text().splitlines() if l.strip()] if sj.exists() else []
report("scoring stops promptly", stopped < 10, f"{stopped:.1f}s after the signal")
report("an interrupted grading writes nothing: the next one starts over, fresh", kept == [],
       f"{len(kept)} of 60 rows")
report("every kept row is a finished score, never a partial one",
       all(r.get("status") == "scored" and "accuracy" in r for r in kept))
report("the file is still in document order",
       [r["doc_id"] for r in kept] == sorted(r["doc_id"] for r in kept),
       str([r["doc_id"] for r in kept][:6]))

print("\nAND THE NEXT RUN GRADES ONLY WHAT IS MISSING")
fresh = pathlib.Path(tempfile.mkdtemp())
(fresh / "predictions").mkdir()
schema = {"type": "object", "properties": {"a": {"type": "string"}}}
small = []
for i in range(6):
    g = fresh / f"g{i}.json"; g.write_text(json.dumps({"a": "x"}))
    (fresh / "predictions" / f"d{i}.json").write_text(json.dumps({"a": "x"}))
    small.append(DS.Doc(f"d{i}", "s", fresh / "x.pdf", g, schema))

graded = []
real_score_one = B._score_one
B._score_one = lambda doc, **k: graded.append(doc.doc_id) or real_score_one(doc, **k)
try:
    B.grade(B.Run("datalab", {}, fresh), small, score_workers=1, verdicts=False)
    report("a first pass grades everything", len(graded) == 6, str(len(graded)))

    graded.clear(); summary = B.grade(B.Run("datalab", {}, fresh), small, score_workers=1, verdicts=True)
    report("a second pass grades everything again: grading is fresh", len(graded) == 6, str(len(graded)))
    report("...and counts every row", summary["documents"] == 6, str(summary["documents"]))

    sj = fresh / "scores.jsonl"
    lines = sj.read_text().splitlines()
    lines[2] = "{ this is not json"
    sj.write_text("\n".join(lines) + "\n")
    graded.clear(); B.grade(B.Run("datalab", {}, fresh), small, score_workers=1, verdicts=True)
    report("an unreadable line is written over by the next grading",
           len(graded) == 6 and len(B.read_scores(B.Run("datalab", {}, fresh))) == 6, str(graded))
finally:
    B._score_one = real_score_one

print("\nAND THE SCORES FILE IS WRITTEN ATOMICALLY")
report("no .partial survives a write", not list(fresh.glob("*.partial*")))
report("every line parses", all(json.loads(l) for l in
                                (fresh / "scores.jsonl").read_text().splitlines() if l.strip()))

print(f"\n{'CTRL-C IS CLEAN' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
