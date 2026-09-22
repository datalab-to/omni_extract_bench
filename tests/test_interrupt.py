#!/usr/bin/env python3
"""Ctrl-C stops the run, and leaves it resumable.

THE SUBTLETY THAT MAKES THIS WORTH A TEST. Python delivers a KeyboardInterrupt to the MAIN
thread. Providers run concurrently, so `ProviderRun.predict` is in a worker -- where a
`except KeyboardInterrupt` can never fire. The handler has to be in `run`, on the main thread,
and it has to set a flag every provider shares; otherwise the interrupt escapes into
`ThreadPoolExecutor.__exit__`, which waits for every provider to work through the whole corpus.
Measured on a toy two-level pool that is 8.1s against 0.0s, and on the real corpus it is hours:
a Ctrl-C that appears to do nothing at all.

It went undetected once because the obvious ways to test it do not deliver the signal: a bash
background job has SIGINT set to SIG_IGN, and a pty without a controlling terminal never turns
^C into a signal. `subprocess.send_signal` is what a terminal actually does.

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
from omni_extract_bench.harness.registry import out_name  # noqa: E402

FAILS = []
ROOT = pathlib.Path(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


print("\nA STOP ALREADY SET MEANS NOTHING IS CALLED")
out = pathlib.Path(tempfile.mkdtemp())
docs = []
for i in range(5):
    gt = out / f"g{i}.json"; gt.write_text("{}")
    pdf = out / f"p{i}.pdf"; pdf.write_bytes(b"%PDF")
    docs.append(B.Doc(f"d{i}", "s", pdf, gt, {"type": "object"}))

called = []
_real_predict = B.predict
B.predict = lambda *a, **k: called.append(1) or {"result": {"a": 1}, "cost": {}}
try:
    already = threading.Event()
    already.set()
    B.ProviderRun("datalab", [B.Run("datalab", "datalab", {}, out)], 2)\
     .predict(docs, timeout=5, stop=already)
finally:
    B.predict = _real_predict
report("no vendor call is made", called == [], f"{len(called)} calls")
report("...and nothing is written, so every document stays resumable",
       not list((out / "records").glob("*.json")))

print("\nAND A REAL SIGINT STOPS A RUN, ON A DOCUMENT BOUNDARY")
script = out / "run_it.py"
script.write_text(f'''
import sys, json, pathlib, time, types
sys.path.insert(0, {str(ROOT)!r})
import omni_extract_bench.benchmark as B
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
    docs.append(B.Doc(f"d{{i:02d}}", "s", p, g, {{"type": "object",
                "properties": {{"a": {{"type": "string"}}}}}}))
B.fetch = lambda *_: out
B.read_manifest = lambda *a, **k: docs
D.adapter = as_adapter(lambda prov: lambda pdf, schema, *, timeout, **o:
    time.sleep(0.4) or Extraction(result={{"a": "x"}}, cost=Cost(usd=0.5)))
B.run(["datalab", "reducto"], out=out, score_workers=1, predict_workers={{"*": 2}})
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
           all(not B.needs_run(recs / f"{d}.json") for d in done))

report("no half-written file survives anywhere",
       not list(work.rglob("*.partial*")), str(list(work.rglob("*.partial*"))))

print("\nONE VENDOR'S BILLING PROBLEM IS NOT EVERY VENDOR'S")
import collections                                                             # noqa: E402
import omni_extract_bench.harness.document as D
import omni_extract_bench.harness.registry as V                                  # noqa: E402
from omni_extract_bench.harness.contract import Cost, Extraction
from omni_extract_bench.harness.errors import AccountFailure

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

acct = pathlib.Path(tempfile.mkdtemp())
pair = []
for i in range(6):
    g = acct / f"g{i}.json"; g.write_text(json.dumps({"a": "x"}))
    pdf = acct / f"p{i}.pdf"; pdf.write_bytes(b"%PDF")
    pair.append(B.Doc(f"d{i}", "s", pdf, g, {"type": "object",
                                             "properties": {"a": {"type": "string"}}}))
calls = collections.Counter()


def broke_adapter(prov):
    def extract(pdf, schema, *, timeout, config):
        calls[prov] += 1
        if prov == "datalab":
            raise AccountFailure("out of credits")
        return Extraction(result={"a": "x"}, cost=Cost(usd=0.1))
    return extract


saved_adapter, saved_fetch, saved_manifest = D.adapter, B.fetch, B.read_manifest
D.adapter = as_adapter(broke_adapter)
B.fetch = lambda *_: acct
B.read_manifest = lambda *a, **k: pair
try:
    B.run(["datalab", "reducto"], out=acct, score_workers=1, predict_workers={"*": 1})
    report("an account failure stops the run rather than publishing", False,
           "run() returned a summary")
except AccountFailure:
    report("an account failure stops the run rather than publishing", True)
finally:
    V.adapter, B.fetch, B.read_manifest = saved_adapter, saved_fetch, saved_manifest
report("the healthy vendor still ran every document", calls["reducto"] == 6,
       f"{calls['reducto']} of 6 -- a shared flag stops it after one")
report("...and no run published a summary from a half-run corpus",
       not list(acct.glob("*/summary.json")))
report("...while its predictions are on disk for --score-only",
       len(list((acct / out_name("reducto") / "predictions").glob("*.json"))) == 6)

print("\nA NARROWED RUN DOES NOT TRUNCATE THE SCORES FILE")
narrow = pathlib.Path(tempfile.mkdtemp())
(narrow / "predictions").mkdir()
five = []
for i in range(5):
    g = narrow / f"g{i}.json"; g.write_text(json.dumps({"a": "x"}))
    (narrow / "predictions" / f"d{i}.json").write_text(json.dumps({"a": "x"}))
    five.append(B.Doc(f"d{i}", "s", narrow / "x.pdf", g,
                      {"type": "object", "properties": {"a": {"type": "string"}}}))
B.Run("v", "v", {}, narrow).score(five, workers=1)
lines = lambda: len((narrow / "scores.jsonl").read_text().strip().splitlines())   # noqa: E731
report("a full run writes every row", lines() == 5, str(lines()))
B.Run("v", "v", {}, narrow).score(five[:2], workers=1)
report("a narrowed run keeps the rows it did not select", lines() == 5, str(lines()))
B.Run("v", "v", {}, narrow).score(five[:2], workers=1, rescore=True)
report("...and so does a narrowed rescore", lines() == 5, str(lines()))

back = B.Run("v", "v", {}, narrow).score(five[:2], workers=1)
report("the return value is this run's selection, not the file",
       [r["doc_id"] for r in back] == ["d0", "d1"], str([r["doc_id"] for r in back]))
report("...so the summary counts what was asked for",
       B.summarise(back)["documents"] == 2, str(B.summarise(back)["documents"]))

print("\nA FULL DISK STOPS THE VENDOR INSTEAD OF BUYING THE REST")
disk = pathlib.Path(tempfile.mkdtemp())
twenty = []
for i in range(20):
    g = disk / f"g{i}.json"; g.write_text(json.dumps({"a": "x"}))
    pdf = disk / f"p{i}.pdf"; pdf.write_bytes(b"%PDF")
    twenty.append(B.Doc(f"d{i}", "s", pdf, g,
                        {"type": "object", "properties": {"a": {"type": "string"}}}))
paid = collections.Counter()
D.adapter = as_adapter(lambda prov: lambda pdf, schema, *, timeout, **o: (
    paid.update([prov]), Extraction(result={"a": "x"}, cost=Cost(usd=0.1)))[1])
real_write_json = B.write_json_atomic
B.write_json_atomic = lambda path, obj, **kw: (
    (_ for _ in ()).throw(OSError("No space left on device")))
try:
    B.ProviderRun("datalab", [B.Run("datalab", "datalab", {}, disk / "x")], 4)\
     .predict(twenty, timeout=5)
    report("a failed write stops the vendor", False, "it returned normally")
except OSError:
    report("a failed write stops the vendor", True)
finally:
    B.write_json_atomic = real_write_json
    D.adapter = saved_adapter
report("...after a couple of documents, not all twenty", paid["datalab"] <= 8,
       f"{paid['datalab']} of 20 paid for with nothing stored")

print("\nA FAILURE INSIDE ONE DOCUMENT IS ONE ROW, NOT THE PASS")
vd = pathlib.Path(tempfile.mkdtemp())
(vd / "predictions").mkdir()
some = []
for i in range(5):
    g = vd / f"g{i}.json"; g.write_text(json.dumps({"a": "x"}))
    (vd / "predictions" / f"d{i}.json").write_text(json.dumps({"a": "x"}))
    some.append(B.Doc(f"d{i}", "s", vd / "x.pdf", g,
                      {"type": "object", "properties": {"a": {"type": "string"}}}))
real_write = pathlib.Path.write_text


def failing_write(self, *a, **k):
    if "verdicts" in str(self) and self.stem == "d4":
        raise OSError("no space left on device")
    return real_write(self, *a, **k)


pathlib.Path.write_text = failing_write
try:
    got = B.Run("v", "v", {}, vd).score(some, workers=1, verdicts=True)
finally:
    pathlib.Path.write_text = real_write
report("the other four documents survive", sum(r["status"] == "scored" for r in got) == 4,
       str([r["status"] for r in got]))
report("the failing one is a row, not an escape",
       sum(r["status"] == "error" for r in got) == 1)
report("...and the file holds all five", 
       len((vd / "scores.jsonl").read_text().strip().splitlines()) == 5)

print("\nAND AN INTERRUPT WHILE SCORING KEEPS WHAT IT GRADED")
score_script = out / "score_it.py"
score_script.write_text(f"""
import sys, json, pathlib
sys.path.insert(0, {str(ROOT)!r})
import omni_extract_bench.benchmark as B

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
        docs.append(B.Doc(f"d{{i:02d}}", "s", out / "x.pdf", g, schema))
    B.Run("v", "v", {{}}, out).score(docs, workers=4)

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
report("what was graded is written, not discarded", 0 < len(kept) < 60,
       f"{len(kept)} of 60 rows -- 0 means the work was thrown away")
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
    small.append(B.Doc(f"d{i}", "s", fresh / "x.pdf", g, schema))

graded = []
real_score_one = B.score_one
B.score_one = lambda doc, **k: graded.append(doc.doc_id) or real_score_one(doc, **k)
try:
    # `verdicts=False` said out loud, because the point of the last two is what happens when
    # rows that HAVE none are asked for them, and the default now writes them.
    B.Run("v", "v", {}, fresh).score(small, workers=1, verdicts=False)
    report("a first pass grades everything", len(graded) == 6, str(len(graded)))

    graded.clear(); rows = B.Run("v", "v", {}, fresh).score(small, workers=1, verdicts=False)
    report("a second pass grades nothing", graded == [], str(graded))
    report("...and still returns every row", len(rows) == 6, str(len(rows)))

    graded.clear()
    B.Run("v", "v", {}, fresh).score(small, workers=1, verdicts=False, rescore=True)
    report("rescore=True grades everything again", len(graded) == 6, str(len(graded)))

    graded.clear(); B.Run("v", "v", {}, fresh).score(small, workers=1, verdicts=True)
    report("asking for verdicts re-grades rows that have none", len(graded) == 6, str(len(graded)))
    graded.clear(); B.Run("v", "v", {}, fresh).score(small, workers=1, verdicts=True)
    report("...and those rows are reusable once they do", graded == [], str(graded))

    sj = fresh / "scores.jsonl"
    lines = sj.read_text().splitlines()
    lines[2] = "{ this is not json"
    sj.write_text("\n".join(lines) + "\n")
    graded.clear(); B.Run("v", "v", {}, fresh).score(small, workers=1, verdicts=True)
    report("an unreadable row costs one document, not the file", len(graded) == 1, str(graded))
finally:
    B.score_one = real_score_one

print("\nAND THE SCORES FILE IS WRITTEN ATOMICALLY")
report("no .partial survives a write", not list(fresh.glob("*.partial*")))
report("every line parses", all(json.loads(l) for l in
                                (fresh / "scores.jsonl").read_text().splitlines() if l.strip()))

print(f"\n{'CTRL-C IS CLEAN' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
