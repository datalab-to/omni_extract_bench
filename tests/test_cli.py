#!/usr/bin/env python3
"""The command line, end to end, on a temporary corpus.

`cli.py` is what actually runs the benchmark. The behaviours worth guarding are the ones that
were wrong at some point:

  * the package's `grade` is the address-based scorer, not the old paired walker
  * a schema is loaded, its `$ref`s resolved, and a missing one is an error rather than `{}`
  * one document this harness cannot score does not take the run with it, and is kept
    DISTINCT from a provider that returned nothing -- merging them lets a broken harness read
    as a poor vendor

The last one survived a rewrite: it used to be tested through `score-dir`, which paired files
by basename across flat per-type directories and has been removed. `bench` reads the benchmark
layout instead, and has to keep the same promise.

Run: python3 tests/test_cli.py
"""
import json
import os as _os
import shutil
import sys as _sys
import tempfile
from pathlib import Path

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
from omni_extract_bench import corpus as corpus_atlas                       # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


_RUNS = [0]


def _next():
    _RUNS[0] += 1
    return _RUNS[0]


def run(*argv):
    """Invoke the CLI in-process, capturing both streams."""
    import contextlib
    import io

    from omni_extract_bench.cli import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main([str(a) for a in argv])
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


SCHEMA = {"type": "object", "properties": {
    "name": {"$ref": "#/$defs/str"}, "n": {"type": "number"}, "z": {"type": "string"}},
    "$defs": {"str": {"type": "string"}}}
DOC = {"name": "Acme", "n": 2, "z": "keep"}
NEARLY = {"name": "Acme", "n": 2, "z": "WRONG"}

print("\nTHE PACKAGE EXPORTS THE SCORER IT CLAIMS TO")

import omni_extract_bench as pkg                                          # noqa: E402
from omni_extract_bench import score as score_mod                         # noqa: E402

report("omni_extract_bench.grade is score.grade", pkg.grade is score_mod.grade)
r = pkg.grade({"a": 1}, {"a": 1}, {"properties": {"a": {"type": "number"}}})
report("it reports the keys the new scorer reports, not `leaf_accuracy`",
       "accuracy" in r and "leaf_accuracy" not in r, str(sorted(r))[:120])

TMP = Path(tempfile.mkdtemp())
try:
    corpus, preds = TMP / "corpus", TMP / "preds"
    preds.mkdir()
    for doc_id in ("good", "broken", "empty", "noschema"):
        (corpus / doc_id).mkdir(parents=True)
        (corpus / doc_id / "ground_truth.json").write_text(json.dumps(DOC))
        if doc_id != "noschema":
            (corpus / doc_id / "schema.json").write_text(json.dumps(SCHEMA))

    print("\nA CORPUS IS ITS ATLAS")
    (preds / "good.json").write_text(json.dumps(NEARLY))
    code, out, err = run("score", "--corpus", corpus, "--predictions", preds,
                                 "--out", str(TMP / f"run{_next()}"))
    report("scoring a corpus with no atlas stops, and says how to make one",
           code == 1 and "build-corpus" in err, (out + err).strip()[:200])
    note("a directory listing is not a benchmark; membership has to be declared")

    print("\nA MISSING SCHEMA IS AN ERROR, NOT AN EMPTY ONE")
    code, out, err = run("build-corpus", "--corpus", corpus)
    report("declaring a corpus with a schema-less document stops, naming the file",
           code == 1 and "schema.json" in err, (out + err).strip()[:200])
    note("an empty schema would grade an additionalProperties subtree silently")

    # Now make it a real corpus and test the rest.
    (corpus / "noschema" / "schema.json").write_text(json.dumps(SCHEMA))
    code, out, err = run("build-corpus", "--corpus", corpus)
    report("declaring it works once every document has both files", code == 0, (out + err)[:200])

    print("\nSCORING RESOLVES $ref AND CHARGES THE RIGHT THINGS")
    code, out, err = run("score", "--corpus", corpus, "--predictions", preds,
                                 "--out", str(TMP / f"run{_next()}"))
    report("`bench` succeeds on a schema containing $ref", code == 0, (out + err)[:200])
    report("the imperfect document scores 2 of 3", "66.67" in out, out.strip()[:200])

    print("\nONE UNSCORABLE DOCUMENT DOES NOT TAKE THE RUN WITH IT")
    (preds / "broken.json").write_text('{"name": {not json')
    (preds / "empty.json").write_text("{}")
    code, out, err = run("score", "--corpus", corpus, "--predictions", preds,
                                 "--out", str(TMP / f"run{_next()}"))
    report("the run completes despite an unparseable prediction", code == 0,
           (out + err).strip()[:200])
    report("the good document is still scored", "66.67" in out, out.strip()[:200])
    report("the unparseable one is named, with a reason",
           "broken" in out and "not JSON" in out, out.strip()[:300])
    note("a benchmark that stops on its first bad document is one you cannot run")

    print("\nA PROVIDER FAILURE IS NOT A HARNESS FAILURE")
    report("an empty prediction is unusable, not graded 0 into the mean",
           "2 unusable" in out and "excluded from that mean" in out, out.strip()[:300])
    report("the mean is taken over the graded only",
           "over the 1 graded" in out, out.strip()[:300])
    note("averaging in a rate-limited run makes it look like a bad model")

    print("\nA HARNESS FAILURE IS REPORTED SEPARATELY, AND LOUDLY")
    # A schema the scorer cannot see through: ours to fix, and it must not read as a vendor
    # scoring badly. Recorded in the atlas first, because changing data is deliberate.
    (corpus / "noschema" / "schema.json").write_text(json.dumps({"$ref": "#/nowhere"}))
    (preds / "noschema.json").write_text(json.dumps(DOC))
    run("build-corpus", "--corpus", corpus)
    code, out, err = run("score", "--corpus", corpus, "--predictions", preds,
                                 "--out", str(TMP / f"run{_next()}"))
    report("a document this harness cannot score is NOT SCORED, not 0",
           "NOT SCORED" in err, (out + err).strip()[:300])
    report("it is named with the reason, on stderr", "noschema" in err, err.strip()[:200])
    report("and the run exits non-zero so CI notices", code == 1, f"exit {code}")
    report("it is excluded from the mean, like an unusable one but counted apart",
           "over the 1 graded" in out and "NOT SCORED" in err, out.strip()[:200])
    note("merging the two would let a broken corpus read as a poor vendor")

    print("\nTHE ARGUMENT PARSER DEMANDS WHAT IS REQUIRED")
    for cmd, missing in ((("score-one", "--pred", "p", "--gt", "g"), "--schema"),
                         (("score", "--predictions", "p"), "--out"),
                         (("build-corpus",), "--corpus"),
                         (("explain", "--run", "r"), "--doc"),
                         (("verify",), "--corpus")):
        code, _o, _e = run(*cmd)
        report(f"`{cmd[0]}` refuses to run without {missing}", code == 2, f"exit {code}")
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{'CLI HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
