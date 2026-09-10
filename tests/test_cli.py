#!/usr/bin/env python3
"""The command line, end to end, on a temporary corpus.

`cli.py` is what actually runs the benchmark, and it had no test. The three behaviours worth
guarding are the ones that were wrong until recently:

  * the package's `grade` is the address-based scorer, not the old paired walker
  * a schema is loaded, its `$ref`s resolved, and a missing one is an error rather than `{}`
  * a document this harness cannot score is EXCLUDED from the mean and reported, never
    counted as the provider having returned nothing

Run: python3 tests/test_cli.py
"""
import json
import os as _os
import shutil
import subprocess
import sys as _sys
import tempfile
from pathlib import Path

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


def run(*argv):
    """Invoke the CLI in-process, capturing both streams."""
    import contextlib
    import io

    from omni_extract_bench import cli
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(list(argv))
        except SystemExit as exc:                                   # argparse on bad args
            code = exc.code
    return code, out.getvalue(), err.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
print("\nTHE PACKAGE EXPORTS THE ADDRESS-BASED SCORER")
# Nothing else in the suite imports `grade` from the package root, so without this the
# switch in __init__.py is invisible to the tests.
import omni_extract_bench as pkg                                        # noqa: E402

report("omni_extract_bench.grade is score.grade",
       pkg.grade.__module__ == "omni_extract_bench.score"
       and pkg.grade.__name__ == "grade",
       f"got {pkg.grade.__module__}.{pkg.grade.__name__}")
report("omni_extract_bench.explain is exported too",
       pkg.explain.__module__ == "omni_extract_bench.score")
report("it reports the keys the new scorer reports, not `leaf_accuracy`",
       "accuracy" in pkg.grade({"a": 1}, {"a": 1},
                               {"properties": {"a": {"type": "number"}}})
       and "leaf_accuracy" not in pkg.grade({"a": 1}, {"a": 1},
                                            {"properties": {"a": {"type": "number"}}}))

# ═══════════════════════════════════════════════════════════════════════════════
TMP = Path(tempfile.mkdtemp(prefix="oeb_cli_"))
try:
    gt, pred, sch = TMP / "gt", TMP / "pred", TMP / "sch"
    for d in (gt, pred, sch):
        d.mkdir()
    DOC = {"invoice_no": "INV-1", "lines": [{"sku": "a", "qty": 1}, {"sku": "b", "qty": 2}]}
    # a schema that uses $ref, so an unresolved one would raise in the scorer
    SCHEMA = {"$defs": {"L": {"type": "object", "properties": {
                  "sku": {"type": "string"}, "qty": {"type": "number"}}}},
              "type": "object", "properties": {
                  "invoice_no": {"type": "string"},
                  "lines": {"type": "array", "items": {"$ref": "#/$defs/L"}}}}
    (gt / "good.json").write_text(json.dumps(DOC))
    (sch / "good.json").write_text(json.dumps(SCHEMA))
    # a prediction in the {"result": ...} envelope, one row moved and one value wrong
    (pred / "good.json").write_text(json.dumps({"result": {
        "invoice_no": "INV-1",
        "lines": [{"sku": "b", "qty": 2}, {"sku": "a", "qty": 9}],
        "vendor": "X"}}))

    print("\nA $ref SCHEMA IS RESOLVED RATHER THAN REFUSED")
    code, out, err = run("score", "--pred", str(pred / "good.json"),
                         "--gt", str(gt / "good.json"), "--schema", str(sch / "good.json"))
    report("`score` succeeds on a schema containing $ref", code == 0, err.strip()[:200])
    grade_out = json.loads(out) if code == 0 and out.strip().startswith("{") else {}
    report("...and reports the new keys",
           {"accuracy", "f1", "fabricated", "invented_field"} <= set(grade_out),
           f"got {sorted(grade_out)[:8]}")
    report("...and the invented top-level key is charged as invented_field",
           grade_out.get("invented_field") == 1, f"got {grade_out.get('invented_field')}")

    print("\nA MISSING SCHEMA IS AN ERROR, NOT AN EMPTY DICT")
    (gt / "noschema.json").write_text(json.dumps(DOC))
    (pred / "noschema.json").write_text(json.dumps(DOC))
    code, out, err = run("score", "--pred", str(pred / "noschema.json"),
                         "--gt", str(gt / "noschema.json"),
                         "--schema", str(sch / "noschema.json"))
    report("`score` fails loudly when the schema file is absent",
           code == 1 and "no schema" in err, f"code {code}, err {err.strip()[:120]}")
    note("it used to pass {} silently, which disabled open-map skipping")

    print("\nONE UNSCORABLE DOCUMENT DOES NOT TAKE THE RUN WITH IT")
    (gt / "broken.json").write_text(json.dumps(DOC))
    (sch / "broken.json").write_text(json.dumps(SCHEMA))
    (pred / "broken.json").write_text('{"result": {not json')
    (gt / "empty.json").write_text(json.dumps(DOC))
    (sch / "empty.json").write_text(json.dumps(SCHEMA))
    (pred / "empty.json").write_text("{}")
    code, out, err = run("score-dir", "--pred-dir", str(pred), "--gt-dir", str(gt),
                         "--schema-dir", str(sch))
    report("the run completes despite two unscorable documents", code == 0, err.strip()[:160])
    report("the good document is still scored", "good" in out and "66.67" in out,
           out.strip()[:200])
    report("both failures are named, with a reason",
           "broken" in err and "noschema" in err and "could not be scored" in err,
           err.strip()[:200])

    print("\nA HARNESS FAILURE IS NOT A VENDOR FAILURE")
    # `empty.json` is a provider returning nothing: scores 0, stays in the mean.
    # `broken`/`noschema` are this harness failing: excluded from the mean, reported.
    report("an empty prediction scores 0 and stays in the mean",
           "no output -> 0" in out, out.strip()[:200])
    report("unscorable documents are excluded from the mean and labelled as harness",
           "NOT SCORED" in out and "harness" in out, out.strip()[:300])
    report("the two are counted separately",
           "2 " in out.split("NOT SCORED")[1][:20] if "NOT SCORED" in out else False,
           "expected 2 not-scored (broken, noschema)")
    note("merging them would let a broken harness read as a poor vendor")

    print("\nTHE LEADERBOARD SCORES EVERY PROVIDER OVER THE SAME DOCUMENTS")
    base = TMP / "base"
    for prov in ("provA", "provB"):
        (base / prov).mkdir(parents=True)
        shutil.copy(pred / "good.json", base / prov / "good.json")
    (base / "provB" / "good.json").write_text(json.dumps({"result": DOC}))   # perfect
    for stray in ("broken.json", "noschema.json", "empty.json"):
        (gt / stray).unlink()
        (pred / stray).unlink()
        if (sch / stray).exists():
            (sch / stray).unlink()
    code, out, err = run("leaderboard", "--pred-root", str(base), "--gt-dir", str(gt),
                         "--schema-dir", str(sch))
    report("`leaderboard` runs and ranks both providers", code == 0
           and "provA" in out and "provB" in out, (out + err).strip()[:200])
    report("the perfect provider outranks the imperfect one",
           out.index("provB") < out.index("provA"), out.strip()[:200])

    print("\nA SCHEMA IS REQUIRED BY THE ARGUMENT PARSER")
    for cmd in (("score", "--pred", "p", "--gt", "g"),
                ("score-dir", "--pred-dir", "p", "--gt-dir", "g"),
                ("leaderboard", "--pred-root", "p", "--gt-dir", "g")):
        code, _out, _err = run(*cmd)
        report(f"`{cmd[0]}` refuses to run without a schema argument", code == 2,
               f"exit {code}")
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{'CLI HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
