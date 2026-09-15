#!/usr/bin/env python3
"""The viewer's generator: what it writes, and what it refuses to.

`oeb ui` exists because the first version of this viewer inlined every address into the page
and stopped working somewhere past forty documents. The properties worth guarding are the ones
that make 660 documents open at all:

  * the index carries no addresses -- it is what you load to CHOOSE a document
  * a document's addresses are one fetchable file, and the union across vendors
  * a literal match does not repeat the value it matched, which is most of the payload
  * two runs cannot both claim the same vendor name, because the viewer labels columns with it

Run: python3 tests/test_ui.py
"""
import json
import os as _os
import shutil
import sys as _sys
import tempfile
from pathlib import Path

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
from omni_extract_bench import ui                                           # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


def run(*argv):
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


print("\nADDRESSES ARE READ IN THE ORDER A PERSON READS THE TABLE")
rows = ["lines[10].sku", "lines[2].sku", "lines[1].sku", "total"]
report("row 2 sorts before row 10, which lexical order gets wrong",
       sorted(rows, key=ui.natural) == ["lines[1].sku", "lines[2].sku", "lines[10].sku", "total"],
       str(sorted(rows, key=ui.natural)))
note("the sidebar is the reading order; interleaving a table with itself makes it unusable")

print("\nA LITERAL MATCH DOES NOT REPEAT THE VALUE IT MATCHED")
report('an identical match is ["m"], with no second slot',
       ui.cell("match", '"Acme"', '"Acme"') == ["m"])
report('a folded match keeps the prediction, because that is the whole point of the bucket',
       ui.cell("match", '"1250.00"', '"1,250.00"') == ["m", '"1,250.00"'])
report("a charge always carries what was predicted, including nothing",
       ui.cell("missing", '"Acme"', None) == ["x", None]
       and ui.cell("wrong value", '"a"', '"b"') == ["w", '"b"'])
report("a verdict this module has no code for passes through as itself",
       ui.cell("something new", None, '"x"') == ["something new", '"x"'])
note("an unknown verdict must show up wrong in the viewer, not vanish from it")

print("\nTHE SAME TEXT CHARGED TWICE IS ONE PAIRING FAILURE")
verds = [{"verdict": "missing", "gold": json.dumps("Wear safety boots"), "pred": None},
         {"verdict": "invented item", "gold": None, "pred": json.dumps("• Wear safety boots")},
         {"verdict": "missing", "gold": json.dumps("Unrelated"), "pred": None}]
near = ui.near_misses(verds)
report("a bullet glyph in front of a sentence is found as the near miss it is",
       len(near) == 1 and near[0]["gold"] == "Wear safety boots", str(near))
report("a genuinely missing value is not paired with anything",
       all(n["gold"] != "Unrelated" for n in near))
note("`loose` answers only this question; `canon_key` stays the one function that scores")

TMP = Path(tempfile.mkdtemp())
try:
    corpus, preds, preds2 = TMP / "corpus", TMP / "a", TMP / "b"
    preds.mkdir()
    preds2.mkdir()
    schema = {"type": "object", "properties": {"name": {"type": "string"},
                                               "n": {"type": "number"}}}
    for doc_id in ("one", "two"):
        (corpus / doc_id).mkdir(parents=True)
        (corpus / doc_id / "ground_truth.json").write_text(json.dumps({"name": "Acme", "n": 2}))
        (corpus / doc_id / "schema.json").write_text(json.dumps(schema))
        (corpus / doc_id / "document.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        (preds / f"{doc_id}.json").write_text(json.dumps({"name": "Acme", "n": 2}))
        (preds2 / f"{doc_id}.json").write_text(json.dumps({"name": "Acme", "n": 9}))
    run("build-corpus", "--corpus", corpus)
    run("score", "--corpus", corpus, "--predictions", preds, "--out", TMP / "run-a",
        "--source", "alpha")
    run("score", "--corpus", corpus, "--predictions", preds2, "--out", TMP / "run-b",
        "--source", "beta")

    print("\nTHE SITE IS AN INDEX PLUS ONE FILE PER DOCUMENT")
    site = TMP / "site"
    code, out, err = run("ui", "--run", TMP / "run-a", "--run", TMP / "run-b",
                         "--corpus", corpus, "--out", site)
    report("it writes a site from two runs", code == 0, (out + err).strip()[:300])
    for name in ("index.html", "index.js", "doc/one.json", "schema/one.json", "pdf/one.pdf"):
        report(f"{name} is there", (site / name).exists())
    report("the schema and the PDF are symlinks, not 774 MB of copies",
           (site / "schema/one.json").is_symlink() and (site / "pdf/one.pdf").is_symlink())

    index = json.loads((site / "index.js").read_text().split("= ", 1)[1].rstrip(";\n"))
    report("the index names both sources, in the order they were given",
           index["sources"] == ["alpha", "beta"], str(index["sources"]))
    report("the index carries NO addresses -- that is what makes it loadable at 660 documents",
           not any("addrs" in d for d in index["docs"]), str(index["docs"])[:200])
    report("every document carries a score per source",
           all(set(d["by_source"]) == {"alpha", "beta"} for d in index["docs"]))

    doc = json.loads((site / "doc/one.json").read_text())
    got = {a[0]: a[2] for a in doc["addrs"]}
    report("a document's file holds every address, with each source's answer",
           set(got) == {"name", "n"}, str(sorted(got)))
    report("a literal match stores one slot; a charge stores the value and its canonical form",
           got["n"]["alpha"] == ["m"] and got["n"]["beta"][:2] == ["w", "9"]
           and len(got["n"]["beta"]) == 3, str(got["n"]))
    note("at nine vendors most addresses match all nine; repeating the value is most of the file")
    only_match = next(r for r in doc["addrs"] if r[0] == "name")
    report("and an address every vendor matched literally carries no canonical form either",
           len(only_match) == 3, str(only_match))
    note("the canonical forms answer 'why did these count as equal'; where nothing disagreed "
         "there is no question to answer")

    print("\nTWO RUNS CANNOT BOTH BE 'alpha'")
    code, out, err = run("ui", "--run", TMP / "run-a", "--run", TMP / "run-a",
                         "--corpus", corpus, "--out", TMP / "site2")
    report("a repeated source is refused, and says to name them apart",
           code == 1 and "--source" in err, (out + err).strip()[:250])
    note("the viewer labels its columns with the source; two of one name loses a column")
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{'THE VIEWER HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
