#!/usr/bin/env python3
"""What must be true of scoring a whole benchmark, stated as properties.

The one that matters most is that an unusable prediction is not graded. `prediction_io.py`
exists because that distinction was once made two different ways, and a provider read as 100%
coverage while 37 of its 45 outputs were empty. The corpus path is where that mistake is easy
to make again, because an error blob grades perfectly happily as a document where every field
is missing.

The rest are about the contract: two files per document, nothing else required. Our published
benchmark carries `source.json`, PDFs and a parquet atlas beside the documents, and none of
that may leak into what someone else's corpus has to look like.

Run: python3 tests/test_bench.py
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
from omni_extract_bench.bench import (                                     # noqa: E402
    cases, document, documents, key_of, predictions, score, summary_row, verdict_rows)

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


SCHEMA = {"type": "object", "properties": {
    "invoice_number": {"type": "string"}, "total": {"type": "number"},
    "lines": {"type": "array", "items": {"type": "object", "properties": {
        "sku": {"type": "string"}, "qty": {"type": "number"}}}}}}
GOLD = {"invoice_number": "A-1", "total": 42.0, "lines": [{"sku": "x", "qty": 2}]}


def corpus_with(**docs):
    """A corpus, declared. The atlas is what makes a directory a benchmark."""
    root = Path(tempfile.mkdtemp())
    for doc_id, gt in docs.items():
        d = root / doc_id
        d.mkdir()
        (d / "ground_truth.json").write_text(json.dumps(gt))
        (d / "schema.json").write_text(json.dumps(SCHEMA))
    corpus_atlas.write(root, corpus_atlas.discover(root))
    return root


def preds_with(**preds):
    root = Path(tempfile.mkdtemp())
    for doc_id, body in preds.items():
        (root / f"{doc_id}.json").write_text(json.dumps(body))
    return root


print("\nBENCHMARK PIPELINE\n")

# ── an unusable prediction is never graded ────────────────────────────────────────────
corpus = corpus_with(a=GOLD)
try:
    for blob, what in (({"__error__": "rate limited"}, "an error blob"),
                       ({}, "an empty object")):
        p = preds_with(a=blob)
        case = next(cases(documents(corpus), predictions(p)))
        out = score(case)
        report(f"{what} is unusable, not a document where everything is missing",
               out.kind == "unusable" and out.summary is None,
               f"kind={out.kind}")
        row = summary_row(case, out)
        report(f"{what} records no accuracy at all",
               "accuracy" not in row or row.get("accuracy") is None)
        shutil.rmtree(p)
    note("a provider once read as 100% coverage with 37 of 45 outputs empty")

    # A prediction that is merely wrong IS graded -- the point is telling them apart.
    p = preds_with(a={"invoice_number": "WRONG", "total": 42.0, "lines": []})
    case = next(cases(documents(corpus), predictions(p)))
    out = score(case)
    report("a wrong prediction is graded, not called unusable",
           out.kind == "graded" and 0 < out.summary["accuracy"] < 100,
           f"kind={out.kind}")
    shutil.rmtree(p)
finally:
    shutil.rmtree(corpus)

# ── the contract is two files, and nothing else ───────────────────────────────────────
corpus = corpus_with(a=GOLD)
try:
    # Everything the published benchmark carries alongside, which no one else should need.
    (corpus / "a" / "source.json").write_text('{"suite": "ours"}')
    (corpus / "a" / "document.pdf").write_bytes(b"%PDF-1.4 not really")
    docs = list(documents(corpus))
    report("extra files beside a document are ignored", len(docs) == 1)
    report("a file in the corpus root is not mistaken for a document",
           [d.doc_id for d in docs] == ["a"])
    note("only ground_truth.json and schema.json are hashed; the rest is ours to carry")

    (corpus / "b").mkdir()
    (corpus / "b" / "ground_truth.json").write_text("{}")
    report("scoring follows the atlas, not a directory listing",
           [d.doc_id for d in documents(corpus)] == ["a"])
    note("so a half-copied document cannot join a benchmark by being present")
    try:
        corpus_atlas.discover(corpus)
        report("a document without a schema stops a rebuild", False, "accepted")
    except FileNotFoundError as exc:
        report("a document without a schema stops a rebuild", "schema.json" in str(exc))
        note("a schema is required and never inferred")
finally:
    shutil.rmtree(corpus)

# ── identity records what was scored ──────────────────────────────────────────────────
corpus = corpus_with(a=GOLD)
p = preds_with(a=GOLD)
try:
    case = next(cases(documents(corpus), predictions(p)))
    report("a case is keyed by the document and the prediction's bytes",
           set(key_of(case)) == {"doc_id", "prediction_id"}, str(sorted(key_of(case))))

    # A row naming a file that is not there is a stop: the run follows the atlas, and a row
    # pointing at nothing is not something to quietly skip.
    (corpus / "a" / "ground_truth.json").unlink()
    try:
        list(documents(corpus))
        report("a row naming a missing file stops the run", False, "accepted silently")
    except FileNotFoundError as exc:
        report("a row naming a missing file stops the run", "missing" in str(exc))
        note("the atlas is what the run follows, so it has to resolve")
finally:
    shutil.rmtree(corpus); shutil.rmtree(p)

# ── predictions join by name, and say so when they do not ─────────────────────────────
corpus = corpus_with(inv_001=GOLD, inv_002=GOLD)
p = preds_with(inv_001=GOLD)
try:
    report("a document with no prediction yields no case, not a zero",
           len(list(cases(documents(corpus), predictions(p)))) == 1)
    bad = preds_with(**{"inv_003": GOLD})
    try:
        list(cases(documents(corpus), predictions(bad)))
        report("a prediction naming no document is caught", False, "accepted")
    except ValueError as exc:
        report("a prediction naming no document is caught, with near matches",
               "inv_001" in str(exc) or "inv_002" in str(exc), str(exc)[:90])
        note("the usual cause is a filename convention, so show what was close")
    shutil.rmtree(bad)
finally:
    shutil.rmtree(corpus); shutil.rmtree(p)

# ── verdicts are an audit trail, so they must round-trip ──────────────────────────────
corpus = corpus_with(a={"invoice_number": "None", "total": 42.0, "lines": []})
p = preds_with(a={"total": 42.0, "lines": []})
try:
    case = next(cases(documents(corpus), predictions(p)))
    rows = verdict_rows(case, score(case))
    by_addr = {r["address"]: r for r in rows}
    missing = by_addr["invoice_number"]
    report('a gold string "None" is not confused with a missing value',
           missing["gold_raw"] == '"None"' and missing["pred_raw"] is None,
           f"gold={missing['gold_raw']!r} pred={missing['pred_raw']!r}")
    note("str() would render both as the text None; JSON keeps them apart")
    report("every verdict row carries the key it belongs to",
           all(set(key_of(case)) <= set(r) for r in rows))

    case2 = next(cases(documents(corpus), predictions(p)))
    report("--no-verdicts produces none, and still grades",
           score(case2, verdicts=False).verdicts is None
           and score(case2, verdicts=False).summary is not None)
finally:
    shutil.rmtree(corpus); shutil.rmtree(p)

print(f"\n{'BENCHMARK PIPELINE HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
