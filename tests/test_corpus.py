#!/usr/bin/env python3
"""What must be true of the corpus and of prediction identity, stated as properties.

`prediction_id` is a contract that `scores` joins on, so these are not unit tests of an
implementation detail -- a change that makes one of them fail orphans every historical score.

The checks that matter most are the ones a probe found the hard way: that a stored extraction
is the vendor's bytes rather than ours, and that `verify` sees a one-byte edit, which nothing
else would.

Run: python3 tests/test_corpus.py
"""
import hashlib
import json
import os as _os
import shutil
import sys as _sys
import tempfile
from pathlib import Path

_ROOT_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT_DIR)
_sys.path.insert(0, _os.path.join(_ROOT_DIR, "scripts"))
from omni_extract_bench.bench import PREDICTION_ID_VERSION, prediction_id  # noqa: E402
from omni_extract_bench.corpus import check_doc_id, check_unique, verify   # noqa: E402

_sys.path.insert(0, _os.path.join(_ROOT_DIR, "scripts")) if False else None
from build_vendors import extract_result                                   # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


print("\nCORPUS AND IDENTITY\n")

# ── the extraction is the vendor's bytes, not ours ────────────────────────────────────
# The whole reason extract_result exists. Vendors write `", "`; json.dumps writes `","`.
envelope = '{"result": {"b": 1, "a": [1, 2]}, "_secs": 92.3}'
span = extract_result(envelope)
report("the extracted span is the vendor's own formatting",
       span == '{"b": 1, "a": [1, 2]}', f"got {span!r}")
report("...and it is NOT what reserialising would produce",
       span != json.dumps(json.loads(span), separators=(",", ":")),
       "reserialising happened to match; the test cannot see the difference")
report("the span still parses to the same value",
       json.loads(span) == {"b": 1, "a": [1, 2]})

# Key order is preserved, which a canonicalising hash would destroy.
report("key order survives, so two orderings are different predictions",
       extract_result('{"result": {"a": 1, "b": 2}}')
       != extract_result('{"result": {"b": 2, "a": 1}}'))

# ── shapes that must fail rather than guess ───────────────────────────────────────────
bad = []
for text, why in [('{"_secs": 1}', "no result key"),
                  ('{"result": }', "unparseable value"),
                  ('{}', "empty object")]:
    try:
        extract_result(text)
        bad.append(why)
    except ValueError:
        pass
report("a payload with no usable result raises instead of inventing an id",
       not bad, f"silently accepted: {bad}")

# ── prediction_id ─────────────────────────────────────────────────────────────────────
body = b'{"a": 1}'
report("prediction_id is a plain sha256 of the stored bytes",
       prediction_id(body) == hashlib.sha256(body).hexdigest())
report("...full digest, not truncated", len(prediction_id(body)) == 64)

# _secs differing must not change the id -- the property the whole scheme exists for.
a = extract_result('{"result": {"x": 1}, "_secs": 92.3}')
b = extract_result('{"result": {"x": 1}, "_secs": 1804.0}')
report("a re-run with a different wall time keeps its id",
       prediction_id(a.encode()) == prediction_id(b.encode()))

# ...and so must a metadata key nobody has seen yet. This is the one that bit twice.
c = extract_result('{"result": {"x": 1}, "_secs": 5, "some_future_flag": true}')
report("an unknown envelope key does not change the id either",
       prediction_id(a.encode()) == prediction_id(c.encode()))

report("the version is recorded, so a change is visible in the data",
       PREDICTION_ID_VERSION == "v1")

# ── doc_id safety, now that the suite directory is gone ───────────────────────────────
ok_ids = ["research__[zhao25] a survey of LLMs", "short__02-10121 H-12 7-2-2024 F-22628",
          "m1__Manufacturing_Industry_Datasets"]
bad_ids = ["", ".hidden", "a/b", "a\\b"]
survived = [d for d in ok_ids if (check_doc_id(d) or True)]
report("real doc_ids with spaces and brackets are accepted", len(survived) == len(ok_ids))
rejected = []
for d in bad_ids:
    try:
        check_doc_id(d)
    except ValueError:
        rejected.append(d)
report("ids that would collide with a path, or hide, are rejected",
       len(rejected) == len(bad_ids), f"accepted: {set(bad_ids) - set(rejected)}")

try:
    check_unique(["a", "b", "a"])
    report("a duplicate doc_id is a build error", False, "accepted silently")
except ValueError:
    report("a duplicate doc_id is a build error", True)

# ── verify sees all three kinds of drift, in BOTH layouts ─────────────────────────────
# The first version guessed that a payload lived at `<row_id>.json`, which made it useless
# for the corpus tree, where a document is a directory holding several files. Paths come
# from the caller now, so one function serves both shapes.
tmp = Path(tempfile.mkdtemp())
try:
    # the corpus shape: a directory per document, several payloads inside
    corpus = tmp / "corpus"
    expected = []
    for name in ("alpha", "beta"):
        d = corpus / name
        d.mkdir(parents=True)
        for payload in ("ground_truth.json", "schema.json"):
            body = f'{{"{name}": "{payload}"}}'.encode()
            (d / payload).write_bytes(body)
            expected.append((f"{name}/{payload}", hashlib.sha256(body).hexdigest()))

    report("a consistent corpus tree reports nothing", verify(expected, corpus) == [])

    body = (corpus / "alpha" / "schema.json").read_bytes()
    (corpus / "alpha" / "schema.json").unlink()
    probs = verify(expected, corpus)
    report("a row with no file is caught, nested one level down",
           len(probs) == 1 and "alpha/schema.json" in probs[0], f"{probs}")
    (corpus / "alpha" / "schema.json").write_bytes(body)

    (corpus / "beta" / "stray.json").write_text("{}")
    probs = verify(expected, corpus)
    report("a file with no row is caught",
           len(probs) == 1 and "beta/stray.json" in probs[0], f"{probs}")
    (corpus / "beta" / "stray.json").unlink()

    # The one nothing else would notice, and the reason the atlas carries hashes.
    original = (corpus / "beta" / "ground_truth.json").read_bytes()
    (corpus / "beta" / "ground_truth.json").write_bytes(original + b" ")
    probs = verify(expected, corpus)
    report("a ONE-BYTE edit is caught",
           len(probs) == 1 and "hash mismatch" in probs[0], f"{probs}")
    (corpus / "beta" / "ground_truth.json").write_bytes(original)
    report("restoring the byte clears it", verify(expected, corpus) == [])

    # an atlas living inside its own tree must not read as an orphan
    (corpus / "corpus.parquet").write_bytes(b"not json")
    report("a non-payload file in the tree is ignored", verify(expected, corpus) == [])

    # A payload type left out of `patterns` is invisible the same way the atlas is, so
    # every row claiming one reads as a missing file. This shipped as a real bug: the
    # corpus builder emitted PDFs and verified only *.json, and reported all 660 as absent
    # while they sat on disk.
    (corpus / "alpha" / "document.pdf").write_bytes(b"%PDF-1.4 fake")
    with_pdf = expected + [("alpha/document.pdf",
                            hashlib.sha256(b"%PDF-1.4 fake").hexdigest())]
    probs = verify(with_pdf, corpus)
    report("a payload type missing from `patterns` reads as absent -- the trap",
           len(probs) == 1 and "row with no file" in probs[0], f"{probs}")
    report("...and naming the pattern fixes it",
           verify(with_pdf, corpus, ("**/*.json", "**/*.pdf")) == [])
    (corpus / "alpha" / "document.pdf").unlink()

    # the predictions shape: one flat file per document, same function
    preds = tmp / "predictions"
    preds.mkdir()
    flat = []
    for name in ("alpha", "beta"):
        body = f'{{"doc": "{name}"}}'.encode()
        (preds / f"{name}.json").write_bytes(body)
        flat.append((f"{name}.json", prediction_id(body)))
    report("the same function verifies the flat predictions shape",
           verify(flat, preds) == [])
finally:
    shutil.rmtree(tmp)

note("drift is checked in both directions, and by content, not just by listing")

# ── one definition of identity, not two ───────────────────────────────────────────────
# Four leaderboard bugs came from several implementations of "are these two equal?" that had
# to agree and did not. `prediction_id` is what `scores` joins on, so a second copy of it is
# the same mistake in the same place.
import omni_extract_bench.bench as _bench                                 # noqa: E402
import omni_extract_bench.corpus as _corpus                               # noqa: E402

report("prediction identity is defined exactly once",
       _bench.prediction_id is prediction_id
       and not hasattr(_corpus, "prediction_id"))
note("a join key with two definitions is a disagreement waiting to be written down")

print(f"\n{'CORPUS AND IDENTITY HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
