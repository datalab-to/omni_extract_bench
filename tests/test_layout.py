#!/usr/bin/env python3
"""What must be true of the storage layout, stated as properties.

`prediction_id` is a contract that `scores` joins on, so these are not unit tests of an
implementation detail -- a change that makes one of them fail orphans every historical score.

The checks that matter most are the ones a probe found the hard way: that a stored extraction
is the vendor's bytes rather than ours, and that `verify` sees a one-byte edit, which nothing
else would.

Run: python3 tests/test_layout.py
"""
import hashlib
import json
import os as _os
import shutil
import sys as _sys
import tempfile
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench.layout import (                                   # noqa: E402
    PREDICTION_ID_VERSION, check_doc_id, check_unique, extract_result,
    prediction_id, verify)

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


print("\nSTORAGE LAYOUT\n")

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

# ── verify sees all three kinds of drift ──────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp())
try:
    payloads = tmp / "predictions"
    payloads.mkdir()
    rows = []
    for name in ("alpha", "beta", "gamma"):
        body = f'{{"doc": "{name}"}}'.encode()
        (payloads / f"{name}.json").write_bytes(body)
        rows.append({"doc_id": name, "prediction_id": prediction_id(body)})
    name_of = lambda r: r["doc_id"]

    report("a consistent tree reports nothing",
           verify(rows, payloads, "prediction_id", name_of) == [])

    body = (payloads / "alpha.json").read_bytes()
    (payloads / "alpha.json").unlink()
    probs = verify(rows, payloads, "prediction_id", name_of)
    report("a row with no file is caught",
           len(probs) == 1 and "row with no file" in probs[0], f"{probs}")
    (payloads / "alpha.json").write_bytes(body)

    (payloads / "orphan.json").write_text("{}")
    probs = verify(rows, payloads, "prediction_id", name_of)
    report("a file with no row is caught",
           len(probs) == 1 and "file with no row" in probs[0], f"{probs}")
    (payloads / "orphan.json").unlink()

    # The one nothing else would notice, and the reason the atlas carries hashes.
    original = (payloads / "beta.json").read_bytes()
    (payloads / "beta.json").write_bytes(original + b" ")
    probs = verify(rows, payloads, "prediction_id", name_of)
    report("a ONE-BYTE edit is caught",
           len(probs) == 1 and "hash mismatch" in probs[0], f"{probs}")
    (payloads / "beta.json").write_bytes(original)

    report("restoring the byte clears it",
           verify(rows, payloads, "prediction_id", name_of) == [])
finally:
    shutil.rmtree(tmp)

note("drift is checked in both directions, and by content, not just by listing")

print(f"\n{'STORAGE LAYOUT HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
