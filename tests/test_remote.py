#!/usr/bin/env python3
"""Staging a corpus out of a bucket, against a mock one.

`oeb score` takes paths. `remote.py` is what turns a bucket into paths, and the properties
worth guarding are the ones that were wrong when it was first written:

  * **the atlas decides what comes down.** Fetching the prefix instead made a spot check of 8
    documents pull all 660 -- and on a fan-out every container would pay for the whole corpus
    to score its share. This is the test that pins it.
  * a filtered atlas in a bucket means the same thing it means locally: a subset
  * a bucket run and a local run agree, down to the prediction_id
  * PDFs are never fetched; the scorer does not open one
  * `--source` comes from the prefix, not from the temporary directory it was staged into
  * a destination that already holds a run is refused BEFORE anything transfers

Run: uv run --with moto python tests/test_remote.py
"""
import json
import os as _os
import shutil
import sys as _sys
import tempfile
from pathlib import Path

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)

try:
    from moto import mock_aws
except ImportError:                                                         # pragma: no cover
    print("  SKIP  tests/test_remote.py needs moto:\n"
          "        uv run --with moto python tests/test_remote.py")
    _sys.exit(0)

# Moto refuses to run against real credentials, and must never see any.
for _v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    _os.environ[_v] = "testing"
for _v in ("AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL", "S3_ENDPOINT_URL", "AWS_PROFILE"):
    _os.environ.pop(_v, None)
_os.environ["AWS_REGION"] = _os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

from omni_extract_bench import remote, s3                                   # noqa: E402

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


print("\nA PREFIX IS READ AS A PATH, NOT AS A STRING")
report("an s3:// value is recognised and a local one is not",
       s3.is_uri("s3://b/k") and not s3.is_uri("/tmp/x") and not s3.is_uri(None))
report("bucket and key come apart", s3.parse("s3://bench/runs/x") == ("bench", "runs/x"))
report("a trailing slash is not a key segment", s3.parse("s3://bench/runs/") == ("bench", "runs"))
try:
    s3.parse("s3://")
    report("a bucketless URI is refused", False, "no error raised")
except ValueError as exc:
    report("a bucketless URI is refused, saying so", "names no bucket" in str(exc))

print("\nTHE SOURCE STAMP COMES FROM WHAT YOU TYPED")
report("an s3 prefix stamps its last segment",
       remote.source_name("s3://bench/vendors/datalab") == "datalab")
report("a trailing slash changes nothing",
       remote.source_name("s3://bench/vendors/datalab/") == "datalab")
note("the stamp labels a vendor; a temporary directory's name would be a silent rename")

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"},
                                           "n": {"type": "number"}}}

with mock_aws():
    import boto3

    cl = boto3.client("s3", region_name="us-east-1")
    cl.create_bucket(Bucket="bench")
    TMP = Path(tempfile.mkdtemp())
    try:
        corpus, preds = TMP / "corpus", TMP / "preds"
        preds.mkdir(parents=True)
        for doc_id in ("one", "two", "three"):
            (corpus / doc_id).mkdir(parents=True)
            (corpus / doc_id / "ground_truth.json").write_text(
                json.dumps({"name": "Acme", "n": 2}))
            (corpus / doc_id / "schema.json").write_text(json.dumps(SCHEMA))
            (corpus / doc_id / "document.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 4000)
            (preds / f"{doc_id}.json").write_text(json.dumps({"name": "Acme", "n": 2}))
        run("build-corpus", "--corpus", corpus)
        s3.upload(corpus, "s3://bench/corpus")
        s3.upload(preds, "s3://bench/vendors/datalab")

        print("\nTHE ATLAS DECIDES WHAT COMES DOWN")
        import pyarrow as pa
        import pyarrow.parquet as pq

        from omni_extract_bench import corpus as corpus_atlas
        rows = pq.read_table(corpus / corpus_atlas.ATLAS).to_pylist()
        pq.write_table(pa.Table.from_pylist([r for r in rows if r["doc_id"] == "one"]),
                       corpus / "just-one.parquet")
        s3.upload(corpus / "just-one.parquet", "s3://bench/corpus/just-one.parquet")

        into = TMP / "staged"
        atlas = remote.stage_corpus("s3://bench/corpus/just-one.parquet", into)
        staged = sorted(p.relative_to(into).as_posix() for p in into.rglob("*") if p.is_file())
        report("a one-row atlas stages one document, not the whole prefix",
               staged == ["just-one.parquet", "one/ground_truth.json", "one/schema.json"],
               str(staged))
        note("this is the bug that made a spot check of 8 documents pull 132 MB")
        report("and no PDF, though all three are in the bucket",
               not any(p.endswith(".pdf") for p in staged))

        pdfs = [o["Key"] for o in cl.list_objects_v2(Bucket="bench", Prefix="corpus")["Contents"]
                if o["Key"].endswith(".pdf")]
        report("the PDFs really are there to have been fetched", len(pdfs) == 3, str(pdfs))

        pdir, extra = remote.stage_predictions("s3://bench/vendors/datalab", TMP / "sp",
                                               {e.doc_id for e in corpus_atlas.read(atlas)})
        got = sorted(p.name for p in pdir.iterdir())
        report("only the prediction the atlas asks for is fetched", got == ["one.json"], str(got))
        report("the rest are named, not silently dropped",
               extra == ["three", "two"], str(extra))
        note("a misnamed prediction has to stay visible even though it is never downloaded")

        print("\nA BUCKET RUN IS THE SAME RUN")
        code = remote.score("s3://bench/corpus", "s3://bench/vendors/datalab",
                            "s3://bench/scores/datalab", jobs=1)
        report("scoring straight from the bucket works", code == 0, f"exit {code}")
        landed = {o["Key"] for o in
                  cl.list_objects_v2(Bucket="bench", Prefix="scores/datalab")["Contents"]}
        report("summary and one verdict file per document land in the bucket",
               {"scores/datalab/summary.parquet", "scores/datalab/verdicts/one.parquet",
                "scores/datalab/verdicts/two.parquet",
                "scores/datalab/verdicts/three.parquet"} <= landed, str(sorted(landed)))

        run("score", "--corpus", corpus, "--predictions", preds, "--out", TMP / "local",
            "--source", "datalab")
        back = TMP / "back"
        s3.fetch("s3://bench/scores/datalab", back, ["summary.parquet"])
        keyed = lambda rs: {r["doc_id"]: (r["kind"], r["accuracy"], r["prediction_id"])
                            for r in rs}
        report("every document scores identically to the local run, down to prediction_id",
               keyed(pq.read_table(back / "summary.parquet").to_pylist())
               == keyed(pq.read_table(TMP / "local/summary.parquet").to_pylist()))
        note("the bucket is where the bytes came from, not a different scorer")

        print("\nA TAKEN DESTINATION IS REFUSED BEFORE ANYTHING MOVES")
        code = remote.score("s3://bench/corpus", "s3://bench/vendors/datalab",
                            "s3://bench/scores/datalab", jobs=1)
        report("scoring into a finished run stops", code == 1, f"exit {code}")
        try:
            remote.score(str(corpus), "s3://bench/vendors/datalab", "s3://bench/scores/x")
            report("a local path here is refused", False, "no error raised")
        except ValueError as exc:
            report("a local path here is refused, pointing at `oeb score`",
                   "oeb score" in str(exc), str(exc)[:160])

        print("\nONE FILE UPLOADS TO ONE KEY")
        report("a single file is one object, not a silent no-op",
               s3.upload(corpus / "just-one.parquet", "s3://bench/corpus/copy.parquet")[0] == 1)
        try:
            s3.upload(TMP / "nothing-here", "s3://bench/x")
            report("an empty upload raises rather than reporting success", False, "no error")
        except FileNotFoundError:
            report("an empty upload raises rather than reporting success", True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{'REMOTE HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
