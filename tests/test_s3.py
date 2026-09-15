#!/usr/bin/env python3
"""`s3://` for `score`'s three paths, against a mock bucket.

The point of the S3 path is that a run happens next to the predictions instead of on a laptop
that has to download them first. What has to hold for that to be worth having:

  * a corpus and a prediction set in a bucket score to the same thing local ones do
  * the finished run lands in the bucket, whole
  * a corpus prefix's PDFs are NOT downloaded -- they are most of its bytes and the scorer
    never opens one
  * `--source` defaults to the prefix's last segment, not the temporary directory it was
    staged into, because that stamp is what labels a vendor
  * a destination that already holds a run is refused BEFORE anything is transferred

Run: uv run --with moto python tests/test_s3.py
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
    print("  SKIP  tests/test_s3.py needs moto:\n"
          "        uv run --with moto python tests/test_s3.py")
    _sys.exit(0)

# Moto refuses to run against real credentials, and must never see any.
for _v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    _os.environ[_v] = "testing"
for _v in ("AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL", "S3_ENDPOINT_URL", "AWS_PROFILE"):
    _os.environ.pop(_v, None)
_os.environ["AWS_REGION"] = _os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

from omni_extract_bench import s3                                           # noqa: E402
from omni_extract_bench.run import source_name                              # noqa: E402

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
report("bucket and key come apart", s3.parse("s3://bench/runs/datalab") == ("bench", "runs/datalab"))
report("a trailing slash is not a key segment", s3.parse("s3://bench/runs/") == ("bench", "runs"))
try:
    s3.parse("s3://")
    report("a bucketless URI is refused", False, "no error raised")
except ValueError as exc:
    report("a bucketless URI is refused, saying so", "names no bucket" in str(exc))

print("\nTHE SOURCE STAMP COMES FROM WHAT YOU TYPED")
report("an s3 prefix stamps its last segment",
       source_name("s3://bench/preds/datalab") == "datalab")
report("a trailing slash changes nothing",
       source_name("s3://bench/preds/datalab/") == "datalab")
report("a local directory still stamps its name", source_name("preds/reducto") == "reducto")
note("the stamp labels a vendor; a temporary directory's name would be a silent rename")

print("\nONLY WHAT THE SCORER READS IS DOWNLOADED")
report("a corpus keeps its atlas and a document's two files",
       all(s3.corpus_wanted(k) for k in
           ("corpus.parquet", "invoices-only.parquet", "one/ground_truth.json",
            "one/schema.json")))
report("and NOT the PDFs, which are most of a corpus's bytes",
       not any(s3.corpus_wanted(k) for k in
               ("one/document.pdf", "one/source.json", "one/pages/1.png")))
report("a prediction prefix keeps <doc_id>.json and the optional sidecar",
       s3.predictions_wanted("one.json") and s3.predictions_wanted("predictions.parquet")
       and not s3.predictions_wanted("logs/one.json"))

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
        for doc_id in ("one", "two"):
            (corpus / doc_id).mkdir(parents=True)
            (corpus / doc_id / "ground_truth.json").write_text(
                json.dumps({"name": "Acme", "n": 2}))
            (corpus / doc_id / "schema.json").write_text(json.dumps(SCHEMA))
            (corpus / doc_id / "document.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 5000)
            (preds / f"{doc_id}.json").write_text(json.dumps({"name": "Acme", "n": 2}))
        run("build-corpus", "--corpus", corpus)

        # The same bytes, local and in the bucket.
        s3.upload(corpus, "s3://bench/corpus")
        s3.upload(preds, "s3://bench/preds/datalab")

        print("\nSCORING FROM A BUCKET, TO A BUCKET")
        code, out, err = run("score", "--corpus", "s3://bench/corpus",
                             "--predictions", "s3://bench/preds/datalab",
                             "--out", "s3://bench/runs/datalab", "--jobs", 1)
        report("it runs", code == 0, (out + err).strip()[:400])
        report("both documents were graded", "{'graded': 2}" in out, out.strip()[:300])
        report("the vendor is stamped from the prefix, not from a temporary directory",
               "source datalab" in out, out.strip()[:200])

        landed = {o["Key"] for o in
                  cl.list_objects_v2(Bucket="bench", Prefix="runs/datalab").get("Contents", ())}
        report("summary.parquet landed in the bucket",
               "runs/datalab/summary.parquet" in landed, str(sorted(landed)))
        report("so did one verdict file per document",
               {"runs/datalab/verdicts/one.parquet",
                "runs/datalab/verdicts/two.parquet"} <= landed, str(sorted(landed)))

        # No PDF was ever transferred, though both documents have one in the bucket.
        report("the PDFs are in the bucket to begin with",
               any(k.endswith(".pdf") for k in
                   (o["Key"] for o in cl.list_objects_v2(Bucket="bench",
                                                         Prefix="corpus").get("Contents", ()))))
        # 7 objects are in the corpus prefix: the atlas, and three files per document.
        # Five are worth fetching. The count staging printed is the check.
        fetched = next((l for l in out.splitlines() if "from s3://bench/corpus" in l), "")
        report("but staging fetched 5 of the prefix's 7 objects, leaving both PDFs",
               "corpus: 5 files" in fetched, fetched.strip() or out.strip()[:300])

        print("\nA PREFIX CAN NAME ONE ATLAS, THE WAY A LOCAL PATH CAN")
        import pyarrow as pa
        import pyarrow.parquet as pq

        from omni_extract_bench import corpus as corpus_atlas
        full = pq.read_table(corpus / corpus_atlas.ATLAS).to_pylist()
        pq.write_table(pa.Table.from_pylist([r for r in full if r["doc_id"] == "one"]),
                       corpus / "just-one.parquet")
        report("one file uploads to one key, rather than silently doing nothing",
               s3.upload(corpus / "just-one.parquet",
                         "s3://bench/corpus/just-one.parquet") == (1, (corpus / "just-one.parquet").stat().st_size))
        code, out, err = run("score", "--corpus", "s3://bench/corpus/just-one.parquet",
                             "--predictions", "s3://bench/preds/datalab",
                             "--out", "s3://bench/runs/subset2", "--jobs", 1)
        report("a filtered atlas in a bucket scores only its rows",
               code == 0 and "1 documents in the atlas" in out, (out + err).strip()[:400])
        report("and the prediction the atlas drops is skipped, not an error",
               "1 prediction(s) skipped" in out, out.strip()[:400])
        note("filtering the atlas is how a subset is chosen; a bucket must not change that")

        print("\nTHE ANSWER IS THE SAME ONE A LOCAL RUN GIVES")
        code, lout, _e = run("score", "--corpus", corpus, "--predictions", preds,
                             "--out", TMP / "local", "--source", "datalab", "--jobs", 1)
        import pyarrow.parquet as pq

        got = TMP / "got"
        s3.download("s3://bench/runs/datalab", got)
        remote_rows = pq.read_table(got / "summary.parquet").to_pylist()
        local_rows = pq.read_table(TMP / "local/summary.parquet").to_pylist()
        keyed = lambda rs: {r["doc_id"]: (r["kind"], r["accuracy"], r["prediction_id"])
                            for r in rs}
        report("every document scores identically, down to the prediction_id",
               keyed(remote_rows) == keyed(local_rows),
               f"{keyed(remote_rows)}\n          {keyed(local_rows)}")
        note("the bucket is where the bytes came from, not a different scorer")

        print("\nA TAKEN DESTINATION IS REFUSED BEFORE ANYTHING MOVES")
        code, out, err = run("score", "--corpus", "s3://bench/corpus",
                             "--predictions", "s3://bench/preds/datalab",
                             "--out", "s3://bench/runs/datalab", "--jobs", 1)
        report("scoring into a finished run stops", code == 1, (out + err).strip()[:300])
        report("and says to name a different --out",
               "already have an answer" in err, err.strip()[:300])
        report("nothing was staged first -- the check is the first thing it does",
               "MB from" not in out, out.strip()[:300])
        note("an hour of scoring should not end at a name that was already taken")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{'S3 HOLDS' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
