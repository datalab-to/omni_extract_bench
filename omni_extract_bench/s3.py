"""`s3://` for the paths `oeb score` takes.

    oeb score --corpus s3://bench/corpus --predictions s3://bench/preds/datalab \
              --out s3://bench/runs/datalab

**Staged, not streamed.** A bucket is downloaded to a temporary directory, scored exactly as a
local corpus is, and the finished run is uploaded. This module is the only place that knows
what a bucket is; everything below it takes a `Path` and cannot tell the difference.

That is the point. Three things make a streaming object-store filesystem the wrong first
version here:

* **Scoring is a process pool.** A streamed path would need a client per worker and
  credentials crossing the fork, on the code path where a mistake is silent.
* **A run is written atomically, by rename.** `write_run` writes a temporary file and renames
  it so a crash leaves the previous table rather than a truncated one. Object stores have no
  rename. Staging keeps that property and uploads a run that is already complete.
* **The corpus is mostly bytes the scorer never reads.** A corpus prefix can hold 774 MB of
  PDFs; the two files a document is are 160 KB. Downloads are filtered, the same way
  `published_corpus` filters its HuggingFace snapshot.

What it costs is disk and one transfer up front. What it buys is that local scoring cannot be
broken by a change to this file.

**Credentials** come from the environment, the way every other AWS tool takes them --
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, a profile, or an instance role. Nothing is read
from a config of ours and nothing is ever printed. For an S3-compatible store (R2, MinIO,
Ceph) pass `--endpoint-url`, or set `AWS_ENDPOINT_URL`; those stores ignore the region, and
`auto` is sent when an endpoint is given and no region is set.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator

SCHEME = "s3://"

#: Environment variables naming an S3-compatible endpoint, in the order they are consulted.
#: `AWS_ENDPOINT_URL` is the one the AWS SDKs themselves read; the others are conventions that
#: predate it and cost nothing to honour.
ENDPOINT_VARS = ("AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL", "S3_ENDPOINT_URL")

#: What a corpus prefix is worth downloading: its atlas, and the two files a document is.
#: A corpus may hold PDFs and whatever else the collection carried; the scorer opens neither.
CORPUS_FILES = ("ground_truth.json", "schema.json")

#: What a prediction prefix is worth downloading: one JSON per document, and the optional
#: sidecar of what the vendor recorded about producing them.
PREDICTION_SIDECAR = "predictions.parquet"


def is_uri(value: object) -> bool:
    """Whether a `--corpus`, `--predictions` or `--out` names a bucket rather than a directory."""
    return isinstance(value, str) and value.startswith(SCHEME)


def parse(uri: str) -> tuple[str, str]:
    """`s3://bucket/some/prefix` -> `("bucket", "some/prefix")`.

    Raises:
        ValueError: for a URI with no bucket, which is almost always a typed `s3:/`.
    """
    rest = uri[len(SCHEME):].strip("/")
    if not rest:
        raise ValueError(f"{uri!r} names no bucket")
    bucket, _, key = rest.partition("/")
    return bucket, key


def client(endpoint: str | None = None):
    """A boto3 S3 client, for AWS or for anything that speaks its API.

    Raises:
        SystemExit: when boto3 is not installed, naming the extra that provides it. A missing
            optional dependency is a setup mistake, not a traceback.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:                                     # pragma: no cover - env specific
        raise SystemExit(
            "s3:// paths need boto3: pip install 'omni-extract-bench[s3]', "
            "or pass local directories instead")
    endpoint = endpoint or next((os.environ[v] for v in ENDPOINT_VARS if os.environ.get(v)),
                                None)
    kwargs = {"config": Config(retries={"max_attempts": 5, "mode": "standard"})}
    if endpoint:
        kwargs["endpoint_url"] = endpoint
        # R2 and friends ignore the region but the signer still needs one.
        kwargs["region_name"] = os.environ.get("AWS_REGION") or "auto"
    return boto3.client("s3", **kwargs)


def keys(s3, uri: str) -> Iterator[tuple[str, int]]:
    """Every object under a prefix, as `(key, size)`. Paginated, because a corpus is 660 of them."""
    bucket, prefix = parse(uri)
    token = None
    while True:
        page = s3.list_objects_v2(Bucket=bucket, Prefix=prefix,
                                  **({"ContinuationToken": token} if token else {}))
        for obj in page.get("Contents", ()):
            yield obj["Key"], obj["Size"]
        if not page.get("IsTruncated"):
            return
        token = page["NextContinuationToken"]


def exists(uri: str, endpoint: str | None = None) -> bool:
    """Whether one object is there. Used to refuse a run whose destination is already taken --
    before an hour of scoring, rather than after it.
    """
    from botocore.exceptions import ClientError

    bucket, key = parse(uri)
    try:
        client(endpoint).head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def corpus_wanted(rel: str) -> bool:
    """A corpus object worth downloading: an atlas at the top, or a document's two files."""
    parts = PurePosixPath(rel).parts
    if len(parts) == 1:
        return parts[0].endswith(".parquet")
    return len(parts) == 2 and parts[1] in CORPUS_FILES


def predictions_wanted(rel: str) -> bool:
    """A prediction object worth downloading: `<doc_id>.json`, or the sidecar beside them."""
    parts = PurePosixPath(rel).parts
    return len(parts) == 1 and (parts[0].endswith(".json") or parts[0] == PREDICTION_SIDECAR)


def download(uri: str, into: Path, wanted: Callable[[str], bool] | None = None,
             endpoint: str | None = None) -> tuple[int, int]:
    """Copy a prefix into a local directory, keeping its shape. Returns `(files, bytes)`.

    `wanted` is asked about each key RELATIVE to the prefix, so a caller filters on the layout
    it expects rather than on the bucket's absolute keys.

    Raises:
        FileNotFoundError: when the prefix holds nothing wanted. An empty download that scored
            zero documents would otherwise look like a corpus with nothing in it.
    """
    s3 = client(endpoint)
    bucket, prefix = parse(uri)
    into.mkdir(parents=True, exist_ok=True)
    files = total = 0
    for key, size in keys(s3, uri):
        rel = key[len(prefix):].lstrip("/") if prefix else key
        if not rel or (wanted and not wanted(rel)):
            continue
        dest = into / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))
        files += 1
        total += size
    if not files:
        raise FileNotFoundError(f"{uri} holds nothing to read")
    return files, total


def upload(local: Path, uri: str, endpoint: str | None = None) -> tuple[int, int]:
    """Copy a finished directory to a prefix, keeping its shape. Returns `(files, bytes)`.

    Called once, on a run that is already complete. There is no partial state to reason about:
    either the scoring finished and this uploads it, or it did not and nothing was written.
    """
    s3 = client(endpoint)
    bucket, prefix = parse(uri)
    files = total = 0
    for path in sorted(p for p in local.rglob("*") if p.is_file()):
        rel = path.relative_to(local).as_posix()
        s3.upload_file(str(path), bucket, f"{prefix}/{rel}" if prefix else rel)
        files += 1
        total += path.stat().st_size
    return files, total
