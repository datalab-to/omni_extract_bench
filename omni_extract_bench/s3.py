"""Moving objects. Nothing about benchmarks.

Transport only: parse a URI, list a prefix, fetch named keys, send a directory. What to fetch
is `remote.py`'s decision, because "the two files a document is" is a fact about the corpus
contract and not about S3 -- and a transport that knew it would be a second place the layout
is written down.

`oeb score` does not import this. It takes paths and returns a run, and a container reading a
bucket is `remote.py` staging into a directory and calling it. That boundary is the point:
local scoring cannot be broken by a change to this file.

**Named, not listed, and parallel.** `fetch` is given the keys it must get, because the caller
has an atlas that already says which documents the run will follow -- listing a prefix and
taking all of it makes every container in a fan-out pay for the whole corpus to score its
share. The round trip, not the bandwidth, is what a staged corpus costs: over a thousand small
objects, which sequentially is ten minutes to score eight documents.

**Credentials** come from the environment, the way every other AWS tool takes them --
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, a profile, or an instance role. Nothing is read
from a config of ours and nothing is ever printed. For an S3-compatible store (R2, MinIO,
Ceph) set `AWS_ENDPOINT_URL` or pass one; those stores ignore the region, so `auto` is sent
when an endpoint is given and no region is set.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Iterator

SCHEME = "s3://"

#: Transfers in flight. A staged corpus is over a thousand small objects and the round trip,
#: not the bandwidth, is what it costs: sequentially that is ten minutes to score eight
#: documents. Boto3 clients are thread-safe, and the connection pool is sized to match.
WORKERS = 16

#: Environment variables naming an S3-compatible endpoint, in the order they are consulted.
#: `AWS_ENDPOINT_URL` is the one the AWS SDKs themselves read; the others are conventions that
#: predate it and cost nothing to honour.
ENDPOINT_VARS = ("AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL", "S3_ENDPOINT_URL")

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
    kwargs = {"config": Config(retries={"max_attempts": 5, "mode": "standard"},
                               max_pool_connections=WORKERS * 2)}
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


def _get(s3, bucket: str, key: str, dest: Path) -> int | None:
    """One object to one file, or None if it is not there.

    Absent is a normal answer here, not an error: an atlas may list a document a vendor never
    predicted, and the run records that as a missing prediction rather than stopping.
    """
    from botocore.exceptions import ClientError

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(bucket, key, str(dest))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    return dest.stat().st_size


def fetch_one(uri: str, dest: Path, endpoint: str | None = None) -> int | None:
    """One named object. Returns its size, or None if the key is not there."""
    bucket, key = parse(uri)
    return _get(client(endpoint), bucket, key, dest)


def fetch(uri: str, into: Path, rels: list[str],
          endpoint: str | None = None) -> tuple[int, int, list[str]]:
    """Named objects under a prefix, in parallel. Returns `(files, bytes, missing)`.

    Named rather than listed, because the caller knows what it needs: the atlas says which
    documents the run will follow, and fetching anything else is a container paying for
    another shard's work.
    """
    from concurrent.futures import ThreadPoolExecutor

    s3 = client(endpoint)
    bucket, prefix = parse(uri)

    def one(rel):
        return rel, _get(s3, bucket, f"{prefix}/{rel}" if prefix else rel, into / rel)

    files = total = 0
    missing = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for rel, size in pool.map(one, rels):
            if size is None:
                missing.append(rel)
            else:
                files += 1
                total += size
    return files, total, missing


def names(uri: str, endpoint: str | None = None) -> set[str]:
    """Every key under a prefix, RELATIVE to it. One listing, no download.

    Listing is cheap and downloading is not, so what a bucket HOLDS and what a run NEEDS are
    asked separately: the atlas decides what comes down, and this decides what to report as
    present but not asked for.
    """
    s3 = client(endpoint)
    _bucket, prefix = parse(uri)
    return {key[len(prefix):].lstrip("/") if prefix else key for key, _size in keys(s3, uri)}


def upload(local: Path, uri: str, endpoint: str | None = None) -> tuple[int, int]:
    """Copy a directory to a prefix, or one file to a key. Returns `(files, bytes)`.

    Called once, on a run that is already complete. There is no partial state to reason about:
    either the scoring finished and this uploads it, or it did not and nothing was written.

    Raises:
        FileNotFoundError: when there is nothing to send. Returning `(0, 0)` for a path that
            does not exist, or a directory holding no files, reports a silent no-op as a
            successful upload -- and the caller has already thrown away what it meant to send.
    """
    s3 = client(endpoint)
    bucket, prefix = parse(uri)
    if local.is_file():
        s3.upload_file(str(local), bucket, prefix)
        return 1, local.stat().st_size
    paths = sorted(p for p in local.rglob("*") if p.is_file()) if local.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"nothing to upload at {local}")
    total = 0
    for path in paths:
        rel = path.relative_to(local).as_posix()
        s3.upload_file(str(path), bucket, f"{prefix}/{rel}" if prefix else rel)
        total += path.stat().st_size
    return len(paths), total
