"""Scoring a corpus that lives in a bucket, for a container that has no local copy.

    from omni_extract_bench import remote
    remote.score("s3://bench/corpus", "s3://bench/vendors/datalab",
                 "s3://bench/scores/2026-09-14/datalab", jobs=8)

**Stage, score, upload.** The corpus and predictions come down into a temporary directory, the
ordinary local scorer runs on them, and the finished run goes up. `oeb score` is not involved
in any of that: it takes paths and returns a run, and keeping it that way is why this module
exists rather than an `s3://` branch inside it.

Three reasons staging rather than a streamed object-store filesystem, all of them about the
scorer and none about taste:

* **Scoring is a process pool.** A streamed path needs a client per worker and credentials
  crossing the fork, on the code path where a mistake is silent.
* **A run is written atomically, by rename.** `write_run` renames a temporary file into place
  so a crash leaves the previous table rather than a truncated one. Object stores have no
  rename. Staging keeps that, and uploads a run that is already complete.
* **It costs the same transfers.** Once the atlas decides what comes down, streaming fetches
  the same objects; it just does not land them on disk first. That is not worth the other two.

**The atlas decides what comes down, and that is the whole trick.** Not `sync`: fetch the
atlas, read which documents the run will follow, fetch exactly those. Scoring already follows
the atlas's rows, so a filtered table beside `corpus.parquet` has to cost a filtered download
-- otherwise a shard of 8 documents pulls all 660, and every container in a fan-out pays for
the whole corpus to score its share. Staging 8 documents is 17 objects, not 1,321.

Everything here requires `s3://`. A container has no local corpus, so there is no either/or to
carry, and nothing silently scores a stale directory that happened to be lying around.
"""
from __future__ import annotations

import sys
import tempfile
from argparse import Namespace
from pathlib import Path, PurePosixPath

from . import corpus as corpus_atlas
from . import run as run_mod
from . import s3
from .bench import PREDICTION_META


def source_name(uri: str) -> str:
    """What to call a prediction set that was not given a `--source`.

    The prefix's last segment, so `s3://bench/vendors/datalab` stamps `datalab` and not the
    temporary directory it was staged into. That stamp labels a column in the summary and in
    the viewer; deriving it from a path of ours would be a silent rename.
    """
    return PurePosixPath(uri.rstrip("/")).name


def stage_corpus(uri: str, into: Path, endpoint: str | None = None) -> Path:
    """Download the documents the atlas names, and return the local atlas.

    `uri` is the corpus prefix, or one atlas parquet inside it -- the same two things
    `--corpus` takes locally, because a filtered table written beside `corpus.parquet` is how
    a subset is chosen and a bucket must not change that.

    The atlas comes first and decides the rest. Its rows name their files relative to
    themselves, and `corpus.py` constrains those paths, so what is fetched is exactly two
    files per listed document -- never the PDFs, which are 774 MB the scorer does not open.

    Raises:
        FileNotFoundError: if there is no atlas, or if it lists a file the bucket does not
            have. The atlas is what a run follows, so it has to resolve.
    """
    is_file = uri.endswith(".parquet")
    prefix = uri.rsplit("/", 1)[0] if is_file else uri.rstrip("/")
    name = PurePosixPath(uri).name if is_file else corpus_atlas.ATLAS

    atlas = into / name
    if s3.fetch_one(f"{prefix}/{name}", atlas, endpoint) is None:
        raise FileNotFoundError(
            f"no atlas at {prefix}/{name}. The atlas is what says which documents are in a "
            f"benchmark; write one with `oeb build-corpus` and upload it.")

    entries = corpus_atlas.read(atlas)
    rels = [p for e in entries for p in (e.ground_truth_path, e.schema_path)]
    files, total, missing = s3.fetch(prefix, into, rels, endpoint)
    if missing:
        raise FileNotFoundError(
            f"{prefix} is missing {len(missing)} file(s) the atlas lists, starting with "
            f"{missing[0]}. The atlas is what a run follows, so every row has to resolve.")
    print(f"  corpus: {len(entries)} documents, {files} files, {total / 1e6:.1f} MB")
    return atlas


def stage_predictions(uri: str, into: Path, wanted: set[str],
                      endpoint: str | None = None) -> tuple[Path, list[str]]:
    """Download the predictions the atlas asks about. Returns the directory and what was left.

    Two questions, asked separately because one is cheap: a LISTING says what the prefix holds,
    and the atlas says what the run needs. Anything held and not needed is what filtering an
    atlas means -- reported by name, so a misnamed file is visible rather than silently
    ignored, but never downloaded.

    A document the atlas lists and the vendor has no prediction for is not an error here. It
    is a missing prediction, and the run records it as one.
    """
    held = s3.names(uri, endpoint)
    have = {n[:-len(".json")] for n in held if n.endswith(".json") and "/" not in n}
    rels = sorted(f"{d}.json" for d in have & wanted)
    if PREDICTION_META in held:
        rels.append(PREDICTION_META)
    if not rels:
        raise FileNotFoundError(
            f"{uri} holds no prediction the atlas asks for. Predictions are <doc_id>.json, "
            f"named for the documents in the corpus.")
    files, total, missing = s3.fetch(uri.rstrip("/"), into, rels, endpoint)
    if missing:
        raise FileNotFoundError(f"{uri} lost {len(missing)} object(s) mid-copy: {missing[0]}")
    print(f"  predictions: {files} files, {total / 1e6:.1f} MB")
    return into, sorted(have - wanted)


def score(corpus: str, predictions: str, out: str, *, source: str | None = None,
          jobs: int = 1, verdicts: bool = True, endpoint: str | None = None) -> int:
    """Stage, score, upload. Returns the scorer's exit code.

    The destination is checked before anything is transferred: neither an hour of scoring nor
    a staged corpus should end at a name that was already taken.
    """
    for name, value in (("corpus", corpus), ("predictions", predictions), ("out", out)):
        if not s3.is_uri(value):
            raise ValueError(f"{name} must be an s3:// prefix here, not {value!r}. "
                             f"For local paths use `oeb score`.")

    done = f"{out.rstrip('/')}/{run_mod.SUMMARY}"
    if s3.exists(done, endpoint):
        print(f"  {done} exists; this corpus, scorer and source already have an answer.\n"
              f"  To score it again, name a different destination.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="oeb-") as name:
        tmp = Path(name)
        atlas = stage_corpus(corpus, tmp / "corpus", endpoint)
        entries = corpus_atlas.read(atlas)
        preds, extra = stage_predictions(predictions, tmp / "predictions",
                                         {e.doc_id for e in entries}, endpoint)
        if extra:
            print(f"  {len(extra)} prediction(s) left where they are; the atlas does not "
                  f"list them: {', '.join(extra[:5])}"
                  + (f" and {len(extra) - 5} more" if len(extra) > 5 else ""))

        local = tmp / "run"
        code = run_mod.cmd_score(Namespace(
            corpus=str(atlas), predictions=str(preds), out=str(local),
            source=source or source_name(predictions), jobs=jobs, no_verdicts=not verdicts))

        files, total = s3.upload(local, out, endpoint)
        print(f"  uploaded {files} files, {total / 1e6:.1f} MB -> {out}")
    return code
