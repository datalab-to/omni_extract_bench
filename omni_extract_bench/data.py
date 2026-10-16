"""The dataset: its documents, read from a folder or downloaded from HuggingFace. A dataset folder holds

    manifest.parquet       a row per document: doc_id, suite, doc_path, gt_path, schema
    manifests/*.parquet    other manifests, if any (a sample of the same documents)
    pdfs/<doc_id>.pdf      the document that gets sent
    gold/<doc_id>.json     its ground truth, the extraction, read by `scorer.gold`

A document is a manifest row (`Doc`) whose paths are full local paths, so a document says where its
file and its gold are without the dataset beside it. The functions over a dataset:

    download_dataset(dataset)            a local copy of the whole dataset; a local folder as it is
    read_manifest(dataset, manifest)     the folder and every document, checked
    select(docs, suites, limit, ...)     the documents to run: by suite or id, sampled, limited
    ManifestError                        a manifest, or a row of one, that breaks the dataset's rules
    dataset_id(dataset)                  what a run's dataset is, whatever its revision

`dataset` is a folder's path or `hf://datasets/<org>/<name>[@<revision>]`.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPO = "datalab-to/omni_extract_bench"
MANIFEST = "manifest.parquet"
HF_PREFIX = "hf://datasets/"
DEFAULT_DATASET = f"{HF_PREFIX}{REPO}"
#: Written out rather than read off `Doc`: manifests on disk have this shape, and a field `Doc`
#: grows later must not invalidate them.
COLUMNS = ("doc_id", "suite", "doc_path", "gt_path", "schema")


class ManifestError(ValueError):
    """A manifest, or a row of one, that breaks the dataset's rules; the message names it and the fix."""


@dataclasses.dataclass(frozen=True)
class Doc:
    """One benchmark element: a manifest row, its paths full local paths.

    `extra` holds the row's other columns: omni-extract-bench reads none of them, so a dataset can
    carry columns of its own (a language, a tag, another gold) for scorers of its own. Treat it as
    read-only. A frozen dataclass rather than a NamedTuple so each document gets its own `extra`;
    a copy with a field changed is `dataclasses.replace(doc, ...)`."""

    doc_id: str
    suite: str
    doc_path: Path
    gt_path: Path
    schema: dict
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)   # the manifest's other columns

    @property
    def id(self) -> str:
        """The document's id, by the name every benchmark's rows share (opb's `Page.id` is its page_id)."""
        return self.doc_id


def _hf_dataset(dataset: str | Path) -> tuple[str, str | None] | None:
    """The repo and revision `dataset` addresses, or `None` when it is a folder on disk."""
    form = f"hf://datasets/<org>/<name>[@<revision>], e.g. {DEFAULT_DATASET}"
    s = str(dataset)
    if isinstance(dataset, Path) and s.startswith("hf:"):
        raise ValueError(f"the dataset {s!r} was made a Path, which folds the '//' out of an "
                         f"address. Pass it as a str: {form}")
    if not s.startswith("hf://"):
        return None
    org, _, name = s.removeprefix(HF_PREFIX).partition("/")
    name, at, revision = name.partition("@")
    if not s.startswith(HF_PREFIX) or not org or not name or "/" in name or (at and not revision):
        raise ValueError(f"{s!r} does not name a HuggingFace dataset. Write it as {form}")
    return f"{org}/{name}", revision or None


def dataset_id(dataset: str | Path) -> str:
    """What a run's dataset is, whatever its revision: the repo of an `hf://` dataset, or a
    folder's path."""
    if (at := _hf_dataset(dataset)) is not None:
        return f"{HF_PREFIX}{at[0]}"
    return str(Path(dataset).expanduser().resolve())


def download_dataset(dataset: str | Path) -> Path:
    """The dataset as a folder on disk: a folder as given, or a HuggingFace dataset downloaded whole,
    at one commit, into the HuggingFace cache, named for that commit. The cache is shared between every
    checkout and working directory, so a second call downloads nothing.

    Note: the whole dataset (0.9 GB), so its documents and its gold are always one commit's. Fetching
    each document when it is first predicted would let a `--limit 5` run download five; that is an
    optimization for later.
    """
    if (at := _hf_dataset(dataset)) is None:
        return Path(dataset).expanduser()
    repo, revision = at
    from huggingface_hub import snapshot_download

    log.info("downloading %s into the huggingface cache", dataset)
    return Path(snapshot_download(repo, repo_type="dataset", revision=revision))


@functools.cache
def read_manifest(dataset: str | Path, manifest: str = MANIFEST) -> tuple[Path, tuple[Doc, ...]]:
    """The dataset's folder (`download_dataset`) and every document its manifest `manifest` (a path in
    the dataset) names, checked, in (suite, doc_id) order, each with its file and gold as full paths.
    Its columns must include `COLUMNS`; any other column is carried on each document's `extra`."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is needed to read the benchmark manifest, and is not installed:\n"
            "    pip install 'omni-extract-bench[benchmark]'"
        ) from None
    _on_hub(dataset, manifest)
    root = download_dataset(dataset)
    path = root / manifest
    if not path.exists():
        raise ManifestError(
            f"there is no manifest at {path}. It is a parquet file with one row per "
            f"document, as {', '.join(COLUMNS)}; ours is at "
            f"https://huggingface.co/datasets/{REPO}, and a corpus of your own follows the "
            f"same convention."
        )
    table = pq.read_table(path)
    absent = [c for c in COLUMNS if c not in table.column_names]
    if absent:
        has = ", ".join(table.column_names) or "no columns"
        raise ManifestError(f"{path} has no {', '.join(absent)} column"
                         f"{'s' if len(absent) > 1 else ''}. A manifest names one document "
                         f"per row, as {', '.join(COLUMNS)}; this one has {has}.")
    docs = []
    for n, row in enumerate(table.to_pylist()):
        # A doc_id is a filename (`predictions/<doc_id>.json`); caught here, the row can be named.
        doc_id = row["doc_id"]
        if not doc_id or doc_id in (".", "..") or Path(doc_id).name != doc_id:
            raise ManifestError(f"{path}: row {n} has doc_id {doc_id!r}, which is not a filename. "
                             f"It names this document's prediction, its record and its row in "
                             f"scores.jsonl, so it has to be one path component.")
        try:
            schema = json.loads(row["schema"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ManifestError(f"{path}: row {n} ({doc_id}) has a schema column that is not "
                             f"JSON: {exc}") from None
        docs.append(Doc(doc_id=doc_id, suite=row["suite"], doc_path=root / row["doc_path"],
                        gt_path=root / row["gt_path"], schema=schema,
                        extra={k: v for k, v in row.items() if k not in COLUMNS}))
    docs.sort(key=lambda d: (d.suite, d.doc_id))
    return root, tuple(docs)


def select(docs: Sequence[Doc], suites: Sequence[str] | None = None, limit: int = 0,
           sample: int = 0, only: frozenset[str] | None = None) -> tuple[Doc, ...]:
    """The documents of `docs` (a dataset's, from `read_manifest`) to run, in their order, filtered by
    suite and by `only` (doc ids, each of which must be among them), and limited.

    `sample` picks that many documents spread over every suite: suites in turn, each document evenly
    spaced within its suite, so a smoke test sees every kind of document and the same ones each time."""
    rows = tuple(docs)
    known = {r.suite for r in rows}
    if bad := sorted(set(suites or ()) - known):
        raise ValueError(f"unknown suites {bad}; the dataset's suites are {sorted(known)}")
    rows = tuple(r for r in rows if not suites or r.suite in suites)
    if only is not None:
        if bad := sorted(only - {r.doc_id for r in rows}):
            raise ValueError(f"{len(bad)} of the documents asked for aren't in the dataset"
                             f"{' (or its suites)' if suites else ''}, e.g. {bad[:3]}; nothing has been run")
        rows = tuple(r for r in rows if r.doc_id in only)
    if sample and rows:
        by: dict[str, list] = {}
        for r in rows: by.setdefault(r.suite, []).append(r)
        k = -(-sample // len(by))
        spread = [[xs[i * len(xs) // min(k, len(xs))] for i in range(min(k, len(xs)))] for xs in by.values()]
        turns = [x for i in range(k) for xs in spread if i < len(xs) for x in [xs[i]]]
        chosen = {r.doc_id for r in turns[:sample]}
        rows = tuple(r for r in rows if r.doc_id in chosen)
    return rows[:limit] if limit else rows


def _on_hub(dataset: str | Path, manifest: str) -> None:
    """Refuses an `hf://` dataset without the manifest `manifest`, before a download of gigabytes.

    Note: a hub that will not answer (offline, cached) is not a no: the download runs, and
    `read_manifest` reports a manifest that really is missing."""
    if (at := _hf_dataset(dataset)) is None:
        return
    repo, revision = at
    try:
        from huggingface_hub import file_exists
    except ImportError:
        raise ImportError(
            "huggingface_hub is needed to fetch the benchmark corpus, and is not installed:\n"
            "    pip install 'omni-extract-bench[benchmark]'\n"
            "The scorer itself needs none of it -- `score` and `oeb score` work without."
        ) from None
    try:
        missing = not file_exists(repo, manifest, repo_type="dataset", revision=revision)
    except OSError:
        missing = False
    if missing:
        raise ManifestError(
            f"there is no {manifest} in {dataset}, so there is nothing to benchmark. A dataset "
            f"holds one or more manifests, each following the same structure as ours at "
            f"https://huggingface.co/datasets/{REPO}. Name another with --manifest, or point "
            f"--dataset at a folder of your own.")
