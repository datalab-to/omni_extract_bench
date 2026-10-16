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
    commit(dataset, root)                the commit a HuggingFace dataset was read at; None for a folder
    normal(manifest)                     a manifest's path in its dataset, spelled one way

`dataset` is a folder's path or `hf://datasets/<org>/<name>[@<revision>]`.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from . import errors

log = logging.getLogger(__name__)
REPO = "datalab-to/omni_extract_bench"
HF_PREFIX = "hf://datasets/"
DEFAULT_DATASET = f"{HF_PREFIX}{REPO}"
DOWNLOAD_ATTEMPTS = 5
MANIFEST = "manifest.parquet"


#: Written out rather than read off `Doc`: manifests on disk have this shape, and a field `Doc`
#: grows later must not invalidate them.
COLUMNS = ("doc_id", "suite", "doc_path", "gt_path", "schema")
NO_SUITE = "other"          # the suite of a document whose manifest gives none (null or "")


class ManifestError(errors.UsageError):
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
        raise errors.UsageError(f"the dataset {s!r} was made a Path, which folds the '//' out of an "
                                f"address. Pass it as a str: {form}")
    if not s.startswith("hf://"):
        return None
    org, _, name = s.removeprefix(HF_PREFIX).partition("/")
    name, at, revision = name.partition("@")
    if not s.startswith(HF_PREFIX) or not org or not name or "/" in name or (at and not revision):
        raise errors.UsageError(f"{s!r} does not name a HuggingFace dataset. Write it as {form}")
    return f"{org}/{name}", revision or None


def download_dataset(dataset: str | Path) -> Path:
    """The dataset as a folder on disk: a folder as given, or a HuggingFace dataset downloaded whole,
    at one commit, into the HuggingFace cache, named for that commit. The cache is shared between every
    checkout and working directory, so a second call downloads nothing.

    Note: the whole dataset (0.9 GB), so its documents and its gold are always one commit's. Fetching
    each document when it is first predicted would let a `--limit 5` run download five; that is an
    optimization for later.
    """
    if (at := _hf_dataset(dataset)) is None:
        root = Path(dataset).expanduser()
        if not root.is_dir():
            hint = (f"; a Hugging Face dataset is written {HF_PREFIX}<org>/<name>, with two slashes"
                    if str(dataset).startswith("hf:") else "")
            raise errors.UsageError(f"no dataset folder at {root}; pass --dataset a dataset folder "
                                    f"or {HF_PREFIX}<org>/<name>{hint}")
        return root
    repo, revision = at
    import huggingface_hub
    from huggingface_hub import errors as hf_errors

    log.info("reading %s from the Hugging Face cache, downloading what it lacks", dataset)
    # Note: hundreds of files; the Hub drops some requests under that load, and a download picks up
    # where the last left off, so it is simply tried again. A missing repo, revision or access, or no
    # network with nothing cached, is the same on every attempt, so it fails at once.
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            return Path(huggingface_hub.snapshot_download(repo, repo_type="dataset", revision=revision, max_workers=4))
        except (hf_errors.RepositoryNotFoundError, hf_errors.GatedRepoError,
                hf_errors.RevisionNotFoundError) as exc:
            raise errors.UsageError(f"can't read dataset {dataset}: {type(exc).__name__}. Check the repo and "
                                    "revision, and that you're logged in with access (huggingface-cli login)"
                                    ) from None
        except (hf_errors.LocalEntryNotFoundError, hf_errors.OfflineModeIsEnabled) as exc:
            # Note: offline by the user's own setting, connecting changes nothing; unsetting it does.
            how = ("HF_HUB_OFFLINE is set; unset it" if os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0")
                   else "the Hub can't be reached; connect")
            raise errors.UsageError(f"can't read dataset {dataset}: it isn't in the Hugging Face cache, and "
                                    f"{how}, or pass --dataset a folder ({type(exc).__name__})") from None
        except hf_errors.HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise errors.UsageError(f"can't read dataset {dataset}: HTTP {status}. Log in with access to "
                                        "it (huggingface-cli login)") from None
            if attempt == DOWNLOAD_ATTEMPTS - 1: raise
            log.warning("downloading %s failed (HTTP %s); trying again in %ds", dataset, status,
                        30 * (attempt + 1))
        except Exception as exc:
            if attempt == DOWNLOAD_ATTEMPTS - 1: raise
            log.warning("downloading %s failed (%s: %s); trying again in %ds", dataset, type(exc).__name__,
                        str(exc)[:200], 30 * (attempt + 1))
        time.sleep(30 * (attempt + 1))
    raise AssertionError("unreachable")


def dataset_id(dataset: str | Path) -> str:
    """What a run's dataset is, whatever its revision: the repo of an `hf://` dataset, or a
    folder's path."""
    if (at := _hf_dataset(dataset)) is not None:
        return f"{HF_PREFIX}{at[0]}"
    return str(Path(dataset).expanduser().resolve())


def normal(manifest: str) -> str:
    """A manifest's path in its dataset, spelled one way: `./manifest.parquet` is `manifest.parquet`."""
    return PurePosixPath(manifest).as_posix()


def commit(dataset: str | Path, root: Path) -> str | None:
    """The commit `dataset` was read at, its downloaded folder `root`'s name; None for a folder."""
    return root.name if _hf_dataset(dataset) is not None else None


@functools.cache
def read_manifest(dataset: str | Path, manifest: str = MANIFEST) -> tuple[Path, tuple[Doc, ...]]:
    """The dataset's folder (`download_dataset`) and every document its manifest `manifest` (a path in
    the dataset) names, checked, in the manifest's order, each with its file and gold as full paths.
    Its columns must include `COLUMNS`; any other column is carried on each document's `extra`."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is needed to read the benchmark manifest, and is not installed:\n"
            "    pip install 'omni-extract-bench[benchmark]'"
        ) from None
    manifest = normal(manifest)
    if Path(manifest).is_absolute() or ".." in Path(manifest).parts:
        on_disk = Path(manifest)
        raise ManifestError(
            f"--manifest {manifest!r} must be a relative path inside the dataset (--dataset), without "
            f"'..', e.g. manifests/sample.parquet"
            + (f"; for this one, pass --dataset {on_disk.parent} --manifest {on_disk.name}"
               if on_disk.is_file() else ""))
    _on_hub(dataset, manifest)
    root = download_dataset(dataset)
    path = root / manifest
    if not path.exists():
        hint = (f" {manifest} is a file here, outside the dataset: pass --dataset {Path(manifest).parent} "
                f"--manifest {Path(manifest).name}." if Path(manifest).is_file() else "")
        raise ManifestError(
            f"there is no manifest at {path}. --manifest is a path inside the dataset (--dataset), "
            f"{MANIFEST} by default; a manifest is a parquet file with one row per document, as "
            f"{', '.join(COLUMNS)}, as ours at https://huggingface.co/datasets/{REPO}.{hint}"
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
        if not isinstance(doc_id, str) or not doc_id or doc_id in (".", "..") or Path(doc_id).name != doc_id:
            raise ManifestError(f"{path}: row {n} has doc_id {doc_id!r}, which is not a filename. "
                             f"It names this document's prediction, its record and its row in "
                             f"scores.jsonl, so it has to be one path component.")
        if row["suite"] is not None and not isinstance(row["suite"], str):
            raise ManifestError(f"{path}: row {n} ({doc_id}) has suite {row['suite']!r}; it must be a "
                                f"string, or null for {NO_SUITE!r}")
        if not _inside(row["doc_path"]):
            raise ManifestError(f"{path}: row {n} ({doc_id}) has doc_path {row['doc_path']!r}; it must be a "
                                f"relative path inside the dataset to a file, e.g. docs/<file>.pdf (any "
                                f"name; documents may share one)")
        if not _inside(row["gt_path"], doc_id):
            raise ManifestError(f"{path}: row {n} ({doc_id}) has gt_path {row['gt_path']!r}; it must be a "
                                f"relative path inside the dataset named for its document, "
                                f"e.g. gold/{doc_id}.json")
        try:
            schema = json.loads(row["schema"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ManifestError(f"{path}: row {n} ({doc_id}) has a schema column that is not "
                             f"JSON: {exc}") from None
        if not isinstance(schema, dict) or not schema:
            raise ManifestError(f"{path}: row {n} ({doc_id}) has schema {row['schema']!r:.80}; it must be a "
                                "non-empty JSON object, the JSON Schema of the extraction")
        suite = row["suite"] if (row["suite"] or "").strip() else NO_SUITE
        docs.append(Doc(doc_id=doc_id, suite=suite, doc_path=root / row["doc_path"],
                        gt_path=root / row["gt_path"], schema=schema,
                        extra={k: v for k, v in row.items() if k not in COLUMNS}))
    if not docs:
        raise ManifestError(f"the manifest {manifest} in {root} names no documents")
    # Note: ignoring case, because an id names files, and two ids that differ only in case name one file
    # on a case-insensitive disk (macOS, Windows).
    by: dict[str, list[str]] = {}
    for d in docs: by.setdefault(d.doc_id.casefold(), []).append(d.doc_id)
    if twice := [ids for ids in by.values() if len(ids) > 1]:
        raise ManifestError(f"{path}: doc ids in the manifest more than once (ignoring case, since each "
                            f"names files): {errors.counted(len(twice), 'id')}, e.g. {twice[:3]}")
    return root, tuple(docs)


def select(docs: Sequence[Doc], suites: Iterable[str] | None = None, limit: int = 0,
           sample: int = 0, only: Iterable[str] | None = None) -> tuple[Doc, ...]:
    """The documents of `docs` (a dataset's, from `read_manifest`) to run, in their order, filtered by
    suite and by `only` (doc ids, each of which must be among them), and limited.

    `sample` picks that many documents spread over every suite: suites in turn, each document evenly
    spaced within its suite, so a smoke test sees every kind of document and the same ones each time."""
    suites, only = _names(suites, "suites") or None, _names(only, "only")
    for name, n in (("limit", limit), ("sample", sample)):
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"{name} is a whole number, not {n!r}")
        if n < 0:
            raise errors.UsageError(f"--{name} is {n}, but it counts documents, so it can't be negative; 0 "
                                    "is no limit")
    rows = tuple(docs)
    known = {r.suite for r in rows}
    if suites is not None and (bad := sorted(suites - known)):
        raise errors.UsageError(f"unknown suites {bad}; the dataset's suites are {sorted(known)}")
    rows = tuple(r for r in rows if suites is None or r.suite in suites)
    if only is not None:
        if bad := sorted(only - {r.doc_id for r in rows}):
            where = " (or its suites)" if suites else ""
            raise errors.UsageError(f"documents asked for that aren't in the dataset{where}: "
                                    f"{errors.counted(len(bad), 'document')}, e.g. {bad[:3]}; nothing has "
                                    "been run")
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


def _names(xs: Iterable[str] | None, what: str) -> frozenset[str] | None:
    """`xs` as a set, read once (a generator is a one-pass iterable); a bare str is refused, since iterated it
    is one-letter names."""
    if isinstance(xs, str):
        raise TypeError(f"{what} is a list of names, not the string {xs!r}; pass [{xs!r}]")
    return None if xs is None else frozenset(xs)


def _inside(path: str, doc_id: str | None = None) -> bool:
    """Whether `path` is a non-empty string, relative, inside the dataset and naming a file (it has an
    extension), and, given `doc_id`, is a gold named for it: `<doc_id>.json`.

    Note: a gold is named for its document, in any folder, so one dataset's documents can carry golds
    of several kinds, one manifest each, and no two documents share one. An input is not: several
    documents may read one file, as when one PDF is asked with several schemas."""
    if not isinstance(path, str) or not path.strip(): return False
    p = Path(path)
    return (not p.is_absolute() and ".." not in p.parts and bool(p.suffix)
            and (doc_id is None or (p.stem == doc_id and p.suffix == ".json")))


def _on_hub(dataset: str | Path, manifest: str) -> None:
    """Refuses an `hf://` dataset without the manifest `manifest`, before a download of gigabytes.

    Note: a hub that will not answer (offline, cached) is not a no: the download runs, and
    `read_manifest` reports a manifest that really is missing."""
    if (at := _hf_dataset(dataset)) is None:
        return
    repo, revision = at
    try:
        import huggingface_hub
    except ImportError:
        raise ImportError(
            "huggingface_hub is needed to fetch the benchmark corpus, and is not installed:\n"
            "    pip install 'omni-extract-bench[benchmark]'\n"
            "The scorer itself needs none of it -- `score` and `oeb score` work without."
        ) from None
    try:
        missing = not huggingface_hub.file_exists(repo, manifest, repo_type="dataset", revision=revision)
    except OSError:
        missing = False
    if missing:
        raise ManifestError(
            f"there is no {manifest} in {dataset}, so there is nothing to benchmark. A dataset "
            f"holds one or more manifests, each following the same structure as ours at "
            f"https://huggingface.co/datasets/{REPO}. Name another with --manifest, or point "
            f"--dataset at a folder of your own.")
