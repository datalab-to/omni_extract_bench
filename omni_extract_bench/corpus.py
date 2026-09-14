"""The corpus atlas: what a benchmark contains, stated rather than discovered.

    <corpus>/corpus.parquet          the atlas -- the source of truth
    <corpus>/<doc_id>/ground_truth.json
    <corpus>/<doc_id>/schema.json

**The atlas defines the benchmark; the directory merely stores it.** Scoring runs over the
rows in `corpus.parquet` and nothing else, so a half-copied document, a leftover directory or
a scratch file cannot silently join a benchmark by being present.

That inversion is what makes curation an ordinary operation. Build an atlas over everything
you have, then delete rows -- in DuckDB, pandas, anything -- and the remaining rows are the
benchmark. The files stay on disk, so nothing is lost and a document returns by re-adding its
row. A filtered corpus is a different corpus and gets a different `version`, which is correct:
it is a different benchmark.

It also makes editing data deliberate. Each row carries the sha256 of the files it names, and
loading a document checks them, so an edited ground truth stops the run instead of quietly
producing different numbers. Rebuilding the atlas is how you say you meant it.

**Paths are stored, and constrained.** Each row names its files relative to the atlas, so a
corpus is relocatable as a unit and you can see what a row points at without knowing a
convention. They must still BE the convention -- `<doc_id>/ground_truth.json` and
`<doc_id>/schema.json` -- because a path column that can say anything is a path column that
can point outside the corpus, at another corpus, or at a file that is not part of the
benchmark at all.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Iterable, NamedTuple

#: The atlas, beside the documents it describes.
ATLAS = "corpus.parquet"

#: The two files a document is. Named in full because they appear in error messages.
GROUND_TRUTH = "ground_truth.json"
SCHEMA = "schema.json"

#: What every atlas must have. A published corpus carries more -- suite, page counts, sizes --
#: and those are additive: anything that reads an atlas depends only on these.
REQUIRED = ("doc_id", "ground_truth_path", "schema_path", "gt_sha256", "schema_sha256")


class Entry(NamedTuple):
    """One row of the atlas: a document, where its files are, and what they hashed to."""

    doc_id: str
    ground_truth_path: str
    schema_path: str
    gt_sha256: str
    schema_sha256: str


class Stale(Exception):
    """A file on disk does not match what the atlas says it is.

    Its own exception because it is the one error with a specific, correct response: if the
    change was intended, rebuild the atlas; if it was not, you have just caught data drifting
    underneath a benchmark.
    """


def expected_paths(doc_id: str) -> tuple[str, str]:
    """Where a document's files must live, relative to the atlas."""
    return f"{doc_id}/{GROUND_TRUTH}", f"{doc_id}/{SCHEMA}"


def check_path(doc_id: str, path: str, filename: str) -> None:
    """Reject a path that is not this document's file, in this corpus.

    Relative, forward-slashed, exactly `<doc_id>/<filename>`. No absolute paths, no `..`, no
    reaching into a sibling document. The column exists to be explicit about where a file is,
    not to make the corpus an arbitrary file list -- an atlas that can point anywhere is one
    that can quietly score a document belonging to something else.
    """
    want = f"{doc_id}/{filename}"
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or "\\" in path:
        raise ValueError(f"{doc_id}: path {path!r} escapes the corpus; must be {want!r}")
    if path != want:
        raise ValueError(f"{doc_id}: path {path!r} does not follow the layout; must be {want!r}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry_for(root: Path, doc_id: str) -> Entry:
    """Hash a document's files as they are now."""
    gt_path, schema_path = expected_paths(doc_id)
    for rel in (gt_path, schema_path):
        if not (root / rel).exists():
            raise FileNotFoundError(
                f"{doc_id}: no {Path(rel).name}. A document is {GROUND_TRUTH} and {SCHEMA}; "
                f"a schema is required and is never inferred, because without it an "
                f"additionalProperties subtree would be graded silently.")
    return Entry(doc_id, gt_path, schema_path,
                 sha256(root / gt_path), sha256(root / schema_path))


def discover(root: Path) -> list[Entry]:
    """Every document the tree contains, for a first atlas.

    Discovery happens exactly here and nowhere else. Every other operation reads the atlas, so
    this is the one moment a benchmark's membership is decided by what is on disk.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"corpus {root} is not a directory")
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / GROUND_TRUTH).exists() or (d / SCHEMA).exists():
            out.append(entry_for(root, d.name))
    return out


def refresh(root: Path, entries: Iterable[Entry]) -> list[Entry]:
    """Re-hash the documents already listed, preserving the selection.

    The counterpart to `discover`, and the reason both exist: re-discovering would resurrect
    every document curation removed. This is what you run after deciding an edit was intended.
    """
    return [entry_for(Path(root), e.doc_id) for e in entries]


def version(entries: Iterable[Entry]) -> str:
    """Identify the corpus by what it selects and what those files contain.

    Curating a row out changes it, because a filtered corpus is a different benchmark.
    Editing a ground truth changes it, because so is a corrected one.
    """
    digest = hashlib.sha256()
    for e in sorted(entries):
        digest.update(f"{e.doc_id}\0{e.gt_sha256}\0{e.schema_sha256}\0".encode())
    return digest.hexdigest()[:16]


def read(root: Path) -> list[Entry]:
    """The atlas, validated.

    Raises:
        FileNotFoundError: if there is no atlas. A corpus without one is not a corpus; run
            `oeb build-corpus` to declare what it contains.
        ValueError: on a duplicate doc_id, a missing column, or a path that is not this
            document's file.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    path = root / ATLAS
    if not path.exists():
        raise FileNotFoundError(
            f"no {ATLAS} in {root}. The atlas is what says which documents are in this "
            f"benchmark; create one with:  oeb build-corpus --corpus {root}")
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError(f"{path} lists no documents")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(missing)}")

    seen, out = set(), []
    for r in rows:
        doc_id = r["doc_id"]
        if doc_id in seen:
            raise ValueError(f"{path}: duplicate doc_id {doc_id!r}")
        seen.add(doc_id)
        check_path(doc_id, r["ground_truth_path"], GROUND_TRUTH)
        check_path(doc_id, r["schema_path"], SCHEMA)
        out.append(Entry(doc_id, r["ground_truth_path"], r["schema_path"],
                         r["gt_sha256"], r["schema_sha256"]))
    return out


def write(root: Path, entries: Iterable[Entry], extra: dict | None = None) -> Path:
    """Write the atlas, preserving any extra columns a caller wants to carry.

    Written to a temporary name and renamed: a crash here would otherwise leave a corpus with
    no definition at all.
    """
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    entries = list(entries)
    extra = extra or {}
    rows = [{**e._asdict(), **extra.get(e.doc_id, {})} for e in entries]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    blank = {k: None for k in fields}
    table = pa.Table.from_pylist([{**blank, **r} for r in rows]).replace_schema_metadata(
        {"corpus_version": version(entries), "documents": str(len(entries))})
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (ATLAS + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, root / ATLAS)
    return root / ATLAS


def check(root: Path, entries: Iterable[Entry]) -> list[str]:
    """What no longer matches the atlas. Empty when the corpus is as declared."""
    root = Path(root)
    problems = []
    for e in entries:
        for rel, want, what in ((e.ground_truth_path, e.gt_sha256, "ground truth"),
                                (e.schema_path, e.schema_sha256, "schema")):
            f = root / rel
            if not f.exists():
                problems.append(f"{e.doc_id}: {rel} is listed but missing")
            elif sha256(f) != want:
                problems.append(f"{e.doc_id}: {what} changed since the atlas was written")
    return problems


def undeclared(root: Path, entries: Iterable[Entry]) -> list[str]:
    """Document directories on disk that the atlas does not list.

    Not an error -- curating a document out is exactly this -- but worth seeing, because it is
    also what a half-finished copy looks like.
    """
    listed = {e.doc_id for e in entries}
    return sorted(d.name for d in Path(root).iterdir()
                  if d.is_dir() and d.name not in listed
                  and ((d / GROUND_TRUTH).exists() or (d / SCHEMA).exists()))
