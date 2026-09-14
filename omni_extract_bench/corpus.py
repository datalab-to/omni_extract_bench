"""The corpus atlas: what a benchmark contains, stated rather than discovered.

    <corpus>/corpus.parquet          the atlas -- the source of truth
    <corpus>/<doc_id>/ground_truth.json
    <corpus>/<doc_id>/schema.json

**The atlas is what you explore and what you run.** `build` writes a row per document found in
the tree, and scoring runs over those rows -- so filtering the table is how you choose a subset
to score, and a half-copied document or a scratch directory cannot join a benchmark by being
present.

It is a plain parquet, meant to be queried. Beyond the three columns below it carries whatever
higher-level metadata you have -- which collection a document came from, its page count, how
big its ground truth is -- and those are what make "score only the long-table documents" or
"how much of this corpus is invoices" a `where` clause rather than a script.

It is not a guarantee that the data has not changed. Scoring reads the files the atlas names,
as they are. If you need to know a ground truth was not edited, that is your store's job -- the
published corpus lives on HuggingFace, whose revisions already pin every byte.

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
from typing import Iterable, NamedTuple, Sequence

#: Characters that make a doc_id unusable as a directory name. The layout has no suite level,
#: so uniqueness and filename-safety stop being free and have to be asserted.
_UNSAFE = ("/", "\\", "\0")

#: The atlas, beside the documents it describes.
ATLAS = "corpus.parquet"

#: The two files a document is. Named in full because they appear in error messages.
GROUND_TRUTH = "ground_truth.json"
SCHEMA = "schema.json"

#: What every atlas must have: which documents, and where their two files are. Everything else
#: is metadata a corpus chooses to carry, and anything reading an atlas depends only on these.
REQUIRED = ("doc_id", "ground_truth_path", "schema_path")


class Entry(NamedTuple):
    """One row of the atlas: a document, and where its two files are."""

    doc_id: str
    ground_truth_path: str
    schema_path: str


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
    """One document's row, checking that the files it names are there."""
    gt_path, schema_path = expected_paths(doc_id)
    for rel in (gt_path, schema_path):
        if not (root / rel).exists():
            raise FileNotFoundError(
                f"{doc_id}: no {Path(rel).name}. A document is {GROUND_TRUTH} and {SCHEMA}; "
                f"a schema is required and is never inferred, because without it an "
                f"additionalProperties subtree would be graded silently.")
    return Entry(doc_id, gt_path, schema_path)


def discover(root: Path) -> list[Entry]:
    """Every document the tree contains, hashed as it is now.

    The one place a benchmark's membership is decided by what is on disk; every other operation
    reads the atlas. That split is what lets the atlas be checked against the files it names:
    scoring trusts the atlas, and an edited file stops the run rather than quietly changing a
    number.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"corpus {root} is not a directory")
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / GROUND_TRUTH).exists() or (d / SCHEMA).exists():
            out.append(entry_for(root, d.name))
    return out


def locate(target: Path) -> tuple[Path, Path]:
    """The corpus root and the atlas to read, from either a directory or an atlas file.

    A directory means its `corpus.parquet`; a file means that file, with the documents found
    beside it. Rows name their files relative to the atlas, so a filtered atlas written next to
    the full one describes the same documents and needs no rewriting -- which is the point:
    narrowing a corpus should not mean overwriting the record of what it contains.
    """
    target = Path(target)
    return (target, target / ATLAS) if target.is_dir() else (target.parent, target)


def read(target: Path) -> list[Entry]:
    """The atlas, validated. Takes a corpus directory or a specific atlas file.

    Raises:
        FileNotFoundError: if there is no atlas. A corpus without one is not a corpus; run
            `oeb build-corpus` to declare what it contains.
        ValueError: on a duplicate doc_id, a missing column, or a path that is not this
            document's file.
    """
    import pyarrow.parquet as pq

    root, path = locate(target)
    if not path.exists():
        raise FileNotFoundError(
            f"no atlas at {path}. The atlas is what says which documents are in this "
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
        out.append(Entry(doc_id, r["ground_truth_path"], r["schema_path"]))
    return out


def extras(target: Path) -> dict:
    """Columns an existing atlas carries that the contract does not define, by doc_id.

    A published corpus records `suite`, page counts, sizes and provenance alongside the five
    required columns. Rebuilding the atlas must not throw those away just because this module
    does not know what they mean -- `suite` decides the subsets the published number averages
    over, and it cannot be recovered from the payloads.
    """
    import pyarrow.parquet as pq

    _root, path = locate(Path(target))
    if not path.exists():
        return {}
    out = {}
    for r in pq.read_table(path).to_pylist():
        rest = {k: v for k, v in r.items() if k not in REQUIRED}
        if rest:
            out[r["doc_id"]] = rest
    return out


def write(root: Path, entries: Iterable[Entry], extra: dict | None = None,
          metadata: dict | None = None) -> Path:
    """Write the atlas. The only thing that does.

    `extra` adds columns per doc_id; `metadata` adds file-level provenance. Both exist so a
    richer builder -- ours records `suite`, page counts and which HuggingFace snapshot it came
    from -- can enrich the atlas without writing one itself. Two writers of the same file is
    how the required columns went missing from the published corpus while the code that
    defined them was already correct.

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
        {"documents": str(len(entries)), **(metadata or {})})
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (ATLAS + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, root / ATLAS)
    return root / ATLAS


def check_doc_id(doc_id: str) -> None:
    """Fail loudly on a doc_id the flat layout cannot hold.

    Raises:
        ValueError: if the id would collide with a path, hide as a dotfile, or be empty.
    """
    if not doc_id:
        raise ValueError("empty doc_id")
    if doc_id.startswith("."):
        raise ValueError(f"doc_id starts with a dot, which hides the directory: {doc_id!r}")
    for ch in _UNSAFE:
        if ch in doc_id:
            raise ValueError(f"doc_id contains {ch!r}, unusable as a filename: {doc_id!r}")


def check_unique(doc_ids: Iterable[str]) -> None:
    """Fail loudly on duplicates.

    With the suite directory gone, two documents sharing an id would silently overwrite each
    other. This turns that into a build error.

    Raises:
        ValueError: listing the ids that appear more than once.
    """
    seen, dupes = set(), []
    for d in doc_ids:
        if d in seen:
            dupes.append(d)
        seen.add(d)
    if dupes:
        raise ValueError(f"{len(dupes)} duplicate doc_id(s): {sorted(set(dupes))[:5]}")


def verify(expected: Iterable[tuple[str, str]], root: Path,
           patterns: Sequence[str] = ("**/*.json",)) -> list[str]:
    """Check an atlas against the files it describes, in both directions.

    An atlas and its payloads can drift with nothing noticing: a row pointing at a deleted
    file, a file no row mentions, or a payload edited in place. The first two are findable by
    listing; the third is invisible without hashes, which is why the atlas carries them.

    Args:
        expected: pairs of (path relative to `root`, expected sha256). Deriving these from the
            atlas rows is the caller's job, because the two layouts differ -- the corpus keeps
            several payloads per document in a directory, predictions keep one flat file each.
            An earlier version guessed `<row_id>.json` and was therefore useless for the
            corpus tree; passing the paths in is what makes one function serve both.
        root: the tree the paths are relative to.
        patterns: which files under `root` are payloads, for finding orphans. Anything not
            matching is ignored, so an atlas sitting inside its own tree is not an orphan --
            but a payload type left out of this list is invisible the same way, and every row
            claiming one then reads as a missing file. Pass every extension the tree holds.

    Returns:
        Complaints, empty when consistent. Returned rather than raised so a caller can report
        all of them at once instead of one per run.
    """
    expected = {str(p): h for p, h in expected}
    on_disk = {str(p.relative_to(root))
               for pat in patterns for p in root.glob(pat) if p.is_file()}

    problems: list[str] = []
    for missing in sorted(set(expected) - on_disk):
        problems.append(f"row with no file: {missing}")
    for orphan in sorted(on_disk - set(expected)):
        problems.append(f"file with no row: {orphan}")
    for shared in sorted(set(expected) & on_disk):
        actual = hashlib.sha256((root / shared).read_bytes()).hexdigest()
        if actual != expected[shared]:
            problems.append(f"hash mismatch: {shared} "
                            f"(atlas {expected[shared][:12]}..., file {actual[:12]}...)")
    return problems
