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

#: Characters that make a doc_id unusable as a directory name. The layout has no suite level,
#: so uniqueness and filename-safety stop being free and have to be asserted.
_UNSAFE = ("/", "\\", "\0")

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


def extras(root: Path) -> dict:
    """Columns an existing atlas carries that the contract does not define, by doc_id.

    A published corpus records `suite`, page counts, sizes and provenance alongside the five
    required columns. Rebuilding the atlas must not throw those away just because this module
    does not know what they mean -- `suite` decides the subsets the published number averages
    over, and it cannot be recovered from the payloads.
    """
    import pyarrow.parquet as pq

    path = Path(root) / ATLAS
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
        {"corpus_version": version(entries), "documents": str(len(entries)),
         **(metadata or {})})
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


def check_unique(doc_ids) -> None:
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


def verify(expected, root: Path, patterns=("**/*.json",)) -> list[str]:
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


def check(root: Path, entries: Iterable[Entry]) -> list[str]:
    """What no longer matches the atlas. Empty when the corpus is as declared.

    `verify` does the work, so there is one implementation of "does this tree match these
    hashes" rather than two that have to agree.

    The globs are exactly what `check_path` allows a row to name, so nothing else in the tree
    is treated as a payload -- a corpus may hold PDFs, page images and provenance beside the
    documents, and those are not drift. Files matching the globs that no row names ARE found,
    and then dropped here: a document curated out of the atlas is the normal case, and
    `undeclared` reports it as information rather than as a problem.
    """
    expected = [(e.ground_truth_path, e.gt_sha256) for e in entries]
    expected += [(e.schema_path, e.schema_sha256) for e in entries]
    problems = verify(expected, Path(root),
                      patterns=(f"*/{GROUND_TRUTH}", f"*/{SCHEMA}"))
    return [p for p in problems if not p.startswith("file with no row")]


def undeclared(root: Path, entries: Iterable[Entry]) -> list[str]:
    """Document directories on disk that the atlas does not list.

    Not an error -- curating a document out is exactly this -- but worth seeing, because it is
    also what a half-finished copy looks like.
    """
    listed = {e.doc_id for e in entries}
    return sorted(d.name for d in Path(root).iterdir()
                  if d.is_dir() and d.name not in listed
                  and ((d / GROUND_TRUTH).exists() or (d / SCHEMA).exists()))
