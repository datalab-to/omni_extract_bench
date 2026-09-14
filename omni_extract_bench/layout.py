"""The benchmark's storage layout: identity, extraction, and verification.

One rule governs the shape: **parquet holds metadata, files hold payloads.** See
`docs/DATA_LAYOUT.md` for why, and for the reasoning behind everything here.

This module is deliberately small and imports nothing from the scorer. The tables it
describes are valid for any scorer version, and keeping this file independent is what makes
that true.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

#: Bumping this orphans every score that joined on the old ids. `scores` keys on
#: `prediction_id`, so a change here is a migration, not a refactor. The version travels with
#: the data rather than living only in this constant.
PREDICTION_ID_VERSION = "v1"

#: Characters that make a doc_id unusable as a directory name. The layout drops the suite
#: level, so uniqueness and filename-safety stop being free and have to be asserted.
_UNSAFE = ("/", "\\", "\0")


def prediction_id(result_bytes: bytes) -> str:
    """Identify an extraction by its stored bytes.

    There is no envelope to see through: a stored prediction is the bare extraction, so this
    is a plain hash of the file. That is what keeps it reproducible in any language, with no
    unwrap rule, no schema dependency and no canonicalisation spec to agree on.
    """
    return hashlib.sha256(result_bytes).hexdigest()


def extract_result(text: str) -> str:
    """The `result` value of a prediction envelope, exactly as the vendor wrote it.

    `json.loads` discards byte offsets, so parsing and re-dumping yields *our* formatting
    rather than theirs. Every one of the 5,936 predictions in the current run differs from
    its reserialised form -- vendors write `", "` where `json.dumps` writes `","` -- so this
    is not a corner to guard against but the normal case, and hashing a reserialised form
    would make `prediction_id` describe us instead of them.

    Decoding from the value's start index returns what the decoder consumed, which is the
    original span.

    Raises:
        ValueError: if there is no `result` key, or its value will not decode. An unexpected
            payload shape should stop a build rather than acquire a plausible-looking id.
    """
    key = '"result"'
    start = text.find(key)
    if start < 0:
        raise ValueError("prediction envelope has no 'result' key")
    colon = text.index(":", start + len(key)) + 1
    while colon < len(text) and text[colon] in " \t\r\n":
        colon += 1
    _value, end = json.JSONDecoder().raw_decode(text, colon)
    return text[colon:end]


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
