"""The dataset's rows: one document and one schema each.

A dataset folder holds:

    manifest.parquet       a row per document: doc_id, suite, doc_path, gt_path, schema
    manifests/*.parquet    other manifests, if any (a sample of the same documents)
    pdfs/<file>.pdf        the document that is sent
    gold/<doc_id>.json     its ground truth, the extraction

Downloading and checking a dataset is omni-bench-core's (`omni_bench_core.data`). This module adds the row type, `Doc`,
and the check that each row's schema is a JSON object.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from omni_bench_core import data, errors

REPO = "datalab-to/omni_extract_bench"
DEFAULT_DATASET = f"{data.HF_PREFIX}{REPO}"
MANIFEST = data.MANIFEST
NO_SUITE = data.NO_SUITE
COLUMNS = ("doc_id", "suite", "doc_path", "gt_path", "schema")


@dataclasses.dataclass(frozen=True)
class Doc:
    """One document of a dataset, with the schema it is asked against.

    Its paths are full paths on this machine. `extra` holds the manifest's other columns, unread, so a
    dataset can carry more for scorers of its own. A copy with a field changed is
    `dataclasses.replace(doc, ...)`."""

    doc_id: str
    suite: str
    doc_path: Path
    gt_path: Path
    schema: dict
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def id(self) -> str:
        """The document's id, by the name every benchmark's rows share."""
        return self.doc_id


def read_manifest(dataset: str | Path = DEFAULT_DATASET, manifest: str = MANIFEST) -> tuple[Path, tuple[Doc, ...]]:
    """The dataset's folder, and every document its manifest names, checked, in the manifest's order."""
    root, rows = data.read_manifest(dataset, manifest, columns=("schema",), id="doc_id", inputs=("doc_path",),
                                    gold="gt_path")
    return root, tuple(doc(r) for r in rows)


def doc(r: dict[str, Any]) -> Doc:
    """One checked manifest row as a `Doc`. Its schema is a JSON string in the manifest, read as an object."""
    try:
        schema = json.loads(r["schema"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise errors.UsageError(f"document {r['doc_id']} has a schema that is not a JSON string ({exc}); the "
                                "schema column holds the JSON Schema as text: write it with json.dumps(schema)"
                                ) from None
    if not isinstance(schema, dict) or not schema:
        raise errors.UsageError(f"document {r['doc_id']} has schema {r['schema']!r:.80}; it must be a non-empty "
                                "JSON object, the JSON Schema of the extraction")
    return Doc(r["doc_id"], r["suite"], r["doc_path"], r["gt_path"], schema, r["extra"])
