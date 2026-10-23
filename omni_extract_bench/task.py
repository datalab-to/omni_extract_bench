"""The extraction task: a PDF and a JSON Schema go in, a JSON object comes out.

Each row is a document and its schema (`data.Doc`). What a system is sent is the PDF and the schema under
the benchmark's conventions (`harness.contract.Question`).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import ClassVar

from omni_bench_core import task

from . import data
from .harness import contract
from .harness import schema as SCHEMA


@dataclasses.dataclass(frozen=True)
class Extract(task.Task[data.Doc, contract.Question]):
    """Extraction: a PDF and a JSON Schema go in, a JSON object comes out. The schema is sent with the
    benchmark's conventions stated in its field descriptions (`harness.schema.apply_overlay`), as every
    published run sends it, and without the benchmark's own annotations."""

    name: ClassVar[str] = "extract"

    def rows(self, dataset: str | Path, manifest: str = data.MANIFEST) -> tuple[data.Doc, ...]:
        return data.read_manifest(dataset, manifest)[1]

    def input(self, row: data.Doc) -> contract.Question:
        if not row.doc_path.exists(): raise FileNotFoundError(row.doc_path)
        return contract.Question(row.doc_path, SCHEMA.strip_benchmark_keys(SCHEMA.apply_overlay(row.schema)))
