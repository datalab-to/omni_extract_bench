"""Omni Extract Bench: the extraction task (`task.Extract`) on a dataset, graded by field matching.

    from omni_extract_bench import benchmark
    from omni_extract_bench.harness.providers.datalab import Datalab

    b = benchmark.default()                                   # the public dataset, graded by field matching
    b.run(Datalab(mode="accurate"), out="runs", sample=20)    # one run is one system
    b.grade("mine/")                                          # a folder of your own predictions

The loop (planning, predicting, resuming, grading, run folders) is omni-bench-core's `Benchmark`.
"""
from __future__ import annotations

from pathlib import Path

from omni_bench_core import benchmark

from . import data, task
from .scorers import field_matching


def default(dataset: str | Path = data.DEFAULT_DATASET, manifest: str = data.MANIFEST, *,
            overlay: bool = True) -> benchmark.Benchmark:
    """Omni Extract Bench: the extraction task on `dataset`'s `manifest`, graded by field matching.

    Its parameters are the benchmark's options, and nothing else: `overlay` is the task's (`task.Extract`).
    Another task or other scorers make another benchmark, built directly:
    `omni_bench_core.benchmark.Benchmark(task.Extract(), dataset, [FieldMatching(), ...])`."""
    return benchmark.Benchmark(task.Extract(overlay), dataset, [field_matching.FieldMatching()], manifest)
