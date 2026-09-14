#!/usr/bin/env python3
"""Build the corpus: one directory per document, plus `corpus.parquet` describing them.

Reads the published dataset in its current shape -- `data/<suite>/<doc_id>/` -- and writes
the shape `docs/DATA_LAYOUT.md` settles on:

    <out>/corpus.parquet
    <out>/<doc_id>/ground_truth.json
    <out>/<doc_id>/schema.json
    <out>/<doc_id>/document.pdf

Every atlas column is a fact about a file or about the JSON's shape, computed without
importing the scorer. That is what lets the table stay valid across scorer versions: nothing
in it has an opinion that a scoring change could invalidate.

The schema is copied raw. `strip_benchmark_keys` then `resolve_refs` are the scorer's
business and change with it; applying them here would bake one version into the corpus.

Usage:
    uv run --with pyarrow --with huggingface_hub python scripts/build_corpus.py --out build/
    uv run ... python scripts/build_corpus.py --out build/ --no-pdfs     # 812 MB lighter
    uv run ... python scripts/build_corpus.py --out build/ --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from omni_extract_bench.layout import check_doc_id, check_unique, verify  # noqa: E402

#: Every payload extension the corpus tree holds. A type missing here is invisible to
#: `verify` -- the same mechanism that correctly ignores `corpus.parquet` would silently
#: ignore real PDFs, and every row claiming one would read as a missing file.
PAYLOAD_PATTERNS = ("**/*.json", "**/*.pdf")

REPO = "datalab-to/omni_extract_bench"
ATLAS = "corpus.parquet"


def biggest_array(node) -> int:
    """Rows in the largest array of objects, at any depth.

    The number that drives matching cost: the scorer solves an assignment over this array and
    both its time and memory go as the square. Walked rather than imported, so the atlas does
    not depend on the scorer to describe itself.
    """
    best, stack = 0, [node]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            stack.extend(n.values())
        elif isinstance(n, list):
            if n and isinstance(n[0], dict):
                best = max(best, len(n))
            stack.extend(n)
    return best


def leaf_values(node) -> int:
    """Scalar values in the document, which approximates flattening cost."""
    total, stack = 0, [node]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            stack.extend(n.values())
        elif isinstance(n, list):
            stack.extend(n)
        else:
            total += 1
    return total


def source_documents(root: Path):
    """(suite, manifest entry) for every document, in manifest order.

    `manifest.json`'s 660 are already the scored set -- the excluded and defective documents
    are absent from it -- so nothing here filters.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    for suite in sorted(manifest):
        for entry in manifest[suite]:
            yield suite, entry


def expected_payloads(rows):
    """(relative path, sha256) for every file the atlas claims, for `verify`.

    The corpus keeps several payloads per document, so the paths cannot be guessed from a row
    id; the caller derives them. PDFs are included only when the row says one exists.
    """
    for r in rows:
        yield f"{r['doc_id']}/ground_truth.json", r["gt_sha256"]
        yield f"{r['doc_id']}/schema.json", r["schema_sha256"]
        if r["pdf_sha256"]:
            yield f"{r['doc_id']}/document.pdf", r["pdf_sha256"]


def build(root: Path, out: Path, with_pdfs: bool):
    docs = list(source_documents(root))
    ids = [e["doc_id"] for _, e in docs]

    # Assert what the flat layout needs, before writing anything. Without the suite
    # directory, a duplicate id silently overwrites its twin.
    check_unique(ids)
    for doc_id in ids:
        check_doc_id(doc_id)

    rows, t0, pdf_bytes = [], time.monotonic(), 0
    for suite, entry in docs:
        doc_id = entry["doc_id"]
        src, dst = root / "data" / suite / doc_id, out / doc_id
        dst.mkdir(parents=True, exist_ok=True)

        gt_bytes = (src / "ground_truth.json").read_bytes()
        schema_bytes = (src / "schema.json").read_bytes()
        (dst / "ground_truth.json").write_bytes(gt_bytes)
        (dst / "schema.json").write_bytes(schema_bytes)

        pdf_sha = None
        pdf = src / "document.pdf"
        if with_pdfs and pdf.exists():
            shutil.copyfile(pdf, dst / "document.pdf")
            body = pdf.read_bytes()
            pdf_sha = hashlib.sha256(body).hexdigest()
            pdf_bytes += len(body)

        gt = json.loads(gt_bytes)
        rows.append({
            "doc_id": doc_id,
            "suite": suite,
            "source_id": entry.get("source_id"),
            "gt_bytes": len(gt_bytes),
            "schema_bytes": len(schema_bytes),
            "gt_sha256": hashlib.sha256(gt_bytes).hexdigest(),
            "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
            "pdf_sha256": pdf_sha,
            "has_pdf": bool(entry.get("has_pdf")),
            "max_array_rows": biggest_array(gt),
            "leaf_values": leaf_values(gt),
        })
    elapsed = time.monotonic() - t0
    print(f"  {len(rows)} documents in {elapsed:.1f}s"
          + (f", including {pdf_bytes / 1e6:.0f} MB of PDFs" if with_pdfs else ", no PDFs"))
    if with_pdfs:
        # Local copies are near-free on a copy-on-write filesystem; uploading them is not.
        print("  (a local copy is cheap on APFS; the 812 MB is real on upload)")
    return rows


def write_atlas(rows, out: Path, snapshot: str):
    table = pa.Table.from_pylist(rows).replace_schema_metadata({
        "corpus_snapshot": snapshot,
        "repo": REPO,
        "rows": str(len(rows)),
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    path = out / ATLAS
    pq.write_table(table, path, compression="zstd")
    print(f"  {ATLAS}: {path.stat().st_size / 1024:.0f} KB, {len(rows)} rows")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-pdfs", action="store_true",
                    help="skip document.pdf; 812 MB lighter, and the scorer never reads them")
    ap.add_argument("--verify-only", action="store_true",
                    help="check an existing tree against its atlas and stop")
    args = ap.parse_args()

    if args.verify_only:
        rows = pq.read_table(args.out / ATLAS).to_pylist()
        problems = verify(expected_payloads(rows), args.out, PAYLOAD_PATTERNS)
        print(f"verify: {len(rows)} rows, {len(problems)} problems")
        for p in problems[:20]:
            print(f"  {p}")
        return 1 if problems else 0

    root = Path(snapshot_download(REPO, repo_type="dataset"))
    print(f"source snapshot {root.name}")
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    rows = build(root, args.out, with_pdfs=not args.no_pdfs)
    write_atlas(rows, args.out, root.name)

    # Verify what was just written, every time. A builder that cannot check its own output
    # is how an atlas starts looking authoritative while being wrong.
    problems = verify(expected_payloads(rows), args.out, PAYLOAD_PATTERNS)
    print(f"  verify: {len(problems)} problems")
    for p in problems[:10]:
        print(f"    {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
