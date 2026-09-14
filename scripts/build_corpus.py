#!/usr/bin/env python3
"""Build the corpus: one directory per document, plus `corpus.parquet` describing them.

Reads the published dataset in its current shape -- `data/<suite>/<doc_id>/` -- and writes
the shape `docs/DATA_LAYOUT.md` settles on:

    <out>/corpus.parquet
    <out>/<doc_id>/ground_truth.json
    <out>/<doc_id>/schema.json
    <out>/<doc_id>/document.pdf
    <out>/<doc_id>/source.json        where this document came from

Every atlas column is a fact about a file or about the JSON's shape, computed without
importing the scorer. That is what lets the table stay valid across scorer versions: nothing
in it has an opinion that a scoring change could invalidate.

`source.json` holds what cannot be derived from the payloads: which collection contributed
the document. `suite` is **declared, never inferred** -- not from the `doc_id` prefix, which
394 of 660 documents carry and 266 do not, and which lies as soon as a document is
reclassified. A builder guessing from it would work for most and quietly mislabel the rest,
which is worse than failing.

Keeping it beside the document rather than in a central manifest is what lets
`corpus.parquet` be rebuilt from the tree alone, and makes adding a document one directory
rather than an edit to a shared file that two people will conflict over.

The schema is copied raw. `strip_benchmark_keys` then `resolve_refs` are the scorer's
business and change with it; applying them here would bake one version into the corpus.

Usage:
    uv run --with pyarrow --with huggingface_hub python scripts/build_corpus.py --out build/
    uv run ... python scripts/build_corpus.py --out build/ --no-pdfs     # 812 MB lighter
    uv run ... python scripts/build_corpus.py --out build/ --verify-only
    uv run ... python scripts/build_corpus.py --out build/ --rebuild-atlas
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
from omni_extract_bench import corpus as corpus_atlas                       # noqa: E402
from omni_extract_bench.corpus import check_doc_id, check_unique, verify  # noqa: E402

#: Every payload extension the corpus tree holds. A type missing here is invisible to
#: `verify` -- the same mechanism that correctly ignores `corpus.parquet` would silently
#: ignore real PDFs, and every row claiming one would read as a missing file.
PAYLOAD_PATTERNS = ("**/*.json", "**/*.pdf")

#: Declared provenance, written beside each document. Everything in it is a fact about where
#: the document came from, which no amount of reading the payloads could recover.
SOURCE = "source.json"

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
    """(suite, manifest entry) for every document in the ORIGINAL dataset, in manifest order.

    Migration only. `manifest.json` is the old layout's declaration of which collection a
    document belongs to, and this is where that fact crosses over into `source.json`.

    Its 660 are already the scored set -- the excluded and defective documents are absent --
    so nothing here filters.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    for suite in sorted(manifest):
        for entry in manifest[suite]:
            yield suite, entry


def read_source(doc_dir: Path) -> dict:
    """The declared provenance for one document.

    Raises:
        ValueError: if `source.json` is missing or declares no suite. Guessing from the
            `doc_id` prefix would succeed for 394 of 660 and mislabel the rest, and a
            plausible wrong answer is worse than a stopped build.
    """
    path = doc_dir / SOURCE
    if not path.exists():
        raise ValueError(f"{doc_dir.name}: no {SOURCE}; suite must be declared, not inferred")
    declared = json.loads(path.read_text())
    if not declared.get("suite"):
        raise ValueError(f"{doc_dir.name}: {SOURCE} declares no suite")
    return declared


def expected_payloads(rows):
    """(relative path, sha256) for every file the atlas claims, for `verify`.

    The corpus keeps several payloads per document, so the paths cannot be guessed from a row
    id; the caller derives them. PDFs are included only when the row says one exists.
    """
    for r in rows:
        yield f"{r['doc_id']}/ground_truth.json", r["gt_sha256"]
        yield f"{r['doc_id']}/schema.json", r["schema_sha256"]
        yield f"{r['doc_id']}/{SOURCE}", r["source_sha256"]
        if r["pdf_sha256"]:
            yield f"{r['doc_id']}/document.pdf", r["pdf_sha256"]


def row_for(doc_id: str, suite: str, source_id, doc_dir: Path, source_bytes: bytes,
            gt_bytes: bytes, schema_bytes: bytes, pdf_sha):
    """One atlas row.

    The first five columns are `corpus.REQUIRED` -- what anything reading an atlas depends on.
    The rest are ours: provenance and shape, additive, and nothing outside this repository
    needs them.
    """
    gt = json.loads(gt_bytes)
    gt_path, schema_path = corpus_atlas.expected_paths(doc_id)
    return {
        "doc_id": doc_id,
        "ground_truth_path": gt_path,
        "schema_path": schema_path,
        "suite": suite,
        "source_id": source_id,
        "gt_bytes": len(gt_bytes),
        "schema_bytes": len(schema_bytes),
        "gt_sha256": hashlib.sha256(gt_bytes).hexdigest(),
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "pdf_sha256": pdf_sha,
        "has_pdf": pdf_sha is not None,
        "max_array_rows": biggest_array(gt),
        "leaf_values": leaf_values(gt),
    }


def rebuild_atlas(out: Path):
    """Regenerate `corpus.parquet` from the document tree, reading declared provenance.

    This is the property `source.json` exists for: the atlas is derivable from the tree alone,
    so it can be deleted and rebuilt without the original dataset, and a document added by
    hand appears without anyone editing a central file.
    """
    docs = sorted(d for d in out.iterdir() if d.is_dir())
    check_unique([d.name for d in docs])
    rows = []
    for d in docs:
        check_doc_id(d.name)
        declared = read_source(d)
        pdf = d / "document.pdf"
        rows.append(row_for(
            d.name, declared["suite"], declared.get("source_id"), d,
            (d / SOURCE).read_bytes(),
            (d / "ground_truth.json").read_bytes(),
            (d / "schema.json").read_bytes(),
            hashlib.sha256(pdf.read_bytes()).hexdigest() if pdf.exists() else None))
    return rows


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

        # The one fact the payloads cannot carry: which collection this came from.
        source_bytes = json.dumps({"suite": suite,
                                   "source_id": entry.get("source_id")},
                                  indent=2).encode()
        (dst / SOURCE).write_bytes(source_bytes)

        rows.append(row_for(doc_id, suite, entry.get("source_id"), dst,
                            source_bytes, gt_bytes, schema_bytes, pdf_sha))
    elapsed = time.monotonic() - t0
    print(f"  {len(rows)} documents in {elapsed:.1f}s"
          + (f", including {pdf_bytes / 1e6:.0f} MB of PDFs" if with_pdfs else ", no PDFs"))
    if with_pdfs:
        # Local copies are near-free on a copy-on-write filesystem; uploading them is not.
        print("  (a local copy is cheap on APFS; the 812 MB is real on upload)")
    return rows


def write_atlas(rows, out: Path, snapshot: str):
    """Enrich and hand off. `corpus.write` is the only thing that writes an atlas.

    This builder knows things the contract does not -- which collection a document came from,
    how big its gold is, whether a PDF exists -- and those go in as extra columns. What it
    must not do is write the file itself: that is how the two required path columns ended up
    missing from the published corpus while `corpus.py` already defined them.
    """
    entries = [corpus_atlas.Entry(r["doc_id"], r["ground_truth_path"], r["schema_path"],
                                  r["gt_sha256"], r["schema_sha256"]) for r in rows]
    extra = {r["doc_id"]: {k: v for k, v in r.items() if k not in corpus_atlas.REQUIRED}
             for r in rows}
    path = corpus_atlas.write(out, entries, extra, metadata={
        "corpus_snapshot": snapshot,
        "repo": REPO,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(f"  {ATLAS}: {path.stat().st_size / 1024:.0f} KB, {len(rows)} rows, "
          f"version {corpus_atlas.version(entries)}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-pdfs", action="store_true",
                    help="skip document.pdf; 812 MB lighter, and the scorer never reads them")
    ap.add_argument("--verify-only", action="store_true",
                    help="check an existing tree against its atlas and stop")
    ap.add_argument("--rebuild-atlas", action="store_true",
                    help="regenerate corpus.parquet from an existing tree, reading source.json")
    args = ap.parse_args()

    if args.verify_only:
        rows = pq.read_table(args.out / ATLAS).to_pylist()
        problems = verify(expected_payloads(rows), args.out, PAYLOAD_PATTERNS)
        print(f"verify: {len(rows)} rows, {len(problems)} problems")
        for p in problems[:20]:
            print(f"  {p}")
        return 1 if problems else 0

    if args.rebuild_atlas:
        rows = rebuild_atlas(args.out)
        write_atlas(rows, args.out, "(rebuilt from tree)")
        problems = verify(expected_payloads(rows), args.out, PAYLOAD_PATTERNS)
        print(f"  verify: {len(problems)} problems")
        for p in problems[:10]:
            print(f"    {p}")
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
