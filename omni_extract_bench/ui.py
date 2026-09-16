"""A viewer for one or more runs, and the command that writes one.

    oeb ui --run run/ --out site/
    oeb ui --run runs/*/ --out site/        several vendors side by side
    python -m http.server -d site/

**A run directory is the only input.** `scores/` carries the schema and, where the manifest
supplied one, `doc_path`; `verdicts/` carries the addresses. Nothing else is needed and nothing
else is accepted -- no corpus, no manifest, no atlas. Hand someone a run and they can audit it.

    <out>/index.html            the viewer
    <out>/index.js              one line per document: accuracy and disagreements per run
    <out>/doc/<doc_id>.json     that document's addresses, fetched when you open it
    <out>/schema/<doc_id>.json  what was asked for, written from the scores table
    <out>/pdf/<doc_id>.pdf      a symlink, where a run carried `doc_path`

**Two files, read on opposite schedules.** Choosing a document needs one number per run for all
of them; reading a document needs every address of that one and of no other. Inlining both caps
out near forty documents: nine vendors over a corpus is ~1.1 GB of addresses, against a few
hundred KB of index and a megabyte or two for the document you opened.

**Symlinks for the PDFs, because the viewer opens one at a time** and a corpus of them is
hundreds of megabytes. `http.server` follows a symlink, and so do `zip` and `tar -h`.

Local paths only. A run in a bucket is one `aws s3 sync` away from being a run on a disk, and
that keeps this module free of a filesystem abstraction it would otherwise need everywhere.
"""
from __future__ import annotations

import glob as _glob
import json
import re
import sys
from argparse import Namespace
from pathlib import Path

#: The viewer itself, shipped with the package. `--out` gets a copy rather than a symlink, so a
#: site keeps working after the package is upgraded under it.
VIEWER = Path(__file__).parent / "viewer" / "index.html"

INDEX, DOCS, SCHEMAS, PDFS = "index.js", "doc", "schema", "pdf"

#: Verdicts, one character each. Every address carries one and a run has millions, so the word
#: would be most of the payload. The viewer prints an unknown code as itself, so a verdict added
#: later shows up wrong rather than disappearing.
CODE = {"matched": "m", "misread": "w", "unfound": "x", "invented_item": "i",
        "fabricated": "f", "invented_field": "n", "skipped_open_map": "s"}

_DIGITS = re.compile(r"(\d+)")


def natural(address: str) -> list:
    """Sort key putting `lines[2]` before `lines[10]`.

    Addresses are the reading order of the whole viewer. Lexical order interleaves a forty-row
    table with itself, and the row you are checking against the page ends up nowhere near the
    one above it.
    """
    return [(1, int(p), "") if p.isdigit() else (0, 0, p) for p in _DIGITS.split(address)]


def decode(cell: str | None) -> object:
    """A stored value back from its JSON. Verdict tables store values encoded, so that `None`
    and the string `"None"` stay different things."""
    if cell is None:
        return None
    try:
        return json.loads(cell)
    except ValueError:
        return cell


def cell(verdict: str, gold: object, pred: object) -> list:
    """One run's answer at one address, as the viewer reads it.

    `["m"]` means it matched the gold literally; `["m", pred]` means `canon_key` folded the two
    -- a date written another way, a number as a string. Dropping the repeated value on a
    literal match is most of the file: at nine runs, most addresses match all nine, and the
    canonical forms are only appended where there is a question to answer.
    """
    code = CODE.get(verdict, verdict)
    if code == "m" and pred == gold:
        return [code]
    return [code, pred]


def runs_from(patterns: list[str]) -> list[tuple[str, Path]]:
    """Every run named, with the label its column gets: the directory's own name.

    A pattern is expanded here as well as by the shell, so `--run 'runs/*'` works quoted or
    not. A directory without a `scores/` is an error rather than a skip -- a glob that sweeps
    in a sibling would otherwise produce a site quietly missing a vendor.
    """
    paths: list[Path] = []
    for pattern in patterns:
        hits = sorted(_glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        if not hits:
            raise ValueError(f"--run {pattern} matched nothing")
        paths.extend(Path(h) for h in hits)

    out, seen = [], {}
    for path in paths:
        if not (path / "scores").exists():
            raise ValueError(f"{path} is not a run: it has no scores/. "
                             f"Produce one with:  oeb score --out {path} ...")
        label = path.name
        if label in seen:
            raise ValueError(f"two runs are both called {label!r}: {seen[label]} and {path}. "
                             f"The directory name is what labels a column, so rename one.")
        seen[label] = path
        out.append((label, path))
    return out


def scores_of(run: Path) -> dict[str, dict]:
    """A run's scores, by document. Every row, including the ones that failed."""
    import pyarrow.parquet as pq

    return {r["doc_id"]: r for r in pq.read_table(run / "scores").to_pylist()}


def addresses(doc_id: str, runs: list[tuple[str, Path]]) -> list:
    """One document's addresses across every run.

    A filtered read per run rather than one pass held in memory: parquet's row-group statistics
    skip almost everything, so it costs ~2 ms against a 2.9M-row table, and only one document
    is ever in memory. The addresses are the union -- a run that invented a field has one the
    gold never had -- and a run missing from an address is one that said nothing there, which
    the viewer draws as a dash.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    merged: dict[str, list] = {}
    interesting: set[str] = set()
    canons: dict[str, str | None] = {}
    for source, run in runs:
        rows = ds.dataset(run / "verdicts", format="parquet").to_table(
            filter=pc.field("doc_id") == doc_id).to_pylist()
        for v in rows:
            gold = decode(v["gold_raw"])
            slot = merged.setdefault(v["address"], [v["address"], gold, {}])
            if slot[1] is None:
                slot[1] = gold
            c = cell(v["verdict"], gold, decode(v["pred_raw"]))
            slot[2][source] = c
            # The canonical forms say WHY a pair did or did not agree, so they are carried only
            # where that is a question. A value matching its gold literally has none, and that
            # is the overwhelming majority of every document.
            if len(c) > 1:
                interesting.add(v["address"])
                c.append(v["pred_canon"])
                canons.setdefault(v["address"], v["gold_canon"])

    out = []
    for a in sorted(merged, key=natural):
        row = merged[a]
        if a in interesting:
            row.append(canons.get(a))
        out.append(row)
    return out


def link(target: Path, at: Path) -> None:
    """Point `at` at `target`, replacing whatever was there."""
    at.parent.mkdir(parents=True, exist_ok=True)
    if at.is_symlink() or at.exists():
        at.unlink()
    at.symlink_to(target.resolve())


def build(runs: list[tuple[str, Path]], out: Path) -> dict:
    """Write the site. Returns what to print.

    Documents are handled one at a time and their addresses are never all in memory: a
    nine-vendor run over a corpus is ~1.1 GB of them, and the machine writing the site is the
    laptop that will read it.
    """
    out.mkdir(parents=True, exist_ok=True)
    (out / DOCS).mkdir(exist_ok=True)
    scored = {label: scores_of(run) for label, run in runs}

    index, written, pdfs, schemas, blank = [], 0, 0, 0, 0
    for doc_id in sorted({d for rows in scored.values() for d in rows}):
        addrs = addresses(doc_id, runs)
        by_source = {}
        for label, _run in runs:
            row = scored[label].get(doc_id)
            # Absence means "this run did not grade it", which the viewer draws as a dash. A
            # failed document keeps its row in the index, so it cannot vanish from a mean.
            if row is None or row["status"] != "scored":
                continue
            answered = [a[2][label] for a in addrs if label in a[2]]
            by_source[label] = {"accuracy": round(row["accuracy"], 2),
                                "bad": sum(1 for c in answered if c[0] != "m"),
                                "seen": len(answered)}
        if not addrs:
            blank += 1
            continue

        payload = json.dumps({"addrs": addrs}, ensure_ascii=False, separators=(",", ":"))
        (out / DOCS / f"{doc_id}.json").write_text(payload, encoding="utf-8")
        written += len(payload.encode("utf-8"))

        # The schema and the PDF come from whichever run has them; they describe the document,
        # not the run, so the first one that carries them wins.
        row = next((scored[l].get(doc_id) for l, _ in runs if scored[l].get(doc_id)), None)
        schema_kb = 0
        if row is not None and row.get("schema"):
            raw = row["schema"]
            body = raw if isinstance(raw, bytes) else str(raw).encode()
            (out / SCHEMAS).mkdir(exist_ok=True)
            (out / SCHEMAS / f"{doc_id}.json").write_bytes(body)
            schema_kb, schemas = round(len(body) / 1024), schemas + 1
        if row is not None and row.get("doc_path"):
            pdf = Path(str(row["doc_path"]))
            if pdf.is_file():
                link(pdf, out / PDFS / f"{doc_id}.pdf")
                pdfs += 1

        index.append({"doc_id": doc_id, "total": len(addrs), "schema_kb": schema_kb,
                      "by_source": by_source})

    head = {"sources": [label for label, _ in runs],
            "verdicts": {v: k for k, v in CODE.items()},
            "docs": index}
    (out / INDEX).write_text(
        "window.INDEX = " + json.dumps(head, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8")
    (out / "index.html").write_text(VIEWER.read_text(encoding="utf-8"), encoding="utf-8")
    return {"docs": len(index), "blank": blank, "pdfs": pdfs, "schemas": schemas,
            "index_kb": (out / INDEX).stat().st_size / 1024, "addrs_mb": written / 1e6}


def cmd_ui(args: Namespace) -> int:
    """Write a browsable site for one or more runs."""
    runs = runs_from(args.run)
    out = Path(args.out)
    stats = build(runs, out)
    if not stats["docs"]:
        print("  nothing to view: no document in those runs has any address.", file=sys.stderr)
        return 1

    print(f"  {stats['docs']} documents, {len(runs)} run(s): "
          f"{', '.join(label for label, _ in runs)}")
    if stats["blank"]:
        print(f"  {stats['blank']} document(s) left out; none of the runs graded them")
    print(f"  {INDEX}: {stats['index_kb']:.0f} KB   "
          f"{DOCS}/: {stats['addrs_mb']:.1f} MB, fetched one document at a time")
    print(f"  {stats['schemas']} schema(s), {stats['pdfs']} PDF(s) linked"
          + ("" if stats["pdfs"] else "; add a doc_path column to a manifest to get them"))
    print(f"\n  python -m http.server -d {out}")
    return 0
