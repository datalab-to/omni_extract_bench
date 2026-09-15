"""A viewer for a run, and the command that writes one.

    oeb ui --run RUN [--run RUN ...] --corpus DIR --out SITE
    python -m http.server -d SITE

One run is one prediction set, so several `--run` are what put several vendors side by side.
The result is a static directory, not a server: there is no state to keep, and a directory
opens on a laptop, behind `http.server`, or from a bucket without any of them being a
different program.

    <out>/index.html            the viewer
    <out>/index.js              one line per document: accuracy and disagreements per source
    <out>/doc/<doc_id>.json     that document's addresses, fetched when you open it
    <out>/schema/<doc_id>.json  a symlink into the corpus
    <out>/pdf/<doc_id>.pdf      a symlink to `document.pdf`, when the corpus has one

**Two files, read on opposite schedules** -- the same split `run.py` makes, for the same
reason. Choosing a document needs one number per vendor for all of them; reading a document
needs every address of that one and of no other. Inlining both is what the first version did,
and it caps out near forty documents: the whole corpus at nine vendors is ~1.1 GB of addresses,
against a 420 KB index over all 660 and ~1.7 MB for the one document you opened.

**Symlinks, so the PDFs cost nothing.** The corpus is 774 MB of them and the viewer opens one
at a time. `http.server` follows a symlink, and so does `zip` and `tar -h` if you want to hand
someone the directory.
"""
from __future__ import annotations

import json
import re
import sys
from argparse import Namespace
from pathlib import Path

from . import corpus as corpus_atlas
from .run import SUMMARY, VERDICTS

#: The viewer itself, shipped with the package. `--out` gets a copy rather than a symlink, so
#: a site keeps working after the package is upgraded under it.
VIEWER = Path(__file__).parent / "viewer" / "index.html"

INDEX = "index.js"
DOCS = "doc"
SCHEMAS = "schema"
PDFS = "pdf"

#: The PDF a document was read from, beside its two required files. Optional: the published
#: corpus omits them, because the scorer never opens one.
DOCUMENT_PDF = "document.pdf"

#: Verdicts, one character each. Every address carries one and a full run has millions, so the
#: word would be most of the payload. The viewer prints an unknown code as itself, so a verdict
#: added later shows up wrong rather than disappearing.
CODE = {"match": "m", "wrong value": "w", "missing": "x", "invented item": "i",
        "fabricated": "f", "invented field": "n", "skipped (open map)": "s"}

#: Charges that are the prediction saying something the gold did not, which is the half of a
#: near miss that pairs with a `missing`.
EXTRA = ("invented item", "invented field", "fabricated")

_DIGITS = re.compile(r"(\d+)")
_ALNUM = re.compile(r"[^0-9a-z]+")


def natural(address: str) -> list:
    """Sort key putting `lines[2]` before `lines[10]`.

    Addresses are the reading order of the whole viewer. Lexical order interleaves a
    forty-row table with itself, and the row you are checking against the page is then nowhere
    near the row above it.
    """
    return [(1, int(p), "") if p.isdigit() else (0, 0, p) for p in _DIGITS.split(address)]


def loose(value: object) -> str | None:
    """A value with everything but its letters and digits removed, or None if it has none.

    Deliberately far looser than `canon_key`, and never used to decide a score -- only to ask
    whether two values the scorer already refused to pair are the same text. `canon_key` is the
    one function that says whether two values are equal; a second one that also did would be
    the divergence its docstring is about.
    """
    if not isinstance(value, str):
        return None
    return _ALNUM.sub("", value.lower()) or None


def decode(cell: str | None) -> object:
    """A stored value back from its JSON. Verdict tables store values encoded, so that `None`
    and the string `"None"` stay different things.
    """
    if cell is None:
        return None
    try:
        return json.loads(cell)
    except ValueError:
        return cell


def near_misses(verds: list[dict]) -> list[dict]:
    """The same text charged twice: once `missing`, once invented, differing only in punctuation.

    A bullet glyph in front of a sentence is enough to do it. Those two charges are one pairing
    failure rather than two extraction errors, and nobody spots the pair by eye across a
    thousand rows -- so the viewer says so above the list.
    """
    unpaired: dict[str, str] = {}
    for v in verds:
        if v["verdict"] == "missing":
            key = loose(decode(v["gold"]))
            if key:
                unpaired.setdefault(key, decode(v["gold"]))
    if not unpaired:
        return []
    out = []
    for v in verds:
        if v["verdict"] in EXTRA:
            pred = decode(v["pred"])
            key = loose(pred)
            if key and key in unpaired:
                out.append({"gold": unpaired[key], "pred": pred})
    return out


def compact(misses: list[dict]) -> dict | None:
    """A near-miss list reduced to what the viewer actually shows: a count and one example.

    The index is the file every reader loads before seeing anything, so nothing belongs in it
    that is not needed to choose a document. Keeping the full lists made it 4.5 MB on the real
    corpus -- 82,113 entries, 93% of the file, one (document, vendor) pair worth 1.8 MB -- to
    render a number and a fragment clipped to 32 characters. Without them it is 339 KB.
    """
    if not misses:
        return None
    return {"n": len(misses), "gold": misses[0]["gold"][:80], "pred": misses[0]["pred"][:80]}


def read_summary(run: Path) -> tuple[str, dict[str, dict]]:
    """A run's summary: what to call its predictions, and its rows by document.

    The source is the run's own stamp rather than its directory name, because the stamp is
    what `score --source` recorded and the directory is whatever the caller typed.
    """
    import pyarrow.parquet as pq

    summary = run / SUMMARY
    if not summary.exists():
        raise FileNotFoundError(
            f"no run at {summary}. Produce one with:  oeb score --out {run} ...")
    table = pq.read_table(summary)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    return meta.get("source") or run.name, {r["doc_id"]: r for r in table.to_pylist()}


def canon(raw: str | None) -> str | None:
    """A stored value's canonical form -- what the scorer actually compared.

    `canon_key` is THE comparison rule, and calling it is the only honest way to show why two
    values were or were not equal: a second notion of canonical form is exactly the divergence
    its docstring is about. (Once the verdict table carries `gold_canon`/`pred_canon` this
    should read them instead of recomputing -- same function, one less place to run it.)
    """
    if raw is None:
        return None
    from .values import canon_key
    try:
        return canon_key(decode(raw))
    except Exception:                                                # noqa: BLE001
        return None


def cell(verdict: str, gold: str | None, pred: str | None) -> list:
    """One vendor's answer at one address, as the viewer reads it.

    `["m"]` means it matched the gold literally; `["m", pred]` means `canon_key` folded the
    two -- a date written another way, a number as a string. Dropping the repeated value on a
    literal match is most of the file: at nine vendors, most addresses match all nine.

    A third slot, where present, is the prediction's canonical form, and the address carries
    the gold's. Together they answer "why did these count as equal" -- or as different --
    without a second opinion about what equal means.
    """
    code = CODE.get(verdict, verdict)
    if code == "m" and pred == gold:
        return [code]
    return [code, pred]


def document_rows(doc_id: str, runs: list[tuple[str, Path]]) -> tuple[list, dict]:
    """One document's addresses across every run, and each source's near misses.

    Addresses are the union: a vendor that invents a field has an address the gold never had,
    and a vendor that skipped the document has none of them. A source missing from an
    address's map is a source that said nothing there, which the viewer draws as a dash.
    """
    import pyarrow.parquet as pq

    merged: dict[str, list] = {}
    interesting: set[str] = set()
    near = {}
    for source, run in runs:
        path = run / VERDICTS / f"{doc_id}.parquet"
        if not path.exists():
            continue
        verds = pq.read_table(path).to_pylist()
        for v in verds:
            slot = merged.setdefault(v["address"], [v["address"], v["gold"], {}])
            if slot[1] is None:
                slot[1] = v["gold"]
            c = cell(v["verdict"], v["gold"], v["pred"])
            slot[2][source] = c
            # The canonical forms are what says WHY a pair did or did not agree, so they are
            # kept only where that is a question. A value matching its gold literally has no
            # question to answer, and it is the overwhelming majority of every document.
            if len(c) > 1:
                interesting.add(v["address"])
                if v["pred"] is not None:
                    c.append(canon(v["pred"]))
        near[source] = near_misses(verds)

    rows = []
    for a in sorted(merged, key=natural):
        row = merged[a]
        if a in interesting:
            row.append(canon(row[1]))
        rows.append(row)
    return rows, near


def link(target: Path, at: Path) -> None:
    """Point `at` at `target`, replacing whatever was there.

    A symlink and not a copy: the PDFs alone are 774 MB, and a site is something you rebuild
    whenever you rescore.
    """
    at.parent.mkdir(parents=True, exist_ok=True)
    if at.is_symlink() or at.exists():
        at.unlink()
    at.symlink_to(target.resolve())


def build(runs: list[tuple[str, Path]], summaries: list[dict[str, dict]],
          root: Path, entries: dict[str, corpus_atlas.Entry], extras: dict,
          out: Path) -> dict:
    """Write the site. Returns what to print.

    Documents are handled one at a time, and their verdicts are never all in memory at once: a
    nine-vendor run over 660 documents is ~0.2 GB of them, and the machine writing the site is
    the same laptop that will read it.
    """
    out.mkdir(parents=True, exist_ok=True)
    (out / DOCS).mkdir(exist_ok=True)

    doc_ids = sorted({d for s in summaries for d in s})
    index, bytes_written, linked_pdfs, ungraded, silent = [], 0, 0, 0, 0
    for doc_id in doc_ids:
        addrs, near = document_rows(doc_id, runs)
        # A run written with `--no-verdicts` has scores and no addresses. There is nothing to
        # look at, and a document drawn with none would read as one where everything matched.
        if not addrs:
            silent += 1
            continue
        by_source = {}
        for (source, _run), summary in zip(runs, summaries):
            row = summary.get(doc_id)
            if row is None or row.get("kind") != "graded":
                continue
            answered = [row[2][source] for row in addrs if source in row[2]]
            by_source[source] = {"accuracy": round(row["accuracy"], 2),
                                 "bad": sum(1 for c in answered if c[0] != "m"),
                                 "seen": len(answered),
                                 "near": compact(near.get(source, []))}
        if not by_source:
            ungraded += 1
            continue

        payload = json.dumps({"addrs": addrs}, ensure_ascii=False, separators=(",", ":"))
        (out / DOCS / f"{doc_id}.json").write_text(payload, encoding="utf-8")
        bytes_written += len(payload.encode("utf-8"))

        entry = entries.get(doc_id)
        schema_kb = 0
        if entry is not None:
            schema = root / entry.schema_path
            if schema.is_file():
                schema_kb = round(schema.stat().st_size / 1024)
                link(schema, out / SCHEMAS / f"{doc_id}.json")
            pdf = root / Path(entry.ground_truth_path).parent / DOCUMENT_PDF
            if pdf.is_file():
                link(pdf, out / PDFS / f"{doc_id}.pdf")
                linked_pdfs += 1

        row = {"doc_id": doc_id, "total": len(addrs), "schema_kb": schema_kb,
               "by_source": by_source}
        suite = (extras.get(doc_id) or {}).get("suite")
        if suite:
            row["suite"] = suite
        index.append(row)

    head = {"sources": [s for s, _ in runs],
            "verdicts": {v: k for k, v in CODE.items()},
            "docs": index}
    (out / INDEX).write_text(
        "window.INDEX = " + json.dumps(head, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8")
    (out / "index.html").write_text(VIEWER.read_text(encoding="utf-8"), encoding="utf-8")
    return {"docs": len(index), "ungraded": ungraded, "silent": silent, "pdfs": linked_pdfs,
            "index_kb": (out / INDEX).stat().st_size / 1024,
            "addrs_mb": bytes_written / 1e6}


def cmd_ui(args: Namespace) -> int:
    """Write a browsable site for one or more runs."""
    from .run import published_corpus

    root, atlas = corpus_atlas.locate(Path(args.corpus) if args.corpus
                                      else published_corpus())
    entries = {e.doc_id: e for e in corpus_atlas.read(atlas)}
    extras = corpus_atlas.extras(atlas)

    runs, summaries, seen = [], [], {}
    for path in args.run:
        run = Path(path)
        source, summary = read_summary(run)
        if source in seen:
            raise ValueError(
                f"two runs both call their predictions {source!r}: {seen[source]} and {run}. "
                f"Name them apart with `oeb score --source`, which is what the viewer labels "
                f"its columns with.")
        seen[source] = run
        runs.append((source, run))
        summaries.append(summary)

    out = Path(args.out)
    stats = build(runs, summaries, root, entries, extras, out)
    if not stats["docs"]:
        print("  nothing to view in those runs." + (
            " Every document was scored with --no-verdicts, so no addresses were kept."
            if stats["silent"] else " No document was graded."), file=sys.stderr)
        return 1
    print(f"  {stats['docs']} documents, {len(runs)} source(s): "
          f"{', '.join(s for s, _ in runs)}")
    if stats["ungraded"]:
        print(f"  {stats['ungraded']} document(s) left out; no run graded them")
    if stats["silent"]:
        print(f"  {stats['silent']} document(s) left out; they were scored with --no-verdicts")
    print(f"  {INDEX}: {stats['index_kb']:.0f} KB   "
          f"{DOCS}/: {stats['addrs_mb']:.1f} MB, fetched one document at a time")
    print(f"  {stats['pdfs']} PDF(s) linked" if stats["pdfs"] else
          f"  no PDFs; the corpus has no {DOCUMENT_PDF} beside its documents")
    print(f"\n  python -m http.server -d {out}")
    return 0
