"""Scoring a corpus from the command line, and writing the result where a query can reach it.

`bench.py` yields plain rows and knows nothing about storage. This is the part that knows
about files: where the published corpus comes from, and what a run looks like on disk.

    <out>/summary.parquet              one row per prediction
    <out>/verdicts/<doc_id>.parquet    every address, with gold and pred

**`--out` is the run directory, exactly as given.** No tree is invented under it. What decides
whether two runs are comparable -- the corpus version, the scorer commit, the source -- is
stamped in `summary.parquet`'s metadata, where it can be read rather than parsed out of a path.
Organising runs is the caller's business, and a convention imposed by the tool would be one
more thing to learn and to work around.

A finished run is not overwritten: the answer for one corpus, one scorer and one set of
predictions is single. Scoring it again means naming a different directory, and comparing the
two is a join on `(doc_id, prediction_id)` -- which is why there is no `--recheck` verb for it.

**One prediction set per run.** Predictions identical across two vendors are therefore scored
twice. Deduplicating them saved 12.4% of a run that is now minutes of fan-out, and cost a
summary table where `count(*)` was not the number of predictions.

Two tables because they are read on opposite schedules. A leaderboard reads every summary row
and no verdicts; an audit reads one document's verdicts and no summary. One file holding both
would drag the audit trail through memory on every leaderboard query -- and the audit trail is
two orders of magnitude larger (~600 KB of summary against ~0.2 GB of verdicts for the full
nine-vendor run; ~20 MB for a single user's 660 documents).

Partitioned per document because that also makes correcting a document cheap: rescoring one
rewrites one small file and touches nothing else. Parquet has no append, so a single verdicts
table would have to be read, concatenated and rewritten whole.

`doc_id` is both the filename and a column, deliberately. Without the column a glob query
would have to parse filenames to group by document, and after compression the repetition costs
nothing:

    select address, gold, pred from 'run/verdicts/*.parquet' where verdict = 'wrong value'
"""
from __future__ import annotations

import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Iterator

from . import corpus as corpus_atlas
from .bench import cases, documents, predictions, score, verdict_rows
from .corpus import Entry

#: The published benchmark. `--corpus` overrides it, which is how you score against your own
#: ground truth: the loader does not care whose corpus it is.
HF_REPO = "datalab-to/omni_extract_bench"

SUMMARY = "summary.parquet"
VERDICTS = "verdicts"


def scorer_version() -> dict[str, str]:
    """Which scorer produced a score.

    The installed version, and nothing else. This used to shell out to git for a commit and a
    dirty flag, which cost a subprocess, a rule for ignoring `__pycache__`, and another for
    vendored installs where the surrounding repository is not ours -- all to record a field
    nothing reads. A commit also only means something inside this repository; anyone who pip
    installs the package never had one.
    """
    from . import __version__
    return {"scorer_version": str(__version__)}


def environment() -> dict[str, str]:
    """What ran, beyond the code. The same commit on x86 and ARM may not agree, and nobody has
    checked -- so record enough to tell those runs apart afterwards rather than assuming.
    """
    import platform

    import numpy
    import scipy
    from .bench import PREDICTION_ID_VERSION
    return {"python": platform.python_version(),
            "platform": f"{platform.system()}-{platform.machine()}",
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "prediction_id_version": PREDICTION_ID_VERSION}


def published_corpus() -> Path:
    """Download the published corpus and return its directory.

    Only the two files a document must have. The PDFs are 812 MB and the scorer never opens
    one, so fetching them by default would cost most users most of the download for nothing.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:                                     # pragma: no cover - env specific
        raise SystemExit(
            "downloading the published corpus needs huggingface_hub: "
            "pip install 'omni-extract-bench[benchmark]', or pass --corpus to use your own")
    return Path(snapshot_download(
        HF_REPO, repo_type="dataset",
        allow_patterns=[corpus_atlas.ATLAS, "*/ground_truth.json", "*/schema.json"]))


def _write(table, path: Path) -> None:
    """Write one parquet, atomically. A crash then leaves the previous file, not a truncated
    one -- which matters most for the summary, the only thing that says a run is finished.
    """
    import os

    import pyarrow.parquet as pq

    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def write_run(out: Path, rows: list[dict], by_doc: dict[str, list[dict]],
              stamp: dict[str, str]) -> None:
    """Write one run: the summary table, and one verdict file per document.

    Two tables because they are read on opposite schedules -- a leaderboard reads every summary
    row and no verdicts, an audit reads one document's verdicts and no summary -- and the
    verdict side is two orders of magnitude larger.

    Per document because that is the unit that produced them, and because a corrected document
    then rewrites one small file. Both are written to a temporary name and renamed, so a crash
    leaves the previous table rather than a truncated one.
    """
    import pyarrow as pa

    out.mkdir(parents=True, exist_ok=True)
    (out / VERDICTS).mkdir(exist_ok=True)

    # An explicit union of keys: a run that graded nothing would otherwise infer a schema with
    # no metric columns at all, and two runs would stop being comparable.
    fields = list(dict.fromkeys(k for r in rows for k in r))
    blank = {k: None for k in fields}
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    meta = {**{k: str(v) for k, v in stamp.items()}, "rows": str(len(rows)),
            "kinds": json.dumps(dict(sorted(kinds.items()))),
            "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write(pa.Table.from_pylist([{**blank, **r} for r in rows]).replace_schema_metadata(meta),
           out / SUMMARY)
    written = 0
    for doc_id, verds in by_doc.items():
        if verds:
            _write(pa.Table.from_pylist(verds), out / VERDICTS / f"{doc_id}.parquet")
            written += 1
    size = (out / SUMMARY).stat().st_size / 1024
    vsize = sum(f.stat().st_size for f in (out / VERDICTS).glob("*.parquet")) / 1e6
    print(f"  {out}/{SUMMARY}: {size:.0f} KB, {len(rows)} rows")
    print(f"  {VERDICTS}/: {written} files, {vsize:.1f} MB")


#: What a user can get wrong, as opposed to what a bug looks like. `cli.main` reports these as
#: a message; anything else keeps its traceback, because it is ours to fix. One boundary rather
#: than a try/except per command, which is four chances to forget one.
USER_ERRORS = (FileNotFoundError, NotADirectoryError, ValueError)


def _jobs(preds: Path, corpus: Path,
          entries: dict[str, Entry]) -> tuple[dict[str, bytes], list[str]]:
    """Every prediction to score, by document, and the ones the atlas does not ask for.

    Filtering the atlas is how you choose a subset to run, so a prediction for a document that
    is no longer listed is the normal result of that -- skipped, and counted. A prediction
    naming nothing at all is still a mistake, usually a filename convention
    (`invoice-003.pdf.json`), and the two are told apart by whether the document is on disk.

    Raises:
        ValueError: for a prediction matching neither the atlas nor the tree, suggesting near
            matches -- the fix is almost always visible once you are shown what was close.
    """
    import difflib

    out, skipped = {}, []
    for doc_id, raw in predictions(preds):
        if doc_id in entries:
            out[doc_id] = raw
        elif (Path(corpus) / doc_id).is_dir():
            skipped.append(doc_id)
        else:
            close = difflib.get_close_matches(doc_id, entries, n=3, cutoff=0.6)
            raise ValueError(
                f"prediction {doc_id!r} matches no document in the corpus."
                + (f" Did you mean: {', '.join(close)}?" if close else "")
                + " A prediction file must be named <doc_id>.json.")
    return out, skipped


def _report(rows: list[dict]) -> int:
    """Say what happened, keeping the three outcomes apart. Returns the exit code.

    The mean is over the graded only. Averaging an unusable prediction in as a zero makes a
    rate-limited run look like a bad model, and averaging in one this harness could not score
    makes our bug look like theirs.
    """
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    graded = [r for r in rows if r["kind"] == "graded"]
    failed = [r for r in rows if r["kind"] == "failed"]
    print(f"\n  {kinds}")
    print(f"  mean accuracy over the {len(graded)} graded: "
          f"{sum(r['accuracy'] for r in graded) / len(graded) if graded else 0.0:.2f}")
    if kinds.get("unusable"):
        print(f"  {kinds['unusable']} unusable (the provider returned nothing scoreable), "
              f"excluded from that mean")
    if failed:
        print(f"  {len(failed)} NOT SCORED -- this harness could not score them:",
              file=sys.stderr)
        for r in failed[:10]:
            print(f"      {r['doc_id']}: {r['error']}", file=sys.stderr)
    return 1 if failed else 0


def cmd_score(args: Namespace) -> int:
    """Score a directory of predictions against a corpus."""
    from .bench import PREDICTION_META, cost, document, prediction_meta

    corpus, atlas = corpus_atlas.locate(Path(args.corpus) if args.corpus
                                        else published_corpus())
    preds = Path(args.predictions)
    entries = {e.doc_id: e for e in corpus_atlas.read(atlas)}
    stamp = {"source": args.source or preds.name, **scorer_version(), **environment()}
    out = Path(args.out) if args.out else None

    carried, collided = prediction_meta(preds)
    if collided:
        print(f"  {PREDICTION_META} columns dropped, they collide with the scorer's own: "
              f"{', '.join(collided)}", file=sys.stderr)

    jobs, skipped = _jobs(preds, corpus, entries)
    if not jobs:
        print("no predictions found; expected <doc_id>.json files", file=sys.stderr)
        return 1
    print(f"  scorer {stamp['scorer_version']}, source {stamp['source']}")
    print(f"  {len(jobs)} predictions, {len(entries)} documents in the atlas")
    if skipped:
        print(f"  {len(skipped)} prediction(s) skipped; the atlas does not list them: "
              f"{', '.join(sorted(skipped)[:5])}"
              + (f" and {len(skipped) - 5} more" if len(skipped) > 5 else ""))

    # A finished run is that corpus, scorer and source's answer, and there is only one.
    if out is not None and (out / SUMMARY).exists():
        print(f"  {out / SUMMARY} exists; this corpus, scorer and source already have an "
              f"answer.\n  To score it again, name a different --out; comparing two runs "
              f"is a join on (doc_id, prediction_id).", file=sys.stderr)
        return 1

    # Longest first: the cost spread is extreme -- a median document has 6 array rows against a
    # largest of 26,725 -- so any other order finishes the cheap work and waits on a straggler.
    ordered = sorted(jobs, key=lambda d: -cost(document(corpus, entries[d]).gt))

    rows, by_doc = [], {}
    for doc_id, row, verds, kind in _scored(ordered, jobs, corpus, entries,
                                            args.jobs, not args.no_verdicts):
        row.update(carried.get(doc_id, {}))
        rows.append(row)
        by_doc[doc_id] = verds
        if out is None:
            shown = f"{row['accuracy']:>7.2f}" if kind == "graded" else f"{kind:>9}"
            print(f"  {doc_id:<52}{shown}  {row.get('error') or ''}".rstrip())

    code = _report(rows)
    if out is not None:
        write_run(out, rows, by_doc, stamp)
    return code


#: One document's work, as a worker receives it: which document, its atlas row, and the
#: prediction's bytes. A plain tuple because it crosses a process boundary.
Job = tuple[str, Entry, bytes]

#: What comes back: the document, its summary row, its verdict rows, and the outcome kind.
Scored = tuple[str, dict, list[dict], str]

#: Set once per worker, because a pool cannot pass these on every call without repeating them.
_CFG: dict = {}


def _init(corpus: str, verdicts: bool) -> None:
    _CFG.update(corpus=Path(corpus), verdicts=verdicts)


def _score_document(job: Job) -> Scored:
    """Score one document, in a worker. The unit of parallelism.

    The document is loaded here rather than sent: a parsed ground truth can be tens of
    megabytes, and pickling one to a worker costs more than reading it there.
    """
    from .bench import Case, document, prediction_id, score, summary_row, verdict_rows

    doc_id, entry, raw = job
    doc = document(_CFG["corpus"], entry)
    case = Case(doc=doc, prediction_id=prediction_id(raw), raw=raw)
    outcome = score(case, verdicts=_CFG["verdicts"])
    return doc_id, summary_row(case, outcome), verdict_rows(case, outcome), outcome.kind


def _scored(ordered: list[str], jobs: dict[str, bytes], corpus: Path,
            entries: dict[str, Entry], workers: int, verdicts: bool) -> Iterator[Scored]:
    """Scored documents, from a pool or from this process.

    One worker runs inline rather than spawning a pool of one: a spawned worker re-imports the
    main module, and a traceback from another process arrives with its frames flattened.
    `--jobs 1` is what you reach for when something is wrong.
    """
    # `corpus.Entry` is a NamedTuple in this package, so it pickles to a worker as itself.
    payload = [(d, entries[d], jobs[d]) for d in ordered]
    if workers == 1:
        _init(str(corpus), verdicts)
        yield from (_score_document(j) for j in payload)
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(str(corpus), verdicts)) as pool:
        yield from pool.map(_score_document, payload, chunksize=1)


def cmd_explain(args: Namespace) -> int:
    """Print every address for one document: what the gold had, and what was predicted."""
    corpus, atlas = corpus_atlas.locate(Path(args.corpus) if args.corpus
                                        else published_corpus())
    docs = [d for d in documents(atlas) if d.doc_id == args.doc]
    if not docs:
        print(f"no document {args.doc!r} in {corpus}", file=sys.stderr)
        return 1
    pred_file = Path(args.predictions) / f"{args.doc}.json"
    if not pred_file.exists():
        print(f"no prediction at {pred_file}", file=sys.stderr)
        return 1
    case = next(cases(docs, [(args.doc, pred_file.read_bytes())]))
    outcome = score(case)
    if outcome.kind != "graded":
        print(f"{args.doc}: {outcome.kind} ({outcome.error})")
        return 0
    for row in verdict_rows(case, outcome):
        if args.all or row["verdict"] != "match":
            print(f"  {row['address']:<44}{row['verdict']:<16}"
                  f"gold={row['gold']}  pred={row['pred']}")
    print(f"\n  accuracy {outcome.summary['accuracy']:.2f}")
    return 0


def cmd_build_corpus(args: Namespace) -> int:
    """Declare what a corpus contains: every document in the tree, and what its files hash to.

    One verb, and it always describes what is on disk now. Which documents are in the
    benchmark is decided by which are in the directory -- so curating is arranging files, and
    this records the result rather than being the place you do it.

    Re-run it whenever documents are added or removed. The atlas is then what a run follows, so
    filtering the table afterwards is how you choose a subset to score.

    Columns the contract does not define are carried over. A published corpus records `suite`,
    page counts and provenance beside the five required ones, and `suite` decides the subsets
    the published number averages over -- rebuilding is not a reason to lose it.
    """
    root = Path(args.corpus)
    entries = corpus_atlas.discover(root)
    if not entries:
        print(f"  no documents under {root}", file=sys.stderr)
        return 1

    carried = corpus_atlas.extras(root)
    corpus_atlas.write(root, entries, carried)
    print(f"  {len(entries)} documents -> {corpus_atlas.ATLAS}")
    kept = sorted({k for v in carried.values() for k in v})
    if kept:
        print(f"  carried {len(kept)} extra column(s) through: {', '.join(kept[:8])}")
    return 0


def cmd_verify(args: Namespace) -> int:
    """Check that every document the atlas lists is there and readable.

    The atlas is what a run follows, so this asks the question a run will ask: does each row
    resolve to two files that parse? It is not a check that the data has not changed -- the
    atlas records what a corpus contains, not what its bytes were.
    """
    root, atlas = corpus_atlas.locate(Path(args.corpus))
    entries = corpus_atlas.read(atlas)
    missing = [f"{e.doc_id}: {rel} is listed but missing"
               for e in entries
               for rel in (e.ground_truth_path, e.schema_path)
               if not (root / rel).is_file()]
    print(f"  {len(entries)} documents listed")
    if missing:
        print(f"  {len(missing)} listed file(s) are not there:", file=sys.stderr)
        for m in missing[:10]:
            print(f"      {m}", file=sys.stderr)
        return 1

    unreadable = []
    for e in entries:
        try:
            documents_one = json.loads((root / e.ground_truth_path).read_bytes())
            json.loads((root / e.schema_path).read_bytes())
            del documents_one
        except ValueError as exc:
            unreadable.append(f"{e.doc_id}: {exc}")
    if unreadable:
        print(f"  {len(unreadable)} document(s) will not parse:", file=sys.stderr)
        for u in unreadable[:10]:
            print(f"      {u}", file=sys.stderr)
        return 1
    print("  every listed document is present and parses")
    return 0
