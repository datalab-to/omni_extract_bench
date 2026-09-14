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
from .corpus import Entry, Stale

#: The published benchmark. `--corpus` overrides it, which is how you score against your own
#: ground truth: the loader does not care whose corpus it is.
HF_REPO = "datalab-to/omni_extract_bench"

SUMMARY = "summary.parquet"
VERDICTS = "verdicts"


def _is_artifact(status_line: str) -> bool:
    """Is this `git status` line a build artifact rather than source?

    Importing the package writes `__pycache__`, and a project that has not ignored it would
    otherwise report the scorer as modified the moment it was used. A `.pyc` is derived from
    the `.py` beside it and cannot change behaviour on its own.
    """
    path = status_line[3:].strip().strip('"')
    return "__pycache__" in path or path.endswith((".pyc", ".pyo"))


def scorer_version(repo: Path | None = None) -> dict[str, str]:
    """Which code produced a score.

    The commit, and whether the tree had uncommitted changes when it ran. Both, because a
    commit alone would be a claim the working tree cannot support: the point of recording it is
    that the same inputs under the same scorer give the same number, and an edited tree cannot
    promise that. `scorer_dirty` says so rather than inventing a name that hides it.

    Untracked files count -- a stray module changes what gets imported and `git diff` cannot
    see it.

    **Dirtiness is asked about this package's source only**, not the whole repository and not
    its build artifacts. Someone who
    vendors this into their own project, or whose virtualenv is committed, would otherwise have
    every edit anywhere in their tree reported as a change to the scorer -- and the one field
    whose job is to say "the commit does not describe what ran" would be true all the time,
    which is the same as being useless.

    The commit stays repository-wide, because for a vendored copy their commit really does pin
    our files' contents.

    Outside a checkout there is no commit, so the installed version stands in. That is weaker,
    since a release covers many working trees, and it is labelled differently for that reason.
    """
    import subprocess

    here = Path(__file__).resolve().parent
    repo = repo or here.parent
    try:
        def git(*a: str) -> str:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                                  text=True, check=True).stdout.strip()

        scope = [str(here)] if repo == here.parent else []
        changed = [line for line in
                   git("status", "--porcelain", "--untracked-files=all", "--",
                       *scope).splitlines()
                   if not _is_artifact(line)]
        return {"scorer_commit": git("rev-parse", "HEAD"),
                "scorer_dirty": str(bool(changed))}
    except Exception:                                       # not a checkout
        from . import __version__ as v
        return {"scorer_version": str(v), "scorer_commit": ""}


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
USER_ERRORS = (FileNotFoundError, NotADirectoryError, ValueError, Stale)


def _jobs(preds: Path, entries: dict[str, Entry]) -> dict[str, bytes]:
    """Every prediction to score, by document.

    Raises:
        ValueError: for a prediction naming no document in the corpus, suggesting near
            matches. The usual cause is a filename convention (`invoice-003.pdf.json`), and
            the fix is almost always visible once you are shown what was close.
    """
    import difflib

    out = {}
    for doc_id, raw in predictions(preds):
        if doc_id not in entries:
            close = difflib.get_close_matches(doc_id, entries, n=3, cutoff=0.6)
            raise ValueError(
                f"prediction {doc_id!r} matches no document in the corpus."
                + (f" Did you mean: {', '.join(close)}?" if close else "")
                + " A prediction file must be named <doc_id>.json.")
        out[doc_id] = raw
    return out


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

    corpus = Path(args.corpus) if args.corpus else published_corpus()
    preds = Path(args.predictions)
    entries = {e.doc_id: e for e in corpus_atlas.read(corpus)}
    stamp = {"corpus_version": corpus_atlas.version(entries.values()),
             "source": args.source or preds.name, **scorer_version(), **environment()}
    out = Path(args.out) if args.out else None

    carried, collided = prediction_meta(preds)
    if collided:
        print(f"  {PREDICTION_META} columns dropped, they collide with the scorer's own: "
              f"{', '.join(collided)}", file=sys.stderr)

    jobs = _jobs(preds, entries)
    if not jobs:
        print("no predictions found; expected <doc_id>.json files", file=sys.stderr)
        return 1
    scorer = stamp.get("scorer_commit") or stamp.get("scorer_version", "?")
    print(f"  corpus {stamp['corpus_version']}, scorer {scorer[:12]}"
          f"{' (uncommitted changes)' if stamp.get('scorer_dirty') == 'True' else ''}, "
          f"source {stamp['source']}")
    print(f"  {len(jobs)} predictions, {len(entries)} documents in the corpus")

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
    corpus = Path(args.corpus) if args.corpus else published_corpus()
    docs = [d for d in documents(corpus) if d.doc_id == args.doc]
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
    """Declare what a corpus contains, or re-record files you meant to change.

    Two verbs, because they answer different questions. Without `--refresh` this discovers
    every document in the tree, which is how a corpus starts. With it, it re-hashes exactly
    the rows already listed -- so a corpus you have curated down does not silently regain
    everything you removed.
    """
    root = Path(args.corpus)
    if args.refresh:
        entries, how = corpus_atlas.refresh(root, corpus_atlas.read(root)), "refreshed"
    else:
        if (root / corpus_atlas.ATLAS).exists() and not args.replace:
            print(f"  {root / corpus_atlas.ATLAS} exists. Re-discovering would undo any "
                  f"curation.\n"
                  f"  To re-record files you changed:   --refresh\n"
                  f"  To start again from the tree:     --replace", file=sys.stderr)
            return 1
        entries, how = corpus_atlas.discover(root), "discovered"
    if not entries:
        print(f"  no documents under {root}", file=sys.stderr)
        return 1

    before = None
    if args.refresh:
        before = corpus_atlas.version(corpus_atlas.read(root))
    # Whatever the old atlas recorded beyond the contract is the corpus's, not ours, and
    # rebuilding is not a reason to lose it.
    carried = corpus_atlas.extras(root)
    corpus_atlas.write(root, entries, carried)
    kept = sorted({k for v in carried.values() for k in v})
    if kept:
        print(f"  carried {len(kept)} extra column(s) through: {', '.join(kept[:8])}")
    after = corpus_atlas.version(entries)
    print(f"  {how} {len(entries)} documents -> {corpus_atlas.ATLAS}")
    print(f"  corpus version {after}"
          + (f"  (was {before})" if before and before != after else ""))
    extra = corpus_atlas.undeclared(root, entries)
    if extra:
        print(f"  {len(extra)} document(s) on disk are not listed: {', '.join(extra[:5])}"
              + (f" and {len(extra) - 5} more" if len(extra) > 5 else ""))
    return 0


def cmd_verify(args: Namespace) -> int:
    """Check the corpus against its atlas, and say what has moved.

    The atlas is a snapshot, so this answers the question you actually have while curating:
    what have I changed since I last declared this corpus?
    """
    root = Path(args.corpus)
    entries = corpus_atlas.read(root)
    problems = corpus_atlas.check(root, entries)
    extra = corpus_atlas.undeclared(root, entries)
    print(f"  {len(entries)} documents listed, corpus version "
          f"{corpus_atlas.version(entries)}")
    if extra:
        # Not an error: curating a document out is exactly this. It is also what a
        # half-finished copy looks like, so it is worth seeing either way.
        print(f"  {len(extra)} on disk but not listed (curated out, or not yet added): "
              f"{', '.join(extra[:5])}" + (f" and {len(extra) - 5} more" if len(extra) > 5
                                           else ""))
    if problems:
        print(f"  {len(problems)} file(s) no longer match the atlas:", file=sys.stderr)
        for p in problems[:10]:
            print(f"      {p}", file=sys.stderr)
        print(f"  If those changes were intended:  oeb build-corpus --corpus {root} "
              f"--refresh", file=sys.stderr)
        return 1
    print("  every listed file matches the atlas")
    return 0
