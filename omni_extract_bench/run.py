"""Scoring a corpus from the command line, and writing the result where a query can reach it.

`bench.py` yields plain rows and knows nothing about storage. This is the part that knows
about files: where the published corpus comes from, and what a run looks like on disk.

    <out>/summary.parquet              one row per (doc_id, prediction_id)
    <out>/verdicts/<doc_id>.parquet    every address, with gold and pred

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

import sys
from pathlib import Path

from . import corpus as corpus_atlas
from .bench import (cases, documents, predictions, score, summary_row,
                    verdict_rows)
from .corpus import Stale

#: The published benchmark. `--corpus` overrides it, which is how you score against your own
#: ground truth: the loader does not care whose corpus it is.
HF_REPO = "datalab-to/omni_extract_bench"

SUMMARY = "summary.parquet"
VERDICTS = "verdicts"


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:                                     # pragma: no cover - env specific
        raise SystemExit(
            "writing a run needs pyarrow: pip install 'omni-extract-bench[run]'")
    return pa, pq


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
            "pip install 'omni-extract-bench[run]', or pass --corpus to use your own")
    return Path(snapshot_download(
        HF_REPO, repo_type="dataset",
        allow_patterns=[corpus_atlas.ATLAS, "*/ground_truth.json", "*/schema.json"]))


def write_run(out: Path, summary: list[dict], by_doc: dict[str, list[dict]]) -> None:
    """Write one run: the summary table, and one verdict file per document."""
    pa, pq = _pyarrow()
    out.mkdir(parents=True, exist_ok=True)
    (out / VERDICTS).mkdir(exist_ok=True)

    # An explicit union of keys: a run where nothing graded would otherwise infer a schema
    # with no metric columns at all, and two runs would stop being comparable.
    fields = list(dict.fromkeys(k for r in summary for k in r))
    blank = {k: None for k in fields}
    pq.write_table(pa.Table.from_pylist([{**blank, **r} for r in summary]),
                   out / SUMMARY, compression="zstd")
    for doc_id, rows in by_doc.items():
        if rows:
            pq.write_table(pa.Table.from_pylist(rows),
                           out / VERDICTS / f"{doc_id}.parquet", compression="zstd")


#: What a user can get wrong, as opposed to what a bug looks like. These are reported as a
#: message; anything else keeps its traceback, because it is ours to fix.
USER_ERRORS = (FileNotFoundError, NotADirectoryError, ValueError, Stale)


def cmd_score(args) -> int:
    """Score a directory of predictions against a corpus."""
    try:
        return _score_corpus(args)
    except USER_ERRORS as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1


def _score_corpus(args) -> int:
    corpus = Path(args.corpus) if args.corpus else published_corpus()
    docs = list(documents(corpus))
    want_verdicts = not args.no_verdicts

    # A prediction for a document the atlas does not list is not necessarily a mistake:
    # curating a document out leaves its predictions behind, and that is the normal state of a
    # filtered corpus. A prediction naming nothing at all still is a mistake, and usually a
    # filename convention -- so tell the two apart by whether the document exists on disk.
    listed = {d.doc_id for d in docs}
    curated_out = set(corpus_atlas.undeclared(corpus, corpus_atlas.read(corpus)))
    wanted, skipped = [], []
    for doc_id, raw in predictions(Path(args.predictions)):
        if doc_id in listed:
            wanted.append((doc_id, raw))
        elif doc_id in curated_out:
            skipped.append(doc_id)
        else:
            wanted.append((doc_id, raw))      # `cases` raises, with near matches
    if skipped:
        print(f"  {len(skipped)} prediction(s) skipped; their documents are not in this "
              f"corpus: {', '.join(sorted(skipped)[:5])}"
              + (f" and {len(skipped) - 5} more" if len(skipped) > 5 else ""))

    summary, by_doc, kinds = [], {}, {}
    seen = 0
    for case in cases(docs, wanted):
        outcome = score(case, verdicts=want_verdicts)
        summary.append(summary_row(case, outcome))
        by_doc[case.doc.doc_id] = verdict_rows(case, outcome)
        kinds[outcome.kind] = kinds.get(outcome.kind, 0) + 1
        seen += 1
        if args.out is None and outcome.kind == "graded":
            print(f"  {case.doc.doc_id:<44}{outcome.summary['accuracy']:>7.2f}")
        elif args.out is None:
            print(f"  {case.doc.doc_id:<44}{outcome.kind:>9}  {outcome.error or ''}")

    if not seen:
        print("no predictions found; expected <doc_id>.json files", file=sys.stderr)
        return 1

    graded = [r for r in summary if r["kind"] == "graded"]
    unusable = [r for r in summary if r["kind"] == "unusable"]
    failed = [r for r in summary if r["kind"] == "failed"]
    mean = sum(r["accuracy"] for r in graded) / len(graded) if graded else 0.0

    print(f"\n  {seen} predictions over {len(docs)} documents in the corpus")
    print(f"  mean accuracy over the {len(graded)} graded: {mean:.2f}")
    if unusable:
        # The provider's failure. It returned nothing scoreable, and that counts against it --
        # but as coverage, not as a zero averaged into the accuracy above.
        print(f"  {len(unusable)} unusable (the provider returned nothing scoreable), "
              f"excluded from that mean")
        for r in unusable[:5]:
            print(f"      {r['doc_id']}: {r['error'] or 'empty'}")
    if failed:
        # Ours. Reported separately and loudly, because a broken corpus reading as a poor
        # vendor is the one mistake here that looks like a result.
        print(f"  {len(failed)} NOT SCORED -- this harness could not score them:",
              file=sys.stderr)
        for r in failed[:10]:
            print(f"      {r['doc_id']}: {r['error']}", file=sys.stderr)

    if args.out:
        write_run(Path(args.out), summary, by_doc)
        # Documents with nothing graded have no verdict file, so count what was written
        # rather than what was considered.
        written = {d: v for d, v in by_doc.items() if v}
        n = sum(len(v) for v in written.values())
        print(f"  wrote {args.out}/{SUMMARY}"
              + (f" and {len(written)} verdict files ({n:,} rows)" if want_verdicts else ""))
    return 1 if failed else 0


def cmd_explain(args) -> int:
    """Print every address for one document: what the gold had, and what was predicted."""
    try:
        return _explain(args)
    except USER_ERRORS as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1


def _explain(args) -> int:
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


def cmd_build_corpus(args) -> int:
    """Declare what a corpus contains, or re-record files you meant to change.

    Two verbs, because they answer different questions. Without `--refresh` this discovers
    every document in the tree, which is how a corpus starts. With it, it re-hashes exactly
    the rows already listed -- so a corpus you have curated down does not silently regain
    everything you removed.
    """
    root = Path(args.corpus)
    try:
        if args.refresh:
            entries = corpus_atlas.refresh(root, corpus_atlas.read(root))
            how = "refreshed"
        else:
            if (root / corpus_atlas.ATLAS).exists() and not args.replace:
                print(f"  {root / corpus_atlas.ATLAS} exists. Re-discovering would undo any "
                      f"curation.\n"
                      f"  To re-record files you changed:   --refresh\n"
                      f"  To start again from the tree:     --replace", file=sys.stderr)
                return 1
            entries = corpus_atlas.discover(root)
            how = "discovered"
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1
    if not entries:
        print(f"  no documents under {root}", file=sys.stderr)
        return 1

    before = None
    if args.refresh:
        before = corpus_atlas.version(corpus_atlas.read(root))
    corpus_atlas.write(root, entries)
    after = corpus_atlas.version(entries)
    print(f"  {how} {len(entries)} documents -> {corpus_atlas.ATLAS}")
    print(f"  corpus version {after}"
          + (f"  (was {before})" if before and before != after else ""))
    extra = corpus_atlas.undeclared(root, entries)
    if extra:
        print(f"  {len(extra)} document(s) on disk are not listed: {', '.join(extra[:5])}"
              + (f" and {len(extra) - 5} more" if len(extra) > 5 else ""))
    return 0


def cmd_verify(args) -> int:
    """Check the corpus against its atlas, and say what has moved.

    The atlas is a snapshot, so this answers the question you actually have while curating:
    what have I changed since I last declared this corpus?
    """
    root = Path(args.corpus)
    try:
        entries = corpus_atlas.read(root)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1

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
