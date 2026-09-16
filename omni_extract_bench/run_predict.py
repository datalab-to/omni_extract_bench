"""Producing predictions from a table of documents: one manifest in, files and a table out.

    oeb predict --manifest jobs.parquet --out preds/

    preds/predictions/<doc_id>.json      the bare extraction
    preds/manifest/part-00000.parquet    one row per manifest row, with pred_path filled in

Same shape as `run_score`, deliberately, and built on its table layer. The property worth
designing for: **this module's output table is the next one's input.** Join ground truth onto
it and score it -- no layout to agree on in between, no pairing files up by basename.

Required columns::

    doc_id
    doc_path     the PDF
    schema       the schema to extract against, inline

**The provider is a run, not a column.** One manifest serves every vendor, and nine vendors are
nine runs of it. A per-row provider would only pay off for a manifest that sent each document
to a different one, and that cannot be written anyway: `doc_id` has to be unique, so the same
620 documents cannot appear nine times in one table. The provider, and its options, are
recorded on every output row instead.

Nothing is consumed: every column survives, `schema` included, because the scorer needs it --
this table is meant to be scored without joining anything back onto it.

**Predictions are written bare, not enveloped.** Latency and cost belong in the table, which
is what the table is for; an envelope is one more thing every reader must know to unwrap, and
a reader that forgets grades the envelope. A failure is written as `{"__error__": ...}`, which
is what the scorer reads as `empty`, so a failed document stays a row with a reason rather
than a hole.

Threads, not processes: this is network-bound, and the harness's capture state -- which is
where per-document cost comes from -- is thread-local by design.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from .run_score import carried, check_manifest, open_manifest, open_uri, write_part

log = logging.getLogger(__name__)

REQUIRED = ("doc_id", "doc_path", "schema")

#: Everything this module writes is `pred_*`, and that prefix is load-bearing: `status` and
#: `error` are names the SCORER writes, so unprefixed ones here would make this table collide
#: with the scorer's own columns -- and the whole point is that this table can be scored.
PREDICT_FIELDS: list[tuple[str, str]] = [
    # `provider` is written, not read: the output says who produced the prediction, which is
    # what the next run needs to know and what the manifest deliberately does not carry.
    ("provider", "string"),
    ("pred_path", "string"), ("pred_status", "string"), ("pred_error", "string"),
    ("pred_latency_s", "float64"), ("pred_cost_usd", "float64"), ("pred_attempts", "int64"),
]

RESERVED = frozenset(name for name, _ in PREDICT_FIELDS)


def predict_row(row: dict, out: str, provider: str, options: SimpleNamespace) -> dict:
    """Run one document through the provider. Never raises: a failed document is a row."""
    from .harness.run_provider import _reset_state, cost_record, extract_one

    blank = {name: None for name, _ in PREDICT_FIELDS} | {"provider": provider}
    doc_id = row["doc_id"]

    started = time.time()
    _reset_state()
    try:
        schema = json.loads(row["schema"])
        # A vendor adapter takes a local file path and a manifest may name an object in a
        # bucket, so the document lands in a temporary file on the way through.
        with open_uri(row["doc_path"]) as fh, tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / f"{doc_id}.pdf"
            pdf.write_bytes(fh.read())
            result = extract_one(provider, pdf, schema, options)
    except Exception as exc:                                            # noqa: BLE001
        result = {"__error__": f"{type(exc).__name__}: {exc}"}

    cost = cost_record(time.time() - started)
    path = f"{out.rstrip('/')}/predictions/{doc_id}.json"
    with open_uri(path, "wb") as fh:
        fh.write(json.dumps(result, default=str).encode())
    error = result.get("__error__") if isinstance(result, dict) else None
    return {**blank, "pred_path": path,
            "pred_status": "error" if error else "ok",
            "pred_error": None if error is None else str(error),
            "pred_latency_s": cost["wall_s"], "pred_cost_usd": cost["usd"],
            "pred_attempts": cost["attempts"]}


def run(manifest: str, out: str, provider: str, jobs: int = 4, batch_size: int = 256,
        timeout: float = 1800, mode: str | None = None,
        completion_model: str | None = None) -> dict:
    """Predict a manifest into `<out>/predictions` and `<out>/manifest`. Returns a tally."""
    from .harness.run_provider import PROVIDERS

    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}. Known: {', '.join(PROVIDERS)}")
    options = SimpleNamespace(timeout=float(timeout), mode=mode,
                              completion_model=completion_model)
    data = open_manifest(manifest)
    check_manifest(data, REQUIRED, RESERVED)
    tally = {"ok": 0, "error": 0}
    total = data.count_rows()
    log.info("predicting %d documents with %s into %s", total, provider, out)
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        for n, batch in enumerate(data.to_batches(batch_size=batch_size)):
            done = list(pool.map(lambda r: predict_row(r, out, provider, options),
                                 batch.to_pylist()))
            for row in done:
                tally[row["pred_status"]] += 1
            write_part(done, PREDICT_FIELDS, f"{out.rstrip('/')}/manifest/part-{n:05d}.parquet",
                       carried(batch, consumed=()))
            log.info("part-%05d: %d/%d documents  %s", n, sum(tally.values()), total,
                     "  ".join(f"{k}={v}" for k, v in tally.items()))
    return tally
