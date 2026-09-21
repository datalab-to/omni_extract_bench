"""TEMPORARY, FOR RECORDING A DEMO. DELETE THIS FILE AND THE THREE HOOKS THAT CALL IT.

    OEB_STUB=1 oeb predict --provider datalab --doc x.pdf --schema s.json
    OEB_STUB=0.3 oeb benchmark --out runs-demo/ --providers datalab reducto

Set `OEB_STUB` and no vendor is called, no key is read, no money is spent and nothing is
graded: a run is a film set. Set it to a number rather than `1` and each prediction takes that
many seconds, so the progress table is slow enough to watch. Grading is always instant.

EVERYTHING FAKE LIVES HERE. The package holds three lines that call into this file and nothing
else, so `rm omni_extract_bench/_stub.py` and the failing imports are the whole removal:

    harness/vendor.py   `predict`        -- one document, one vendor, from the command line
    benchmark.py        `_predict_one`   -- the same, in a run, where the gold is also known
    benchmark.py        `score_one`      -- one grade

WHAT IT FAKES, AND WHY EACH ONE IS SHAPED THE WAY IT IS. A prediction is the schema filled in,
or -- in a benchmark, where the document's gold is at hand -- the gold itself with a fraction
of its leaves misread. A grade is a plausible row rather than a real one. Both the fraction and
the row come from a digest of the PROVIDER NAME, so the legs of a run land on different
numbers that hold still between takes instead of all scoring alike or all scoring zero.

IT WILL OVERWRITE REAL WORK. A stubbed grade never reads the prediction on disk, so pointing
`--out` at a directory holding a real run rewrites its `scores.jsonl` with invented rows. Use a
directory of its own.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

ENV = "OEB_STUB"


def on() -> bool:
    return bool(os.environ.get(ENV))


def _rate(provider: str) -> float:
    """How much this provider gets wrong, fixed by its name: 0.04 to 0.25."""
    return 0.04 + int(hashlib.sha256(provider.encode()).hexdigest()[:4], 16) % 22 / 100


def _fill(schema: dict):
    """One placeholder for one resolved subschema, keyed to what it says it wants."""
    for branch in schema.get("anyOf") or schema.get("oneOf") or []:
        if branch.get("type") != "null":
            return _fill(branch)
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {k: _fill(v) for k, v in (schema.get("properties") or {}).items()}
    if kind == "array":
        return [_fill(schema.get("items") or {})]
    if kind in ("number", "integer"):
        return 1
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    return "sample"


def _degrade(value, rng: random.Random, rate: float):
    """The gold answer with a fraction of its leaves misread, so a fake run scores like a run."""
    if isinstance(value, dict):
        return {k: _degrade(v, rng, rate) for k, v in value.items()}
    if isinstance(value, list):
        return [_degrade(v, rng, rate) for v in value]
    if rng.random() >= rate:
        return value
    if isinstance(value, str):
        return "sample"
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    return None


def prediction(provider: str, schema: dict, gold: Path | None = None) -> dict:
    """What `vendor.predict` would have returned, without the vendor."""
    from .metric import resolve_refs

    started = time.time()
    time.sleep(float(os.environ[ENV]) if os.environ[ENV] != "1" else 0)
    if gold is not None and Path(gold).exists():
        result = _degrade(json.loads(Path(gold).read_text()),
                          random.Random(f"{provider}:{gold}"), _rate(provider))
    else:
        result = _fill(resolve_refs(schema))
    seed = int(hashlib.sha256(provider.encode()).hexdigest()[:4], 16)
    return {
        "result": result,
        "raw": {"stub": True},
        "error": None,
        "provider": provider,
        "cost": {"usd": round(0.004 + seed % 90 / 10000, 4), "source": ENV,
                 "wall_s": round(time.time() - started, 1), "attempts": 1,
                 "billed_out_of_band": False},
        "job_id": None,
        "schema_sent": schema,
        "run_manifest": {"timeout_s": 0.0, "settings": {}, "model": None, "timed_out": False,
                         "conventions_applied": False,
                         "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
    }


def score_row(doc, provider: str) -> dict:
    """What `score_one` would have returned, without reading or grading anything.

    The counts are built from `matched` outwards so they still add up the way a real row does:
    every gold field is matched, misread or unfound, precision is measured against what was
    asserted, and accuracy sits just under recall because it is the one that an addition counts
    against. Three metrics drawn independently would not survive anyone reading the row.
    """
    rng = random.Random(f"{provider}:{doc.doc_id}")
    total = rng.randint(24, 400)
    matched = round(total * (1 - min(0.9, max(0.0, rng.gauss(_rate(provider), 0.04)))))
    misread = round((total - matched) * 0.85)
    fabricated, invented_item = rng.randint(0, 2), rng.randint(0, 1)
    asserted = matched + misread + fabricated + invented_item
    recall, precision = matched / total, matched / asserted
    return {
        "doc_id": doc.doc_id, "suite": doc.suite, "provider": provider, "status": "scored",
        "accuracy": matched / (total + fabricated + invented_item),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if matched else 0.0,
        "total": total, "matched": matched, "misread": misread,
        "unfound": total - matched - misread, "fabricated": fabricated,
        "invented_item": invented_item, "invented_field": 0,
        "asserted": asserted, "matching_exact": True,
    }
