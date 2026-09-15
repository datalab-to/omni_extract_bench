#!/usr/bin/env python3
"""Re-poll documents that hit the uniform per-document limit, using the captured poll URL.

These are async APIs. A run that was still `processing` when our 1800s limit expired has, in
several cases, finished since -- and we were billed for it. Scoring it zero states that the
vendor could not extract the document, when what actually happened is that we stopped asking.

Two rules make this legitimate rather than favourable treatment:

  UNIFORM   every provider with a timed-out document and a captured id is retried, not just the
            one whose number we care about. Recovering our own product's timeouts while leaving
            a competitor's on the floor would be the most self-serving thing this benchmark
            could do, and it would be invisible in the final table.
  MARKED    a recovered document carries `recovered_after_timeout: true`. Latency is a real
            product property -- a provider that needs more than the limit IS slower -- so the
            headline can keep the timeout rule while the recovered figure sits beside it. The
            two numbers answer different questions and both are honest; conflating them is not.

No new inference: this re-issues the GET that the harness already made, so the vendor bills
nothing further and a nondeterministic provider cannot return a different answer than the one
we paid for.

Run: python -m harness.recover_timeouts --out-root predictions/ [--apply]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HARNESS))

from omni_extract_bench.prediction_io import usable  # noqa: E402

TIMEOUT_MARKERS = ("timeout", "did not finish", "still processing", '"status":"processing"',
                   '"status": "Running"', '"status":"Running"')


# Credentials come from the environment, the same names the runner uses. This script once had
# its own laptop-path copy and died on the VM with FileNotFoundError, the same way the R2
# durability sync had already died with the same bug an hour earlier.


def _env():
    """Credentials from the environment; nothing machine-specific."""
    return dict(os.environ)


def headers_for(prov, env):
    if prov.startswith("datalab"):
        return {"X-Api-Key": env["DATALAB_API_KEY"]}
    if prov == "reducto":
        return {"Authorization": f"Bearer {env['REDUCTO_API_KEY']}"}
    if prov == "extend":
        from harness.providers import extend as EXP
        h = dict(EXP.headers(env["EXTEND_API_KEY"]))
        if env.get("EXTEND_WORKSPACE_ID"):
            h["X-Extend-Workspace-Id"] = env["EXTEND_WORKSPACE_ID"]
        return h
    if prov.startswith("azure"):
        return {"Ocp-Apim-Subscription-Key": env["AZURE_CU_KEY"]}
    if prov == "llamaextract":
        key = env.get("LLAMA_CLOUD_API_KEY")
        return {"Authorization": f"Bearer {key}"}
    return {}


def extraction_of(prov, body):
    """Pull the schema-shaped extraction out of the vendor's envelope.

    Provider-specific ON PURPOSE. A generic "first non-empty dict under result/json/data" walk
    looks reasonable and is wrong: datalab's `json` key holds the parsed BLOCK TREE, so the
    generic version recovered a document's layout instead of its extraction, scored it 0.0, and
    -- worse than leaving it a timeout -- marked it a successful extraction that produced
    nothing. A wrong recovery is more damaging than no recovery, because it converts "did not
    finish" into "answered incorrectly".
    """
    if not isinstance(body, dict):
        return None
    if prov.startswith("datalab"):
        raw = body.get("extraction_schema_json")   # the extraction, despite the name
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:  # noqa: BLE001
                return None
        return raw if isinstance(raw, dict) and raw else None
    if prov == "extend":
        for path in (("output", "value"), ("value",), ("data", "value")):
            cur = body
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, dict) and cur:
                from harness.providers import extend as EXP
                # `id` was sent to Extend under an alias (reserved key); restore the schema's name
                return EXP.restore_reserved(cur)
        return None
    if prov.startswith("azure"):
        cur = ((body.get("result") or {}).get("contents") or [{}])
        return cur[0].get("fields") if cur and isinstance(cur[0], dict) else None
    for path in (("result",), ("data",), ("output",)):
        cur = body
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, dict) and cur:
            return cur
    return None


def main():
    import httpx

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True,
                    help="predictions root: <out-root>/<provider>/<subset>/<doc>.json")
    ap.add_argument("--apply", action="store_true", help="write recovered results back")
    a = ap.parse_args()
    env = _env()

    print(f"{'provider':13}{'document':44}{'status':>12}{'recovered':>11}")
    totals = {}
    for d in sorted(glob.glob(f"{a.out_root}/*")):
        prov = os.path.basename(d)
        hdrs = headers_for(prov, env)
        got = tried = 0
        for f in glob.glob(f"{d}/*/*.json"):
            if "_raw" in f:
                continue
            res = json.load(open(f)).get("result")
            if usable(res):
                continue
            err = str((res or {}).get("__error__") if isinstance(res, dict) else "")
            if not any(m.lower() in err.lower() for m in TIMEOUT_MARKERS):
                continue
            raw_path = f.replace(f"/{prov}/", f"/{prov}/_raw/")
            rec = json.load(open(raw_path)) if os.path.exists(raw_path) else {}
            polls = [c["url"] for c in (rec.get("http") or [])
                     if c.get("method") == "GET" and re.search(r"/(job|extract|runs|results|"
                                                               r"extract_runs|analyzerResults)/",
                                                               c["url"])]
            if not polls:
                continue
            tried += 1
            url = polls[-1]
            try:
                r = httpx.get(url, headers=hdrs, timeout=180)
                body = r.json() if r.status_code < 400 else {}
            except Exception as exc:  # noqa: BLE001
                print(f"{prov:13}{os.path.basename(f)[:42]:44}{'ERR':>12}"
                      f"{type(exc).__name__:>11}")
                continue
            status = str(body.get("status") or r.status_code)
            extraction = extraction_of(prov, body)
            ok = bool(extraction)
            print(f"{prov:13}{os.path.basename(f)[:42]:44}{status[:11]:>12}{('YES' if ok else 'no'):>11}")
            if ok:
                got += 1
                if a.apply:
                    blob = json.load(open(f))
                    blob["result"] = extraction
                    blob["recovered_after_timeout"] = True
                    json.dump(blob, open(f, "w"), default=str)
                    rec["recovered_after_timeout"] = True
                    cost = rec.get("cost") or {}
                    cents = (body.get("cost_breakdown") or {}).get("final_cost_cents")
                    if cents is not None:
                        cost["usd"] = round(float(cents) / 100.0, 6)
                        cost["source"] = "vendor response (recovered after timeout)"
                        cost["billed_out_of_band"] = False
                    rec["cost"] = cost
                    json.dump(rec, open(raw_path, "w"), default=str)
        if tried:
            totals[prov] = (got, tried)
    print()
    for prov, (got, tried) in sorted(totals.items()):
        print(f"  {prov:13} recovered {got}/{tried} timed-out documents")
    if not a.apply:
        print("\n(dry run — nothing written; re-run with --apply)")


if __name__ == "__main__":
    main()
