"""Azure AI Content Understanding adapter.

Content Understanding is analyzer-based: a JSON Schema becomes a `fieldSchema`, the analyzer is
created once per schema (cached by schema hash, because creating one per document would both be
slow and litter the resource), then each document is analysed against it.

Two things worth knowing before comparing its numbers with anyone else's:

  * the `completion` model is a deployment choice, not a product tier. `gpt-4.1-mini` and
    `gpt-4.1` are different systems behind the same API, so which one was used has to be
    recorded and published beside the score -- see `--completion-model`.
  * its field types are a smaller set than JSON Schema's. Nested objects become `object`,
    arrays of objects become `array` of `object`, and anything else degrades to `string`.
    That is the vendor's surface, not a benchmark choice, but it means a schema this adapter
    sends is a lossier statement of the task than the one other vendors receive.

Auth: AZURE_CU_KEY. The `endpoint` is an OPTION, not a credential -- see `Config`.

    oeb predict --provider azure-cu --doc doc.pdf --schema schema.json
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import httpx

from ..budget import Budget, PollRetry

from ..contract import Cost, Extraction

from ..errors import MissingCredential, VendorError

API_VERSION = "2025-05-01-preview"
DEFAULT_COMPLETION_MODEL = "gpt-4.1-mini"
_TERMINAL_OK = {"succeeded", "completed"}
_TERMINAL_BAD = {"failed", "cancelled"}


@dataclasses.dataclass(frozen=True)
class Config:
    """What azure-cu can be asked.

    `completion_model` is a DEPLOYMENT CHOICE, not a product tier: `gpt-4.1-mini` and `gpt-4.1`
    are different systems behind one API, so which one ran has to be published beside the score.

    `endpoint` is a setting for the same reason: it selects an Azure resource and region, and
    two runs against different deployments used to produce records nothing could tell apart.
    It is an OPTION with a literal default, exactly like datalab's and extend's `base_url` --
    it was read from `$AZURE_CU_ENDPOINT` for one commit, and that made the run DIRECTORY a
    function of the shell, so a resume from a terminal without the variable set looked in a
    directory with no records and re-bought the corpus. There is no public Azure endpoint to
    default to, so the default is empty and `extract` says what to pass.

    The KEY stays in the environment -- that is a credential and belongs in no record.
    """

    endpoint: str = dataclasses.field(
        default="", metadata={"help": "the Azure resource to analyse against, e.g. "
                                      "https://<resource>.cognitiveservices.azure.com"})
    completion_model: str = dataclasses.field(
        default=DEFAULT_COMPLETION_MODEL,
        metadata={"help": "the deployment behind the analyzer; publish it with the score"})
    api_version: str = API_VERSION
    poll_interval: float = 3.0

    def __post_init__(self):
        """Normalise here, so the Config holds what the URLs are actually built from.

        Empty is allowed rather than required: `config_for` runs for `oeb providers` and for a
        run's directory name, and a `Config` with a required field cannot be built at all --
        every adapter writes `config: Config = Config()` as a default argument, evaluated at
        import. `extract` is where a missing endpoint is a failure, and it says so.
        """
        object.__setattr__(self, "endpoint", self.endpoint.rstrip("/"))


def _field(prop: dict) -> dict:
    """One JSON-Schema property as a Content Understanding field definition."""
    prop = prop or {}
    t = prop.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    desc = prop.get("description") or ""
    if t == "object":
        return {"type": "object", "description": desc,
                "properties": {k: _field(v) for k, v in (prop.get("properties") or {}).items()}}
    if t == "array":
        items = prop.get("items") or {}
        it = items.get("type")
        if isinstance(it, list):
            it = next((x for x in it if x != "null"), "string")
        if it == "object":
            return {"type": "array", "description": desc,
                    "items": {"type": "object",
                              "properties": {k: _field(v) for k, v in (items.get("properties") or {}).items()}}}
        return {"type": "array", "description": desc, "items": {"type": "string"}}
    if t in ("number", "integer"):
        return {"type": "number", "description": desc}
    if t == "boolean":
        return {"type": "boolean", "description": desc}
    return {"type": "string", "description": desc}


def prepare_schema(schema: dict) -> dict:
    """Azure takes a `fieldSchema`, not a JSON Schema -- so this vendor's payload is not a
    schema at all, and `extract` sends whatever shape the API takes."""
    return {"fields": {k: _field(v) for k, v in ((schema or {}).get("properties") or {}).items()}}


def _value(node):
    """Unwrap one Content Understanding value node to a plain Python value."""
    if not isinstance(node, dict):
        return node
    for key in ("valueString", "valueNumber", "valueInteger", "valueBoolean", "valueDate"):
        if key in node:
            return node[key]
    if "valueArray" in node:
        return [_value(x) for x in node["valueArray"] or []]
    if "valueObject" in node:
        return {k: _value(v) for k, v in (node["valueObject"] or {}).items()}
    if "content" in node:
        return node["content"]
    return None


def fields_to_dict(fields: dict) -> dict:
    return {k: _value(v) for k, v in (fields or {}).items()}


_ANALYZERS: dict[tuple[str, str], str] = {}
_ANALYZER_LOCK = threading.Lock()


def analyzer_id(fields: dict, completion_model: str, api_version: str) -> str:
    """The analyzer's name: a digest of EVERYTHING the analyzer is built from.

    An analyzer is a server-side object, reused by name, and a `409` on the PUT means it
    already exists -- which this treats as success. So anything that changes what the analyzer
    IS has to change its name, or the 409 silently hands back somebody else's.

    Naming it after the field schema alone did exactly that with `completion_model`, which goes
    into the PUT body and not into the name. Run the corpus with `gpt-4.1-mini`, run it again
    with `gpt-4.1`: same name, 409, every document analysed against the first deployment while
    the record says the second. Two published numbers, one model, and nothing to show for it.
    """
    spelled = json.dumps({"fieldSchema": fields, "completion": completion_model,
                          "api_version": api_version}, sort_keys=True)
    return f"oeb-{hashlib.sha256(spelled.encode()).hexdigest()[:16]}"


def _ensure_analyzer(client, endpoint: str, api_version: str, analyzer: str,
                     fields: dict, completion_model: str, budget, poll_interval: float) -> None:
    """Create the analyzer once per process, not once per thread.

    KEYED BY (endpoint, analyzer) -- the cache is process-wide and an analyzer lives inside one
    Azure resource, so keying it by the analyzer alone let a second leg on a different endpoint
    skip its own creation and then 404 on every document.

    UNDER THE LOCK FOR THE WHOLE CREATE-AND-WAIT, not just the cache lookup. A 409 says the
    analyzer EXISTS, not that it is READY -- so a thread that lost the race would skip the
    readiness wait below and analyse against an analyzer still provisioning. Checking the cache
    without holding anything is what let several threads reach the PUT at once, and widening
    `--predict-workers` makes that likelier rather than rarer.

    Threads that block here are waiting for something they need anyway, and the wait is bounded
    by the holding document's own budget.
    """
    with _ANALYZER_LOCK:
        if (endpoint, analyzer) in _ANALYZERS:
            return
        r = client.put(
            f"{endpoint}/contentunderstanding/analyzers/{analyzer}"
            f"?api-version={api_version}",
            json={"baseAnalyzerId": "prebuilt-documentAnalyzer",
                  "config": {"returnDetails": False, "completion": completion_model},
                  "fieldSchema": fields})
        if r.status_code != 409:
            if r.status_code >= 400:
                raise VendorError(f"creating analyzer: HTTP {r.status_code}: {r.text[:300]}",
                                  status=r.status_code, body=r.text)
            if r.headers.get("Operation-Location"):
                _await(client, r.headers["Operation-Location"], budget=budget,
                       poll_interval=poll_interval, want_result=False)
        _ANALYZERS[(endpoint, analyzer)] = analyzer


def _await(client, op_url: str, *, budget, poll_interval: float, want_result: bool):
    """Poll one operation. Takes the DOCUMENT's budget, not a fresh timeout: this is
    called twice per document -- once for the analyzer, once for the analysis -- and with
    a timeout each it gave azure-cu two full budgets where every other vendor got one."""
    polls = 0
    retry = PollRetry(budget)
    while True:
        budget.check(f"still analysing after {polls} polls")
        try:
            r = client.get(op_url)
        except httpx.TransportError as exc:
            if retry.again():
                continue
            raise VendorError(f"polling failed: {exc}"[:300], status=None) from None
        if r.status_code >= 400:
            if retry.again(r.status_code):
                continue
            raise VendorError(f"HTTP {r.status_code}: {r.text[:300]}",
                              status=r.status_code, body=r.text)
        retry.ok()
        polls += 1
        body = r.json()
        status = str(body.get("status", "")).lower()
        if status in _TERMINAL_OK:
            return body if want_result else True
        if status in _TERMINAL_BAD:
            raise VendorError(f"azure-cu {status}: {json.dumps(body.get('error') or {})[:300]}",
                              status=200, body=r.text)
        time.sleep(poll_interval)


def extract(pdf: Path, schema: dict, *, timeout: float = 1800.0,
            config: Config = Config()) -> Extraction:
    """Create (or reuse) an analyzer for this field schema, analyse the document, poll for it.

    `schema` here is already the Azure `fieldSchema` -- `DIALECT` is applied by the caller, so
    what arrives is what goes on the wire.

    Azure does not report a per-call cost, so `cost.usd` is None and the record says
    `billed_out_of_band` -- rather than inventing a figure from a price list.
    """
    endpoint = config.endpoint
    key = os.environ.get("AZURE_CU_KEY")
    if not endpoint or not key:
        raise MissingCredential(
            "azure-cu needs AZURE_CU_KEY in the environment and an endpoint in its options:\n"
            "    --options '{\"azure-cu\": {\"endpoint\": "
            "\"https://<resource>.cognitiveservices.azure.com\"}}'")
    budget = Budget(timeout)

    with httpx.Client(headers={"Ocp-Apim-Subscription-Key": key}, timeout=120) as client:
        analyzer = analyzer_id(schema, config.completion_model, config.api_version)
        _ensure_analyzer(client, endpoint, config.api_version, analyzer, schema,
                         config.completion_model, budget, config.poll_interval)
        r = client.post(f"{endpoint}/contentunderstanding/analyzers/{analyzer}:analyze"
                        f"?api-version={config.api_version}",
                        content=pdf.read_bytes(),
                        headers={"Content-Type": "application/octet-stream"})
        if r.status_code >= 400:
            raise VendorError(f"HTTP {r.status_code}: {r.text[:300]}",
                              status=r.status_code, body=r.text)
        op = r.headers.get("Operation-Location")
        if not op:
            raise VendorError("no Operation-Location header on :analyze",
                              status=r.status_code, body=r.text)

        body = _await(client, op, budget=budget, poll_interval=config.poll_interval,
                      want_result=True)

    contents = ((body or {}).get("result") or {}).get("contents") or []
    if not contents:
        raise VendorError("analysis returned no contents", status=200,
                          body=json.dumps(body)[:300])
    return Extraction(result=fields_to_dict(contents[0].get("fields") or {}),
                      raw=body,
                      cost=Cost(),
                      job_id=analyzer)

