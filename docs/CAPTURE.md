# Capturing runs so you pay for them once

Benchmarking extraction vendors costs real money on calls that are slow, often async, and
sometimes nondeterministic. What you keep from each call decides whether a failure costs you an
explanation or another invoice — so `omni_extract_bench.harness.capture` records responses at the
**transport layer**, before anything parses them.

```python
from omni_extract_bench.harness import capture

capture.install_taps()                 # in-process SDK calls (httpx + requests)
capture.reset()                        # per document; records are thread-local
result = run_one_document(pdf, schema)
record = {
    "result": result,
    "http": capture.records(),         # status, body, timing, and the request minus its payload
    "job_ids": capture.job_ids(),      # what makes cost recoverable WITHOUT re-running
    "usage": capture.usage_from_records(),
}
```

For adapters you launch as their own process, patching the parent reaches nothing:

```python
env = capture.subprocess_env(os.environ, log_path)   # sitecustomize taps the child
subprocess.run(cmd, env=env, ...)
capture.merge_subprocess_log(log_path)               # merge on failure too — especially then
```

Three things are worth stating plainly, because each was learned the expensive way:

**Capture below the parser, or you capture the parser's opinion.** Wrapping the adapter records
its return value, so a model's text is gone whenever `json.loads` failed, an HTTP error body
becomes a status code, and the vendor's billing fields vanish with the envelope. Those look like
three bugs and are one placement error.

**Test the contents, not the key.** An audit here counted capture files, found a populated field
named `raw` on every one, and reported full coverage — while the field held parsed output. The
same audit passed while *subprocess* providers captured no HTTP whatsoever.

**Tap every transport the code *could* use, not the one you believe it uses.** Four are covered
— `httpx.Client`, `httpx.AsyncClient`, `requests.Session`, `urllib.request` — plus a
`sitecustomize` for subprocesses. Each gap was found separately, *after* the previous one had
supposedly fixed capture, because they all look identical from the outside: the file exists, the
key is present, the list is empty. The last one was a provider adapter that used no SDK at all
and called the REST API with the standard library.

They live in one module (`harness/_tap/oeb_capture.py`) for the same reason: maintained as near-copies,
each gap has to be found and fixed once per copy, which is how the fourth one survived the
first three fixes.

**Keep job ids.** Async APIs hand back an id and keep the job, so usage can usually be recovered
from job history later for free. Re-running a document to recover a number the vendor will still
hand you is the worst available trade, and for a nondeterministic provider it does not even
reproduce the answer you scored.

**A parsing mistake should cost a re-parse, not another invoice.** Because the bodies are on
disk, a bug in how usage is *read* is fixed by re-reading them. That happened here: the usage
parser filtered keys against a list of known names and so discarded a vendor's
`num_pages_billed` for not being on the list — page counts recovered afterwards from stored
responses, with nothing re-run. Keep whatever the vendor puts in its usage block; a known-names
filter is the hard-coded-path mistake one level down.

Units deserve the same care as values: several vendors report `credits`, which is not dollars —
the rate is contract-specific. `usage_from_records` keeps the vendor's own field names so no
conversion happens by accident, and `dialects.cost_from_response` converts only where the unit
is known (a field named in cents reported as dollars overstates by 100x).

