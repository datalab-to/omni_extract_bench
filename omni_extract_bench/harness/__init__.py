"""Everything that is not the scorer: producing predictions, and preparing them to be scored.

The package above this one is the metric. `score.py` reaches exactly `matching`, `normalize`
and `values` and nothing else -- `tests/test_score_standalone.py` fails if that ever widens --
so a grade cannot come to depend on a transport, a vendor dialect, or an envelope convention.
This directory is where those live.

    capture.py       records vendor calls at the transport layer, so a run is paid for once
    _tap/            the taps themselves, imported by PATH rather than by name: a subprocess
                     `sitecustomize` must load them before any SDK, without this package being
                     installed in the child. Invisible to an import scanner; not dead.
    dialects.py      reshapes a JSON Schema into what a given vendor will accept
    prediction_io.py `usable()` -- is this payload something to score, or a recorded failure
    providers/       vendor adapters

WHAT IS NOT HERE, AND WHERE IT WENT
-----------------------------------
The runner. Nothing in this package calls `capture.install_taps()`, and `providers/` holds one
adapter of the nine vendors on the board. The code that drove them is not in this repository,
not in the datalab monorepo, and not on any machine we have looked at.

It did leave a complete account of itself. Every prediction in
`s3://example-bucket/omni-extract-bench/runs/full/baselines/<vendor>/_raw/` carries
the schema actually sent, every HTTP call with method, url, status, elapsed and body, the job
ids, the cost, and a `run_manifest` giving timeout, tier and max output tokens. Those records
are in THIS module's format -- `http`, `job_ids`, the 20,000-character `truncated` flag are
`capture.record` and `capture.MAX_BODY` -- which is the evidence that a caller existed and what
it would take to write another.

So `capture.py` is not scaffolding for something hypothetical. It is half of a system whose
other half is missing, kept because it is also the specification for rebuilding that half.
"""
