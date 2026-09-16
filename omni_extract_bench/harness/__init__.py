"""Everything that is not the scorer: producing predictions, and preparing them to be scored.

The package above this one is the metric. `score.py` reaches exactly `matching`, `normalize`
and `values` and nothing else -- `tests/test_score_standalone.py` fails if that ever widens --
so a grade cannot come to depend on a transport, a vendor dialect, or an envelope convention.
This directory is where those live.

    run_provider.py  the runner: one provider over a document set, under the parity rules
    capture.py       records vendor calls at the transport layer, so a run is paid for once
    _tap/            the taps themselves, imported by PATH rather than by name: a subprocess
                     `sitecustomize` must load them before any SDK, without this package being
                     installed in the child. Invisible to an import scanner; not dead.
    dialects.py      reshapes a JSON Schema into what a given vendor will accept
    schema_overlay.py writes a gold convention into the field descriptions every vendor sees
    prediction_io.py `usable()` -- is this payload something to score, or a recorded failure
    recover_timeouts.py  re-reads a job a vendor finished after the adapter stopped waiting
    providers/       vendor adapters, one per vendor, each runnable as `python -m`

    pip install 'omni-extract-bench[harness]'

The scorer needs none of it. Grading reads JSON off a disk, so a machine that only scores
never installs a vendor SDK -- which is why the adapters' dependencies are an extra rather
than a dependency, and why nothing above this directory imports anything in it.

WHAT A RUN LEAVES BEHIND
------------------------
Every prediction this package produces is accompanied by a raw record carrying the schema
actually sent, every HTTP call, the job ids, the cost and a `run_manifest`, in
`capture.record`'s own format. That is what makes a vendor's score checkable after the fact:
what it was asked, what it answered, and what the call cost, rather than the parsed result
alone.
"""
