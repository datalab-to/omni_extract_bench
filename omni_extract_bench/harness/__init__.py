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

WHAT WAS MISSING, AND WHERE IT CAME BACK FROM
---------------------------------------------
This file used to record that the runner did not exist: that nothing called
`capture.install_taps()`, that `providers/` held one adapter of the nine vendors on the board,
and that the code which drove them was not in this repository or on any machine we had looked
at. What it had instead was the evidence a caller must have existed -- every prediction in
`s3://datalab-training-pipelines/omni-extract-bench/runs/full/baselines/<vendor>/_raw/` carries
the schema actually sent, every HTTP call, the job ids, the cost, and a `run_manifest`, all in
`capture.record`'s own format.

The runner was on `fix/scorer-parity-660` the whole time. It is now here, and the account
above stands as written: those records are what it produces.
"""
