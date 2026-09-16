"""Vendor adapters: one module per vendor, each runnable on its own.

    python -m omni_extract_bench.harness.providers.<vendor> --pdf X.pdf --schema S.json --out O.json

Each is a CLI rather than a library because `run_provider` launches them as subprocesses: the
transport tap has to be installed before the vendor's SDK is imported, and a monkey-patch in
this process cannot reach a child. Running one by hand is then also how you reproduce a single
document without the runner.

No `__all__` and nothing imported here. Importing a vendor adapter pulls its HTTP client, and
several are only installed with the `harness` extra -- so naming them at package level would
make `import omni_extract_bench.harness.providers` fail on a scoring-only machine.
"""
