"""Vendor adapters: one module per vendor, each an `extract()` function.

    extract(pdf, schema, *, timeout, **options) -> Extraction

It makes the vendor call, parses the answer, and returns both. It does not RETURN a failure --
it raises `VendorError` (or `VendorTimeout`) from where the failure happened, while it still
knows what happened. `harness/extraction.py` has the contract.

Each also keeps a `main()`, so one document can still be reproduced by hand:

    python -m omni_extract_bench.harness.providers.<vendor> --pdf X.pdf --schema S.json --out O.json

They were once CLIs *only*, launched as subprocesses so a transport tap could be installed in
the child before the vendor's SDK loaded. The tap is gone and so is the subprocess; the command
line survives because reproducing a single document is genuinely useful.

No `__all__` and nothing imported here. Importing a vendor adapter pulls its HTTP client, and
those come with the `harness` extra -- so naming them at package level would make
`import omni_extract_bench.harness.providers` fail on a scoring-only machine.
"""
