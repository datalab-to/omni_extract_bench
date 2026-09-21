"""Vendor adapters: one module per vendor, each satisfying `harness.contract.Adapter`.

    Config          what the vendor can be asked
    prepare_schema  the JSON Schema -> whatever this vendor's API takes
    extract         makes the call, parses the answer, returns both

`extract` does not RETURN a failure -- it raises `VendorError` (or `VendorTimeout`) from where
the failure happened, while it still knows what happened. `harness/contract.py` states the
contract, and `registry.ADAPTERS` maps a provider name to one of these modules.

One document is reproduced by hand with `oeb predict`, which goes through `document.predict`
and so applies the same parity rules and writes the same record a benchmark run would:

    oeb predict --provider <vendor> --doc X.pdf --schema S.json

The adapters used to carry a `main()` each, generating their own flags from the same `Config`.
That was a second way in, and it prepared the schema differently from the benchmark it existed
to explain -- it applied no overlay and stripped no benchmark-only keys.

No `__all__` and nothing imported here -- but that no longer makes this package cheap to
import. `harness/__init__.py` runs first and imports `registry`, which names every adapter, so
reaching this module at all pulls httpx and openai with it. That is why the `harness` extra is
checked once, there, instead of per adapter. The scorer is what stays free of all this: it
never imports `harness`, and `tests/test_score_standalone.py` fails if that changes.
"""
