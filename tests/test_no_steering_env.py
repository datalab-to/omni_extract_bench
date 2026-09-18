"""The adapters read the environment for CREDENTIALS ONLY.

Everything that steers a vendor -- mode, tier, array strategy, api version, base url,
completion model -- is a keyword argument, so it arrives through `options` and is recorded in
`run_manifest.overrides`. An environment variable steers a run without appearing in its
record: `LLAMAEXTRACT_TIER=cost_effective` once produced a run indistinguishable from the
maxed-out one the benchmark claims to publish. Recording the variables was the first fix;
removing them is the real one, and this test is what keeps them gone.

Credentials are the exception on purpose. A key changes whether a call is ALLOWED, not what it
asks, and it must not be on a command line or in a record -- so it stays in the environment.
"""
import ast
import pathlib

PROVIDERS = pathlib.Path(__file__).resolve().parents[1] / "omni_extract_bench/harness/providers"

#: Reading one of these is reading a secret (or, for the two Azure/Extend account fields, the
#: coordinates the secret is valid at). Anything else is a setting and belongs in a parameter.
CREDENTIAL = {
    "DATALAB_API_KEY", "REDUCTO_API_KEY", "EXTEND_API_KEY", "MISTRAL_API_KEY",
    "LLAMA_CLOUD_API_KEY", "LLAMAPARSE_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY",
    "AZURE_CU_ENDPOINT", "AZURE_CU_KEY", "EXTEND_WORKSPACE_ID",
}


def env_reads(path: pathlib.Path) -> list[tuple[str, int]]:
    """Every `os.environ[...]`/`.get(...)`/`os.getenv(...)` name in a module, with its line."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        src = ""
        if isinstance(node, ast.Call):
            src = ast.unparse(node.func)
            if not (src.endswith("environ.get") or src.endswith("getenv")):
                continue
            arg = node.args[0] if node.args else None
        elif isinstance(node, ast.Subscript) and "environ" in ast.unparse(node.value):
            arg = node.slice
        else:
            continue
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            found.append((arg.value, node.lineno))
    return found


modules = sorted(p for p in PROVIDERS.glob("*.py") if p.stem != "__init__")
assert modules, "no adapters found"

offenders = {}
for path in modules:
    for name, line in env_reads(path):
        if name not in CREDENTIAL:
            offenders.setdefault(path.name, []).append(f"{name} (line {line})")

assert not offenders, (
    "adapters must take settings as arguments, not out of the environment:\n"
    + "\n".join(f"  {f}: {', '.join(v)}" for f, v in offenders.items())
    + "\n\nAdd a keyword argument with the default you want and expose it in `main()`; then it "
      "reaches the adapter through `oeb benchmark --options` and shows up in "
      "`run_manifest.overrides`, which is what makes a non-stock run say so."
)
print(f"{len(modules)} adapters: no steering read from the environment")

# The settings that USED to be environment variables are all reachable as arguments -- removing
# the variable without promoting it to a parameter would just make the setting unreachable.
import inspect  # noqa: E402
import importlib  # noqa: E402

PROMOTED = {
    "datalab": ("mode", "base_url"),
    "reducto": ("agentic_table_mode",),
    "extend": ("array_strategy", "api_version", "base_url"),
    "llamaextract": ("tier",),
    "azure_cu": ("completion_model",),
    "llm_single_shot": ("base_url",),
}
for module, params in PROMOTED.items():
    sig = inspect.signature(
        importlib.import_module(f"omni_extract_bench.harness.providers.{module}").extract)
    missing = [p for p in params if p not in sig.parameters]
    assert not missing, f"{module}.extract cannot be steered through options: {missing}"
print(f"{sum(len(v) for v in PROMOTED.values())} former env settings are keyword arguments")

# And none of them is optional-with-a-None-default pretending to have one: a `None` default is
# a second place the real default can hide.
for module in ("datalab", "reducto", "llamaextract", "azure_cu"):
    sig = inspect.signature(
        importlib.import_module(f"omni_extract_bench.harness.providers.{module}").extract)
    for name in PROMOTED[module]:
        got = sig.parameters[name].default
        assert got is not inspect.Parameter.empty and got is not None, \
            f"{module}.extract({name}=) should carry its literal default, got {got!r}"
print("every promoted setting carries its literal default in the signature")
