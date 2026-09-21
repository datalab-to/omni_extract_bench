"""The adapters read the environment for CREDENTIALS ONLY.

Everything that steers a vendor -- mode, tier, array strategy, api version, base url,
completion model -- is a field on the adapter's `Config`, so it arrives through `--options` and
is recorded in `run_manifest.settings`. An environment variable steers a run without appearing
in its record: `LLAMAEXTRACT_TIER=cost_effective` once produced a run indistinguishable from
the maxed-out one the benchmark claims to publish.

Credentials are the exception on purpose. A key changes whether a call is ALLOWED, not what it
asks, and it must not be on a command line or in a record -- so it stays in the environment.
"""
import ast
import pathlib

PROVIDERS = pathlib.Path(__file__).resolve().parents[1] / "omni_extract_bench/harness/providers"

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
    + "\n\nAdd a field to the adapter's `Config` with the default you want; then it "
      "reaches the adapter through `oeb benchmark --options` and shows up in "
      "`run_manifest.settings`, which is what makes a run state what it asked."
)
print(f"{len(modules)} adapters: no steering read from the environment")

import dataclasses  # noqa: E402
import importlib  # noqa: E402

PROMOTED = {
    "datalab": ("mode", "base_url"),
    "reducto": ("agentic_table_mode",),
    "extend": ("array_strategy", "api_version", "base_url"),
    "llamaextract": ("tier",),
    "azure_cu": ("completion_model",),
    "llm_single_shot": ("base_url",),
}
def config_fields(module):
    """The adapter's `Config` fields -- the one declaration of what it can be asked."""
    return {f.name: f for f in dataclasses.fields(
        importlib.import_module(f"omni_extract_bench.harness.providers.{module}").Config)}


for module, params in PROMOTED.items():
    missing = [p for p in params if p not in config_fields(module)]
    assert not missing, f"{module}.Config cannot be steered through options: {missing}"
print(f"{sum(len(v) for v in PROMOTED.values())} former env settings are Config fields")

for module in ("datalab", "reducto", "llamaextract", "azure_cu"):
    fields = config_fields(module)
    for name in PROMOTED[module]:
        got = fields[name].default
        assert got is not dataclasses.MISSING and got is not None, \
            f"{module}.Config.{name} should carry its literal default, got {got!r}"
print("every promoted setting carries its literal default on the Config")
