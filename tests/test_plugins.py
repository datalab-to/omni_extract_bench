#!/usr/bin/env python3
"""Adapters from outside the harness -- the `omni_extract_bench.adapters` entry-point group.

A package installed beside the harness declares `name = module` under the group, and the
registry lists the name from metadata alone and imports the module the first time the name
is asked for. Each case below is a fake installed package: a `dist-info` folder with an
`entry_points.txt` beside an adapter module, on `sys.path` and nowhere else.

An adapter defined in code, with no package, is filed with `add_adapter` instead, under the
same rules for names.

`registry` reads the group once, at import, so every case runs in a fresh interpreter.

Run: python3 tests/test_plugins.py
"""
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []

ADAPTER = """\
import dataclasses

@dataclasses.dataclass(frozen=True)
class Config:
    depth: int = 3

def prepare_schema(schema):
    return schema

def extract(pdf, schema, *, timeout, config):
    raise AssertionError("not called")
"""


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


def site(dists: dict[str, dict[str, str]], modules: dict[str, str]) -> Path:
    """A directory that looks like site-packages: `dists` maps a package to its entry points."""
    root = Path(tempfile.mkdtemp())
    for dist, eps in dists.items():
        info = root / f"{dist}-0.1.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.1\n")
        lines = "".join(f"{k} = {v}\n" for k, v in eps.items())
        (info / "entry_points.txt").write_text(f"[omni_extract_bench.adapters]\n{lines}")
    for name, src in modules.items():
        (root / f"{name}.py").write_text(src)
    return root


def run(root: Path, code: str, *args: str) -> subprocess.CompletedProcess:
    argv = [sys.executable, "-c", textwrap.dedent(code)] if code else [sys.executable, *args]
    return subprocess.run(argv, capture_output=True, text=True, cwd=root,
                          env={"PYTHONPATH": f"{root}:{ROOT}", "PATH": "/usr/bin:/bin"})


print("A DECLARED ADAPTER IS LISTED, AND IMPORTED ONLY WHEN ASKED FOR")
good = site({"fakepkg": {"fake-agent": "fake_adapter"}}, {"fake_adapter": ADAPTER})
out = run(good, """
    import sys
    from omni_extract_bench.harness import registry
    print("listed", "fake-agent" in registry.PROVIDERS)
    print("lazy", "fake_adapter" not in sys.modules)
    print("loaded", registry.adapter("fake-agent").__name__)
    print("settings", registry.settings_for("fake-agent", {"depth": 5}))
    print("resolve", registry.resolve("fake-agent"))
""")
report("the name is in PROVIDERS", "listed True" in out.stdout, out.stdout + out.stderr)
report("...without importing its module", "lazy True" in out.stdout, out.stdout + out.stderr)
report("adapter() imports and returns the module", "loaded fake_adapter" in out.stdout,
       out.stdout + out.stderr)
report("its Config takes --options", "settings {'depth': 5}" in out.stdout,
       out.stdout + out.stderr)
report("resolve() names it by its module", "resolve fake_adapter" in out.stdout,
       out.stdout + out.stderr)

cli = run(good, "", "-m", "omni_extract_bench.cli", "providers")
report("`oeb providers` lists it", "fake-agent" in cli.stdout.splitlines(),
       cli.stdout + cli.stderr)
cli = run(good, "", "-m", "omni_extract_bench.cli", "providers", "fake-agent")
report("`oeb providers fake-agent` shows its options", "depth" in cli.stdout,
       cli.stdout + cli.stderr)

print("\nA NAME THAT WOULD BE AMBIGUOUS IS REFUSED")
for label, name in (("a built-in's name", "datalab"), ("a model id's '/'", "org/model")):
    root = site({"fakepkg": {name: "fake_adapter"}}, {"fake_adapter": ADAPTER})
    out = run(root, "from omni_extract_bench.harness import registry")
    report(f"{label} raises, naming the package",
           out.returncode != 0 and "ValueError" in out.stderr and "'fakepkg'" in out.stderr,
           out.stderr)

root = site({"pkg-a": {"fake-agent": "fake_adapter"}, "pkg-b": {"fake-agent": "fake_adapter"}},
            {"fake_adapter": ADAPTER})
out = run(root, "from omni_extract_bench.harness import registry")
report("one name from two packages raises, naming both",
       out.returncode != 0 and "'pkg-a'" in out.stderr and "'pkg-b'" in out.stderr, out.stderr)

print("\nA BROKEN ADAPTER FAILS WHEN ASKED FOR, SAYING WHOSE IT IS")
root = site({"fakepkg": {"fake-agent": "fake_adapter"}},
            {"fake_adapter": "import a_module_that_is_not_installed\n"})
out = run(root, """
    from omni_extract_bench.harness import registry
    print("listed", "fake-agent" in registry.PROVIDERS)
    registry.adapter("fake-agent")
""")
report("a module that fails to import is still listed", "listed True" in out.stdout,
       out.stdout + out.stderr)
raised = next((l for l in out.stderr.splitlines() if l.startswith("ImportError:")), "")
report("...and raises naming provider, package and module",
       "'fake-agent'" in raised and "'fakepkg'" in raised and "'fake_adapter'" in raised,
       out.stderr)
report("...keeping the original error", "a_module_that_is_not_installed" in out.stderr,
       out.stderr)

root = site({"fakepkg": {"fake-agent": "fake_adapter"}},
            {"fake_adapter": "class Config: pass\n"})
out = run(root, """
    from omni_extract_bench.harness import registry
    registry.adapter("fake-agent")
""")
report("a module missing contract names raises, listing them",
       "TypeError" in out.stderr and "prepare_schema, extract" in out.stderr, out.stderr)

print("\nWITH NO PLUGINS INSTALLED, NOTHING CHANGES")
root = site({}, {})
out = run(root, """
    from omni_extract_bench.harness import registry
    print(registry.PROVIDERS == sorted(registry.ADAPTERS), registry.PLUGINS)
""")
report("PROVIDERS is the built-ins", "True {}" in out.stdout, out.stdout + out.stderr)

print("\nAN ADAPTER DEFINED IN CODE IS FILED WITH add_adapter")
root = site({"fakepkg": {"pkg-agent": "fake_adapter"}}, {"fake_adapter": ADAPTER})
out = run(root, """
    import types
    from omni_extract_bench.harness import PROVIDERS, add_adapter, adapter, registry
    import fake_adapter

    def outcome(f):
        try:
            f()
            return "ok"
        except (TypeError, ValueError) as exc:
            return f"{type(exc).__name__}: {exc}"

    add_adapter("code-agent", fake_adapter)
    print("listed", "code-agent" in PROVIDERS, PROVIDERS == sorted(PROVIDERS))
    print("loaded", adapter("code-agent").__name__)
    print("unknown", outcome(lambda: adapter("nonesuch")))
    print("builtin", outcome(lambda: add_adapter("datalab", fake_adapter)))
    print("plugin", outcome(lambda: add_adapter("pkg-agent", fake_adapter)))
    print("again", outcome(lambda: add_adapter("code-agent", fake_adapter)))
    print("slash", outcome(lambda: add_adapter("org/model", fake_adapter, replace=True)))
    print("missing", outcome(lambda: add_adapter("bare", types.ModuleType("bare"))))
    print("bare listed", "bare" in PROVIDERS)
    add_adapter("datalab", fake_adapter, replace=True)
    print("replaced", adapter("datalab").__name__, PROVIDERS.count("datalab"))
""")
lines = dict(l.split(" ", 1) for l in out.stdout.splitlines() if " " in l)
detail = out.stdout + out.stderr
report("add_adapter is exported from the harness", out.returncode == 0, detail)
report("the name joins PROVIDERS, the list callers already imported, kept sorted",
       lines.get("listed") == "True True", detail)
report("adapter() returns it", lines.get("loaded") == "fake_adapter", detail)
report("the unknown-provider error lists it", "code-agent" in lines.get("unknown", ""), detail)
report("a built-in's name is refused, naming replace=True",
       "ValueError" in lines.get("builtin", "") and "replace=True" in lines["builtin"], detail)
report("a name an installed package declares is refused",
       "ValueError" in lines.get("plugin", ""), detail)
report("a name already added in code is refused", "ValueError" in lines.get("again", ""), detail)
report("a name with '/' is refused even with replace=True",
       "ValueError" in lines.get("slash", ""), detail)
report("a module missing contract names raises, listing them",
       "TypeError" in lines.get("missing", "") and "prepare_schema, extract" in lines["missing"],
       detail)
report("...and is not listed", lines.get("bare") == "listed False",
       detail)
report("replace=True replaces, without listing the name twice",
       lines.get("replaced") == "fake_adapter 1", detail)

print(f"\n{'ADAPTERS FROM OTHER PACKAGES PLUG IN' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
sys.exit(1 if FAILS else 0)
