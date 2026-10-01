#!/usr/bin/env python3
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
# Note: every case runs in a fresh interpreter, because `registry` reads entry points once.
PRELUDE = """\
import sys, types
from omni_extract_bench.harness import PROVIDERS, add_adapter, adapter, registry
def outcome(f):
    try: return f"ok {f()}"
    except Exception as exc: return f"{type(exc).__name__}: {exc} {getattr(exc, '__notes__', '')}"
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


def run(root: Path, code: str = "", *args: str) -> str:
    argv = [sys.executable, *args] if args else [sys.executable, "-c",
                                                 PRELUDE + textwrap.dedent(code)]
    p = subprocess.run(argv, capture_output=True, text=True, cwd=root,
                       env={"PYTHONPATH": f"{root}:{ROOT}", "PATH": "/usr/bin:/bin"})
    return p.stdout + p.stderr


print("A DECLARED ADAPTER IS LISTED, AND IMPORTED ONLY WHEN ASKED FOR")
good = site({"fakepkg": {"fake-agent": "fake_adapter"}}, {"fake_adapter": ADAPTER})
out = run(good, """
    print("lazy", "fake-agent" in PROVIDERS, "fake_adapter" not in sys.modules)
    print("loaded", adapter("fake-agent").__name__, registry.settings_for("fake-agent", {"depth": 5}),
          registry.resolve("fake-agent"))
""")
report("listed in PROVIDERS without importing its module", "lazy True True" in out, out)
report("adapter(), settings_for and resolve work on it",
       "loaded fake_adapter {'depth': 5} fake_adapter" in out, out)
out = run(good, "", "-m", "omni_extract_bench.cli", "providers")
report("`oeb providers` lists it", "fake-agent" in out.splitlines(), out)
out = run(good, "", "-m", "omni_extract_bench.cli", "providers", "fake-agent")
report("`oeb providers fake-agent` shows its options", "depth" in out, out)

print("\nA NAME THAT WOULD BE AMBIGUOUS IS REFUSED, NAMING THE PACKAGE")
for label, dists in (("a built-in's name", {"fakepkg": {"datalab": "fake_adapter"}}),
                     ("a model id's '/'", {"fakepkg": {"org/model": "fake_adapter"}}),
                     ("one name from two packages", {"pkg-a": {"x": "fake_adapter"},
                                                     "fakepkg": {"x": "fake_adapter"}})):
    out = run(site(dists, {"fake_adapter": ADAPTER}))
    report(label, "ValueError: package '" in out, out)

print("\nA BROKEN ADAPTER FAILS WHEN ASKED FOR, SAYING WHOSE IT IS")
broken = {"fake_adapter": "import a_module_that_is_not_installed\n", "bare": "class Config: pass\n"}
out = run(site({"fakepkg": {"fake-agent": "fake_adapter", "bare-agent": "bare"}}, broken), """
    print("listed", "fake-agent" in PROVIDERS)
    print("import", outcome(lambda: adapter("fake-agent")))
    print("contract", outcome(lambda: adapter("bare-agent")))
""")
report("a module that fails to import is still listed", "listed True" in out, out)
report("...and raises the original error, noting provider and package",
       "a_module_that_is_not_installed" in out and "adapter 'fake-agent' from package 'fakepkg'"
       in out, out)
report("a module missing contract names raises, listing them",
       "TypeError" in out and "prepare_schema, extract" in out, out)

print("\nWITH NO PLUGINS INSTALLED, NOTHING CHANGES")
out = run(site({}, {}), 'print("same", PROVIDERS == sorted(registry.ADAPTERS), registry.PLUGINS)')
report("PROVIDERS is the built-ins", "same True {}" in out, out)

print("\nAN ADAPTER DEFINED IN CODE IS FILED WITH add_adapter")
out = run(site({"fakepkg": {"pkg-agent": "fake_adapter"}}, {"fake_adapter": ADAPTER}), """
    import fake_adapter as m
    add_adapter("code-agent", m)
    print("listed", "code-agent" in PROVIDERS, PROVIDERS == sorted(PROVIDERS))
    print("loaded", adapter("code-agent").__name__)
    print("unknown", outcome(lambda: adapter("nonesuch")))
    print("builtin", outcome(lambda: add_adapter("datalab", m)))
    print("plugin", outcome(lambda: add_adapter("pkg-agent", m)))
    print("again", outcome(lambda: add_adapter("code-agent", m)))
    print("slash", outcome(lambda: add_adapter("org/model", m, replace=True)))
    print("missing", outcome(lambda: add_adapter("bare", types.ModuleType("bare"))))
    print("bare", "bare" in PROVIDERS)
    add_adapter("datalab", m, replace=True)
    print("replaced", adapter("datalab").__name__, PROVIDERS.count("datalab"))
""")
got = dict(l.split(" ", 1) for l in out.splitlines() if " " in l)
report("the name joins PROVIDERS, the list callers already imported, kept sorted",
       got.get("listed") == "True True", out)
report("adapter() returns it", got.get("loaded") == "fake_adapter", out)
report("the unknown-provider error lists it", "code-agent" in got.get("unknown", ""), out)
report("a built-in's name is refused, naming replace=True",
       got.get("builtin", "").startswith("ValueError") and "replace=True" in got["builtin"], out)
for key, label in (("plugin", "a name an installed package declares"),
                   ("again", "a name already added in code"),
                   ("slash", "a name with '/' (even with replace=True)")):
    report(f"{label} is refused", got.get(key, "").startswith("ValueError"), out)
report("a module missing contract names is refused, and not listed",
       "prepare_schema, extract" in got.get("missing", "") and got.get("bare") == "False", out)
report("replace=True replaces, without listing the name twice",
       got.get("replaced") == "fake_adapter 1", out)

print(f"\n{'ADAPTERS FROM OTHER PACKAGES PLUG IN' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
sys.exit(1 if FAILS else 0)
