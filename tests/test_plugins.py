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
from omni_extract_bench import benchmark
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
    root = Path(tempfile.mkdtemp())
    for dist, eps in dists.items():
        info = root / f"{dist}-0.1.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.1\n")
        lines = "".join(f"{k} = {v}\n" for k, v in eps.items())
        (info / "entry_points.txt").write_text(f"[omni_extract_bench.adapters]\n{lines}")
    for name, src in modules.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(src)
    return root


def run(root: Path, code: str = "", *args: str) -> str:
    argv = [sys.executable, *args] if args else [sys.executable, "-c",
                                                 PRELUDE + textwrap.dedent(code)]
    p = subprocess.run(argv, capture_output=True, text=True, cwd=root,
                       env={"PYTHONPATH": f"{root}:{ROOT}", "PATH": "/usr/bin:/bin"})
    return p.stdout + p.stderr


root = site({"fakepkg": {"fake-agent": "fake_adapter", "broken": "broken", "bare": "bare"},
             "pkg-a": {"agent-a": "pkg_a.adapter", "my-datalab": "pkg_a.datalab"},
             "pkg-b": {"agent-b": "pkg_b.adapter"}},
            {"fake_adapter.py": ADAPTER, "broken.py": "import not_installed\n",
             "bare.py": "class Config: pass\n", "pkg_a/__init__.py": "",
             "pkg_a/adapter.py": ADAPTER, "pkg_a/datalab.py": ADAPTER, "pkg_b/__init__.py": "",
             "pkg_b/adapter.py": ADAPTER})
out = run(root, """
    print("lazy", "fake-agent" in PROVIDERS, "fake_adapter" not in sys.modules)
    print("loaded", adapter("fake-agent").__name__, registry.settings_for("fake-agent", {"depth": 5}))
    print("import", outcome(lambda: adapter("broken")))
    print("contract", outcome(lambda: adapter("bare")))
    print("keys", *(registry.resolve(p) for p in ("agent-a", "agent-b", "my-datalab", "datalab")))
    print("workers", benchmark.workers_for("my-datalab", None))
    import fake_adapter as m
    add_adapter("code-agent", m)
    print("added", "code-agent" in PROVIDERS, PROVIDERS == sorted(PROVIDERS))
    print("taken", outcome(lambda: add_adapter("datalab", m)), outcome(lambda: add_adapter("fake-agent", m)))
    print("slash", outcome(lambda: add_adapter("org/model", m, replace=True)))
    add_adapter("datalab", m, replace=True)
    print("replaced", adapter("datalab").__name__)
""")
got = dict(l.split(" ", 1) for l in out.splitlines() if " " in l)
report("a plugin is listed without importing its module", got.get("lazy") == "True True", out)
report("adapter() imports it, and its Config takes options",
       got.get("loaded") == "fake_adapter {'depth': 5}", out)
report("an import failure keeps its error, noting provider and package",
       "not_installed" in got.get("import", "") and "from package 'fakepkg'" in got["import"], out)
report("a module missing contract names is refused, naming them",
       "TypeError" in got.get("contract", "") and "prepare_schema, extract" in got["contract"], out)
report("modules that end alike get separate keys, and no built-in's cap",
       got.get("keys") == "pkg_a.adapter pkg_b.adapter pkg_a.datalab datalab"
       and got.get("workers") == "5", out)
report("add_adapter lists the name in PROVIDERS, kept sorted", got.get("added") == "True True", out)
report("add_adapter refuses a built-in's or a plugin's name",
       got.get("taken", "").count("ValueError") == 2 and "replace=True" in got["taken"], out)
report("add_adapter refuses '/' even with replace=True", "ValueError" in got.get("slash", ""), out)
report("replace=True replaces", got.get("replaced") == "fake_adapter", out)

out = run(root, "", "-m", "omni_extract_bench.cli", "providers", "fake-agent")
report("`oeb providers fake-agent` shows its options", "depth" in out, out)

for label, dists in (("a built-in's name", {"fakepkg": {"datalab": "fake_adapter"}}),
                     ("a model id's '/'", {"fakepkg": {"org/model": "fake_adapter"}}),
                     ("one name from two packages", {"pkg-a": {"x": "fake_adapter"},
                                                     "fakepkg": {"x": "fake_adapter"}})):
    out = run(site(dists, {"fake_adapter.py": ADAPTER}))
    report(f"a plugin with {label} is refused, naming the package",
           "ValueError: package '" in out, out)

print(f"\n{'ADAPTERS FROM OTHER PACKAGES PLUG IN' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
sys.exit(1 if FAILS else 0)
