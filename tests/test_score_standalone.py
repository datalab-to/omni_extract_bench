#!/usr/bin/env python3
"""`score.py` must not depend on `grading.py`.

The old scorer is kept for now as an independent cross-check (see
`tests/test_score_equivalence.py`), but it is on its way out. This test fails the moment
`score.py` -- or the package `__init__` -- reaches back into it, so the deletion stays a
one-line change rather than an archaeology exercise.

The shared value layer lives in `values.py`: `canon_key` must be importable from exactly one
place, or the two jobs that ask "are these equal?" can silently disagree, which has happened
twice.

Run: python3 tests/test_score_standalone.py
"""
import ast
import os as _os
import sys as _sys
from pathlib import Path

_ROOT = Path(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, str(_ROOT))

PKG = _ROOT / "omni_extract_bench"
FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def imports_of(path):
    """Every module inside the package that this file imports, as a path from the package root.

    Careful about two forms that are easy to get wrong, and both appear here:
    `from . import normalize as N` names a MODULE in its aliases rather than in `.module`,
    and `from .vendor.longextract_bench import grading as G` is a different file from
    `grading.py` -- so the full path matters, not the last segment. Anything that does not
    resolve to a real file under the package is dropped, since it is a name, not a module.
    """
    out = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        candidates = []
        if isinstance(node, ast.ImportFrom) and node.level:              # relative
            base = node.module or ""
            for a in node.names:
                candidates.append(f"{base}.{a.name}" if base else a.name)
            if base:
                candidates.append(base)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("omni_extract_bench."):
                    candidates.append(a.name.split(".", 1)[1])
        for c in candidates:
            rel = c.replace(".", "/")
            if (PKG / f"{rel}.py").exists() or (PKG / rel / "__init__.py").exists():
                out.add(c)
    return out


print("\nscore.py DOES NOT IMPORT grading.py")
score_imports = imports_of(PKG / "score.py")
report("score.py imports neither grading nor anything from it",
       "grading" not in score_imports, f"imports {sorted(score_imports)}")
report("score.py takes the value layer from values.py",
       "values" in score_imports, f"imports {sorted(score_imports)}")

print("\nTHE PACKAGE __init__ DOES NOT IMPORT grading.py")
init_imports = imports_of(PKG / "__init__.py")
report("__init__.py imports nothing from grading",
       "grading" not in init_imports, f"imports {sorted(init_imports)}")

print("\nNOR DOES ANYTHING score.py DEPENDS ON")
seen, stack = set(), list(score_imports | init_imports)
while stack:
    mod = stack.pop()
    path = PKG / f"{mod.replace('.', '/')}.py"
    if mod in seen or not path.exists():
        continue
    seen.add(mod)
    stack.extend(imports_of(path))
report("the whole transitive closure is free of grading",
       "grading" not in seen, f"closure: {sorted(seen)}")
print(f"          closure: {sorted(seen)}")

print("\nAND canon_key COMES FROM EXACTLY ONE PLACE")
import omni_extract_bench as pkg                                        # noqa: E402
from omni_extract_bench import grading, score, values                   # noqa: E402

report("values.canon_key is the one definition",
       score.canon_key is values.canon_key
       and grading.canon_key is values.canon_key
       and pkg.canon_key is values.canon_key)
report("...and so is cmp_leaf",
       score.cmp_leaf is values.cmp_leaf and grading.cmp_leaf is values.cmp_leaf)
report("values.py itself imports no grader",
       "grading" not in imports_of(PKG / "values.py"),
       f"imports {sorted(imports_of(PKG / 'values.py'))}")

print(f"\n{'score.py IS STANDALONE' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
