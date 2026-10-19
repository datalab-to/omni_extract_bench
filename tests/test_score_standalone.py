#!/usr/bin/env python3
"""The scoring code stands alone, and `canon_key` has one definition. It checks that:

  * scoring reaches only `documents`, `pairing`, `matching`, `values` and `recognise`, and `field_matching.py`
    is the one file that imports outside the package;
  * there is one scorer: no other module is named for grading or defines a score; and
  * `canon_key` is defined in one place, because two definitions of "are these equal" eventually disagree.

Run: python3 tests/test_score_standalone.py
"""
import ast
import os as _os
import sys as _sys
from pathlib import Path

_ROOT = Path(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, str(_ROOT))

PKG = _ROOT / "omni_extract_bench"
FM = PKG / "scorers" / "field_matching"   # the one scorer: imports inside it resolve here
SCORING_PATH = {"matching", "values", "recognise", "documents", "pairing"}
BRIDGE = "field_matching"   # the scorer as an omni-bench-core `Scorer`: the one file allowed to import outside
FAILS = []


def note(text):
    print(f"          {text}")


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def imports_of(path):
    """Every module inside the package that this file imports, as a path from the package root.

    Careful about two forms that are easy to get wrong: `from . import matching as OM` names a
    module in its aliases rather than in `.module`, and a dotted path like
    `from .a.b import c` is a different file from `c.py` -- so the full path matters, not the
    last segment. Anything that does not resolve to a real file under the package is dropped,
    since it is a name, not a module.
    """
    out = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        candidates = []
        if isinstance(node, ast.ImportFrom) and node.level:
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
            if (FM / f"{rel}.py").exists() or (FM / rel / "__init__.py").exists():
                out.add(c)
    return out


def closure(start):
    seen, stack = set(), list(start)
    while stack:
        mod = stack.pop()
        path = FM / f"{mod.replace('.', '/')}.py"
        if mod in seen or not path.exists():
            continue
        seen.add(mod)
        stack.extend(imports_of(path))
    return seen


print("\nThe scoring path is closed")
direct = imports_of(FM / "compare.py") | imports_of(FM / "pairing.py") | imports_of(FM / "documents.py")
reach = closure(direct)
report("the scorer imports nothing outside the scoring path",
       direct <= SCORING_PATH, f"imports {sorted(direct)}")
report("...and the closure is exactly the scoring path, nothing wider",
       reach == SCORING_PATH, f"closure {sorted(reach)}")
note(f"the scorer reaches {sorted(direct)} directly; `recognise` arrives through `values`")


def outside(path):
    """The imports of `path` that leave this package and the standard library: another package (by its
    full module name, so `omni_bench_core.errors` and `omni_bench_core.scorer` differ), or the rest of oeb."""
    out = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level > 1:
            out.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
    return {m for m in out if m.startswith(".") or m.split(".")[0] not in _sys.stdlib_module_names | {"numpy", "scipy"}}


# Note: the math may raise the core's `UsageError` for a gold it can't read, and `omni_bench_core.errors` imports
# nothing, so it keeps the math standalone.
ALLOWED_OUTSIDE = {"omni_bench_core.errors"}
math_files = [p for p in FM.glob("*.py") if p.stem not in (BRIDGE, "__init__")]
leaks = {p.name: sorted(outside(p) - ALLOWED_OUTSIDE) for p in math_files if outside(p) - ALLOWED_OUTSIDE}
report("only field_matching.py imports outside the package; the math imports nothing of omni-bench-core or oeb",
       not leaks, f"imports outside: {leaks}")
report("values.py, the shared layer, stays below the scorer",
       "score" not in imports_of(FM / "values.py"),
       f"values imports {sorted(imports_of(FM / 'values.py'))}")
report("recognise.py depends on nothing in the package (is this text a date / a number)",
       not imports_of(FM / "recognise.py"),
       f"recognise imports {sorted(imports_of(FM / 'recognise.py'))}")
report("values.py reaches recognise and nothing else",
       imports_of(FM / "values.py") == {"recognise"},
       f"values imports {sorted(imports_of(FM / 'values.py'))}")
report("nothing the scorer reaches lives under harness/",
       not any(m.startswith("harness") for m in reach),
       f"closure {sorted(reach)}")

print("\nThe two schema questions the scorer asks")
from omni_extract_bench.scorers.field_matching.documents import is_open_map, unwrap_schema       # noqa: E402
for _br in ("anyOf", "oneOf", "allOf"):
    report(f"{_br} resolves to the non-null branch",
           unwrap_schema({_br: [{"type": "null"}, {"type": "string"}]}) == {"type": "string"},
           f"{unwrap_schema({_br: [{'type': 'null'}, {'type': 'string'}]})}")
report("a plain node is returned unchanged",
       unwrap_schema({"type": "string"}) == {"type": "string"})
report("a non-dict is not a schema", unwrap_schema("nope") == {} and unwrap_schema(None) == {})
report("open maps are detected from explicit additionalProperties only",
       is_open_map({"additionalProperties": True})
       and is_open_map({"additionalProperties": {"type": "string"}})
       and not is_open_map({"type": "object", "properties": {}}),
       "default-true semantics would make every object an open map")
report("...and through a union branch",
       is_open_map({"anyOf": [{"type": "null"}, {"additionalProperties": True}]}))

print("\nThere is exactly one grader")
graders = sorted(p.relative_to(PKG).as_posix() for p in PKG.rglob("*.py")
                 if "grad" in p.stem.lower() or "scor" in p.stem.lower())
scorers = sorted(p.name for p in (PKG / "scorers").iterdir() if p.is_dir() and not p.name.startswith("_"))
report("one scorer package, scorers/field_matching, and no module named for scoring",
       scorers == ["field_matching"] and graders == [], f"scorers {scorers}, modules {graders}")
defines = sorted(p.relative_to(PKG).as_posix() for p in PKG.rglob("*.py")
                 if p != FM / "compare.py" and "\ndef score(" in p.read_text())
report("nothing else defines a score", not defines, f"a second grader in {defines}")
report("no vendored grader tree", not (PKG / "vendor").exists(),
       "omni_extract_bench/vendor/ is back")

print("\nAnd canon_key comes from exactly one place")
import omni_extract_bench as pkg                                        # noqa: E402
from omni_extract_bench.scorers import field_matching as scorer             # noqa: E402
from omni_extract_bench.scorers.field_matching import values                 # noqa: E402

report("the package exports the function, not the module",
       pkg.score is scorer.score and callable(pkg.score))
report("values.canon_key is the one definition: the scorer holds no other",
       callable(values.canon_key) and not hasattr(scorer, "canon_key"))
report("...and so is cmp_leaf", callable(values.cmp_leaf) and not hasattr(scorer, "cmp_leaf"))

print(f"\n{'The scorer is standalone' if not FAILS else 'Failures:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
