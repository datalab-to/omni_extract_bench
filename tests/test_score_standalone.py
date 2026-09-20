#!/usr/bin/env python3
"""The scoring path is a closed set of two modules, and `canon_key` has one definition.

This test used to say "`metric.py` must not depend on `grading.py`", so that deleting the old
paired walker would stay a one-line change rather than an archaeology exercise. That worked:
`grading.py` and the vendored upstream grader are both gone, and this file is what
kept the path clear enough to remove them in one go.

What it guards now is the state that made the removal possible, so it cannot quietly erode:

  * scoring reaches exactly `matching` and `values` -- no CLI, no capture, no vendor
    tree, nothing that would drag a transport or a provider into a score;
  * no second grader comes back. Any module whose name says it grades is a second answer to
    "what is the score", and this repo scores through `metric.py` alone;
  * `canon_key` is importable from exactly one place. Two definitions of "are these equal?"
    that must agree eventually will not -- it has happened twice here, and `values.py`'s
    docstring records both.

Run: python3 tests/test_score_standalone.py
"""
import ast
import os as _os
import sys as _sys
from pathlib import Path

_ROOT = Path(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, str(_ROOT))

PKG = _ROOT / "omni_extract_bench"
#: Every module a score is allowed to depend on. Four, each with one subject:
#:   matching    optimal row assignment
#:   values      the comparison rule -- folds, canon_key, states_nothing
#:   recognise   is this text a date, a time, a number
#: The set is the point, not its size: nothing here reaches a transport, a vendor dialect or
#: an envelope convention, so a score cannot move when one of those changes.
SCORING_PATH = {"matching", "values", "recognise"}
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
    MODULE in its aliases rather than in `.module`, and a dotted path like
    `from .a.b import c` is a different file from `c.py` -- so the full path matters, not the
    last segment. Anything that does not resolve to a real file under the package is dropped,
    since it is a name, not a module.
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


def closure(start):
    seen, stack = set(), list(start)
    while stack:
        mod = stack.pop()
        path = PKG / f"{mod.replace('.', '/')}.py"
        if mod in seen or not path.exists():
            continue
        seen.add(mod)
        stack.extend(imports_of(path))
    return seen


print("\nTHE SCORING PATH IS CLOSED")
direct = imports_of(PKG / "metric.py")
reach = closure(direct)
report("metric.py imports nothing outside the scoring path",
       direct <= SCORING_PATH, f"imports {sorted(direct)}")
report("...and the closure is exactly the scoring path, nothing wider",
       reach == SCORING_PATH, f"closure {sorted(reach)}")
note(f"metric.py reaches {sorted(direct)} directly; `recognise` arrives through `values`")
report("values.py, the shared layer, stays below the scorer",
       "score" not in imports_of(PKG / "values.py"),
       f"values imports {sorted(imports_of(PKG / 'values.py'))}")
# The split is only worth having if the pieces stay separable. `recognise` answers a question
# that has nothing to do with the other two, so it may not reach back up.
report("recognise.py depends on nothing in the package (is this text a date / a number)",
       not imports_of(PKG / "recognise.py"),
       f"recognise imports {sorted(imports_of(PKG / 'recognise.py'))}")
report("values.py reaches recognise and nothing else",
       imports_of(PKG / "values.py") == {"recognise"},
       f"values imports {sorted(imports_of(PKG / 'values.py'))}")
# Implied by the closure above, but named so the directory's purpose survives a reader who
# does not derive it from a set equality: a score must not come to depend on a transport, a
# vendor dialect, or an envelope convention.
report("nothing the scorer reaches lives under harness/",
       not any(m.startswith("harness") for m in reach),
       f"closure {sorted(reach)}")

print("\nTHE TWO SCHEMA QUESTIONS THE SCORER ASKS")
# `unwrap_schema` resolves a union to its non-null branch. `allOf` was handled and asserted
# nowhere; `evaluation_config` used to be carried through it, propagating a concept the
# harness strips before a vendor ever sees a schema and that nothing here reads.
from omni_extract_bench.metric import is_open_map, unwrap_schema       # noqa: E402
for _br in ("anyOf", "oneOf", "allOf"):
    report(f"{_br} resolves to the non-null branch",
           unwrap_schema({_br: [{"type": "null"}, {"type": "string"}]}) == {"type": "string"},
           f"{unwrap_schema({_br: [{'type': 'null'}, {'type': 'string'}]})}")
report("a plain node is returned unchanged",
       unwrap_schema({"type": "string"}) == {"type": "string"})
report("a non-dict is not a schema", unwrap_schema("nope") == {} and unwrap_schema(None) == {})
report("open maps are detected from EXPLICIT additionalProperties only",
       is_open_map({"additionalProperties": True})
       and is_open_map({"additionalProperties": {"type": "string"}})
       and not is_open_map({"type": "object", "properties": {}}),
       "default-true semantics would make every object an open map")
report("...and through a union branch",
       is_open_map({"anyOf": [{"type": "null"}, {"additionalProperties": True}]}))

print("\nTHERE IS EXACTLY ONE GRADER")
# The check that matters is that no second module DEFINES a score. The filename check stands in
# front of it because a module named for scoring is how the second one arrives: the runners
# that used to sit here (`run_score.py`, `run_score_modal.py`) resolved a manifest row before
# calling the metric, and a resolver is one edit away from deciding something the metric should
# decide. The metric itself is `metric.py`, named for what it computes rather than for the verb,
# so NOTHING in the package should be named for scoring -- including the metric.
graders = sorted(p.relative_to(PKG).as_posix() for p in PKG.rglob("*.py")
                 if "grad" in p.stem.lower() or "scor" in p.stem.lower())
report("no module is named for scoring; the metric is metric.py", graders == [],
       f"found {graders}")
defines = sorted(p.relative_to(PKG).as_posix() for p in PKG.rglob("*.py")
                 if p.name != "metric.py" and "\ndef score(" in p.read_text())
report("nothing else defines a score", not defines, f"a second grader in {defines}")
report("no vendored grader tree", not (PKG / "vendor").exists(),
       "omni_extract_bench/vendor/ is back")

print("\nAND canon_key COMES FROM EXACTLY ONE PLACE")
import omni_extract_bench as pkg                                        # noqa: E402
# `metric`, not `score`: the package exports a FUNCTION called `score`, so that name no longer
# reaches the module. Which is the point of naming the module for what it holds instead.
from omni_extract_bench import metric, values                           # noqa: E402

report("the package exports the function, not the module",
       pkg.score is metric.score and callable(pkg.score))
report("values.canon_key is the one definition",
       metric.canon_key is values.canon_key and pkg.canon_key is values.canon_key)
report("...and so is cmp_leaf", metric.cmp_leaf is values.cmp_leaf)

print(f"\n{'metric.py IS STANDALONE' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
