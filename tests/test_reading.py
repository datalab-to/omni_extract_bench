"""The line between a system's failure and our bug: an answer is read inside `omni_bench_core.adapter.reading`.

Inside `reading`, an answer that can't be read fails one document and the run goes on. Anywhere else in an
adapter's call, an exception is a bug, and stops the run. So every place that reads what a system sent back
while calling it (a job id, a status, a cost) must sit inside `reading`.

The guard walks each provider module from the functions that call a system (`extract` and `call`) through
the module's own helpers they call. In those it fails on:

- a `.json()` or `json.loads(...)` call outside a `reading` block; and
- a subscript of a name bound from an answer (`.json()`, `json.loads`, or a helper that returns one, such
  as llamaextract's `_req`) outside a `reading` block.

It is a guard, not a type checker: reading the extraction itself is `read`'s job, where the core already
makes any failure the document's.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

pytest.importorskip("httpx")
pytest.importorskip("openai")

from omni_bench_core import adapter as core  # noqa: E402

from omni_extract_bench import harness  # noqa: E402
from omni_extract_bench.harness import contract  # noqa: E402

PROVIDERS = Path(harness.__file__).parent / "providers"
MODULES = sorted(p for p in PROVIDERS.glob("*.py") if p.name != "__init__.py")
ROOTS = {"extract", "call"}                               # the methods that call a system
ANSWERS = {"json", "loads", "_req", "_poll", "poll", "_await"}   # calls whose result is a system's answer


def reached(tree: ast.Module) -> list[ast.FunctionDef]:
    """The module's functions reached from `ROOTS` through calls to the module's own functions."""
    defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    seen, todo = set(), [n for n in ROOTS if n in defs]
    while todo:
        if (name := todo.pop()) in seen: continue
        seen.add(name)
        todo += [c.func.id for c in ast.walk(defs[name]) if isinstance(c, ast.Call)
                 and isinstance(c.func, ast.Name) and c.func.id in defs and c.func.id != "read"]
    return [defs[n] for n in seen]


def is_reading(node: ast.With) -> bool:
    calls = [i.context_expr for i in node.items if isinstance(i.context_expr, ast.Call)]
    return any(getattr(c.func, "attr", None) == "reading" for c in calls)


def answer_call(node: ast.AST) -> bool:
    return any(isinstance(c, ast.Call) and getattr(c.func, "attr", getattr(c.func, "id", None)) in ANSWERS
               for c in ast.walk(node))


def decodes(node: ast.Call) -> bool:
    """Whether `node` decodes a system's bytes: `.json()`, or `json.loads(...)`."""
    f = node.func
    return isinstance(f, ast.Attribute) and (f.attr == "json" or (
        f.attr == "loads" and isinstance(f.value, ast.Name) and f.value.id == "json"))


def outside(fn: ast.FunctionDef) -> list[str]:
    """Each read of an answer in `fn` that isn't inside a `reading` block, as `line: what`."""
    bound = {t.id for n in ast.walk(fn) for t in (n.targets if isinstance(n, ast.Assign) else
             [n.target] if isinstance(n, ast.NamedExpr) else []) if isinstance(t, ast.Name)
             and answer_call(n.value)}
    out = []

    def visit(node: ast.AST, inside: bool) -> None:
        inside = inside or (isinstance(node, ast.With) and is_reading(node))
        if not inside:
            if isinstance(node, ast.Call) and decodes(node):
                out.append(f"{node.lineno}: {ast.unparse(node.func)}()")
            if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name) and node.value.id in bound):
                out.append(f"{node.lineno}: {node.value.id}[...]")
        for child in ast.iter_child_nodes(node):
            visit(child, inside)

    visit(fn, False)
    return out


@pytest.mark.parametrize("path", MODULES, ids=[p.stem for p in MODULES])
def test_every_answer_is_read_inside_reading(path):
    tree = ast.parse(path.read_text())
    found = [f"{fn.name} {x}" for fn in reached(tree) for x in outside(fn)]
    assert not found, f"{path.name} reads a system's answer outside `adapter.reading`: {found}"


@dataclasses.dataclass(frozen=True)
class Answers(contract.Extractor):
    """A system whose answer lacks its job id. `careful` reads the answer inside `reading`, as an adapter
    must."""

    retries = 0
    careful: bool = True

    def extract(self, pdf, schema, *, timeout):
        answer = {"wrong": "shape"}
        if self.careful:
            with core.reading("the job's id"):
                return core.Call({"extraction": {"id": answer["job_id"]}})
        return core.Call({"extraction": {"id": answer["job_id"]}})


def test_a_malformed_answer_inside_reading_fails_the_document(tmp_path):
    (pdf := tmp_path / "d.pdf").write_bytes(b"%PDF")
    rec = core.predict(Answers(), contract.Question(pdf, {"type": "object"}))
    assert rec["output"] is None and "couldn't be read (the job's id)" in rec["error"]["message"], rec


def test_a_bug_outside_reading_stops_the_run(tmp_path):
    (pdf := tmp_path / "d.pdf").write_bytes(b"%PDF")
    with pytest.raises(KeyError):
        core.predict(Answers(careful=False), contract.Question(pdf, {"type": "object"}))
