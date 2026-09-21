#!/usr/bin/env python3
"""The command line, checked against the handlers behind it and against the README.

Nothing else here drives the CLI: the other tests call `score` directly, which is why `oeb
predict` shipped in 0.1.1 with no `--provider` flag at all while `cmd_predict` read
`args.provider` on its first line. Every invocation raised AttributeError, and the form the
README documented failed one step earlier on an unrecognized argument.

So the first test reads each `cmd_*` handler's source for every `args.X` it touches and
asserts the parser produces it -- a flag a handler needs and the parser does not define is
caught whoever adds it, without anyone having to remember this file exists. The third test
keeps that honest as the CLI grows: a handler with no invocation listed below is a verb
nothing checks, and it fails here rather than in someone's shell.

The second reads the README's own `oeb ...` lines and parses them. A documented command the
parser rejects is a bug in one of the two, and which one does not matter to whoever types it.

Nothing is executed: the handlers are swapped for a spy, so this needs no network, no files
and no extras.

Run: python3 tests/test_cli_surface.py
"""
import ast
import contextlib
import inspect
import io
import re
import shlex
import sys
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from omni_extract_bench import cli  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HANDLERS = tuple(sorted(n for n in dir(cli) if n.startswith("cmd_")))
VERBS = ("score", "benchmark", "providers", "predict")
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILS.append(name)


def attrs_read(fn):
    """Every `args.X` the handler reads, from its source."""
    tree = ast.parse(inspect.getsource(fn).lstrip())
    arg = tree.body[0].args.args[0].arg
    return {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == arg}


def parse(argv):
    """The namespace the real parser builds, with dispatch replaced by a spy.

    `main` builds the parser, parses, then calls `args.fn`, which `set_defaults` bound by
    name at parse time -- so swapping the module attribute first is enough to stop before
    any work. A rejected line exits, and that is the failure this test exists to see.
    """
    captured = {}
    saved = {name: getattr(cli, name) for name in HANDLERS}
    buf = io.StringIO()
    try:
        for name in HANDLERS:
            setattr(cli, name, lambda args: (captured.setdefault("args", args), 0)[1])
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(io.StringIO()):
            cli.main(argv)
    except SystemExit:
        raise AssertionError(f"parser rejected `oeb {' '.join(argv)}`: "
                             f"{buf.getvalue().strip().splitlines()[-1:]}")
    finally:
        for name, fn in saved.items():
            setattr(cli, name, fn)
    return captured.get("args")


INVOCATIONS = {
    "score": (["score", "--gt", "g.json", "--pred", "p.json",
               "--schema", "s.json"], "cmd_score"),
    "benchmark": (["benchmark", "--providers", "datalab"], "cmd_benchmark"),
    "providers": (["providers"], "cmd_providers"),
    "predict": (["predict", "--provider", "datalab", "--doc", "d.pdf",
                 "--schema", "s.json"], "cmd_predict"),
}

for verb, (argv, handler_name) in INVOCATIONS.items():
    readers = [getattr(cli, handler_name)]
    try:
        ns = parse(argv)
    except AssertionError as exc:
        check(f"`oeb {verb}` parses", False, str(exc))
        continue
    check(f"`oeb {verb}` parses", ns is not None, "the handler was never reached")
    if ns is None:
        continue
    missing = sorted({a for fn in readers for a in attrs_read(fn) if not hasattr(ns, a)})
    check(f"`oeb {verb}` supplies every flag it reads", not missing,
          f"reads args.{', args.'.join(missing)}; the parser defines none of it")


def _incomplete(buf: str) -> bool:
    try:
        shlex.split(buf)
        return False
    except ValueError:
        return True


_joined, _buf = [], ""
for _doc in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
    for _ln in _doc.read_text().splitlines():
        _ln = _ln.strip()
        if _buf or re.match(rf"^oeb\s+({'|'.join(VERBS)})\b", _ln):
            _buf = (_buf + " " if _buf else "") + _ln.rstrip("\\")
            if not _ln.endswith("\\") and not _incomplete(_buf):
                _cmd = re.split(r"\s(?:\||>>?|&&|;)\s", _buf)[0]
                _joined.append((_doc.name, " ".join(_cmd.split())))
                _buf = ""
    _buf = ""
lines = [(doc, ln) for doc, ln in _joined if "..." not in ln]
check("the docs show commands to check", len(lines) >= 1, f"found {len(lines)}")
for doc, line in lines:
    try:
        parse(shlex.split(line)[1:])
        check(f"{doc}: {line}", True)
    except AssertionError as exc:
        check(f"{doc}: {line}", False, str(exc))


unchecked = sorted(set(HANDLERS) - {h for _, h in INVOCATIONS.values()})
check("every cmd_* handler has an invocation in this file", not unchecked,
      f"no line checks {', '.join(unchecked)}")

import logging                                                          # noqa: E402

parse(["score", "--gt", "g.json", "--pred", "p.json", "--schema", "s.json"])
for lib in ("httpx", "openai._base_client", "urllib3", "a_library_added_next_year"):
    check(f"{lib} does not narrate at INFO",
          logging.getLogger(lib).getEffectiveLevel() > logging.INFO)
check("...but a third-party WARNING still gets through",
      logging.getLogger("httpx").getEffectiveLevel() <= logging.WARNING)
check("our own logger reports progress",
      logging.getLogger("omni_extract_bench.benchmark").getEffectiveLevel() <= logging.INFO)

print("\nA BENCHMARK ASKS BEFORE IT SPENDS")
import builtins                                                          # noqa: E402


class _Stdin:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


def _asked(answer, tty=True):
    """What `confirm_plan` decides, given a terminal that answers `answer`."""
    saved_in, saved_input = cli.sys.stdin, builtins.input
    shown = io.StringIO()
    cli.sys.stdin = _Stdin(tty)
    builtins.input = ((lambda _="": (_ for _ in ()).throw(EOFError))
                      if answer is EOFError else (lambda _="": answer))
    try:
        with contextlib.redirect_stderr(shown):
            return cli.confirm_plan("benchmark: 1 run"), shown.getvalue()
    finally:
        cli.sys.stdin, builtins.input = saved_in, saved_input


for answer, want in (("y", True), ("yes", True), ("Y", True),
                     ("n", False), ("", False), ("nonsense", False)):
    got, _ = _asked(answer)
    check(f"a terminal answering {answer!r} -> {want}", got is want, str(got))
check("the plan is shown before the question", "benchmark: 1 run" in _asked("y")[1])
check("...on stderr, so stdout stays the summary", _asked("y")[1].strip() == "benchmark: 1 run")
check("no terminal proceeds without asking", _asked("n", tty=False)[0] is True)
check("...and a closed terminal does not", _asked(EOFError)[0] is False)

print(f"\n{'ALL CLI TESTS PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
