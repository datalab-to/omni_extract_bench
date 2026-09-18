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

# run from anywhere: `python tests/x.py` puts tests/ on the path, not the repo root
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from omni_extract_bench import cli  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
#: Read off the module rather than listed, so a new verb is in scope here the moment it exists.
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


# A representative invocation of each verb: every required flag and nothing optional, so
# what the namespace carries is what the parser defaults rather than what the line supplied.
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


# ── every `oeb ...` line in the README parses ───────────────────────────────────
# A command in prose wraps two ways, and both have to be followed or an example reads as a
# truncated command and fails for the wrong reason: a trailing `\`, and an argument whose
# quote is still open -- which is how the `--options` examples span lines, their JSON being
# easier to read laid out than on one line.
def _incomplete(buf: str) -> bool:
    try:
        shlex.split(buf)
        return False
    except ValueError:                 # "No closing quotation"
        return True


# EVERY markdown file that documents the CLI, not only the README: `docs/API.md` spells the
# flags out one by one, which is exactly the kind of list that goes quietly out of date.
_joined, _buf = [], ""
for _doc in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
    for _ln in _doc.read_text().splitlines():
        _ln = _ln.strip()
        if _buf or re.match(rf"^oeb\s+({'|'.join(VERBS)})\b", _ln):
            _buf = (_buf + " " if _buf else "") + _ln.rstrip("\\")
            if not _ln.endswith("\\") and not _incomplete(_buf):
                # Cut shell plumbing: a doc may pipe or redirect a command, and `> pred.json`
                # is the shell's business, not argparse's.
                _cmd = re.split(r"\s(?:\||>>?|&&|;)\s", _buf)[0]
                _joined.append((_doc.name, " ".join(_cmd.split())))
                _buf = ""
    _buf = ""                       # a command cannot run on past the end of its own file
# A `...` is a SHAPE, not an invocation -- `oeb providers datalab` prints one to show where
# the options go, and `{"mode": ...}` is not JSON anyone should be able to run.
lines = [(doc, ln) for doc, ln in _joined if "..." not in ln]
check("the docs show commands to check", len(lines) >= 1, f"found {len(lines)}")
for doc, line in lines:
    try:
        # `shlex`, not `split()`: `--options '{"datalab": {"mode": "accurate"}}'` is ONE
        # argument, and split on whitespace it arrives as five that argparse cannot place.
        parse(shlex.split(line)[1:])
        check(f"{doc}: {line}", True)
    except AssertionError as exc:
        check(f"{doc}: {line}", False, str(exc))


# ── no verb goes unchecked ──────────────────────────────────────────────────────
# The point of this file is that nothing else drives the CLI. A handler added without an
# invocation above would be exactly as unchecked as `cmd_predict` was when it shipped
# reading a flag that did not exist.
unchecked = sorted(set(HANDLERS) - {h for _, h in INVOCATIONS.values()})
check("every cmd_* handler has an invocation in this file", not unchecked,
      f"no line checks {', '.join(unchecked)}")

# ── the libraries under us do not narrate ───────────────────────────────────────
# The root logger stays at WARNING and only our own namespace is turned up. `basicConfig` on
# root would switch on INFO for every library in the process, and then each has to be muted by
# name -- httpx logs a line per request, and `openai` logs its own copy of that line through
# `openai._base_client`, so silencing httpx was not enough. This asserts the property that
# replaces the list: a library nobody has heard of is quiet WITHOUT being named.
import logging                                                          # noqa: E402

parse(["score", "--gt", "g.json", "--pred", "p.json", "--schema", "s.json"])   # configures logging
for lib in ("httpx", "openai._base_client", "urllib3", "a_library_added_next_year"):
    check(f"{lib} does not narrate at INFO",
          logging.getLogger(lib).getEffectiveLevel() > logging.INFO)
check("...but a third-party WARNING still gets through",
      logging.getLogger("httpx").getEffectiveLevel() <= logging.WARNING)
check("our own logger reports progress",
      logging.getLogger("omni_extract_bench.benchmark").getEffectiveLevel() <= logging.INFO)

print(f"\n{'ALL CLI TESTS PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
