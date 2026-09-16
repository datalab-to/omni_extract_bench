#!/usr/bin/env python3
"""The command line, checked against the handlers behind it and against the README.

Nothing else here drives the CLI: the other tests call `run_score.run` and `grade` directly,
which is why `oeb predict` shipped in 0.1.1 with no `--provider` flag at all while
`cmd_predict` read `args.provider` on its first line. Every invocation raised AttributeError,
and the form the README documents failed one step earlier on an unrecognized argument.

So the first test reads each `cmd_*` handler's source for every `args.X` it touches and
asserts the parser produces it -- a flag a handler needs and the parser does not define is
caught whoever adds it, without anyone having to remember this file exists.

The second reads the README's own `oeb ...` lines and parses them. A documented command the
parser rejects is a bug in one of the two, and which one does not matter to whoever types it.

Nothing is executed: the handlers are swapped for a spy, so this needs no manifest, no
network and no extras.

Run: python3 tests/test_cli_surface.py
"""
import ast
import contextlib
import inspect
import io
import re
import sys
from pathlib import Path

# run from anywhere: `python tests/x.py` puts tests/ on the path, not the repo root
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from omni_extract_bench import cli  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HANDLERS = ("cmd_score", "cmd_predict", "cmd_score_one")
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
    "score": (["score", "--manifest", "m.parquet", "--out", "run/"], cli.cmd_score),
    "predict": (["predict", "--manifest", "m.parquet", "--out", "p/",
                 "--provider", "datalab"], cli.cmd_predict),
    "score-one": (["score-one", "--gt", "g.json", "--pred", "p.json",
                   "--schema", "s.json"], cli.cmd_score_one),
}

for verb, (argv, handler) in INVOCATIONS.items():
    try:
        ns = parse(argv)
    except AssertionError as exc:
        check(f"`oeb {verb}` parses", False, str(exc))
        continue
    check(f"`oeb {verb}` parses", ns is not None, "the handler was never reached")
    if ns is None:
        continue
    missing = sorted(a for a in attrs_read(handler) if not hasattr(ns, a))
    check(f"`oeb {verb}` supplies every flag its handler reads", not missing,
          f"handler reads args.{', args.'.join(missing)}; the parser defines none of it")


# ── every `oeb ...` line in the README parses ───────────────────────────────────
lines = [ln.strip() for ln in (ROOT / "README.md").read_text().splitlines()
         if re.match(r"^oeb\s+(score|predict|score-one)\b", ln.strip())]
check("the README shows commands to check", len(lines) >= 3, f"found {len(lines)}")
for line in lines:
    try:
        parse(line.split()[1:])
        check(f"README: {line}", True)
    except AssertionError as exc:
        check(f"README: {line}", False, str(exc))

print(f"\n{'ALL CLI TESTS PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
