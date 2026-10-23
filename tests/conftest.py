"""Collect each script-style test file as one pytest test.

Most files here are scripts: they print a PASS/FAIL line per check and exit nonzero if any
failed. Until a file is migrated to `def test_...` functions, pytest runs it in a subprocess
and the whole file is one test; a failure shows the script's output. A file that defines a
`test_` function is collected by pytest as usual, so files can migrate one at a time.

    pytest                  # every file
    pytest -n auto          # in parallel (pytest-xdist)
    pytest -k order         # files whose name matches
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_TEST_FUNCTION = re.compile(r"^def test_", re.MULTILINE)


def pytest_pycollect_makemodule(module_path: Path, parent: pytest.Collector) -> pytest.File | None:
    # Note: importing a script would run it, and its `sys.exit` would abort collection, so a
    # script must never reach pytest's own module collector.
    if _TEST_FUNCTION.search(module_path.read_text()):
        return None
    return Script.from_parent(parent, path=module_path)


class ScriptFailed(Exception):
    pass


class Script(pytest.File):
    def collect(self):
        yield ScriptRun.from_parent(self, name=self.path.stem)


class ScriptRun(pytest.Item):
    def runtest(self) -> None:
        # Note: added to, not replaced, so a local omni-bench-core put on PYTHONPATH reaches the script too.
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
        out = subprocess.run([sys.executable, str(self.path)], cwd=ROOT, env=env,
                             capture_output=True, text=True)
        if out.returncode:
            raise ScriptFailed(f"{self.path.name} exited {out.returncode}\n{out.stdout}{out.stderr}")

    def repr_failure(self, excinfo, style=None):
        if isinstance(excinfo.value, ScriptFailed):
            return str(excinfo.value)
        return super().repr_failure(excinfo, style)

    def reportinfo(self):
        return self.path, None, self.path.name
