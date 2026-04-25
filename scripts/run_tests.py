#!/usr/bin/env python3
"""Run project tests.

Tries to use pytest first; if pytest is not available, falls back to a tiny test runner
that imports `tests.test_*.py` modules and executes functions named `test_*`.

Usage:
    python scripts/run_tests.py
"""
from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS_DIR = ROOT / "tests"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))


def run_pytest() -> bool:
    try:
        import pytest  # type: ignore
    except Exception:
        return False
    print("Running tests with pytest...")
    return subprocess.call([sys.executable, "-m", "pytest", "-q"]) == 0


def fallback_runner() -> bool:
    print("pytest not available; running fallback test runner...")
    failures = []

    for path in sorted(TESTS_DIR.glob("test_*.py")):
        mod_name = f"tests.{path.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            failures.append((mod_name, "IMPORT_ERROR", traceback.format_exc()))
            continue
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                print(f"OK: {mod_name}.{name}")
            except AssertionError:
                failures.append((f"{mod_name}.{name}", "ASSERTION", traceback.format_exc()))
            except Exception:
                failures.append((f"{mod_name}.{name}", "ERROR", traceback.format_exc()))

    if failures:
        print(f"\n{len(failures)} test(s) failed:\n")
        for what, kind, tb in failures:
            print(f"--- {what} ({kind}) ---")
            print(tb)
        return False

    print("\nAll tests passed (fallback runner).")
    return True


if __name__ == "__main__":
    ok = run_pytest()
    if not ok:
        ok = fallback_runner()
    sys.exit(0 if ok else 1)
