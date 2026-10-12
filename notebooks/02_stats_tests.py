"""Notebook 02 - Hypothesis tests (Module 2).

Delegates to src.stats_tests.run(); saves a JSON summary and prints it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stats_tests import run  # noqa: E402

if __name__ == "__main__":
    out = run()
    print("\n=== SUMMARY ===")
    print(json.dumps(out, indent=2, default=str)[:3500])
