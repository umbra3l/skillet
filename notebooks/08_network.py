"""Notebook 08 - Skill co-occurrence network (Module 6).

Delegates to src.network.build_network(); renders the spring-layout figure and
exports the node-link JSON + meta for the Skill Radar app.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.network import build_network  # noqa: E402

if __name__ == "__main__":
    meta = build_network()
    print("\nNetwork meta:")
    print(json.dumps(meta, indent=2))
