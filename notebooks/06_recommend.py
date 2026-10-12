"""Notebook 06 - Recommender demo + persona walkthrough."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recommend import SkillRadar  # noqa: E402

if __name__ == "__main__":
    sr = SkillRadar()

    # A couple of demo personas from the cluster file
    meta = json.loads((ROOT / "data" / "cluster_meta.json").read_text())
    print("Personas discovered:")
    for p in meta["personas"]:
        print(f"  #{p['cluster']} (n={p['n']:,}): "
              + ", ".join(s["skill"] for s in p["top_skills"][:6]))

    # Pretend a user already knows the top 4 skills of persona #0 minus a few
    p0 = meta["personas"][0]
    known = [s["skill"] for s in p0["top_skills"][:4]]
    print(f"\nDemo user knows: {known}")
    out = sr.recommend(current_skills=known, top_n=5)
    print(json.dumps(out, indent=2, default=str))
