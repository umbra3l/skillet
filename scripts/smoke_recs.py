"""End-to-end recommender smoke: the three canonical profiles.

The numbers here are data-derived and must not drift across UI edits.
Run:  .venv/bin/python scripts/smoke_recs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recommend import SkillRadar  # noqa: E402
from src.salary_calc import expected_uplift  # noqa: E402

# Baselines from the USD-pay model (ConvertedCompYearly + a fixed effect
# per country). The earlier local-currency model gave 49.4 / 38.9 / 188.3.
CASES = [
    ("python+js+react", ["lang_Python", "lang_JavaScript", "web_React"], 54.3),
    (".net dev",        ["lang_C_2", "web_ASP_NET", "plat_Microsoft_Azure"], 69.8),
    ("cold start",      [], 215.2),
]

if __name__ == "__main__":
    eng = SkillRadar()
    failed = False
    for label, skills, expect in CASES:
        rec = eng.recommend(current_skills=skills, top_n=5)
        recs = rec["recommendations"]
        hero = expected_uplift([r["skill"] for r in recs])
        # tolerance: the brief quotes ~+38.9 for the .NET stack; the exact
        # value depends on the precise stack picked, so allow 0.5pp.
        ok = abs(hero["total_pct"] - expect) < 0.5
        failed |= not ok
        flag = "OK  " if ok else "DRIFT"
        print(f"[{flag}] {label}: +{hero['total_pct']:.1f}% "
              f"({hero['n_significant']}/{hero['n_skills']} sig)  expected ~{expect}%")
    sys.exit(1 if failed else 0)
