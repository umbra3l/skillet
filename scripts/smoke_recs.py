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

# Baselines: the five skills' joint uplift (one regression with all five,
# USD pay, a fixed effect per country -- src/salary_calc.py). Summing the
# separate per-skill coefficients gave 54.3 / 69.8 / 215.2 for the same
# skills; the local-currency model before that gave 49.4 / 38.9 / 188.3.
CASES = [
    ("python+js+react", ["lang_Python", "lang_JavaScript", "web_React"], 24.8),
    (".net dev",        ["lang_C_2", "web_ASP_NET", "plat_Microsoft_Azure"], 32.8),
    ("cold start",      [], 81.7),
]

if __name__ == "__main__":
    eng = SkillRadar()
    failed = False
    for label, skills, expect in CASES:
        rec = eng.recommend(current_skills=skills, top_n=5)
        recs = rec["recommendations"]
        hero = expected_uplift([r["skill"] for r in recs])
        # tolerance: allow 0.5pp.
        ok = abs(hero["total_pct"] - expect) < 0.5
        failed |= not ok
        flag = "OK  " if ok else "DRIFT"
        print(f"[{flag}] {label}: +{hero['total_pct']:.1f}% "
              f"({hero['n_significant']}/{hero['n_skills']} sig)  expected ~{expect}%")
    sys.exit(1 if failed else 0)
