"""Salary calculator: 'What if I learn this?' estimator.

Uses the per-skill MLR coefficients in data/skill_roi.csv. For a set of
candidate skills, the predicted log-comp uplift is the sum of their
regression coefficients (controlling for work experience and country).
We report the total as a percent and break it down per skill.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROI_CSV = ROOT / "data" / "skill_roi.csv"


def expected_uplift(candidate_skills: Iterable[str], min_p_value: float = 1.0) -> dict:
    """Sum per-skill MLR coefficients for the candidates.

    Parameters
    ----------
    candidate_skills : iterable of column names (e.g. 'lang_Python')
    min_p_value      : if < 1.0, exclude skills whose ROI p-value is above
                       this cutoff from the total (still shown individually).

    Returns
    -------
    dict with total_pct, n_skills, n_significant, and a per-skill breakdown
    (sorted by impact, descending).
    """
    roi = pd.read_csv(ROI_CSV).set_index("skill")
    rows = []
    total_log = 0.0
    n_sig = 0
    for s in candidate_skills:
        if s not in roi.index:
            continue
        r = roi.loc[s]
        rows.append({
            "skill": s,
            "pct_impact": float(r["pct_salary_impact"]),
            "p_value": float(r["p_value"]),
            "ci_lo": float(r["ci_lo"]),
            "ci_hi": float(r["ci_hi"]),
            "n_users": int(r["users"]),
            "significant": bool(r["p_value"] < 0.05),
        })
        if r["p_value"] <= min_p_value:
            total_log += float(r["coef_log_comp"])
            if r["p_value"] < 0.05:
                n_sig += 1
    rows.sort(key=lambda r: r["pct_impact"], reverse=True)
    total_pct = (float(np.exp(total_log)) - 1.0) * 100.0
    return {
        "total_pct": round(total_pct, 2),
        "n_skills": len(rows),
        "n_significant": n_sig,
        "breakdown": rows,
    }


if __name__ == "__main__":
    import json, sys
    skills = sys.argv[1:] or ["lang_Python", "plat_Kubernetes", "plat_Terraform"]
    out = expected_uplift(skills)
    print(json.dumps(out, indent=2))
