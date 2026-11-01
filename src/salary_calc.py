"""Salary calculator: 'What if I learn this?' estimator.

For a set of candidate skills, fit ONE regression with all of them together:

    log_comp ~ WorkExp + country fixed effects + skill_1 + ... + skill_k

The modelled uplift is the sum of the k joint coefficients, and its 95%
interval comes from the coefficient covariance (Var = 1' Sigma 1), so the
interval is a real interval for the total. Fitting the skills jointly also
stops overlapping skills (Docker and Kubernetes, say) from each claiming the
same pay premium, which summing the separate per-skill coefficients in
data/skill_roi.csv would do. The per-skill rows report the joint
coefficients, plus each skill's stand-alone ROI for reference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
ROI_CSV = ROOT / "data" / "skill_roi.csv"
PROCESSED = ROOT / "data" / "processed.parquet"

_sample: tuple[pd.DataFrame, pd.DataFrame] | None = None


def _pct(log_pts: float) -> float:
    return (float(np.exp(log_pts)) - 1.0) * 100.0


def _model_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The salary-model rows and their WorkExp + country design (cached)."""
    global _sample
    if _sample is None:
        from src.regression import country_effects
        df = pd.read_parquet(PROCESSED)
        sub = df[(df["comp_inlier"] == 1) & df["log_comp"].notna()].dropna(
            subset=["WorkExp", "Country"])
        base = pd.concat([sub[["WorkExp"]].astype(float), country_effects(sub)], axis=1)
        _sample = (sub, base)
    return _sample


def expected_uplift(candidate_skills: Iterable[str]) -> dict:
    """Joint modelled uplift of learning all `candidate_skills`.

    Returns
    -------
    dict with total_pct, ci_lo_pct, ci_hi_pct, p_value (of the total),
    n_skills, n_significant (joint p < 0.05) and a per-skill breakdown
    sorted by joint impact, descending.
    """
    sub, base = _model_sample()
    skills = [s for s in dict.fromkeys(candidate_skills)
              if s in sub.columns and sub[s].nunique() == 2]
    if not skills:
        return {"total_pct": 0.0, "ci_lo_pct": 0.0, "ci_hi_pct": 0.0,
                "p_value": float("nan"), "n_skills": 0, "n_significant": 0,
                "breakdown": []}

    X = sm.add_constant(pd.concat([base, sub[skills].astype(float)], axis=1))
    m = sm.OLS(sub["log_comp"], X).fit()

    # Total effect = sum of the joint coefficients, tested as one contrast.
    contrast = np.zeros(len(m.params))
    contrast[[m.params.index.get_loc(s) for s in skills]] = 1.0
    tt = m.t_test(contrast)
    total = float(np.squeeze(tt.effect))
    lo, hi = (float(x) for x in np.squeeze(tt.conf_int()))

    roi = pd.read_csv(ROI_CSV).set_index("skill")
    ci = m.conf_int()
    rows = []
    for s in skills:
        p = float(m.pvalues[s])
        rows.append({
            "skill": s,
            "pct_impact": _pct(m.params[s]),
            "p_value": p,
            "ci_lo": _pct(ci.loc[s, 0]),
            "ci_hi": _pct(ci.loc[s, 1]),
            "n_users": int(sub[s].sum()),
            "significant": p < 0.05,
            "pct_impact_alone": (float(roi.loc[s, "pct_salary_impact"])
                                 if s in roi.index else float("nan")),
        })
    rows.sort(key=lambda r: r["pct_impact"], reverse=True)
    return {
        "total_pct": round(_pct(total), 2),
        "ci_lo_pct": round(_pct(lo), 2),
        "ci_hi_pct": round(_pct(hi), 2),
        "p_value": float(np.squeeze(tt.pvalue)),
        "n_skills": len(rows),
        "n_significant": sum(r["significant"] for r in rows),
        "breakdown": rows,
    }


if __name__ == "__main__":
    import json, sys
    skills = sys.argv[1:] or ["lang_Python", "plat_Kubernetes", "plat_Terraform"]
    out = expected_uplift(skills)
    print(json.dumps(out, indent=2))
