"""Module 3: Simple & Multiple Linear Regression.

  SLR  : log(CompUSD) ~ WorkExp            -- goodness of fit, assumptions
  MLR  : log(CompUSD) ~ WorkExp + Country   -- full model with controls
  Per-skill ROI: for each skill s, fit
         log(CompUSD) ~ WorkExp + Country + skill_s
  and record the coefficient of skill_s as that skill's "salary ROI",
  holding experience and country constant.  This avoids the
  multicollinearity that a kitchen-sink MLR would suffer, and gives a
  single interpretable number per skill -- the M3 application requirement.

  CompUSD is the survey's ConvertedCompYearly (annual pay in USD). Country
  enters as a fixed effect for EVERY country (not just the ten largest),
  so cross-country pay levels never leak into a skill's coefficient.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed.parquet"
OUT_DIR = ROOT / "data"
ASSETS = ROOT / "assets"

SKILL_PREFIXES = ("lang_", "db_", "plat_", "web_", "devenv_", "aimodel_")


def skill_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if any(c.startswith(p) for p in SKILL_PREFIXES) and df[c].nunique() == 2]


def country_effects(df: pd.DataFrame) -> pd.DataFrame:
    """One dummy per country, United States as the reference level."""
    ref = df["Country"].value_counts().index[0]
    cat = pd.Categorical(df["Country"],
                         categories=[ref] + sorted(set(df["Country"].dropna()) - {ref}))
    return pd.get_dummies(cat, prefix="fe", drop_first=True, dtype=float).set_index(df.index)


def fit_slr(df: pd.DataFrame) -> dict:
    sub = df[(df["comp_inlier"] == 1) & df["log_comp"].notna() & df["WorkExp"].notna()]
    X = sm.add_constant(sub[["WorkExp"]])
    y = sub["log_comp"]
    m = sm.OLS(y, X).fit()
    return {
        "n": int(m.nobs), "r2": float(m.rsquared), "adj_r2": float(m.rsquared_adj),
        "f_stat": float(m.fvalue), "f_pvalue": float(m.f_pvalue),
        "intercept": float(m.params["const"]), "work_exp_coef": float(m.params["WorkExp"]),
        "work_exp_p": float(m.pvalues["WorkExp"]),
        "aic": float(m.aic), "bic": float(m.bic),
    }


def fit_full_mlr(df: pd.DataFrame, skills: list[str]) -> dict:
    """log_comp ~ WorkExp + country dummies + ALL skills  (assumptions / VIF demo)."""
    cols = ["WorkExp"] + skills
    sub = df[(df["comp_inlier"] == 1) & df["log_comp"].notna()].dropna(subset=cols)
    fe = country_effects(sub)
    X = sm.add_constant(pd.concat([sub[cols], fe], axis=1))
    y = sub["log_comp"]
    m = sm.OLS(y, X).fit()
    # VIF on a sample of skill columns to keep the table readable
    vif = {}
    sample = ["WorkExp"] + skills[:15] + list(fe.columns[:5])
    sample = [c for c in sample if c in X.columns and c != "const"]
    Xv = sm.add_constant(X[sample].astype(float))
    for i, c in enumerate(sample):
        try:
            vif[c] = float(variance_inflation_factor(Xv.values, i + 1))
        except Exception:
            vif[c] = None
    return {
        "n": int(m.nobs), "r2": float(m.rsquared), "adj_r2": float(m.rsquared_adj),
        "f_pvalue": float(m.f_pvalue), "aic": float(m.aic), "bic": float(m.bic),
        "n_features": int(X.shape[1]),
        "vif_sample": vif,
    }


def per_skill_roi(df: pd.DataFrame) -> pd.DataFrame:
    """For each skill, fit log_comp ~ WorkExp + country + skill, record coef & t-stat."""
    sub = df[(df["comp_inlier"] == 1) & df["log_comp"].notna()].dropna(
        subset=["WorkExp", "Country", "log_comp"])
    fe = country_effects(sub)
    rows = []
    for s in skill_columns(df):
        d = sub.dropna(subset=[s])
        if d.shape[0] < 500 or d[s].sum() < 50:
            continue
        X = sm.add_constant(pd.concat([d[["WorkExp", s]], fe.loc[d.index]], axis=1))
        y = d["log_comp"]
        m = sm.OLS(y, X).fit()
        coef = float(m.params[s])
        p = float(m.pvalues[s])
        # Convert to approximate percent salary impact
        pct = (np.exp(coef) - 1) * 100
        rows.append({
            "skill": s,
            "n": int(m.nobs),
            "users": int(d[s].sum()),
            "coef_log_comp": coef,
            "pct_salary_impact": pct,
            "p_value": p,
            "t_stat": float(m.tvalues[s]),
            "ci_lo": float(np.exp(m.conf_int().loc[s, 0]) - 1) * 100,
            "ci_hi": float(np.exp(m.conf_int().loc[s, 1]) - 1) * 100,
            "r2": float(m.rsquared),
        })
    out = pd.DataFrame(rows).sort_values("coef_log_comp", ascending=False).reset_index(drop=True)
    return out


def run() -> dict:
    df = pd.read_parquet(PROCESSED)
    print(f"loaded {df.shape}")
    skills = skill_columns(df)
    print(f"  {len(skills)} skill columns")

    slr = fit_slr(df)
    print(f"  SLR  R^2 = {slr['r2']:.3f}  WorkExp coef = {slr['work_exp_coef']:.4f} (p={slr['work_exp_p']:.2e})")
    full = fit_full_mlr(df, skills)
    print(f"  Full MLR  R^2 = {full['r2']:.3f}  adj_R^2 = {full['adj_r2']:.3f}  features = {full['n_features']}")

    roi = per_skill_roi(df)
    roi.to_csv(OUT_DIR / "skill_roi.csv", index=False)
    print(f"  per-skill ROI: {roi.shape[0]} skills -> skill_roi.csv  (top 3 below)")
    print(roi.head(3)[["skill", "pct_salary_impact", "p_value", "users"]].to_string(index=False))

    out = {
        "slr_experience_vs_log_comp": slr,
        "full_mlr_skills_country_workexp": full,
        "skill_roi_top10": roi.head(10).to_dict(orient="records"),
        "skill_roi_bottom10": roi.tail(10).to_dict(orient="records"),
    }
    (OUT_DIR / "regression_summary.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    run()
