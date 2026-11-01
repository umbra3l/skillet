"""Module 2: hypothesis tests on the processed survey.

Two-sample t-tests (AI users vs non-users) and chi-square tests of independence
(AI usage vs remote-work, top skills vs developer type) with Cramér's V effect size.

The raw salary t-test is confounded by country (AI use is more common in
lower-paying countries), so it is paired with an OLS test of the same
difference holding work experience and country fixed:
    log_comp ~ ai_user + WorkExp + country fixed effects
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.regression import country_effects

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed.parquet"
OUT_JSON = ROOT / "data" / "stats_tests.json"
ASSETS = ROOT / "assets"


def cramers_v(confusion: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(confusion).statistic
    n = confusion.to_numpy().sum()
    if n == 0:
        return float("nan")
    r, k = confusion.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if min(r, k) > 1 else 0.0


def _ai_pay_gap_controlled(df: pd.DataFrame) -> dict:
    """AI users' log-comp gap holding work experience and country fixed.

    Same sample as the salary models (USD pay, IQR inliers).
    """
    sub = df[(df["comp_inlier"] == 1) & df["log_comp"].notna()].dropna(
        subset=["WorkExp", "Country"])
    X = sm.add_constant(pd.concat(
        [sub[["ai_user", "WorkExp"]].astype(float), country_effects(sub)], axis=1))
    m = sm.OLS(sub["log_comp"], X).fit()
    coef, p = float(m.params["ai_user"]), float(m.pvalues["ai_user"])
    lo, hi = (float(x) for x in m.conf_int().loc["ai_user"])
    return {
        "model": "log_comp ~ ai_user + WorkExp + country fixed effects (OLS)",
        "n": int(m.nobs),
        "coef_log_comp": coef,
        "pct_gap": (np.exp(coef) - 1) * 100,
        "ci_pct": [(np.exp(lo) - 1) * 100, (np.exp(hi) - 1) * 100],
        "t": float(m.tvalues["ai_user"]),
        "p_value": p,
        "conclusion": ("Reject H0; AI users earn more at the same experience and country"
                       if p < 0.05 and coef > 0 else
                       "Reject H0; AI users earn less at the same experience and country"
                       if p < 0.05 else "Fail to reject H0"),
    }


def run() -> dict:
    ASSETS.mkdir(exist_ok=True)
    df = pd.read_parquet(PROCESSED)
    print(f"loaded {df.shape}")

    results: dict = {}

    # --- Two-sample t-tests (Module 2: hypothesis testing, p-value) ---
    users = df.loc[df["ai_user"] == 1, "log_comp"].dropna()
    nonusers = df.loc[df["ai_user"] == 0, "log_comp"].dropna()
    t_sal, p_sal = stats.ttest_ind(users, nonusers, equal_var=False, nan_policy="omit")

    users_sat = df.loc[df["ai_user"] == 1, "JobSat_num"].dropna()
    nonusers_sat = df.loc[df["ai_user"] == 0, "JobSat_num"].dropna()
    t_sat, p_sat = stats.ttest_ind(users_sat, nonusers_sat, equal_var=False, nan_policy="omit")

    # Welch's t (unequal variance) -- the modern default
    results["t_tests"] = {
        "log_comp_ai_users_vs_non": {
            "n_users": int(users.shape[0]), "n_non": int(nonusers.shape[0]),
            "mean_log_comp_users": float(users.mean()), "mean_log_comp_non": float(nonusers.mean()),
            "t": float(t_sal), "p_value": float(p_sal),
            "conclusion": "Reject H0; AI users' log compensation differs" if p_sal < 0.05
                           else "Fail to reject H0",
        },
        "log_comp_ai_users_vs_non_controlled": _ai_pay_gap_controlled(df),
        "JobSat_ai_users_vs_non": {
            "n_users": int(users_sat.shape[0]), "n_non": int(nonusers_sat.shape[0]),
            "mean_sat_users": float(users_sat.mean()), "mean_sat_non": float(nonusers_sat.mean()),
            "t": float(t_sat), "p_value": float(p_sat),
            "conclusion": "Reject H0; AI users' job satisfaction differs" if p_sat < 0.05
                           else "Fail to reject H0",
        },
    }

    # --- Chi-square tests of independence (Module 2: relationship among variables) ---
    # AI usage vs RemoteWork
    rw = df["RemoteWork"].fillna("(missing)")
    ct1 = pd.crosstab(df["ai_user"], rw)
    chi2_1, p1, dof1, _ = stats.chi2_contingency(ct1)
    results["chi_square"] = {
        "ai_user_vs_remote": {
            "contingency": ct1.to_dict(),
            "chi2": float(chi2_1), "dof": int(dof1), "p_value": float(p1),
            "cramers_v": cramers_v(ct1),
            "conclusion": "Reject H0; AI usage and remote work are associated" if p1 < 0.05
                           else "Fail to reject H0",
        }
    }

    # Top role group vs the most common language (use argmax over role one-hots
    # to recover each respondent's *primary* role label).
    role_cols = [c for c in df.columns if c.startswith("role_")]
    lang_cols = [c for c in df.columns if c.startswith("lang_")]
    if role_cols and lang_cols:
        role_arr = np.array(role_cols)
        primary = pd.Series(role_arr[np.argmax(df[role_cols].values, axis=1)])
        # Restrict to roles with enough respondents so the chi-square is meaningful
        keep_roles = primary.value_counts()
        keep_roles = keep_roles[keep_roles > 200].index
        primary = primary.where(primary.isin(keep_roles), other="(other)")
        top_lang = df[lang_cols].sum().sort_values(ascending=False).index[0]
        ct2 = pd.crosstab(primary, df[top_lang])
        ct2 = ct2.loc[ct2.sum(axis=1) > 100]
        if ct2.shape[0] > 1 and ct2.shape[1] > 1:
            chi2_2, p2, dof2, _ = stats.chi2_contingency(ct2)
            results["chi_square"][f"role_vs_{top_lang}"] = {
                "contingency": ct2.to_dict(),
                "chi2": float(chi2_2), "dof": int(dof2), "p_value": float(p2),
                "cramers_v": cramers_v(ct2),
                "conclusion": "Reject H0" if p2 < 0.05 else "Fail to reject H0",
            }

    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"saved -> {OUT_JSON}")
    return results


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str)[:2000])
