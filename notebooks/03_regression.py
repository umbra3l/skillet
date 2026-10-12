"""Notebook 03 - Regression (Module 3).

Delegates to src.regression.run(); plots the top-15 skill ROI.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.regression import per_skill_roi, fit_slr, fit_full_mlr  # noqa: E402
from src.preprocess import skill_columns  # noqa: E402

DATA = ROOT / "data"
ASSETS = ROOT / "assets"


def main() -> None:
    df = pd.read_parquet(DATA / "processed.parquet")
    slr = fit_slr(df)
    full = fit_full_mlr(df, skill_columns(df))
    roi = per_skill_roi(df)
    roi.to_csv(DATA / "skill_roi.csv", index=False)

    print("SLR (log_comp ~ WorkExp):")
    print(f"  R^2 = {slr['r2']:.3f}  coef = {slr['work_exp_coef']:+.4f}  p = {slr['work_exp_p']:.2e}")
    print("Full MLR (log_comp ~ WorkExp + country + skills):")
    print(f"  R^2 = {full['r2']:.3f}  adj_R^2 = {full['adj_r2']:.3f}  features = {full['n_features']}")
    print(f"  Sample VIFs (first 8): {list(full['vif_sample'].items())[:8]}")
    print(f"\nPer-skill ROI: {len(roi)} skills estimated; saved -> data/skill_roi.csv")
    print(roi.head(15)[["skill", "pct_salary_impact", "p_value", "users"]].to_string(index=False))

    # plot top 15 and bottom 15
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, data, title in [
        (axes[0], roi.head(15), "Top 15 skills by salary impact"),
        (axes[1], roi.tail(15).iloc[::-1], "Bottom 15 skills by salary impact"),
    ]:
        ax.barh(data["skill"], data["pct_salary_impact"],
                color=["#4C72B0" if v >= 0 else "#C44E52"
                       for v in data["pct_salary_impact"]])
        ax.set_xlabel("Salary impact (%) - per-skill MLR, control = work exp + country")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(ASSETS / "03_skill_roi.png", dpi=140)
    plt.close(fig)
    print(f"\nFigure saved -> {ASSETS / '03_skill_roi.png'}")


if __name__ == "__main__":
    main()
