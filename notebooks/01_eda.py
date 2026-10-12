"""Notebook 01 - Exploratory data analysis.

Run:  python -m notebooks.01_eda
Saves figures to assets/ and prints a summary to stdout.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def main() -> None:
    df = pd.read_parquet(DATA / "processed.parquet")
    print(f"processed shape: {df.shape}\ncolumns: {df.shape[1]}")
    print(df[["WorkExp", "YearsCode", "CompTotal", "log_comp",
              "JobSat_num", "JobSat_high", "ai_user"]].describe().round(2))

    # ---- 1. Compensation distribution (histogram) ----------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = df.loc[df["comp_inlier"] == 1, "CompTotal"].dropna()
    sub = sub[sub.between(sub.quantile(0.01), sub.quantile(0.99))]
    ax.hist(sub, bins=60, color="#4C72B0", edgecolor="white")
    ax.set_title("Annual compensation (inliers, 1-99% trimmed)")
    ax.set_xlabel("CompTotal (raw currency)")
    ax.set_ylabel("Respondents")
    fig.tight_layout()
    fig.savefig(ASSETS / "01_comp_hist.png", dpi=140)
    plt.close(fig)

    # ---- 2. Job-satisfaction distribution (bar) ------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.5))
    js = df["JobSat_num"].value_counts().sort_index()
    labels = {1: "Very dissatisfied", 2: "Slightly dissatisfied",
              3: "Neutral", 4: "Slightly satisfied", 5: "Very satisfied"}
    ax.bar(js.index, js.values, color="#55A868", edgecolor="white")
    ax.set_xticks(list(js.index))
    ax.set_xticklabels([labels[i] for i in js.index], rotation=20, ha="right")
    ax.set_title("Overall job satisfaction (1-5)")
    ax.set_ylabel("Respondents")
    fig.tight_layout()
    fig.savefig(ASSETS / "01_jobsat_bar.png", dpi=140)
    plt.close(fig)

    # ---- 3. Scatter: WorkExp vs log_comp ------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sub = df[(df["comp_inlier"] == 1) & df["WorkExp"].notna()].sample(
        min(8000, df.shape[0]), random_state=0)
    ax.scatter(sub["WorkExp"], sub["log_comp"], s=4, alpha=0.25, color="#4C72B0")
    # simple trend
    coef = np.polyfit(sub["WorkExp"], sub["log_comp"], 1)
    xs = np.linspace(0, 50, 50)
    ax.plot(xs, coef[0] * xs + coef[1], color="black", lw=2,
            label=f"slope = {coef[0]:.3f}")
    ax.set_xlabel("Years of professional experience")
    ax.set_ylabel("log(1 + compensation)")
    ax.set_title("Compensation rises with experience (cross-section)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "01_scatter_exp_comp.png", dpi=140)
    plt.close(fig)

    # ---- 4. AI user share (pie) ---------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4))
    counts = df["ai_user"].value_counts()
    ax.pie(counts, labels=["Non-user", "AI user"], autopct="%1.1f%%",
           colors=["#C44E52", "#4C72B0"], startangle=90)
    ax.set_title("Do you currently use AI tools in development?")
    fig.tight_layout()
    fig.savefig(ASSETS / "01_ai_pie.png", dpi=140)
    plt.close(fig)

    # ---- 5. Box: log_comp by experience bucket -------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    df["exp_bucket"] = pd.cut(df["WorkExp"],
                              bins=[-1, 2, 5, 10, 20, 100],
                              labels=["0-2", "3-5", "6-10", "11-20", "20+"])
    sub = df[(df["comp_inlier"] == 1) & df["exp_bucket"].notna()]
    sns.boxplot(data=sub, x="exp_bucket", y="log_comp", ax=ax, color="#4C72B0")
    ax.set_xlabel("Years of professional experience")
    ax.set_ylabel("log(1 + compensation)")
    ax.set_title("Compensation distribution by experience bucket")
    fig.tight_layout()
    fig.savefig(ASSETS / "01_box_exp_comp.png", dpi=140)
    plt.close(fig)

    # ---- 6. Correlation heatmap of top numeric features ----------------------
    top_skill_cols = [c for c in df.columns
                      if c.startswith(("lang_", "db_", "plat_"))
                      and df[c].nunique() == 2]
    top_skill_cols = df[top_skill_cols].sum().sort_values(ascending=False).head(8).index.tolist()
    num = df[["WorkExp", "YearsCode", "log_comp", "JobSat_num", "ai_user"] + top_skill_cols]
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, vmin=-0.4, vmax=0.4, center=0, cmap="RdBu_r", ax=ax,
                annot=True, fmt=".2f", annot_kws={"size": 7})
    ax.set_title("Correlation heatmap (key numeric + top-8 most common skills)")
    fig.tight_layout()
    fig.savefig(ASSETS / "01_corr_heatmap.png", dpi=140)
    plt.close(fig)

    print("\nEDA figures written to", ASSETS)


if __name__ == "__main__":
    main()
