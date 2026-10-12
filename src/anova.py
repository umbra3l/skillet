"""Module 7b: One-way ANOVA of log compensation across K-means personas.

Tests whether mean log compensation differs across the three developer
personas discovered by ``src.cluster`` (cluster labels in
``data/cluster_model.joblib`` under the ``labels`` key), with a Bonferroni
corrected pairwise post-hoc comparison.

Outputs:
  - data/anova_personas.json  : F, df, p, eta_squared, pairwise results
  - assets/anova_personas.png : box plot of log_comp by persona
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed.parquet"
MODEL_PATH = ROOT / "data" / "cluster_model.joblib"
META_PATH = ROOT / "data" / "cluster_meta.json"
OUT_JSON = ROOT / "data" / "anova_personas.json"
ASSETS = ROOT / "assets"

# Cluster id -> persona name (matches the top-skill personas in cluster_meta.json)
PERSONA_NAMES = {0: ".NET", 1: "Systems/enterprise", 2: "JS/web"}


def _persona_names() -> dict[int, str]:
    """Resolve cluster -> persona-name mapping from cluster_meta.json."""
    names = dict(PERSONA_NAMES)
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        for p in meta.get("personas", []):
            names[int(p["cluster"])] = names.get(int(p["cluster"]), f"cluster {p['cluster']}")
    return names


def run() -> dict:
    """One-way ANOVA of log_comp across the K-means personas."""
    from scipy import stats

    ASSETS.mkdir(exist_ok=True)
    bundle = joblib.load(MODEL_PATH)
    labels = np.asarray(bundle["labels"])
    df = pd.read_parquet(PROCESSED)
    log_comp = df["log_comp"].to_numpy()

    if len(labels) != len(log_comp):
        raise ValueError(f"label/row mismatch: {len(labels)} vs {len(log_comp)}")

    names = _persona_names()
    clusters = sorted(int(c) for c in np.unique(labels))
    groups = [log_comp[labels == c] for c in clusters]
    groups = [g[np.isfinite(g)] for g in groups]

    f_stat, p_val = stats.f_oneway(*groups)
    df1 = len(groups) - 1
    df2 = int(sum(len(g) for g in groups)) - len(groups)

    # eta^2 = SS_between / SS_total
    all_vals = np.concatenate(groups)
    grand = all_vals.mean()
    ss_between = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    ss_total = float(((all_vals - grand) ** 2).sum())
    eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")

    # Pairwise Welch t-tests with Bonferroni correction
    pairs = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            a, b = clusters[i], clusters[j]
            ga, gb = groups[i], groups[j]
            t, p = stats.ttest_ind(ga, gb, equal_var=False, nan_policy="omit")
            pairs.append({
                "cluster_a": a, "cluster_b": b,
                "persona_a": names.get(a, f"cluster {a}"),
                "persona_b": names.get(b, f"cluster {b}"),
                "mean_a": float(ga.mean()), "mean_b": float(gb.mean()),
                "mean_diff": float(ga.mean() - gb.mean()),
                "t": float(t), "p_value": float(p),
                "n_comparisons": len(clusters) * (len(clusters) - 1) // 2,
            })
    n_comp = len(pairs)
    for pr in pairs:
        pr["p_value_bonferroni"] = float(min(pr["p_value"] * n_comp, 1.0))

    # Headline pair = largest absolute mean difference
    headline_pair = max(pairs, key=lambda pr: abs(pr["mean_diff"]))
    diff = abs(headline_pair["mean_diff"])
    headline = (
        f"F({df1}, {df2}) = {f_stat:.2f}, p = {p_val:.3g}; "
        f"clusters {headline_pair['persona_a']} and {headline_pair['persona_b']} "
        f"differ by {diff:.2f} log-points "
        f"(p_adj = {headline_pair['p_value_bonferroni']:.3g})"
    )

    result = {
        "test": "one-way ANOVA",
        "dependent_variable": "log_comp",
        "factor": "kmeans_cluster (persona)",
        "groups": [
            {"cluster": c, "persona": names.get(c, f"cluster {c}"),
             "n": int(len(g)), "mean": float(g.mean()),
             "std": float(g.std(ddof=1)), "median": float(np.median(g))}
            for c, g in zip(clusters, groups)
        ],
        "F": float(f_stat),
        "df1": int(df1),
        "df2": int(df2),
        "p_value": float(p_val),
        "eta_squared": float(eta_sq),
        "pairwise": pairs,
        "pairwise_correction": "Bonferroni",
        "headline": headline,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))

    # --- Box plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    data = [g for g in groups]
    tick_labels = [f"{names.get(c, f'cluster {c}')}\n(n={len(g):,})"
                   for c, g in zip(clusters, groups)]
    bp = ax.boxplot(data, showmeans=True, patch_artist=True,
                    medianprops={"color": "black"})
    ax.set_xticks(range(1, len(tick_labels) + 1))
    ax.set_xticklabels(tick_labels)
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("log compensation (log_comp)")
    ax.set_title(
        "log compensation by persona (one-way ANOVA): "
        f"F({df1}, {df2}) = {f_stat:.2f}, p = {p_val:.3g}"
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "anova_personas.png", dpi=140)
    plt.close(fig)

    print(f"loaded {len(log_comp):,} rows; personas: "
          + ", ".join(f"{names.get(c, c)}={len(g):,}"
                      for c, g in zip(clusters, groups)))
    print(f"saved -> {OUT_JSON}, {ASSETS / 'anova_personas.png'}")
    print("group means (log_comp):")
    for c, g in zip(clusters, groups):
        print(f"  {names.get(c, f'cluster {c}'):>20}: "
              f"mean={g.mean():.3f}  sd={g.std(ddof=1):.3f}  n={len(g):,}")
    print("\nHeadline:")
    print("  " + headline)
    return result


# Explicit public name expected by notebooks / callers.
run_anova = run


if __name__ == "__main__":
    run()
