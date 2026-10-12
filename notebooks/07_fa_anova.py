"""Notebook 07 - Factor Analysis + one-way ANOVA across personas (Module 7).

Delegates to src.fa.run_factor and src.anova.run_anova.  The factor analysis
compares latent skill structure against the PCA in src.cluster; the ANOVA tests
whether log compensation differs across the three K-means personas.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anova import run_anova  # noqa: E402
from src.fa import run_factor  # noqa: E402

if __name__ == "__main__":
    print("=" * 72)
    print("Factor Analysis on the 86-skill matrix")
    print("=" * 72)
    fa_meta = run_factor()
    print(f"\nchosen n_factors = {fa_meta['n_factors']} "
          f"(Kaiser={fa_meta['kaiser_factors']}, "
          f"parallel={fa_meta['parallel_analysis_factors']}); "
          f"cumulative variance = {fa_meta['cumulative_variance_total']:.2%}")

    print("\n" + "=" * 72)
    print("One-way ANOVA of log_comp across personas")
    print("=" * 72)
    anova = run_anova()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"FA:    {fa_meta['n_factors']} factors, "
          f"{fa_meta['cumulative_variance_total']:.2%} cumulative variance")
    print(f"ANOVA: {anova['headline']}")
