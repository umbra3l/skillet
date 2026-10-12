"""Module 7a: Exploratory Factor Analysis on the binary skill matrix.

NOTE: ``factor_analyzer`` 0.5.1 calls ``sklearn.utils.check_array`` with the
legacy ``force_all_finite`` keyword, which scikit-learn >= 1.6 removed (renamed
to ``ensure_all_finite``).  A small compatibility shim is installed below so the
module works on the current environment without pinning/downgrading sklearn.


Complements the PCA in ``src.cluster``: PCA maximises variance explained by
orthogonal components, whereas factor analysis models the *latent* structure
that generates the observed skill co-occurrences.  We fit the same 86-skill
matrix the clustering uses (skills known by >= 5% of the sample), determine the
number of factors with the Kaiser criterion (eigenvalue > 1) checked against a
parallel-analysis baseline, and rotate with Varimax for interpretability.

Outputs:
  - data/factor_loadings.csv : factor x skill loadings (rotated)
  - data/fa_meta.json        : n_factors, variance explained, method notes
  - assets/factor_scree.png  : scree plot of eigenvalues + Kaiser cutoff
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
from factor_analyzer import FactorAnalyzer  # noqa: E402

# --- sklearn >= 1.6 compatibility shim for factor_analyzer 0.5.1 ----------
import factor_analyzer.factor_analyzer as _fa_mod  # noqa: E402
import inspect  # noqa: E402

from sklearn.utils import check_array as _sk_check_array  # noqa: E402

if "force_all_finite" not in inspect.signature(_sk_check_array).parameters:
    def _check_array_compat(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _sk_check_array(*args, **kwargs)

    _fa_mod.check_array = _check_array_compat

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed.parquet"
MODEL_PATH = ROOT / "data" / "cluster_model.joblib"
OUT_CSV = ROOT / "data" / "factor_loadings.csv"
OUT_META = ROOT / "data" / "fa_meta.json"
ASSETS = ROOT / "assets"

KAISER_CUTOFF = 1.0
N_PARALLEL_ITER = 50


def _random_eigenvalues(n_rows: int, n_cols: int, n_iter: int,
                        seed: int = 42) -> np.ndarray:
    """Mean eigenvalues of correlation matrices from column-permuted data.

    This is the random-data baseline used for a parallel-analysis-style check
    of how many factors retain signal beyond chance.
    """
    rng = np.random.default_rng(seed)
    eig = np.empty((n_iter, n_cols))
    for i in range(n_iter):
        rand = rng.integers(0, 2, size=(n_rows, n_cols)).astype(float)
        # Permute each column independently to destroy cross-skill structure
        rand = np.apply_along_axis(rng.permutation, 0, rand)
        corr = np.corrcoef(rand, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        eig[i] = np.linalg.eigvalsh(corr)
    return eig.mean(axis=0)


def run(n_factors: int | None = None) -> dict:
    """Run exploratory factor analysis on the clustering skill matrix.

    Parameters
    ----------
    n_factors:
        Override the Kaiser/parallel-analysis choice.  When ``None`` the number
        of factors is set to the Kaiser count (eigenvalue > 1).
    """
    ASSETS.mkdir(exist_ok=True)
    bundle = joblib.load(MODEL_PATH)
    skills = list(bundle["skills"])
    df = pd.read_parquet(PROCESSED)
    X = df[skills].astype(float).values
    print(f"loaded skill matrix: {X.shape}  ({len(skills)} skills)")

    # --- Correlation structure & Kaiser criterion -------------------------
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    eigvals = np.sort(np.linalg.eigvalsh(corr))[::-1]
    kaiser_k = int(np.sum(eigvals > KAISER_CUTOFF))

    # Parallel-analysis-style baseline: retain factors whose eigenvalue beats
    # the mean eigenvalue of random (permuted) data at the same rank.
    rand_eig = _random_eigenvalues(X.shape[0], X.shape[1], N_PARALLEL_ITER)
    rand_eig = np.sort(rand_eig)[::-1]
    pa_k = int(np.sum(eigvals > rand_eig))

    chosen = int(n_factors) if n_factors is not None else kaiser_k
    chosen = max(1, chosen)
    print(f"  Kaiser criterion (eig > {KAISER_CUTOFF}): {kaiser_k} factors")
    print(f"  parallel-analysis baseline:            {pa_k} factors")
    print(f"  -> fitting {chosen} factors (varimax rotation)")

    # --- Fit FactorAnalyzer ----------------------------------------------
    fa = FactorAnalyzer(n_factors=chosen, rotation="varimax", method="principal")
    fa.fit(X)

    loadings = pd.DataFrame(
        fa.loadings_, index=skills,
        columns=[f"Factor{i + 1}" for i in range(chosen)],
    )
    variance, prop_var, cum_var = fa.get_factor_variance()
    loadings.to_csv(OUT_CSV, float_format="%.4f")

    # --- Scree plot -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    idx = np.arange(1, len(eigvals) + 1)
    ax.plot(idx, eigvals, marker="o", ms=4, lw=1.3, color="#1f77b4",
            label="Eigenvalue (observed)")
    ax.plot(idx, rand_eig, ls="--", lw=1.1, color="#7f7f7f",
            label=f"Parallel analysis ({N_PARALLEL_ITER} perms)")
    ax.axhline(KAISER_CUTOFF, ls=":", lw=1.4, color="#d62728",
               label=f"Kaiser cutoff (eigenvalue = {KAISER_CUTOFF})")
    for i in range(min(5, len(eigvals))):
        ax.annotate(f"{eigvals[i]:.2f}", (idx[i], eigvals[i]),
                    textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xlabel("Factor index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree plot of the 86-skill correlation matrix "
                 f"(Kaiser retains {kaiser_k} factors)")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(ASSETS / "factor_scree.png", dpi=140)
    plt.close(fig)

    # --- Top skills per factor -------------------------------------------
    top_by_factor = {}
    for f in loadings.columns:
        top = loadings[f].sort_values(ascending=False).head(5)
        top_by_factor[f] = [
            {"skill": s, "loading": float(v)} for s, v in top.items()
        ]

    meta = {
        "n_skills": int(len(skills)),
        "n_samples": int(X.shape[0]),
        "n_factors": int(chosen),
        "kaiser_factors": kaiser_k,
        "parallel_analysis_factors": pa_k,
        "rotation": "varimax",
        "method": "principal factor extraction on the Pearson correlation matrix",
        "kaiser_cutoff": KAISER_CUTOFF,
        "parallel_analysis_iterations": N_PARALLEL_ITER,
        "eigenvalues_observed": [float(e) for e in eigvals],
        "eigenvalues_parallel_baseline": [float(e) for e in rand_eig],
        "variance_explained": [float(v) for v in variance],
        "proportion_variance_explained": [float(v) for v in prop_var],
        "cumulative_variance_explained": [float(v) for v in cum_var],
        "cumulative_variance_total": float(cum_var[-1]),
        "top_skills_per_factor": top_by_factor,
        "notes": (
            "Binary skill matrix (skills known by >= 5% of sample), identical "
            "to the K-means clustering input. Number of factors follows the "
            "Kaiser criterion (eigenvalue > 1); the parallel-analysis baseline "
            "is reported for comparison. Varimax rotation applied for "
            "interpretability of skill groupings."
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, default=str))

    # --- Console summary --------------------------------------------------
    print(f"\nsaved -> {OUT_CSV}, {OUT_META}, {ASSETS / 'factor_scree.png'}")
    print(f"cumulative variance explained by {chosen} factors: "
          f"{cum_var[-1]:.2%}")
    print("\nTop 5 skills per factor:")
    for f, skills_top in top_by_factor.items():
        print(f"  {f}: "
              + ", ".join(f"{s['skill']} ({s['loading']:.2f})" for s in skills_top))
    return meta


# Explicit public name expected by notebooks / callers.
run_factor = run


if __name__ == "__main__":
    run()
