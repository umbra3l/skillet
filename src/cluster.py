"""Module 5: PCA + K-means + Hierarchical clustering on the skill space.

  - Standardize the binary skill matrix
  - PCA: keep components explaining >= 80% of variance
  - K-means: choose k by silhouette score over a small grid
  - Hierarchical: dendrogram on a 1500-row subsample (for visualization)
  - Personas: top distinguishing skills per cluster (prevalence lift vs overall)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed.parquet"
OUT_DIR = ROOT / "data"
ASSETS = ROOT / "assets"

MODEL_PATH = OUT_DIR / "cluster_model.joblib"
META_PATH = OUT_DIR / "cluster_meta.json"

SKILL_PREFIXES = ("lang_", "db_", "plat_", "web_", "devenv_", "aimodel_")


def run(min_k: int = 3, max_k: int = 8, dendro_n: int = 1500, pca_var: float = 0.80) -> dict:
    OUT_DIR.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    df = pd.read_parquet(PROCESSED)
    print(f"loaded {df.shape}")

    skills = [c for c in df.columns if any(c.startswith(p) for p in SKILL_PREFIXES)
              and df[c].nunique() == 2]
    X = df[skills].astype(float).values
    print(f"  skill matrix: {X.shape}")

    sc = StandardScaler(with_mean=True).fit(X)
    Xs = sc.transform(X)

    pca = PCA(n_components=pca_var, random_state=42).fit(Xs)
    Xp = pca.transform(Xs)
    print(f"  PCA: {pca.n_components_} components explain "
          f"{pca.explained_variance_ratio_.sum():.2%} of variance")

    # Pick k via silhouette
    best_k, best_sil = min_k, -1
    sil_scores = {}
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xp)
        s = float(silhouette_score(Xp, km.labels_))
        sil_scores[k] = s
        if s > best_sil:
            best_sil, best_k = s, k
    print(f"  silhouette: {sil_scores}  -> best k = {best_k}  (silhouette = {best_sil:.3f})")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xp)
    labels = km.labels_

    # Hierarchical on a subsample (for the dendrogram figure)
    rng = np.random.default_rng(42)
    idx = rng.choice(Xp.shape[0], size=min(dendro_n, Xp.shape[0]), replace=False)
    Z = linkage(Xp[idx], method="ward")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z, truncate_mode="lastp", p=20, leaf_rotation=90, ax=ax)
        ax.set_title("Hierarchical clustering (Ward linkage) on PCA skill components")
        ax.set_ylabel("distance")
        fig.tight_layout()
        fig.savefig(ASSETS / "dendrogram.png", dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"  dendrogram skipped: {e}")

    # Personas: top skills by lift (prevalence_in_cluster / prevalence_overall)
    overall = X.mean(axis=0)
    personas = []
    for c in range(best_k):
        mask = labels == c
        if mask.sum() < 50:
            continue
        prev = X[mask].mean(axis=0)
        lift = prev / np.maximum(overall, 1e-6)
        top_idx = np.argsort(-lift)[:8]
        personas.append({
            "cluster": int(c),
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "top_skills": [
                {"skill": skills[i], "prevalence": float(prev[i]),
                 "lift": float(lift[i])}
                for i in top_idx
            ],
        })

    joblib.dump({
        "scaler": sc, "pca": pca, "kmeans": km, "skills": skills,
        "labels": labels, "Xp": Xp,
    }, MODEL_PATH)

    meta = {
        "n_samples": int(X.shape[0]),
        "n_skills": int(X.shape[1]),
        "pca_components": int(pca.n_components_),
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "silhouette_by_k": sil_scores,
        "best_k": best_k,
        "best_silhouette": best_sil,
        "personas": personas,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, default=str))
    print(f"  saved -> {MODEL_PATH}, {META_PATH}")
    return meta


if __name__ == "__main__":
    m = run()
    print(json.dumps({k: v for k, v in m.items() if k != "personas"}, indent=2))
    print("\nPersonas:")
    for p in m["personas"]:
        print(f"  cluster {p['cluster']} (n={p['n']}, share={p['share']:.2%}): "
              + ", ".join(s["skill"] for s in p["top_skills"][:5]))
