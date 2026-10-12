"""Notebook 05 - Clustering + PCA (Module 5).

Delegates to src.cluster.run(); plots PCA scatter coloured by cluster and the
hierarchical dendrogram (already saved by src.cluster).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cluster import run as run_cluster  # noqa: E402

DATA = ROOT / "data"
ASSETS = ROOT / "assets"


def pca_scatter() -> None:
    bundle = joblib.load(DATA / "cluster_model.joblib")
    Xp = bundle["Xp"]
    labels = bundle["labels"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = plt.get_cmap("tab10")
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(Xp[m, 0], Xp[m, 1], s=4, alpha=0.35, color=cmap(int(c) % 10), label=f"Persona {c}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Developer personas in PCA skill-space")
    ax.legend(markerscale=4, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(ASSETS / "05_pca_scatter.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    meta = run_cluster()
    pca_scatter()
    print("\nClustering summary:")
    print(json.dumps({k: v for k, v in meta.items() if k != "personas"}, indent=2, default=str))
    print("\nPersonas (top 5 skills each):")
    for p in meta["personas"]:
        print(f"  cluster {p['cluster']}  n={p['n']:,}  ({p['share']:.1%})  ->  "
              + ", ".join(s["skill"] for s in p["top_skills"][:5]))
