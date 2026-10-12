"""Notebook 04 - Logistic regression (Module 4).

Delegates to src.classification.run(); plots ROC + confusion matrix.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classification import run  # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib
import pandas as pd

DATA = ROOT / "data"
ASSETS = ROOT / "assets"


def plot_roc_and_cm() -> None:
    df = pd.read_parquet(DATA / "processed.parquet")
    sk = [c for c in df.columns if c.startswith(("lang_", "db_", "plat_", "web_", "devenv_", "aimodel_"))]
    top = df[sk].sum().sort_values(ascending=False).head(30).index.tolist()
    feats = ["WorkExp", "ai_user"] + top
    sub = df.dropna(subset=["JobSat_high"] + feats).copy()
    sub["JobSat_high"] = sub["JobSat_high"].astype(int)
    X = sub[feats].astype(float).values
    y = sub["JobSat_high"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(yte, proba, ax=axes[0], name="Logistic (satisfaction)")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set_title("ROC curve")
    # confusion matrix at Youden threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(yte, proba)
    opt = thr[int(np.argmax(tpr - fpr))]
    yhat = (proba >= opt).astype(int)
    cm = confusion_matrix(yte, yhat)
    im = axes[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        axes[1].text(j, i, f"{v:,}", ha="center", va="center", color="black")
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Not satisfied", "Satisfied"]); axes[1].set_yticklabels(["Not satisfied", "Satisfied"])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].set_title(f"Confusion matrix @ threshold = {opt:.2f}")
    fig.tight_layout()
    fig.savefig(ASSETS / "04_roc_cm.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    metrics = run()
    plot_roc_and_cm()
    print("\nLogistic regression metrics:")
    print(json.dumps({k: v for k, v in metrics.items() if k != "top_features_by_abs_coef"},
                     indent=2, default=str))
    print(f"\nTop features by |coef|:")
    for f in metrics["top_features_by_abs_coef"][:10]:
        print(f"  {f['feature']:<30} {f['coef']:+.3f}")
