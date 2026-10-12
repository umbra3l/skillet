"""Notebook 04 - Classification models (Module 4).

Delegates to src.classification: the logistic baseline (`run`), then a
Decision Tree and Gaussian Naive Bayes (`run_tree_and_nb`) on the same design
matrix and split.  Plots the logistic ROC + confusion matrix, a side-by-side
ROC/AUC comparison across all three models, and the tree's feature importances.
"""
from __future__ import annotations

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

from src.classification import _modeling_set, run, run_tree_and_nb  # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, confusion_matrix, roc_auc_score, roc_curve
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


def _test_split_and_probas() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Rebuild the exact train/test split and return test-set P(y=1) for all 3 models.

    Loads the saved joblib bundles so the plotted curves are the *same* models
    that produced the reported metrics (no retraining).
    """
    _, X, y = _modeling_set()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    log = joblib.load(DATA / "satisfaction_model.joblib")
    dt = joblib.load(DATA / "satisfaction_model_dt.joblib")
    nb = joblib.load(DATA / "satisfaction_model_nb.joblib")
    probas = {
        "Logistic": log["model"].predict_proba(log["scaler"].transform(Xte))[:, 1],
        "Decision tree": dt["model"].predict_proba(Xte)[:, 1],
        "Naive Bayes": nb["model"].predict_proba(Xte)[:, 1],
    }
    return yte, probas


def plot_roc_compare(yte: np.ndarray, probas: dict[str, np.ndarray]) -> None:
    """2-panel comparison: overlaid ROC curves + an AUC bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    aucs: dict[str, float] = {}
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(yte, p)
        aucs[name] = float(roc_auc_score(yte, p))
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {aucs[name]:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves - satisfaction target")
    axes[0].legend(loc="lower right")

    names = list(aucs)
    vals = [aucs[n] for n in names]
    bars = axes[1].bar(names, vals, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].axhline(0.5, ls="--", color="grey")
    axes[1].set_ylim(min(vals) - 0.03, max(vals) + 0.03)
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC comparison")
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(ASSETS / "04_roc_compare.png", dpi=140)
    plt.close(fig)


def plot_tree_importance(tree_metrics: dict) -> None:
    """Horizontal bar chart of the tree's top-15 Gini importances."""
    top = tree_metrics["top_features_by_importance"][::-1]
    feats = [t["feature"] for t in top]
    imps = [t["importance"] for t in top]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(feats, imps, color="#DD8452")
    ax.set_xlabel("Gini importance")
    ax.set_title("Decision tree - top-15 feature importances")
    fig.tight_layout()
    fig.savefig(ASSETS / "04_tree_importance.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    # 1) logistic baseline (unchanged) -> saves flat metrics + model
    run()
    # 2) tree + NB on the same split -> appends "models" block
    models = run_tree_and_nb()
    # 3) figures
    plot_roc_and_cm()
    yte, probas = _test_split_and_probas()
    plot_roc_compare(yte, probas)
    plot_tree_importance(models["decision_tree"])

    print("\nModel comparison (Youden-optimal threshold):")
    print(f"{'model':<16}{'AUC':>9}{'F1':>9}{'accuracy':>10}")
    for key, label in [("logistic", "Logistic"),
                       ("decision_tree", "Decision tree"),
                       ("naive_bayes", "Naive Bayes")]:
        r = models[key]
        print(f"{label:<16}{r['auc']:>9.3f}{r['f1_at_opt']:>9.3f}{r['accuracy_at_opt']:>10.3f}")

    print("\nTop-10 decision-tree features by importance:")
    for f in models["decision_tree"]["top_features_by_importance"][:10]:
        print(f"  {f['feature']:<30} {f['importance']:.4f}")
