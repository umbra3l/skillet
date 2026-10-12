"""Module 4: Binary Logistic Regression + ROC/confusion matrix.

  Target : JobSat_high  (1 = 'Slightly satisfied' or 'Very satisfied')
  Features: WorkExp + ai_user + top-K skills (selected by frequency)
  Eval   : ROC curve, AUC, optimal threshold (Youden's J), confusion matrix
  Output : data/satisfaction_model.joblib, data/satisfaction_metrics.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed.parquet"
OUT_DIR = ROOT / "data"
MODEL_PATH = OUT_DIR / "satisfaction_model.joblib"
META_PATH = OUT_DIR / "satisfaction_metrics.json"
ASSETS = ROOT / "assets"

SKILL_PREFIXES = ("lang_", "db_", "plat_", "web_", "devenv_", "aimodel_")


def run(top_n_skills: int = 30) -> dict:
    OUT_DIR.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    df = pd.read_parquet(PROCESSED)
    print(f"loaded {df.shape}")

    # Top-N most common skills
    skill_cols = [c for c in df.columns if any(c.startswith(p) for p in SKILL_PREFIXES)
                  and df[c].nunique() == 2]
    top = (df[skill_cols].sum().sort_values(ascending=False).head(top_n_skills).index.tolist())

    feat_cols = ["WorkExp", "ai_user"] + top
    sub = df.dropna(subset=["JobSat_high"] + feat_cols).copy()
    sub["JobSat_high"] = sub["JobSat_high"].astype(int)
    print(f"  modeling set: {sub.shape}; positive rate: {sub['JobSat_high'].mean():.3f}")

    X = sub[feat_cols].astype(float).values
    y = sub["JobSat_high"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Xtr_s, ytr)
    proba = clf.predict_proba(Xte_s)[:, 1]

    fpr, tpr, thr = roc_curve(yte, proba)
    auc = float(roc_auc_score(yte, proba))
    j = tpr - fpr
    opt = float(thr[int(np.argmax(j))])

    yhat = (proba >= opt).astype(int)
    cm = confusion_matrix(yte, yhat).tolist()
    metrics = {
        "n": int(len(y)),
        "n_features": int(X.shape[1]),
        "auc": auc,
        "optimal_threshold_youden": opt,
        "accuracy_at_opt": float(accuracy_score(yte, yhat)),
        "precision_at_opt": float(precision_score(yte, yhat, zero_division=0)),
        "recall_at_opt": float(recall_score(yte, yhat, zero_division=0)),
        "f1_at_opt": float(f1_score(yte, yhat, zero_division=0)),
        "confusion_matrix_at_opt": cm,
        "top_features_by_abs_coef": sorted(
            [{"feature": f, "coef": float(c)} for f, c in zip(feat_cols, clf.coef_[0])],
            key=lambda r: abs(r["coef"]), reverse=True)[:15],
    }
    print(f"  AUC = {auc:.3f}  opt threshold = {opt:.3f}  F1 = {metrics['f1_at_opt']:.3f}")

    # Save the per-skill satisfaction association too, for the recommender
    coef_map = {f: float(c) for f, c in zip(feat_cols, clf.coef_[0])}
    pd.DataFrame([{"feature": k, "coef": v} for k, v in coef_map.items()]) \
        .to_csv(OUT_DIR / "satisfaction_coefs.csv", index=False)

    joblib.dump({"model": clf, "scaler": sc, "features": feat_cols}, MODEL_PATH)
    META_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"  saved model -> {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    m = run()
    print(json.dumps({k: v for k, v in m.items() if k != "top_features_by_abs_coef"}, indent=2))
