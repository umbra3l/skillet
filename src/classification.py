"""Module 4: Binary Logistic Regression + Decision Tree + Gaussian NB.

  Target : JobSat_high  (1 = JobSat >= 7 on the 0-10 scale)
  Features: WorkExp + ai_user + RemoteWork dummies + OrgSize + top-K skills
  Eval   : ROC curve, AUC, optimal threshold (Youden's J), confusion matrix

`run()` is the logistic baseline.  `run_tree_and_nb()` refits a Decision Tree
and a Gaussian Naive Bayes on the *identical* design matrix and 75/25
stratified split so all three models are directly comparable.

  Output : data/satisfaction_model.joblib, data/satisfaction_model_dt.joblib,
           data/satisfaction_model_nb.joblib, data/satisfaction_metrics.json
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
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed.parquet"
OUT_DIR = ROOT / "data"
MODEL_PATH = OUT_DIR / "satisfaction_model.joblib"
META_PATH = ROOT / "data" / "satisfaction_metrics.json"
ASSETS = ROOT / "assets"

SKILL_PREFIXES = ("lang_", "db_", "plat_", "web_", "devenv_", "aimodel_")
REMOTE_DUMMIES = [
    "rw_Remote",
    "rw_Hybrid_remote_heavy",
    "rw_Hybrid_inperson_heavy",
    "rw_In_person",
    "rw_Flexible",
]
ORG_DUMMIES = ["org_Small", "org_Medium", "org_Large", "org_Very_large"]


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Build RemoteWork and OrgSize dummies in place."""
    rw = df["RemoteWork"].fillna("")
    df["rw_Remote"] = (rw == "Remote").astype(int)
    df["rw_Hybrid_remote_heavy"] = rw.str.contains("Hybrid.*remote", case=False, regex=True).astype(int)
    df["rw_Hybrid_inperson_heavy"] = rw.str.contains("Hybrid.*in-person", case=False, regex=True).astype(int)
    df["rw_In_person"] = (rw == "In-person").astype(int)
    df["rw_Flexible"] = rw.str.contains("flexible", case=False, regex=True).astype(int)

    org = df["OrgSize"].fillna("") if "OrgSize" in df.columns else pd.Series("", index=df.index)
    df["org_Small"] = org.str.contains(r"2-9|10-19|20-99", case=False, regex=True).astype(int)
    df["org_Medium"] = org.str.contains(r"100-499|500-999", case=False, regex=True).astype(int)
    df["org_Large"] = org.str.contains(r"1,000-4,999|5,000-9,999", case=False, regex=True).astype(int)
    df["org_Very_large"] = org.str.contains(r"10,000 or more", case=False, regex=True).astype(int)
    return df


def _modeling_set(top_n_skills: int = 30) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load processed data and return (feature names, X, y) for JobSat_high.

    Shared by the logistic baseline and the tree/NB models so every classifier
    is trained on exactly the same rows and columns.
    """
    df = pd.read_parquet(PROCESSED)
    df = _engineer(df)
    print(f"loaded {df.shape}")

    skill_cols = [c for c in df.columns if any(c.startswith(p) for p in SKILL_PREFIXES)
                  and df[c].nunique() == 2]
    top = (df[skill_cols].sum().sort_values(ascending=False).head(top_n_skills).index.tolist())

    feat_cols = ["WorkExp", "ai_user"] + REMOTE_DUMMIES + ORG_DUMMIES + top
    sub = df.dropna(subset=["JobSat_high"] + feat_cols).copy()
    sub["JobSat_high"] = sub["JobSat_high"].astype(int)
    print(f"  modeling set: {sub.shape}; positive rate: {sub['JobSat_high'].mean():.3f}")

    X = sub[feat_cols].astype(float).values
    y = sub["JobSat_high"].values
    return feat_cols, X, y


def _threshold_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """ROC/AUC + Youden-optimal threshold + accuracy/precision/recall/F1 + confusion matrix."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    auc = float(roc_auc_score(y_true, proba))
    opt = float(thr[int(np.argmax(tpr - fpr))])
    yhat = (proba >= opt).astype(int)
    return {
        "auc": auc,
        "optimal_threshold_youden": opt,
        "accuracy_at_opt": float(accuracy_score(y_true, yhat)),
        "precision_at_opt": float(precision_score(y_true, yhat, zero_division=0)),
        "recall_at_opt": float(recall_score(y_true, yhat, zero_division=0)),
        "f1_at_opt": float(f1_score(y_true, yhat, zero_division=0)),
        "confusion_matrix_at_opt": confusion_matrix(y_true, yhat).tolist(),
    }


def run(top_n_skills: int = 30) -> dict:
    OUT_DIR.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    feat_cols, X, y = _modeling_set(top_n_skills)

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

    coef_map = {f: float(c) for f, c in zip(feat_cols, clf.coef_[0])}
    pd.DataFrame([{"feature": k, "coef": v} for k, v in coef_map.items()]) \
        .to_csv(OUT_DIR / "satisfaction_coefs.csv", index=False)

    joblib.dump({"model": clf, "scaler": sc, "features": feat_cols}, MODEL_PATH)
    META_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"  saved model -> {MODEL_PATH}")
    return metrics


def run_tree_and_nb(top_n_skills: int = 30) -> dict:
    """Fit a Decision Tree and a Gaussian Naive Bayes on the logistic design matrix.

    Reuses `_modeling_set()` and the same 75/25 stratified split (random_state=42)
    as `run()`, so ROC/AUC/F1 are directly comparable across the three models.
    Writes data/satisfaction_model_dt.joblib and data/satisfaction_model_nb.joblib,
    and appends a "models" block (logistic + decision tree + naive bayes) to
    data/satisfaction_metrics.json.
    """
    OUT_DIR.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    feat_cols, X, y = _modeling_set(top_n_skills)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    dt = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42).fit(Xtr, ytr)
    nb = GaussianNB().fit(Xtr, ytr)

    dt_metrics = _threshold_metrics(yte, dt.predict_proba(Xte)[:, 1])
    nb_metrics = _threshold_metrics(yte, nb.predict_proba(Xte)[:, 1])

    dt_metrics["top_features_by_importance"] = sorted(
        [{"feature": f, "importance": float(imp)}
         for f, imp in zip(feat_cols, dt.feature_importances_)],
        key=lambda r: r["importance"], reverse=True)[:15]

    print(f"  tree: AUC = {dt_metrics['auc']:.3f}  opt threshold = "
          f"{dt_metrics['optimal_threshold_youden']:.3f}  F1 = {dt_metrics['f1_at_opt']:.3f}")
    print(f"  NB  : AUC = {nb_metrics['auc']:.3f}  opt threshold = "
          f"{nb_metrics['optimal_threshold_youden']:.3f}  F1 = {nb_metrics['f1_at_opt']:.3f}")

    joblib.dump({"model": dt, "features": feat_cols}, OUT_DIR / "satisfaction_model_dt.joblib")
    joblib.dump({"model": nb, "features": feat_cols}, OUT_DIR / "satisfaction_model_nb.joblib")

    # Keep the logistic metrics saved by run() at the top level, and nest all
    # three models under "models" for a self-contained comparison.
    existing = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    existing.pop("models", None)
    models = {
        "logistic": existing,
        "decision_tree": dt_metrics,
        "naive_bayes": nb_metrics,
    }
    META_PATH.write_text(json.dumps({**existing, "models": models}, indent=2, default=str))
    print(f"  saved models -> {OUT_DIR / 'satisfaction_model_dt.joblib'}, "
          f"{OUT_DIR / 'satisfaction_model_nb.joblib'}")
    return models


if __name__ == "__main__":
    m = run()
    models = run_tree_and_nb()
    print(json.dumps({k: v for k, v in m.items() if k != "top_features_by_abs_coef"}, indent=2))
    print("\nModel comparison (Youden-optimal threshold):")
    print(f"{'model':<16}{'AUC':>9}{'F1':>9}{'accuracy':>10}")
    for key, label in [("logistic", "Logistic"),
                       ("decision_tree", "Decision tree"),
                       ("naive_bayes", "Naive Bayes")]:
        r = models[key]
        print(f"{label:<16}{r['auc']:>9.3f}{r['f1_at_opt']:>9.3f}{r['accuracy_at_opt']:>10.3f}")
